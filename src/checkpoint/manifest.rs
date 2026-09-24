//! The Parameter Manifest: the parameter names and shapes the Checkpoint's own configuration
//! describes, derived before any weight is read (ADR-0005).

use std::collections::BTreeMap;
use std::fmt::{self, Write as _};
use std::path::Path;
use std::str::FromStr;

use serde::Deserialize;

use crate::error::{Error, Result};

use crate::model_source::{AGENT_CONFIG_FILE, ENCODER_CONFIG_FILE};

/// The most layers either configuration may declare. Both drive a loop that inserts about fifteen
/// Manifest entries per layer, so a hostile or corrupt count is out-of-memory rather than an error
/// unless it is rejected before the loop.
const MAX_LAYERS: usize = 1024;

#[derive(Debug, Deserialize)]
struct EncoderConfig {
    hidden_size: usize,
    intermediate_size: usize,
    vocab_size: usize,
    num_hidden_layers: usize,
}

#[derive(Debug, Deserialize)]
struct AgentConfig {
    head_layers: usize,
    act_costs: BTreeMap<String, serde_json::Value>,
}

/// A dtype safetensors defines. The weights header carries the dtype as text; parsing it once is
/// what lets an expected dtype and an actual one be compared as values rather than as strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Dtype {
    Bool,
    U8,
    I8,
    F8E4M3,
    F8E5M2,
    U16,
    I16,
    F16,
    Bf16,
    U32,
    I32,
    F32,
    U64,
    I64,
    F64,
}

impl Dtype {
    /// The bytes one element of this dtype occupies.
    pub(super) fn bytes(self) -> u64 {
        match self {
            Self::Bool | Self::U8 | Self::I8 | Self::F8E4M3 | Self::F8E5M2 => 1,
            Self::U16 | Self::I16 | Self::F16 | Self::Bf16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
        }
    }

    /// The name safetensors spells this dtype with.
    fn name(self) -> &'static str {
        match self {
            Self::Bool => "BOOL",
            Self::U8 => "U8",
            Self::I8 => "I8",
            Self::F8E4M3 => "F8_E4M3",
            Self::F8E5M2 => "F8_E5M2",
            Self::U16 => "U16",
            Self::I16 => "I16",
            Self::F16 => "F16",
            Self::Bf16 => "BF16",
            Self::U32 => "U32",
            Self::I32 => "I32",
            Self::F32 => "F32",
            Self::U64 => "U64",
            Self::I64 => "I64",
            Self::F64 => "F64",
        }
    }
}

impl fmt::Display for Dtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A dtype name safetensors does not define.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct UnknownDtype;

impl FromStr for Dtype {
    type Err = UnknownDtype;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        Ok(match name {
            "BOOL" => Self::Bool,
            "U8" => Self::U8,
            "I8" => Self::I8,
            "F8_E4M3" => Self::F8E4M3,
            "F8_E5M2" => Self::F8E5M2,
            "U16" => Self::U16,
            "I16" => Self::I16,
            "F16" => Self::F16,
            "BF16" => Self::Bf16,
            "U32" => Self::U32,
            "I32" => Self::I32,
            "F32" => Self::F32,
            "U64" => Self::U64,
            "I64" => Self::I64,
            "F64" => Self::F64,
            _ => return Err(UnknownDtype),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Parameter {
    pub(super) dtype: Dtype,
    pub(super) shape: Vec<u64>,
}

/// The parameters a Checkpoint is expected to hold, by name.
#[derive(Debug, Default)]
pub(super) struct Manifest {
    pub(super) parameters: BTreeMap<String, Parameter>,
}

/// Derive the Parameter Manifest of the Checkpoint in `directory` from its own configuration.
pub(super) fn read(directory: &Path) -> Result<Manifest> {
    let encoder: EncoderConfig = read_json(&directory.join(ENCODER_CONFIG_FILE))?;
    let agent: AgentConfig = read_json(&directory.join(AGENT_CONFIG_FILE))?;
    Manifest::from_configs(directory, &encoder, &agent)
}

impl Manifest {
    /// The Parameter Manifest the two configurations describe (ADR-0005), built from the encoder
    /// dimensions, the decision-head depth and the action costs, so it cannot drift away from the
    /// architecture it describes. `directory` locates the configurations, so a dimension that
    /// cannot be a shape is reported against the file that carries it.
    fn from_configs(
        directory: &Path,
        encoder: &EncoderConfig,
        agent: &AgentConfig,
    ) -> Result<Self> {
        let encoder_config = directory.join(ENCODER_CONFIG_FILE);
        let agent_config = directory.join(AGENT_CONFIG_FILE);
        let hidden = dimension(&encoder_config, encoder.hidden_size)?;
        let intermediate = dimension(&encoder_config, encoder.intermediate_size)?;
        let vocab = dimension(&encoder_config, encoder.vocab_size)?;
        let layers = encoder.num_hidden_layers;
        if layers > MAX_LAYERS {
            return Err(Error::invalid_checkpoint(&encoder_config, "num_hidden_layers is too large"));
        }
        let head_layers = agent.head_layers;
        if head_layers > MAX_LAYERS {
            return Err(Error::invalid_checkpoint(&agent_config, "head_layers is too large"));
        }
        let doubled_intermediate = intermediate
            .checked_mul(2)
            .ok_or_else(|| Error::invalid_checkpoint(&encoder_config, "intermediate_size is too large"))?;
        let head_width = hidden
            .checked_mul(4)
            .ok_or_else(|| Error::invalid_checkpoint(&encoder_config, "hidden_size is too large"))?;
        // The query, key and value projection width, in the encoder and in a head layer.
        let tripled = hidden
            .checked_mul(3)
            .ok_or_else(|| Error::invalid_checkpoint(&encoder_config, "hidden_size is too large"))?;
        // The action head reads the hidden state plus the four Question Type features.
        let act_input = hidden
            .checked_add(4)
            .ok_or_else(|| Error::invalid_checkpoint(&encoder_config, "hidden_size is too large"))?;
        let actions = dimension(&agent_config, agent.act_costs.len().saturating_add(1))?;

        let mut manifest = Self::default();
        manifest.add("temperature", Dtype::F32, vec![3]);
        manifest.add("encoder.embeddings.norm.weight", Dtype::F16, vec![hidden]);
        manifest.add(
            "encoder.embeddings.tok_embeddings.weight",
            Dtype::F16,
            vec![vocab, hidden],
        );
        manifest.add("encoder.final_norm.weight", Dtype::F16, vec![hidden]);
        // One layer-prefix buffer for both loops; only the per-parameter key is its own String.
        let mut prefix = String::with_capacity(64);
        for layer in 0..layers {
            prefix.clear();
            let _ = write!(prefix, "encoder.layers.{layer}");
            if layer > 0 {
                manifest.add(
                    &format!("{prefix}.attn_norm.weight"),
                    Dtype::F16,
                    vec![hidden],
                );
            }
            manifest.add(
                &format!("{prefix}.attn.Wo.weight"),
                Dtype::F16,
                vec![hidden, hidden],
            );
            manifest.add(
                &format!("{prefix}.attn.Wqkv.weight"),
                Dtype::F16,
                vec![tripled, hidden],
            );
            manifest.add(
                &format!("{prefix}.mlp.Wi.weight"),
                Dtype::F16,
                vec![doubled_intermediate, hidden],
            );
            manifest.add(
                &format!("{prefix}.mlp.Wo.weight"),
                Dtype::F16,
                vec![hidden, intermediate],
            );
            manifest.add(
                &format!("{prefix}.mlp_norm.weight"),
                Dtype::F16,
                vec![hidden],
            );
        }
        for layer in 0..head_layers {
            prefix.clear();
            let _ = write!(prefix, "head.layers.{layer}");
            manifest.add(
                &format!("{prefix}.linear1.weight"),
                Dtype::F16,
                vec![head_width, hidden],
            );
            manifest.add(
                &format!("{prefix}.linear1.bias"),
                Dtype::F16,
                vec![head_width],
            );
            manifest.add(
                &format!("{prefix}.linear2.weight"),
                Dtype::F16,
                vec![hidden, head_width],
            );
            manifest.add(&format!("{prefix}.linear2.bias"), Dtype::F16, vec![hidden]);
            for norm in ["norm1", "norm2"] {
                manifest.add(&format!("{prefix}.{norm}.weight"), Dtype::F16, vec![hidden]);
                manifest.add(&format!("{prefix}.{norm}.bias"), Dtype::F16, vec![hidden]);
            }
            manifest.add(
                &format!("{prefix}.self_attn.in_proj_weight"),
                Dtype::F16,
                vec![tripled, hidden],
            );
            manifest.add(
                &format!("{prefix}.self_attn.in_proj_bias"),
                Dtype::F16,
                vec![tripled],
            );
            manifest.add(
                &format!("{prefix}.self_attn.out_proj.weight"),
                Dtype::F16,
                vec![hidden, hidden],
            );
            manifest.add(
                &format!("{prefix}.self_attn.out_proj.bias"),
                Dtype::F16,
                vec![hidden],
            );
        }
        manifest.add("type_emb.weight", Dtype::F16, vec![3, hidden]);
        manifest.add("scorer.0.weight", Dtype::F16, vec![hidden]);
        manifest.add("scorer.0.bias", Dtype::F16, vec![hidden]);
        manifest.add("scorer.1.weight", Dtype::F16, vec![hidden, hidden]);
        manifest.add("scorer.1.bias", Dtype::F16, vec![hidden]);
        manifest.add("scorer.3.weight", Dtype::F16, vec![1, hidden]);
        manifest.add("scorer.3.bias", Dtype::F16, vec![1]);
        manifest.add("act_head.0.weight", Dtype::F16, vec![256, act_input]);
        manifest.add("act_head.0.bias", Dtype::F16, vec![256]);
        manifest.add("act_head.2.weight", Dtype::F16, vec![actions, 256]);
        manifest.add("act_head.2.bias", Dtype::F16, vec![actions]);
        Ok(manifest)
    }

    fn add(&mut self, name: &str, dtype: Dtype, shape: Vec<u64>) {
        self.parameters
            .insert(name.to_string(), Parameter { dtype, shape });
    }
}

fn dimension(path: &Path, value: usize) -> Result<u64> {
    u64::try_from(value).map_err(|_| Error::invalid_checkpoint(path, "a dimension is too large"))
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let bytes = std::fs::read(path).map_err(|error| Error::io("read", path, error))?;
    serde_json::from_slice(&bytes).map_err(|error| Error::json(path.display().to_string(), error))
}
