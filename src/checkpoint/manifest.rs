//! The Parameter Manifest: the parameter names and shapes the Checkpoint's own configuration
//! describes, derived before any weight is read (ADR-0005).

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

use crate::error::{Error, Result};

use super::{AGENT_CONFIG, ENCODER_CONFIG, invalid};

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Parameter {
    pub(super) dtype: &'static str,
    pub(super) shape: Vec<u64>,
}

/// The parameters a Checkpoint is expected to hold, by name.
#[derive(Debug, Default)]
pub(super) struct Manifest {
    pub(super) parameters: BTreeMap<String, Parameter>,
}

/// Derive the Parameter Manifest of the Checkpoint in `directory` from its own configuration.
pub(super) fn read(directory: &Path) -> Result<Manifest> {
    let encoder: EncoderConfig = read_json(&directory.join(ENCODER_CONFIG))?;
    let agent: AgentConfig = read_json(&directory.join(AGENT_CONFIG))?;
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
    ) -> Result<Manifest> {
        let encoder_config = directory.join(ENCODER_CONFIG);
        let agent_config = directory.join(AGENT_CONFIG);
        let hidden = dimension(&encoder_config, encoder.hidden_size)?;
        let intermediate = dimension(&encoder_config, encoder.intermediate_size)?;
        let vocab = dimension(&encoder_config, encoder.vocab_size)?;
        let layers = encoder.num_hidden_layers;
        let doubled_intermediate = intermediate
            .checked_mul(2)
            .ok_or_else(|| invalid(&encoder_config, "intermediate_size is too large"))?;
        let head_width = hidden
            .checked_mul(4)
            .ok_or_else(|| invalid(&encoder_config, "hidden_size is too large"))?;
        // The query, key and value projection width, in the encoder and in a head layer.
        let tripled = hidden
            .checked_mul(3)
            .ok_or_else(|| invalid(&encoder_config, "hidden_size is too large"))?;
        // The action head reads the hidden state plus the four Question Type features.
        let act_input = hidden
            .checked_add(4)
            .ok_or_else(|| invalid(&encoder_config, "hidden_size is too large"))?;
        let actions = dimension(&agent_config, agent.act_costs.len().saturating_add(1))?;

        let mut manifest = Manifest::default();
        manifest.add("temperature", "F32", vec![3]);
        manifest.add("encoder.embeddings.norm.weight", "F16", vec![hidden]);
        manifest.add(
            "encoder.embeddings.tok_embeddings.weight",
            "F16",
            vec![vocab, hidden],
        );
        manifest.add("encoder.final_norm.weight", "F16", vec![hidden]);
        for layer in 0..layers {
            let prefix = format!("encoder.layers.{layer}");
            if layer > 0 {
                manifest.add(&format!("{prefix}.attn_norm.weight"), "F16", vec![hidden]);
            }
            manifest.add(
                &format!("{prefix}.attn.Wo.weight"),
                "F16",
                vec![hidden, hidden],
            );
            manifest.add(
                &format!("{prefix}.attn.Wqkv.weight"),
                "F16",
                vec![tripled, hidden],
            );
            manifest.add(
                &format!("{prefix}.mlp.Wi.weight"),
                "F16",
                vec![doubled_intermediate, hidden],
            );
            manifest.add(
                &format!("{prefix}.mlp.Wo.weight"),
                "F16",
                vec![hidden, intermediate],
            );
            manifest.add(&format!("{prefix}.mlp_norm.weight"), "F16", vec![hidden]);
        }
        for layer in 0..agent.head_layers {
            let prefix = format!("head.layers.{layer}");
            manifest.add(
                &format!("{prefix}.linear1.weight"),
                "F16",
                vec![head_width, hidden],
            );
            manifest.add(&format!("{prefix}.linear1.bias"), "F16", vec![head_width]);
            manifest.add(
                &format!("{prefix}.linear2.weight"),
                "F16",
                vec![hidden, head_width],
            );
            manifest.add(&format!("{prefix}.linear2.bias"), "F16", vec![hidden]);
            for norm in ["norm1", "norm2"] {
                manifest.add(&format!("{prefix}.{norm}.weight"), "F16", vec![hidden]);
                manifest.add(&format!("{prefix}.{norm}.bias"), "F16", vec![hidden]);
            }
            manifest.add(
                &format!("{prefix}.self_attn.in_proj_weight"),
                "F16",
                vec![tripled, hidden],
            );
            manifest.add(
                &format!("{prefix}.self_attn.in_proj_bias"),
                "F16",
                vec![tripled],
            );
            manifest.add(
                &format!("{prefix}.self_attn.out_proj.weight"),
                "F16",
                vec![hidden, hidden],
            );
            manifest.add(
                &format!("{prefix}.self_attn.out_proj.bias"),
                "F16",
                vec![hidden],
            );
        }
        manifest.add("type_emb.weight", "F16", vec![3, hidden]);
        manifest.add("scorer.0.weight", "F16", vec![hidden]);
        manifest.add("scorer.0.bias", "F16", vec![hidden]);
        manifest.add("scorer.1.weight", "F16", vec![hidden, hidden]);
        manifest.add("scorer.1.bias", "F16", vec![hidden]);
        manifest.add("scorer.3.weight", "F16", vec![1, hidden]);
        manifest.add("scorer.3.bias", "F16", vec![1]);
        manifest.add("act_head.0.weight", "F16", vec![256, act_input]);
        manifest.add("act_head.0.bias", "F16", vec![256]);
        manifest.add("act_head.2.weight", "F16", vec![actions, 256]);
        manifest.add("act_head.2.bias", "F16", vec![actions]);
        Ok(manifest)
    }

    fn add(&mut self, name: &str, dtype: &'static str, shape: Vec<u64>) {
        self.parameters
            .insert(name.to_string(), Parameter { dtype, shape });
    }
}

fn dimension(path: &Path, value: usize) -> Result<u64> {
    u64::try_from(value).map_err(|_| invalid(path, "a dimension is too large"))
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let bytes = std::fs::read(path).map_err(|error| Error::io("read", path, error))?;
    serde_json::from_slice(&bytes).map_err(|error| Error::json(path.display().to_string(), error))
}
