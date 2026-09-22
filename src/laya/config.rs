//! Typed checkpoint configuration and compatibility validation.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use mlx_rs::error::Exception;
use serde::Deserialize;
use serde_json::Value;

use crate::error::{Error, Result};

const ENCODER_KEYS: &[&str] = &[
    "architectures",
    "attention_bias",
    "attention_dropout",
    "bos_token_id",
    "classifier_activation",
    "classifier_bias",
    "classifier_dropout",
    "classifier_pooling",
    "cls_token_id",
    "decoder_bias",
    "deterministic_flash_attn",
    "dtype",
    "embedding_dropout",
    "eos_token_id",
    "global_attn_every_n_layers",
    "gradient_checkpointing",
    "hidden_activation",
    "hidden_size",
    "initializer_cutoff_factor",
    "initializer_range",
    "intermediate_size",
    "layer_norm_eps",
    "layer_types",
    "local_attention",
    "max_position_embeddings",
    "mlp_bias",
    "mlp_dropout",
    "model_type",
    "norm_bias",
    "norm_eps",
    "num_attention_heads",
    "num_hidden_layers",
    "pad_token_id",
    "position_embedding_type",
    "repad_logits_with_grad",
    "rope_parameters",
    "sep_token_id",
    "sparse_pred_ignore_index",
    "sparse_prediction",
    "tie_word_embeddings",
    "transformers_version",
    "vocab_size",
];

const AGENT_KEYS: &[&str] = &[
    "act_costs",
    "amp_dtype",
    "cost_wrong_act",
    "encoder",
    "head_layers",
    "head_max_len",
    "max_len",
    "max_prefixes",
    "model_name",
    "temperature",
    "temperature_by_options",
    "training",
];

#[derive(Debug, Deserialize)]
pub(super) struct EncoderConfig {
    pub(super) vocab_size: usize,
    pub(super) hidden_size: usize,
    pub(super) intermediate_size: usize,
    pub(super) num_hidden_layers: usize,
    pub(super) num_attention_heads: usize,
    pub(super) local_attention: usize,
    pub(super) max_position_embeddings: usize,
    layer_types: Vec<AttentionType>,
    rope_parameters: BTreeMap<AttentionType, RopeParameters>,
    pub(super) norm_eps: f32,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum AttentionType {
    #[serde(rename = "full_attention")]
    Full,
    #[serde(rename = "sliding_attention")]
    Sliding,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RopeParameters {
    rope_type: String,
    rope_theta: f32,
}

#[derive(Debug, Deserialize)]
pub(super) struct AgentConfig {
    pub(super) max_len: usize,
    pub(super) head_max_len: usize,
    pub(super) head_layers: usize,
    pub(super) temperature: Vec<f32>,
    pub(super) temperature_by_options: BTreeMap<String, f32>,
    pub(super) act_costs: BTreeMap<String, Value>,
}

impl EncoderConfig {
    pub(super) fn layer_type(&self, index: usize) -> std::result::Result<AttentionType, Exception> {
        self.layer_types
            .get(index)
            .copied()
            .ok_or_else(|| Exception::custom(format!("missing attention type for layer {index}")))
    }

    pub(super) fn rope_base(&self, kind: AttentionType) -> std::result::Result<f32, Exception> {
        self.rope_parameters
            .get(&kind)
            .map(|parameters| parameters.rope_theta)
            .ok_or_else(|| Exception::custom("missing RoPE parameters for attention type"))
    }
}

pub(super) fn read_encoder_config(path: &Path) -> Result<EncoderConfig> {
    read_config(path, ENCODER_KEYS)
}

pub(super) fn read_agent_config(path: &Path) -> Result<AgentConfig> {
    read_config(path, AGENT_KEYS)
}

pub(super) fn validate_config(encoder: &EncoderConfig, agent: &AgentConfig) -> Result<()> {
    let valid_temperatures = agent.temperature.len() == 3
        && agent
            .temperature
            .iter()
            .chain(agent.temperature_by_options.values())
            .all(|value| value.is_finite() && *value > 0.0);
    if !valid_temperatures {
        return Err(Error::Inference {
            message: "calibration temperatures must be three finite positive values".to_string(),
        });
    }
    if encoder.vocab_size == 0
        || encoder.hidden_size == 0
        || encoder.intermediate_size == 0
        || encoder.intermediate_size % 2 != 0
        || encoder.num_attention_heads == 0
        || encoder.hidden_size % encoder.num_attention_heads != 0
        || encoder.hidden_size % 64 != 0
        || encoder.num_hidden_layers == 0
        || encoder.local_attention == 0
        || encoder.max_position_embeddings == 0
        || agent.head_layers == 0
    {
        return Err(Error::Inference {
            message: "encoder configuration has invalid dimensions".to_string(),
        });
    }
    if encoder.layer_types.len() != encoder.num_hidden_layers {
        return Err(Error::Inference {
            message: "layer_types must contain one entry per encoder layer".to_string(),
        });
    }
    let valid_rope = [AttentionType::Full, AttentionType::Sliding]
        .iter()
        .all(|kind| {
            encoder.rope_parameters.get(kind).is_some_and(|parameters| {
                parameters.rope_type == "default"
                    && parameters.rope_theta.is_finite()
                    && parameters.rope_theta > 0.0
            })
        });
    if !valid_rope || !encoder.norm_eps.is_finite() || encoder.norm_eps <= 0.0 {
        return Err(Error::Inference {
            message: "encoder normalization or RoPE configuration is invalid".to_string(),
        });
    }
    if i32::try_from(encoder.hidden_size).is_err()
        || i32::try_from(encoder.max_position_embeddings).is_err()
    {
        return Err(Error::Inference {
            message: "encoder dimensions exceed MLX limits".to_string(),
        });
    }
    if !(4 < agent.head_max_len
        && agent.head_max_len < agent.max_len
        && agent.max_len <= encoder.max_position_embeddings)
    {
        return Err(Error::Inference {
            message: "expected 4 < head_max_len < max_len <= max_position_embeddings".to_string(),
        });
    }
    Ok(())
}

fn read_config<T: for<'de> Deserialize<'de>>(path: &Path, allowed: &[&str]) -> Result<T> {
    let bytes = fs::read(path).map_err(|error| Error::io("read", path, error))?;
    parse_config(path, &bytes, allowed)
}

fn parse_config<T: for<'de> Deserialize<'de>>(
    path: &Path,
    bytes: &[u8],
    allowed: &[&str],
) -> Result<T> {
    let value: Value = serde_json::from_slice(bytes)
        .map_err(|error| Error::json(path.display().to_string(), error))?;
    let object = value.as_object().ok_or_else(|| Error::InvalidCheckpoint {
        path: path.to_path_buf(),
        message: "configuration is not an object".to_string(),
    })?;
    if let Some(key) = object.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Err(Error::InvalidCheckpoint {
            path: path.to_path_buf(),
            message: format!("configuration has unknown field `{key}`"),
        });
    }
    serde_json::from_value(value).map_err(|error| Error::InvalidCheckpoint {
        path: path.to_path_buf(),
        message: format!("configuration is invalid: {error}"),
    })
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    fn encoder_config_json() -> Value {
        serde_json::json!({
            "vocab_size": 128, "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": 2, "num_attention_heads": 1, "local_attention": 128,
            "global_attn_every_n_layers": 3, "max_position_embeddings": 512,
            "layer_types": ["full_attention", "sliding_attention"],
            "rope_parameters": {
                "full_attention": {"rope_type": "default", "rope_theta": 160000.0},
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}
            },
            "norm_eps": 0.00001
        })
    }

    #[test]
    fn encoder_config_rejects_missing_runtime_fields() {
        let mut config = encoder_config_json();
        config
            .as_object_mut()
            .unwrap()
            .shift_remove("local_attention");
        let bytes = serde_json::to_vec(&config).unwrap();
        assert!(
            parse_config::<EncoderConfig>(Path::new("encoder/config.json"), &bytes, ENCODER_KEYS)
                .is_err()
        );
    }

    #[test]
    fn encoder_config_rejects_unknown_fields_and_attention_types() {
        let mut unknown = encoder_config_json();
        unknown["surprise"] = serde_json::json!(true);
        let bytes = serde_json::to_vec(&unknown).unwrap();
        assert!(
            parse_config::<EncoderConfig>(Path::new("encoder/config.json"), &bytes, ENCODER_KEYS)
                .is_err()
        );

        let mut kind = encoder_config_json();
        kind["layer_types"][0] = serde_json::json!("typo_attention");
        let bytes = serde_json::to_vec(&kind).unwrap();
        assert!(
            parse_config::<EncoderConfig>(Path::new("encoder/config.json"), &bytes, ENCODER_KEYS)
                .is_err()
        );
    }
}
