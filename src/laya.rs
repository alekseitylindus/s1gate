//! The native Laya backend.
//!
//! This module wires checkpoint resources into the Laya pipeline. The
//! protocol, checkpoint configuration, Candle runtime, and response rendering
//! live in focused internal modules so they can evolve independently.

mod config;
mod output;
mod prompt;
mod runtime;

use serde_json::Value;
use tokenizers::Tokenizer;

use crate::call::Call;
use crate::checkpoint;
use crate::error::{Error, Result};
use crate::model_source::ModelSource;
use crate::store::Store;

use config::{read_agent_config, read_encoder_config, validate_config};
use output::format_result;
use prompt::{prepare, special_tokens};
use runtime::{forward, load_weights};

#[derive(Debug)]
pub(super) struct SpecialTokens {
    pub(super) cls: u32,
    pub(super) sep: u32,
    pub(super) mask: u32,
    pub(super) pad: u32,
    pub(super) mask_text: String,
}

/// Run one validated System One Call against a local Checkpoint.
pub fn run(store: &Store, source: &ModelSource, call: &Call) -> Result<Value> {
    let directory = store.checkpoint_dir(source.repo)?;
    checkpoint::verify(store, source)?;
    let agent = read_agent_config(&directory.join("rl_agent_config.json"))?;
    let encoder = read_encoder_config(&directory.join("encoder/config.json"))?;
    validate_config(&encoder, &agent)?;
    let tokenizer =
        Tokenizer::from_file(directory.join("tokenizer/tokenizer.json")).map_err(|error| {
            Error::Inference {
                message: format!("loading tokenizer: {error}"),
            }
        })?;
    let special = special_tokens(&directory, &tokenizer)?;
    let prepared = prepare(call, &tokenizer, &special, &agent)?;
    let weights = load_weights(&directory.join("model.safetensors"))?;
    let (logits, actions) = forward(&encoder, &agent, &weights, &prepared, &special)?;
    format_result(call, &agent, prepared, logits, actions)
}
