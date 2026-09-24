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
use crate::error::{Error, Result};
use crate::model_source;
use crate::store::Checkpoint;

use config::{AgentConfig, EncoderConfig, read_agent_config, read_encoder_config, validate_config};
use output::format_result;
use prompt::{prepare, special_tokens};
use runtime::{Forward, Weights, forward, load_weights};

#[derive(Debug)]
pub(super) struct SpecialTokens {
    pub(super) cls: u32,
    pub(super) sep: u32,
    pub(super) mask: u32,
    pub(super) pad: u32,
    pub(super) mask_text: String,
}

/// The local Backend: a Checkpoint's resources, held for repeated Calls.
///
/// A process that judges many Calls loads it once — the HTTP server does, so weights are mapped
/// once and every Call is judged against them.
pub struct Loaded {
    agent: AgentConfig,
    encoder: EncoderConfig,
    tokenizer: Tokenizer,
    special: SpecialTokens,
    weights: Weights,
}

impl Loaded {
    /// Load the resources of `checkpoint`: its configuration, tokenizer, special tokens and
    /// weights, once the Checkpoint has verified against its own Provenance.
    ///
    /// # Errors
    ///
    /// Whatever verification fails with; [`Error::InvalidCheckpoint`] when a configuration the
    /// Backend reads is missing or invalid; and [`Error::Inference`] when a configuration cannot be
    /// parsed, the tokenizer cannot be loaded, or the weights cannot be mapped.
    pub fn load(checkpoint: &Checkpoint) -> Result<Self> {
        let directory = checkpoint.directory();
        checkpoint.verify()?;
        let agent = read_agent_config(&directory.join(model_source::AGENT_CONFIG_FILE))?;
        let encoder = read_encoder_config(&directory.join(model_source::ENCODER_CONFIG_FILE))?;
        validate_config(&encoder, &agent)?;
        let tokenizer = Tokenizer::from_file(directory.join(model_source::TOKENIZER_FILE))
            .map_err(|error| Error::Inference {
                message: format!("loading tokenizer: {error}"),
            })?;
        let special = special_tokens(directory, &tokenizer)?;
        let weights = load_weights(&directory.join(model_source::WEIGHTS_FILE))?;
        Ok(Self {
            agent,
            encoder,
            tokenizer,
            special,
            weights,
        })
    }

    /// Judge one validated System One Call against this Backend's resources, and answer in the
    /// public response shape: one Answer per Question, under the id the caller chose.
    ///
    /// # Errors
    ///
    /// [`Error::Inference`] when the prompt does not fit the token budget, when the model output
    /// does not match the Question batch, or when native inference fails.
    pub fn run(&self, call: &Call) -> Result<Value> {
        let prepared = prepare(call, &self.tokenizer, &self.special, &self.agent)?;
        let Forward { logits, .. } = forward(
            &self.encoder,
            &self.agent,
            &self.weights,
            &prepared,
            &self.special,
        )?;
        format_result(call, &self.agent, &prepared, logits)
    }
}
