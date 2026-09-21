//! s1gate pulls a Checkpoint from a curated Model Source into the local Model Store and judges
//! questions against it.
//!
//! Only [`pull`] reaches the network; the HTTP client lives inside that module alone (ADR-0003).
//! Vocabulary: `CONTEXT.md`. Decisions: `docs/adr/`.

pub mod call;
pub mod cli;
pub mod error;
pub mod model_source;
pub mod provenance;
pub mod pull;
pub mod store;

pub use error::{Error, Result};
