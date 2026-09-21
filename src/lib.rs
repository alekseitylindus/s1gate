//! s1gate pulls a Checkpoint from a curated Model Source into the local Model Store and judges
//! questions against it.
//!
//! Only [`pull`] reaches the network; the HTTP client lives inside that module alone (ADR-0003).
//! Vocabulary: `CONTEXT.md`. Decisions: `docs/adr/`.
//!
//! [`cli`] parses the command line and dispatches; `commands` holds one module per subcommand,
//! with what it takes, what it does and what it reports. The other modules are the domain those
//! commands call.

mod commands;

pub mod call;
pub mod checkpoint;
pub mod cli;
pub mod digest;
pub mod error;
pub mod model_source;
pub mod provenance;
pub mod pull;
pub mod store;

pub use error::{Error, Result};
