//! One module per subcommand: the arguments it takes, what it does, and what it reports. The
//! domain each command calls is the crate's other modules; nothing here judges a model or reads a
//! Checkpoint itself.

use crate::error::Result;
use crate::model_source::{self, ModelSource};
use crate::store::Store;

pub mod infer;
pub mod models;
pub mod pull;
pub mod verify;

/// The curated Model Source `name` addresses. A command that addresses a Checkpoint takes one full
/// Model Source name, `<owner>/<name>` (ADR-0011), and the name is validated before the
/// environment is consulted, so an invalid name stays a usage error on a host with no Model Store.
pub fn checkpoint_name(name: &str) -> Result<&'static ModelSource> {
    Store::at("").checkpoint_dir(name)?;
    model_source::lookup(name)
}
