//! The `infer` command: judge one System One Call from JSON on stdin, without network access.

use std::io::{self, Read};

use super::checkpoint_name;
use crate::call::Call;
use crate::error::{Error, Result};
use crate::store::Store;

/// What `infer` was asked for.
#[derive(clap::Args)]
pub struct Args {
    /// The Model Source whose local Checkpoint should judge the call
    #[arg(long, value_name = "MODEL")]
    pub name: String,
}

pub fn run(args: Args) -> Result<()> {
    // Judging the call is a Backend's job; this command owns the input and error contract alone.
    let _call = read_call()?;
    let name = checkpoint_name(&args.name)?.repo;
    let store = Store::from_env()?;
    if store.provenance(name)?.is_none() {
        return Err(Error::MissingCheckpoint {
            name: name.to_string(),
        });
    }
    Err(Error::InferenceUnavailable)
}

/// Read the System One Call from stdin, the only place `infer` takes input: a reader that fails is
/// this command's failure, while bytes that are not a Call are the caller's.
fn read_call() -> Result<Call> {
    let mut input = Vec::new();
    io::stdin()
        .lock()
        .read_to_end(&mut input)
        .map_err(|error| Error::io("read", "<stdin>", error))?;
    Call::from_bytes(&input)
}
