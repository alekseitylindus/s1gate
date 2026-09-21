//! The `infer` command: judge one System One Call from JSON on stdin, without network access.

use std::io::{self, Read, Write};

use super::checkpoint_name;
use crate::call::Call;
use crate::error::{Error, Result};
use crate::laya;
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
    let source = checkpoint_name(&args.name)?;
    let name = source.repo;
    let store = Store::from_env()?;
    if store.provenance(name)?.is_none() {
        return Err(Error::MissingCheckpoint {
            name: name.to_string(),
        });
    }
    let result = laya::run(&store, source, &_call)?;
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer(&mut stdout, &result).map_err(|error| Error::Inference {
        message: format!("writing result: {error}"),
    })?;
    stdout
        .write_all(b"\n")
        .map_err(|error| Error::io("write", "<stdout>", error))?;
    Ok(())
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
