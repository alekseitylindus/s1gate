//! The `verify` command: verify stored Checkpoints against their Provenance, without network
//! access, one Model Source or the whole Model Store.

use super::checkpoint_name;
use crate::checkpoint;
use crate::error::{Error, Result};
use crate::model_source;
use crate::store::Store;

/// What `verify` was asked for.
#[derive(clap::Args)]
pub struct Args {
    /// Verify one Model Source; without it, verify every stored Checkpoint
    #[arg(long, value_name = "MODEL")]
    pub name: Option<String>,
}

pub fn run(args: Args) -> Result<()> {
    // The name is checked before the environment is consulted, so an invalid name stays a usage
    // error on a host with no Model Store.
    let named = args.name.as_deref().map(checkpoint_name).transpose()?;
    let store = Store::from_env()?;
    match named {
        Some(source) => {
            report(&checkpoint::verify(&store, source)?);
            Ok(())
        }
        None => every_stored(&store),
    }
}

/// Verify every Checkpoint the store holds, reporting each failure rather than stopping at the
/// first: the operator asked about the whole store, and one broken Checkpoint says nothing about
/// the others. Each failure names the Checkpoint it came from, which the errors themselves do not.
fn every_stored(store: &Store) -> Result<()> {
    let names = store.checkpoint_names()?;
    let total = names.len();
    let mut failed = 0;
    for name in names {
        match model_source::lookup(&name).and_then(|source| checkpoint::verify(store, source)) {
            Ok(verified) => report(&verified),
            Err(error) => {
                failed += 1;
                eprintln!("s1gate: {name}: {error}");
            }
        }
    }
    match failed {
        0 => Ok(()),
        failed => Err(Error::VerificationFailed { failed, total }),
    }
}

fn report(report: &checkpoint::Report) {
    println!(
        "verified {}@{} ({} files)",
        report.provenance.source, report.provenance.resolved_revision, report.files
    );
}
