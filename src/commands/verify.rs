//! The `verify` command: verify stored Checkpoints against their Provenance, without network
//! access, one Model Source or the whole Model Store.

use super::checkpoint_name;
use crate::checkpoint::Report;
use crate::error::{Error, Result};
use crate::model_source::{self, ModelSource};
use crate::store::{Checkpoint, Store};

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
            report(&stored(&store, source)?.verify()?);
            Ok(())
        }
        None => every_stored(&store),
    }
}

/// The Checkpoint of `source` in the store, or the error naming the Pull that would provide it.
fn stored(store: &Store, source: &'static ModelSource) -> Result<Checkpoint> {
    store
        .checkpoint(source)?
        .ok_or_else(|| Error::missing_checkpoint(source.repo))
}

/// Verify every Checkpoint the store holds, reporting each failure rather than stopping at the
/// first: the operator asked about the whole store, and one broken Checkpoint says nothing about
/// the others. Each failure names the Checkpoint it came from, which the errors themselves do not.
fn every_stored(store: &Store) -> Result<()> {
    let names = store.checkpoint_names()?;
    let total = names.len();
    let mut failed = 0;
    for name in names {
        let verified = model_source::lookup(&name)
            .and_then(|source| stored(store, source))
            .and_then(|checkpoint| checkpoint.verify());
        match verified {
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

fn report(report: &Report) {
    println!(
        "verified {}@{} ({} files)",
        report.provenance.source, report.provenance.resolved_revision, report.files
    );
}
