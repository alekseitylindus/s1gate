//! The `pull` command: pull a Checkpoint into the Model Store, or, without a Model Source, list
//! the Model Sources s1gate can pull.

use crate::error::Result;
use crate::model_source;
use crate::pull::{self, Hub, PullRequest};
use crate::store::Store;

/// What `pull` was asked for.
#[derive(clap::Args)]
pub struct Args {
    /// The Model Source to pull, e.g. convaiinnovations/laya
    pub model: Option<String>,
    /// The revision to pull; the default branch when absent
    #[arg(long, value_name = "REF", requires = "model")]
    pub revision: Option<String>,
    /// Replace the Checkpoint the Model Store already holds for this Model Source
    #[arg(long, requires = "model")]
    pub force: bool,
}

pub fn run(args: Args) -> Result<()> {
    let Args {
        model,
        revision,
        force,
    } = args;
    // `--revision` and `--force` cannot arrive here: each requires a Model Source.
    let Some(model) = model else {
        return list();
    };

    // Checked before the environment is consulted, so an unsupported Model Source reads as the
    // usage error it is rather than as a missing Model Store. Pull repeats the check for callers
    // that build their own store.
    let source = model_source::lookup(&model)?;
    let store = Store::from_env()?;
    let request = PullRequest {
        source: source.repo.to_string(),
        revision,
        force,
    };
    let outcome = pull::pull(&store, &Hub::public(), &request)?;
    report(&outcome);
    Ok(())
}

/// List the Model Sources s1gate can pull, each with the revision of the Checkpoint the Model Store
/// holds for it. Reaches no network: the curated Model Sources are a value, and the Model Store is
/// on disk.
fn list() -> Result<()> {
    // A Model Store without a location holds no Checkpoint, which is no reason to say nothing about
    // which Model Sources exist.
    let store = Store::from_env().ok();
    for source in model_source::SOURCES {
        let held = match &store {
            Some(store) => store
                .provenance(source.repo)?
                .map(|provenance| provenance.resolved_revision),
            None => None,
        };
        match held {
            Some(revision) => println!("{}  {revision}", source.repo),
            None => println!("{}", source.repo),
        }
    }
    Ok(())
}

fn report(outcome: &pull::Outcome) {
    let provenance = &outcome.provenance;
    if outcome.unchanged() {
        println!(
            "{}@{} is already in {}",
            provenance.source,
            provenance.resolved_revision,
            outcome.directory.display()
        );
    } else {
        println!(
            "pulled {}@{} into {} ({} files)",
            provenance.source,
            provenance.resolved_revision,
            outcome.directory.display(),
            provenance.files.len()
        );
    }
}
