//! The command line: one subcommand per operation.

use clap::{Parser, Subcommand};
use std::io;

use crate::call::Call;
use crate::error::Result;
use crate::model_source;
use crate::pull::{self, Hub, PullRequest};
use crate::store::Store;

#[derive(Parser)]
#[command(
    name = "s1gate",
    version,
    about = "Run local System One decision models natively"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Pull a Checkpoint into the Model Store; the only command that reaches the network
    ///
    /// Without a Model Source, lists the Model Sources s1gate can pull, each with the revision of
    /// the Checkpoint the Model Store already holds for it.
    Pull {
        /// The Model Source to pull, e.g. convaiinnovations/laya
        model: Option<String>,
        /// The revision to pull; the default branch when absent
        #[arg(long, value_name = "REF", requires = "model")]
        revision: Option<String>,
        /// Replace the Checkpoint the Model Store already holds for this Model Source
        #[arg(long, requires = "model")]
        force: bool,
    },
    /// Judge one System One Call from JSON on stdin without network access
    Infer {
        /// The Model Source whose local Checkpoint should judge the call
        #[arg(long, value_name = "MODEL")]
        name: String,
    },
}

/// Parse the command line, run the command, and report what it did. Exit codes are the caller's:
/// 0 success, 1 runtime error, 2 usage error.
pub fn run() -> Result<()> {
    match Cli::parse().command {
        // `--revision` and `--force` cannot arrive here: each requires a Model Source.
        Command::Pull { model: None, .. } => list(),
        Command::Pull {
            model: Some(model),
            revision,
            force,
        } => {
            // Checked before the environment is consulted, so an unsupported Model Source reads as
            // the usage error it is rather than as a missing Model Store. Pull repeats the check
            // for callers that build their own store.
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
        Command::Infer { name } => infer(&name),
    }
}

fn infer(name: &str) -> Result<()> {
    let mut stdin = io::stdin().lock();
    let _call = Call::read(&mut stdin)?;
    model_source::lookup(name)?;

    // Validate the Checkpoint name without consulting the environment. An invalid name remains a
    // usage error even on a host with no configured Model Store.
    Store::at("").checkpoint_dir(name)?;
    let store = Store::from_env()?;
    if store.provenance(name)?.is_none() {
        return Err(crate::error::Error::MissingCheckpoint {
            name: name.to_string(),
        });
    }
    Err(crate::error::Error::InferenceUnavailable)
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
