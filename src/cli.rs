//! The command line: one subcommand per operation.

use clap::{Parser, Subcommand};
use std::io::{self, Read};

use crate::call::Call;
use crate::error::{Error, Result};
use crate::model_source::{self, ModelSource};
use crate::pull::{self, Hub, PullRequest};
use crate::store::Store;
use crate::verify as checkpoint_verify;

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
    /// Verify stored Checkpoints against their Provenance without network access
    Verify {
        /// Verify one Model Source; without it, verify every stored Checkpoint
        #[arg(long, value_name = "MODEL")]
        name: Option<String>,
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
        Command::Verify { name } => verify(name.as_deref()),
    }
}

fn verify(name: Option<&str>) -> Result<()> {
    // The name is checked before the environment is consulted, so an invalid name stays a usage
    // error on a host with no Model Store.
    let named = name.map(checkpoint_name).transpose()?;
    let store = Store::from_env()?;
    match named {
        Some(source) => {
            report_verified(&checkpoint_verify::verify(&store, source)?);
            Ok(())
        }
        None => verify_stored(&store),
    }
}

/// Verify every Checkpoint the store holds, reporting each failure rather than stopping at the
/// first: the operator asked about the whole store, and one broken Checkpoint says nothing about
/// the others. Each failure names the Checkpoint it came from, which the errors themselves do not.
fn verify_stored(store: &Store) -> Result<()> {
    let names = store.checkpoint_names()?;
    let total = names.len();
    let mut failed = 0;
    for name in names {
        match model_source::lookup(&name)
            .and_then(|source| checkpoint_verify::verify(store, source))
        {
            Ok(report) => report_verified(&report),
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

fn report_verified(report: &checkpoint_verify::Report) {
    println!(
        "verified {}@{} ({} files)",
        report.provenance.source, report.provenance.resolved_revision, report.files
    );
}

/// The curated Model Source `name` addresses. A command that addresses a Checkpoint takes one full
/// Model Source name, `<owner>/<name>` (ADR-0011), and the name is validated before the
/// environment is consulted, so an invalid name stays a usage error on a host with no Model Store.
fn checkpoint_name(name: &str) -> Result<&'static ModelSource> {
    Store::at("").checkpoint_dir(name)?;
    model_source::lookup(name)
}

fn infer(name: &str) -> Result<()> {
    // Judging the call is a Backend's job; this command owns the input and error contract alone.
    let _call = read_call()?;
    let name = checkpoint_name(name)?.repo;
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
