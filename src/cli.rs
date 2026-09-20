//! The command line: one subcommand per operation.

use clap::{Parser, Subcommand};

use crate::error::Result;
use crate::model_source;
use crate::pull::{self, Hub, PullRequest};
use crate::store::{self, Store};

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
    Pull {
        /// The Model Source repository, e.g. convaiinnovations/laya
        repo: String,
        /// The name to store the Checkpoint under
        #[arg(long, value_name = "NAME")]
        name: String,
        /// The revision to pull; the default branch when absent
        #[arg(long, value_name = "REF")]
        revision: Option<String>,
        /// Replace the Checkpoint already stored under this name
        #[arg(long)]
        force: bool,
    },
}

/// Parse the command line, run the command, and report what it did. Exit codes are the caller's:
/// 0 success, 1 runtime error, 2 usage error.
pub fn run() -> Result<()> {
    match Cli::parse().command {
        Command::Pull {
            repo,
            name,
            revision,
            force,
        } => {
            let request = PullRequest {
                name,
                source: repo,
                revision,
                force,
            };
            // Checked before the environment is consulted, so an unsupported Model Source or a name
            // that is not one directory reads as the usage error it is rather than as a missing
            // Model Store. Pull repeats both checks for callers that build their own store.
            model_source::lookup(&request.source)?;
            store::validate_name(&request.name)?;
            let store = Store::from_env()?;
            let outcome = pull::pull(&store, &Hub::public(), &request)?;
            report(&outcome);
            Ok(())
        }
    }
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
