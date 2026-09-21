//! The command line: one subcommand per operation. Each subcommand's arguments, behaviour and
//! output live in its own module under [`crate::commands`]; this module declares them, parses
//! them, and dispatches.

use clap::{Parser, Subcommand};

use crate::commands;
use crate::error::Result;

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
    Pull(commands::pull::Args),
    /// Judge one System One Call from JSON on stdin without network access
    Infer(commands::infer::Args),
    /// Verify stored Checkpoints against their Provenance without network access
    Verify(commands::verify::Args),
}

/// Parse the command line, run the command, and report what it did. Exit codes are the caller's:
/// 0 success, 1 runtime error, 2 usage error.
pub fn run() -> Result<()> {
    match Cli::parse().command {
        Command::Pull(args) => commands::pull::run(args),
        Command::Infer(args) => commands::infer::run(args),
        Command::Verify(args) => commands::verify::run(args),
    }
}
