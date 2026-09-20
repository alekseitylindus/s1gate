//! Exit codes: 0 success, 1 runtime error, 2 usage error.

use std::process::ExitCode;

fn main() -> ExitCode {
    match s1gate::cli::run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("s1gate: {error}");
            ExitCode::from(error.exit_code())
        }
    }
}
