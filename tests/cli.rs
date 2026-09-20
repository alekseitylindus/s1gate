//! The command line as the operator meets it: argument handling, messages, and exit codes. None of
//! these reach the Model Source.

mod support;

use std::process::Command;

use support::TempDir;

#[test]
fn pull_without_a_name_is_a_usage_error() {
    let run = run(&["pull", "convaiinnovations/laya"]);

    assert_eq!(run.code, 2);
    assert!(
        run.stderr.contains("--name"),
        "the error names the missing option: {}",
        run.stderr
    );
}

#[test]
fn no_subcommand_is_a_usage_error() {
    let run = run(&[]);

    assert_eq!(run.code, 2);
    assert!(run.stderr.contains("Usage"), "{}", run.stderr);
}

#[test]
fn an_unsupported_model_source_lists_the_supported_one() {
    let run = run(&["pull", "some/other-model", "--name", "laya"]);

    assert_eq!(run.code, 2);
    assert!(
        run.stderr.contains(
            "unsupported Model Source `some/other-model` (supported: convaiinnovations/laya)"
        ),
        "{}",
        run.stderr
    );
}

#[test]
fn a_name_that_is_not_one_directory_is_rejected() {
    let run = run(&["pull", "convaiinnovations/laya", "--name", "../laya"]);

    assert_eq!(run.code, 2);
    assert!(
        run.stderr.contains("invalid Checkpoint name `../laya`"),
        "{}",
        run.stderr
    );
}

#[test]
fn without_a_store_location_the_failure_is_a_runtime_error() {
    let output = Command::new(binary())
        .args(["pull", "convaiinnovations/laya", "--name", "laya"])
        .env_remove("XDG_DATA_HOME")
        .env_remove("HOME")
        .output()
        .expect("the s1gate binary runs");

    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("cannot locate the Model Store"),
        "unexpected stderr: {stderr}"
    );
}

#[test]
fn an_unsupported_model_source_is_reported_before_the_store_is_located() {
    let output = Command::new(binary())
        .args(["pull", "some/other-model", "--name", "laya"])
        .env_remove("XDG_DATA_HOME")
        .env_remove("HOME")
        .output()
        .expect("the s1gate binary runs");

    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("unsupported Model Source `some/other-model`"),
        "the request is checked before the environment: {stderr}"
    );
}

#[test]
fn help_describes_pull() {
    let run = run(&["pull", "--help"]);

    assert_eq!(run.code, 0);
    assert!(run.stdout.contains("--revision"), "{}", run.stdout);
    assert!(run.stdout.contains("--force"), "{}", run.stdout);
}

struct Run {
    code: i32,
    stdout: String,
    stderr: String,
}

fn run(args: &[&str]) -> Run {
    let data_home = TempDir::new("cli");
    let output = Command::new(binary())
        .args(args)
        .env("XDG_DATA_HOME", data_home.path())
        .output()
        .expect("the s1gate binary runs");
    Run {
        code: output.status.code().expect("the binary exits"),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_s1gate")
}
