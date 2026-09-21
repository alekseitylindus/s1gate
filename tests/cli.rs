//! The command line as the operator meets it: argument handling, messages, and exit codes. None of
//! these reach the Model Source.

mod support;

use std::fs;
use std::process::Command;

use s1gate::provenance::{FileRecord, Provenance};
use support::TempDir;

#[test]
fn pull_without_a_model_source_lists_the_one_s1gate_can_pull() {
    let run = run(&["pull"]);

    assert_eq!(run.code, 0);
    assert_eq!(run.stdout, "convaiinnovations/laya\n");
}

#[test]
fn a_stored_checkpoint_is_listed_with_the_revision_it_holds() {
    let data_home = TempDir::new("cli-listing");
    let checkpoint = data_home
        .path()
        .join("s1gate/models/convaiinnovations/laya");
    fs::create_dir_all(&checkpoint).expect("the Checkpoint directory");
    fs::write(
        checkpoint.join("provenance.json"),
        Provenance {
            source: "convaiinnovations/laya".to_string(),
            requested_revision: None,
            resolved_revision: "1c5edc17a7acd8701df6fc341c0d179f1c62c982".to_string(),
            files: Vec::<FileRecord>::new(),
        }
        .to_json(),
    )
    .expect("a record on disk");

    let run = run_in(&data_home, &["pull"]);

    assert_eq!(run.code, 0);
    assert_eq!(
        run.stdout,
        "convaiinnovations/laya  1c5edc17a7acd8701df6fc341c0d179f1c62c982\n"
    );
}

#[test]
fn without_a_store_location_the_listing_still_names_the_model_sources() {
    let output = Command::new(binary())
        .args(["pull"])
        .env_remove("XDG_DATA_HOME")
        .env_remove("HOME")
        .output()
        .expect("the s1gate binary runs");

    assert_eq!(output.status.code(), Some(0));
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "convaiinnovations/laya\n",
        "the curated Model Sources need no Model Store to be listed"
    );
}

#[test]
fn a_flag_without_a_model_source_is_a_usage_error() {
    let cases: [&[&str]; 2] = [&["pull", "--revision", "main"], &["pull", "--force"]];

    for args in cases {
        let run = run(args);

        assert_eq!(run.code, 2, "{args:?}: {}", run.stderr);
        assert!(
            run.stderr.contains("<MODEL>"),
            "{args:?} names the missing argument: {}",
            run.stderr
        );
    }
}

#[test]
fn no_subcommand_is_a_usage_error() {
    let run = run(&[]);

    assert_eq!(run.code, 2);
    assert!(run.stderr.contains("Usage"), "{}", run.stderr);
}

#[test]
fn an_unsupported_model_source_lists_the_supported_one() {
    let run = run(&["pull", "some/other-model"]);

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
fn without_a_store_location_the_failure_is_a_runtime_error() {
    let output = Command::new(binary())
        .args(["pull", "convaiinnovations/laya"])
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
        .args(["pull", "some/other-model"])
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
    assert!(
        !run.stdout.contains("--name"),
        "the Checkpoint name is the Model Source now: {}",
        run.stdout
    );
}

struct Run {
    code: i32,
    stdout: String,
    stderr: String,
}

fn run(args: &[&str]) -> Run {
    run_in(&TempDir::new("cli"), args)
}

fn run_in(data_home: &TempDir, args: &[&str]) -> Run {
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
