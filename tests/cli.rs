//! The command line as the operator meets it: argument handling, messages, and exit codes. None of
//! these reach the Model Source.

mod support;

use std::fs;
use std::io::Write;
use std::process::Command;
use std::process::Stdio;

use s1gate::provenance::{FileRecord, Provenance};
use support::TempDir;
use support::fixture;

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

#[test]
fn an_empty_model_store_verifies_silently() {
    let run = run(&["verify"]);

    assert_eq!(run.code, 0);
    assert!(run.stdout.is_empty());
}

#[test]
fn verify_with_a_name_proves_the_stored_checkpoint() {
    let data_home = TempDir::new("cli-verify");
    fixture::write(&checkpoint_root(&data_home), fixture::header());

    let run = run_in(&data_home, &["verify", "--name", fixture::NAME]);

    assert_eq!(run.code, 0);
    assert_eq!(
        run.stdout,
        format!(
            "verified {}@{} (5 files)\n",
            fixture::NAME,
            fixture::REVISION
        )
    );
    assert!(run.stderr.is_empty(), "{}", run.stderr);
}

#[test]
fn verify_without_a_name_verifies_every_checkpoint_the_store_holds() {
    let data_home = TempDir::new("cli-verify-store");
    let root = checkpoint_root(&data_home);
    fixture::write(&root, fixture::header());
    // Neither of these is a Checkpoint: the first is a stray directory, the second is what an
    // interrupted Pull leaves behind — a directory whose Provenance record was never written.
    fs::create_dir_all(root.join("scratch/notes")).expect("a stray directory");
    fs::create_dir_all(root.join("convaiinnovations/incomplete")).expect("an interrupted Pull");

    let run = run_in(&data_home, &["verify"]);

    assert_eq!(run.code, 0);
    assert_eq!(
        run.stdout,
        format!(
            "verified {}@{} (5 files)\n",
            fixture::NAME,
            fixture::REVISION
        )
    );
    assert!(run.stderr.is_empty(), "{}", run.stderr);
}

#[test]
fn verify_reports_a_failed_checkpoint_and_verifies_the_rest() {
    let data_home = TempDir::new("cli-verify-stale");
    let root = checkpoint_root(&data_home);
    fixture::write(&root, fixture::header());
    // A Checkpoint of a Model Source s1gate no longer supports: the store, not the command line,
    // is what is out of date, so the sweep reports it and carries on.
    fs::create_dir_all(root.join("someone/other")).expect("a stale Checkpoint");
    fs::write(root.join("someone/other/provenance.json"), b"{}").expect("a record");

    let run = run_in(&data_home, &["verify"]);

    assert_eq!(run.code, 1);
    assert_eq!(
        run.stdout,
        format!(
            "verified {}@{} (5 files)\n",
            fixture::NAME,
            fixture::REVISION
        ),
        "the Checkpoints that verify are still reported"
    );
    assert!(
        run.stderr
            .contains("s1gate: someone/other: unsupported Model Source"),
        "{}",
        run.stderr
    );
    assert!(
        run.stderr
            .contains("1 of 2 Checkpoints failed verification"),
        "{}",
        run.stderr
    );
}

#[test]
fn verify_names_the_checkpoint_a_failure_came_from() {
    let data_home = TempDir::new("cli-verify-missing-file");
    let directory = fixture::write(&checkpoint_root(&data_home), fixture::header());
    fs::remove_file(directory.join("model.safetensors")).expect("remove a Checkpoint file");

    let run = run_in(&data_home, &["verify"]);

    assert_eq!(run.code, 1);
    assert!(run.stdout.is_empty(), "{}", run.stdout);
    assert_eq!(
        run.stderr,
        concat!(
            "s1gate: convaiinnovations/laya: Checkpoint file `model.safetensors` is missing\n",
            "s1gate: 1 of 1 Checkpoints failed verification\n"
        )
    );
}

#[test]
fn verify_rejects_a_name_that_is_not_a_model_source_path() {
    let run = run(&["verify", "--name", "laya"]);

    assert_eq!(run.code, 2);
    assert!(
        run.stderr.contains("invalid Checkpoint name `laya`"),
        "{}",
        run.stderr
    );
}

#[test]
fn verify_rejects_an_unsupported_model_source() {
    let run = run(&["verify", "--name", "some/other-model"]);

    assert_eq!(run.code, 2);
    assert!(
        run.stderr
            .contains("unsupported Model Source `some/other-model`"),
        "{}",
        run.stderr
    );
}

#[test]
fn verify_validates_the_name_before_locating_the_store() {
    let output = Command::new(binary())
        .args(["verify", "--name", "laya"])
        .env_remove("XDG_DATA_HOME")
        .env_remove("HOME")
        .output()
        .expect("the s1gate binary runs");

    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("invalid Checkpoint name `laya`"),
        "the name is checked before the environment: {stderr}"
    );
}

/// The Model Store root `data_home` holds, where a Checkpoint is written for the binary to find.
fn checkpoint_root(data_home: &TempDir) -> std::path::PathBuf {
    data_home.path().join("s1gate/models")
}

#[test]
fn infer_requires_a_model_source_name() {
    let run = run_with_input(&["infer"], "{}");

    assert_eq!(run.code, 2);
    assert!(run.stdout.is_empty());
    assert!(run.stderr.contains("--name"), "{}", run.stderr);
}

#[test]
fn infer_rejects_invalid_json_without_writing_stdout() {
    let run = run_with_input(&["infer", "--name", "convaiinnovations/laya"], "not JSON");

    assert_eq!(run.code, 2);
    assert!(run.stdout.is_empty());
    assert!(run.stderr.contains("System One Call"), "{}", run.stderr);
}

#[test]
fn infer_rejects_trailing_json() {
    let run = run_with_input(
        &["infer", "--name", "convaiinnovations/laya"],
        r#"{"state":"x","questions":{}} {}"#,
    );

    assert_eq!(run.code, 2);
    assert!(run.stdout.is_empty());
    assert!(run.stderr.contains("trailing"), "{}", run.stderr);
}

#[test]
fn infer_rejects_a_choice_with_one_option_before_loading_the_checkpoint() {
    let run = run_with_input(
        &["infer", "--name", "convaiinnovations/laya"],
        r#"{
            "state": "x",
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Where should this go?",
                    "criteria": {"billing": "payments"}
                }
            }
        }"#,
    );

    assert_eq!(run.code, 2);
    assert!(run.stdout.is_empty());
    assert!(run.stderr.contains("route"), "{}", run.stderr);
    assert!(run.stderr.contains("at least two"), "{}", run.stderr);
    assert!(run.stderr.contains("Options"), "{}", run.stderr);
}

#[test]
fn infer_rejects_duplicate_choice_options() {
    let run = run_with_input(
        &["infer", "--name", "convaiinnovations/laya"],
        r#"{
            "state": "x",
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Where should this go?",
                    "criteria": ["billing", "billing"]
                }
            }
        }"#,
    );

    assert_eq!(run.code, 2);
    assert!(run.stdout.is_empty());
    assert!(run.stderr.contains("distinct Options"), "{}", run.stderr);
}

#[test]
fn infer_rejects_an_uncurated_model_source_before_locating_the_store() {
    let run = run_with_input(
        &["infer", "--name", "some/other-model"],
        r#"{"state":"x","questions":{"q":{"type":"noul","instructions":"x"}}}"#,
    );

    assert_eq!(run.code, 2);
    let stderr = run.stderr;
    assert!(
        stderr.contains("unsupported Model Source `some/other-model`"),
        "{stderr}"
    );
}

#[test]
fn infer_rejects_a_score_with_choice_criteria() {
    let run = run_with_input(
        &["infer", "--name", "convaiinnovations/laya"],
        r#"{
            "state": "x",
            "questions": {
                "urgency": {
                    "type": "score",
                    "instructions": "How urgent?",
                    "criteria": {"low": "later", "high": "now"}
                }
            }
        }"#,
    );

    assert_eq!(run.code, 2);
    assert!(run.stdout.is_empty());
    assert!(
        run.stderr.contains("score Criteria must be an array"),
        "{}",
        run.stderr
    );
}

#[test]
fn infer_rejects_a_noul_with_only_one_option() {
    let run = run_with_input(
        &["infer", "--name", "convaiinnovations/laya"],
        r#"{
            "state": "x",
            "questions": {
                "risk": {
                    "type": "noul",
                    "instructions": "Is it risky?",
                    "criteria": {"true": "risky"}
                }
            }
        }"#,
    );

    assert_eq!(run.code, 2);
    assert!(run.stdout.is_empty());
    assert!(run.stderr.contains("false and true"), "{}", run.stderr);
}

#[test]
fn infer_rejects_an_unknown_question_type() {
    let run = run_with_input(
        &["infer", "--name", "convaiinnovations/laya"],
        r#"{
            "state": "x",
            "questions": {
                "route": {"type": "ranking", "instructions": "x", "criteria": ["a", "b"]}
            }
        }"#,
    );

    assert_eq!(run.code, 2);
    assert!(run.stdout.is_empty());
    assert!(run.stderr.contains("choice"), "{}", run.stderr);
    assert!(run.stderr.contains("score"), "{}", run.stderr);
    assert!(run.stderr.contains("noul"), "{}", run.stderr);
}

#[test]
fn infer_rejects_duplicate_question_ids() {
    let run = run_with_input(
        &["infer", "--name", "convaiinnovations/laya"],
        r#"{
            "state": "x",
            "questions": {
                "route": {"type": "choice", "instructions": "x", "criteria": ["a", "b"]},
                "route": {"type": "choice", "instructions": "y", "criteria": ["c", "d"]}
            }
        }"#,
    );

    assert_eq!(run.code, 2);
    assert!(run.stdout.is_empty());
    assert!(
        run.stderr.contains("duplicate Question id `route`"),
        "{}",
        run.stderr
    );
}

#[test]
fn infer_rejects_an_empty_question_id() {
    let run = run_with_input(
        &["infer", "--name", "convaiinnovations/laya"],
        r#"{
            "state": "x",
            "questions": {
                "": {"type": "score", "instructions": "x", "criteria": ["low", "high"]}
            }
        }"#,
    );

    assert_eq!(run.code, 2);
    assert!(run.stdout.is_empty());
    assert!(run.stderr.contains("Question id"), "{}", run.stderr);
}

#[test]
fn infer_accepts_all_question_types_then_reports_the_missing_checkpoint() {
    let run = run_with_input(
        &["infer", "--name", "convaiinnovations/laya"],
        r#"{
            "state": {"body": "x"},
            "questions": {
                "route": {"type": "choice", "instructions": "Where?", "criteria": ["a", "b"]},
                "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "high"]},
                "risk": {"type": "noul", "instructions": "Is it risky?"}
            }
        }"#,
    );

    assert_eq!(run.code, 1);
    assert!(run.stdout.is_empty());
    assert!(
        run.stderr.contains("Checkpoint `convaiinnovations/laya`"),
        "{}",
        run.stderr
    );
    assert!(run.stderr.contains("s1gate pull"), "{}", run.stderr);
}

#[test]
fn infer_rejects_input_that_is_not_utf8() {
    let run = run_with_input(
        &["infer", "--name", "convaiinnovations/laya"],
        b"{\"state\":\"\xff\"}",
    );

    assert_eq!(run.code, 2);
    assert!(run.stdout.is_empty());
    assert!(run.stderr.contains("System One Call"), "{}", run.stderr);
    assert!(run.stderr.contains("UTF-8"), "{}", run.stderr);
}

#[test]
fn infer_names_the_call_once_in_a_diagnostic() {
    let run = run_with_input(&["infer", "--name", "convaiinnovations/laya"], "not JSON");

    assert_eq!(run.code, 2);
    assert_eq!(
        run.stderr.matches("System One Call").count(),
        1,
        "{}",
        run.stderr
    );
}

#[test]
fn infer_rejects_a_score_level_that_is_not_a_string() {
    let run = run_with_input(
        &["infer", "--name", "convaiinnovations/laya"],
        r#"{
            "state": "x",
            "questions": {
                "urgency": {
                    "type": "score",
                    "instructions": "How urgent?",
                    "criteria": ["low", 1]
                }
            }
        }"#,
    );

    assert_eq!(run.code, 2);
    assert!(run.stdout.is_empty());
    assert!(run.stderr.contains("urgency"), "{}", run.stderr);
    assert!(run.stderr.contains("string Levels"), "{}", run.stderr);
}

#[test]
fn infer_rejects_duplicate_score_levels() {
    let run = run_with_input(
        &["infer", "--name", "convaiinnovations/laya"],
        r#"{
            "state": "x",
            "questions": {
                "urgency": {
                    "type": "score",
                    "instructions": "How urgent?",
                    "criteria": ["low", "low"]
                }
            }
        }"#,
    );

    assert_eq!(run.code, 2);
    assert!(run.stdout.is_empty());
    assert!(run.stderr.contains("urgency"), "{}", run.stderr);
    assert!(run.stderr.contains("distinct Levels"), "{}", run.stderr);
}

#[test]
fn infer_names_the_question_whose_option_name_is_unusable() {
    let run = run_with_input(
        &["infer", "--name", "convaiinnovations/laya"],
        r#"{
            "state": "x",
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Where should this go?",
                    "criteria": ["billing", ""]
                }
            }
        }"#,
    );

    assert_eq!(run.code, 2);
    assert!(run.stdout.is_empty());
    assert!(
        run.stderr.contains("Question `route` Option name"),
        "{}",
        run.stderr
    );
    assert_eq!(
        run.stderr.matches("invalid System One Call").count(),
        1,
        "{}",
        run.stderr
    );
}

#[test]
fn infer_names_the_question_whose_options_repeat() {
    let run = run_with_input(
        &["infer", "--name", "convaiinnovations/laya"],
        r#"{
            "state": "x",
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Where should this go?",
                    "criteria": {"billing": "payments", "billing": "billing"}
                }
            }
        }"#,
    );

    assert_eq!(run.code, 2);
    assert!(run.stdout.is_empty());
    assert!(run.stderr.contains("route"), "{}", run.stderr);
    assert!(run.stderr.contains("distinct Options"), "{}", run.stderr);
}

#[test]
fn infer_escapes_the_question_id_it_rejects() {
    let run = run_with_input(
        &["infer", "--name", "convaiinnovations/laya"],
        r#"{"state":"x","questions":{"a\u001b[2Kb":{"type":"noul","instructions":"x"}}}"#,
    );

    assert_eq!(run.code, 2);
    assert!(run.stderr.contains(r"a\u{1b}[2Kb"), "{}", run.stderr);
    assert!(!run.stderr.contains('\u{1b}'), "{}", run.stderr);
}

#[test]
fn infer_escapes_a_duplicated_question_id_it_rejects() {
    let run = run_with_input(
        &["infer", "--name", "convaiinnovations/laya"],
        r#"{
            "state": "x",
            "questions": {
                "a\u001b[2Kb": {"type": "noul", "instructions": "x"},
                "a\u001b[2Kb": {"type": "noul", "instructions": "y"}
            }
        }"#,
    );

    assert_eq!(run.code, 2);
    assert!(run.stderr.contains(r"a\u{1b}[2Kb"), "{}", run.stderr);
    assert!(!run.stderr.contains('\u{1b}'), "{}", run.stderr);
}

#[test]
#[ignore = "requires a pulled MLX checkpoint in LAYA_MODEL_DIR and the Metal toolchain"]
fn infer_runs_a_real_laya_checkpoint() {
    let checkpoint = std::env::var_os("LAYA_MODEL_DIR")
        .map(std::path::PathBuf::from)
        .expect("set LAYA_MODEL_DIR to a pulled convaiinnovations/laya checkpoint");
    assert!(
        checkpoint.join("provenance.json").is_file(),
        "LAYA_MODEL_DIR must contain provenance.json"
    );

    let data_home = TempDir::new("cli-laya-e2e");
    let model = data_home
        .path()
        .join("s1gate/models/convaiinnovations/laya");
    fs::create_dir_all(model.parent().expect("the model has a parent"))
        .expect("the test Model Store");
    std::os::unix::fs::symlink(&checkpoint, &model).expect("the checkpoint symlink");

    let run = run_in_with_input(
        &data_home,
        &["infer", "--name", "convaiinnovations/laya"],
        r#"{
            "state": {"message": "The light is on."},
            "questions": {
                "lit": {
                    "type": "choice",
                    "instructions": "Is the light on?",
                    "criteria": {"no": null, "yes": null}
                }
            }
        }"#,
    );
    assert_eq!(run.code, 0, "{}", run.stderr);

    let result: serde_json::Value =
        serde_json::from_str(&run.stdout).expect("inference returns JSON");
    assert_eq!(result["model"], "rl-agent");
    assert_eq!(result["answers"]["lit"]["type"], "choice");
    let probabilities = result["answers"]["lit"]["probabilities"]
        .as_object()
        .expect("choice probabilities");
    let total = probabilities
        .values()
        .map(|value| value.as_f64().expect("a numeric probability"))
        .inspect(|value| assert!((0.0..=1.0).contains(value)))
        .sum::<f64>();
    assert!((total - 1.0).abs() < 0.001, "probabilities sum to {total}");
    let action = result["answers"]["lit"]["rl_agent"]["act_probability"]
        .as_f64()
        .expect("a numeric action probability");
    assert!((0.0..=1.0).contains(&action));
    assert_eq!(result["usage"]["output_tokens"], 0);
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

fn run_with_input(args: &[&str], input: impl AsRef<[u8]>) -> Run {
    let data_home = TempDir::new("cli-infer");
    run_in_with_input(&data_home, args, input)
}

fn run_in_with_input(data_home: &TempDir, args: &[&str], input: impl AsRef<[u8]>) -> Run {
    let mut child = Command::new(binary())
        .args(args)
        .env("XDG_DATA_HOME", data_home.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("the s1gate binary runs");
    child
        .stdin
        .take()
        .expect("stdin is piped")
        .write_all(input.as_ref())
        .expect("the call is written");
    let output = child.wait_with_output().expect("the s1gate process exits");
    Run {
        code: output.status.code().expect("the binary exits"),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_s1gate")
}
