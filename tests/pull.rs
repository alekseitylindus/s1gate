//! Pull, through its public seam: a local Model Source stands in for `convaiinnovations/laya`, so
//! no test reaches the network.

mod support;

use s1gate::provenance::Algorithm;
use s1gate::pull::{self, Hub, PullRequest};
use s1gate::store::Store;
use support::{ALLOWLIST, FakeSource, ServedFile, TempDir};

const LAYA: &str = "convaiinnovations/laya";
const FIRST: &str = "1111111111111111111111111111111111111111";
const SECOND: &str = "2222222222222222222222222222222222222222";

#[test]
fn pull_stores_exactly_the_allowlist_under_its_model_source() {
    let source = FakeSource::new(LAYA);
    source.publish(FIRST, published(FIRST));
    source.point("main", FIRST);
    let root = TempDir::new("allowlist");
    let store = Store::at(root.path());

    let outcome = pull::pull(&store, &Hub::at(source.base_url()), &request()).expect("pull");

    assert_eq!(outcome.directory, root.path().join(LAYA));
    let mut expected: Vec<String> = ALLOWLIST.iter().map(|path| path.to_string()).collect();
    expected.push("provenance.json".to_string());
    expected.sort();
    assert_eq!(
        support::tree(&root.path().join(LAYA)),
        expected,
        "only the allowlist is stored"
    );

    for (path, body) in source.bodies(FIRST) {
        if !ALLOWLIST.contains(&path.as_str()) {
            continue;
        }
        let stored = std::fs::read(root.path().join(LAYA).join(&path)).expect("stored file");
        assert_eq!(stored, body, "{path} holds the published bytes");
    }

    let untouched: Vec<String> = source
        .file_requests()
        .into_iter()
        .map(|request| request.target)
        .filter(|target| target.contains("README") || target.contains("multilingual"))
        .collect();
    assert!(
        untouched.is_empty(),
        "files outside the allowlist are never pulled: {untouched:?}"
    );
}

#[test]
fn pull_records_provenance_for_every_stored_file() {
    let source = FakeSource::new(LAYA);
    source.publish(FIRST, published(FIRST));
    source.point("main", FIRST);
    let root = TempDir::new("provenance");
    let store = Store::at(root.path());

    let outcome = pull::pull(&store, &Hub::at(source.base_url()), &request()).expect("pull");
    let provenance = &outcome.provenance;

    assert_eq!(provenance.source, LAYA);
    assert_eq!(provenance.requested_revision, None);
    assert_eq!(provenance.resolved_revision, FIRST);
    assert_eq!(
        provenance.files.len(),
        ALLOWLIST.len(),
        "every allowlisted file is recorded"
    );

    for (path, body) in source.bodies(FIRST) {
        if !ALLOWLIST.contains(&path.as_str()) {
            continue;
        }
        let record = provenance.file(&path).expect("a record per stored file");
        assert_eq!(record.size, body.len() as u64, "{path} size");
        assert_eq!(record.sha256, support::sha256(&body), "{path} sha256");
        let published = record
            .published
            .as_ref()
            .unwrap_or_else(|| panic!("{path} has a published checksum"));
        let expected = if path.ends_with(".safetensors") {
            assert_eq!(published.algorithm, Algorithm::Sha256);
            support::sha256(&body)
        } else {
            assert_eq!(published.algorithm, Algorithm::GitBlobSha1);
            support::git_blob_sha1(&body)
        };
        assert_eq!(published.checksum, expected, "{path} published checksum");
    }

    // Literals from `sha256sum` and `git hash-object` over this commit's own 58-byte
    // `rl_agent_config.json`, which the Model Source stand-in served above.
    let record = provenance.file("rl_agent_config.json").expect("recorded");
    assert_eq!(record.size, 58);
    assert_eq!(
        record.sha256,
        "2f9c26fd10e5baddc4ea75227c13c2ce5abebca421ab0a335acce847c9afddd5"
    );
    assert_eq!(
        record.published.as_ref().expect("published").checksum,
        "2b20640b0dd371811514ec7faf7798e082922993"
    );

    let recorded: s1gate::provenance::Provenance = serde_json::from_str(
        &std::fs::read_to_string(root.path().join(LAYA).join("provenance.json"))
            .expect("a record on disk"),
    )
    .expect("the record is readable JSON");
    assert_eq!(&recorded, provenance);
}

#[test]
fn pull_resolves_the_default_branch_or_the_requested_revision() {
    let source = FakeSource::new(LAYA);
    source.publish(FIRST, published(FIRST));
    source.publish(SECOND, published(SECOND));
    source.point("main", FIRST);
    source.point("next", SECOND);
    let root = TempDir::new("revisions");
    let store = Store::at(root.path());
    let hub = Hub::at(source.base_url());

    let default = pull::pull(&store, &hub, &request()).expect("pull");
    assert_eq!(default.provenance.resolved_revision, FIRST);
    assert_eq!(default.provenance.requested_revision, None);
    assert_eq!(
        stored(&root, "rl_agent_config.json"),
        source.bodies(FIRST)["rl_agent_config.json"],
        "the default branch's bytes are stored"
    );

    let named = pull::pull(
        &store,
        &hub,
        &PullRequest {
            revision: Some("next".to_string()),
            force: true,
            ..request()
        },
    )
    .expect("pull of a named revision");
    assert_eq!(named.provenance.resolved_revision, SECOND);
    assert_eq!(
        named.provenance.requested_revision.as_deref(),
        Some("next"),
        "the requested ref is recorded beside the commit it resolved to"
    );
    assert_eq!(
        stored(&root, "rl_agent_config.json"),
        source.bodies(SECOND)["rl_agent_config.json"]
    );
}

#[test]
fn pull_rejects_a_revision_the_model_source_does_not_have() {
    let source = FakeSource::new(LAYA);
    source.point("main", FIRST);
    let root = TempDir::new("unknown-revision");
    let store = Store::at(root.path());

    let error = pull::pull(
        &store,
        &Hub::at(source.base_url()),
        &PullRequest {
            revision: Some("no-such-branch".to_string()),
            ..request()
        },
    )
    .expect_err("an unknown revision fails the pull");

    assert!(matches!(error, s1gate::Error::RevisionNotFound { .. }));
    assert_eq!(
        error.to_string(),
        "convaiinnovations/laya has no revision `no-such-branch`"
    );
    assert!(
        !root.path().join(LAYA).exists(),
        "nothing is stored for a revision that does not exist"
    );
}

#[test]
fn pull_fails_when_the_model_source_does_not_publish_an_allowlisted_file() {
    let source = FakeSource::new(LAYA);
    let files: Vec<ServedFile> = published(FIRST)
        .into_iter()
        .filter(|file| file.path != "encoder/config.json")
        .collect();
    source.publish(FIRST, files);
    source.point("main", FIRST);
    let root = TempDir::new("missing-file");
    let store = Store::at(root.path());

    let error = pull::pull(&store, &Hub::at(source.base_url()), &request())
        .expect_err("an unpublished allowlist file fails the pull");

    assert!(matches!(error, s1gate::Error::MissingFile { .. }));
    assert_eq!(
        error.to_string(),
        "convaiinnovations/laya does not publish `encoder/config.json`"
    );
    assert_eq!(
        store.provenance(LAYA).expect("the store is readable"),
        None,
        "an incomplete Checkpoint is never recorded"
    );
}

#[test]
fn an_interrupted_pull_leaves_a_partial_file_the_next_pull_streams_again() {
    let source = FakeSource::new(LAYA);
    source.publish(
        FIRST,
        published(FIRST)
            .into_iter()
            .map(|file| match file.path.as_str() {
                "model.safetensors" => file.truncated_at(4),
                _ => file,
            })
            .collect(),
    );
    source.point("main", FIRST);
    let root = TempDir::new("interrupted");
    let store = Store::at(root.path());
    let hub = Hub::at(source.base_url());
    let checkpoint = root.path().join(LAYA);

    let error = pull::pull(&store, &hub, &request()).expect_err("a truncated Pull fails");
    // The body is shorter than the announced `content-length`, so the read of the stream fails
    // (src/pull/mod.rs: `remote.read(..)` maps to `Error::io("read", &partial, ..)`).
    assert!(
        matches!(error, s1gate::Error::Io { op: "read", .. }),
        "{error}"
    );

    assert!(
        checkpoint.join("model.safetensors.part").exists(),
        "the interrupted Pull is left as a partial file"
    );
    assert!(!checkpoint.join("model.safetensors").exists());
    assert_eq!(store.provenance(LAYA).expect("the store is readable"), None);

    let attempts = source.file_requests().len();
    source.publish(FIRST, published(FIRST));
    let outcome = pull::pull(&store, &hub, &request()).expect("the retry succeeds");

    assert!(
        !checkpoint.join("model.safetensors.part").exists(),
        "the retry leaves no partial file behind"
    );
    assert_eq!(
        stored(&root, "model.safetensors"),
        source.bodies(FIRST)["model.safetensors"],
        "the retry stored the whole file"
    );
    assert_eq!(
        outcome.pulled.first().map(String::as_str),
        Some("model.safetensors")
    );
    let retry: Vec<support::Request> = source.file_requests()[attempts..].to_vec();
    assert!(
        retry
            .iter()
            .any(|request| support::resolve_path(&request.target)
                == Some((FIRST.to_string(), "model.safetensors".to_string()))),
        "the retry asks for the file again: {retry:?}"
    );
    assert!(
        retry
            .iter()
            .all(|request| request.header("range").is_none()),
        "a partial file is never resumed"
    );
}

#[test]
fn pull_rejects_a_file_whose_bytes_do_not_match_the_published_checksum() {
    let source = FakeSource::new(LAYA);
    let root = TempDir::new("checksum");
    let store = Store::at(root.path());
    let hub = Hub::at(source.base_url());
    let checkpoint = root.path().join(LAYA);
    source.point("main", FIRST);

    // An LFS file, published as a sha256 the bytes do not match.
    source.publish(
        FIRST,
        corrupting(published(FIRST), "model.safetensors", &"0".repeat(64)),
    );
    let error = pull::pull(&store, &hub, &request()).expect_err("a corrupt LFS file fails");
    assert!(matches!(error, s1gate::Error::ChecksumMismatch { .. }));
    assert!(
        error
            .to_string()
            .starts_with("`model.safetensors` hashes to "),
        "unexpected message: {error}"
    );
    assert!(!checkpoint.join("model.safetensors").exists());
    assert!(!checkpoint.join("provenance.json").exists());

    // A git-tracked file, published as a git blob object id the bytes do not match.
    source.publish(
        FIRST,
        corrupting(
            published(FIRST),
            "tokenizer/tokenizer_config.json",
            &"1".repeat(40),
        ),
    );
    let error = pull::pull(&store, &hub, &request()).expect_err("a corrupt config file fails");
    assert!(matches!(error, s1gate::Error::ChecksumMismatch { .. }));
    assert!(
        error
            .to_string()
            .starts_with("`tokenizer/tokenizer_config.json` hashes to "),
        "unexpected message: {error}"
    );
    assert!(!checkpoint.join("tokenizer/tokenizer_config.json").exists());
    assert!(!checkpoint.join("provenance.json").exists());
}

#[test]
fn pull_rejects_a_file_served_from_another_commit() {
    let source = FakeSource::new(LAYA);
    source.publish(
        FIRST,
        published(FIRST)
            .into_iter()
            .map(|file| match file.path.as_str() {
                "encoder/config.json" => file.served_from(SECOND),
                _ => file,
            })
            .collect(),
    );
    source.point("main", FIRST);
    let root = TempDir::new("mixed-commits");
    let store = Store::at(root.path());

    let error = pull::pull(&store, &Hub::at(source.base_url()), &request())
        .expect_err("a file from another commit fails the pull");

    assert!(matches!(error, s1gate::Error::UnexpectedCommit { .. }));
    assert_eq!(
        error.to_string(),
        format!(
            "`encoder/config.json` was served from revision {SECOND}, not the resolved revision {FIRST}"
        )
    );
}

#[test]
fn a_repeated_pull_of_the_same_revision_streams_nothing() {
    let source = FakeSource::new(LAYA);
    source.publish(FIRST, published(FIRST));
    source.point("main", FIRST);
    let root = TempDir::new("idempotent");
    let store = Store::at(root.path());
    let hub = Hub::at(source.base_url());

    let first = pull::pull(&store, &hub, &request()).expect("pull");
    let before = source.requests().len();
    let again = pull::pull(&store, &hub, &request()).expect("pull again");

    assert!(again.unchanged(), "nothing was pulled: {:?}", again.pulled);
    assert_eq!(again.provenance, first.provenance);
    let since = source.requests()[before..].to_vec();
    assert_eq!(
        since.len(),
        1,
        "the repeat resolves the revision and nothing else: {since:?}"
    );
    assert!(since[0].target.starts_with("/api/models/"));
    assert_eq!(
        stored(&root, "model.safetensors"),
        source.bodies(FIRST)["model.safetensors"]
    );
}

#[test]
fn a_different_revision_under_the_same_name_needs_force() {
    let source = FakeSource::new(LAYA);
    source.publish(FIRST, published(FIRST));
    source.publish(SECOND, published(SECOND));
    source.point("main", FIRST);
    source.point("next", SECOND);
    let root = TempDir::new("replacement");
    let store = Store::at(root.path());
    let hub = Hub::at(source.base_url());

    pull::pull(&store, &hub, &request()).expect("pull");
    let refusal = pull::pull(
        &store,
        &hub,
        &PullRequest {
            revision: Some("next".to_string()),
            ..request()
        },
    )
    .expect_err("another revision over the same name is refused");

    assert!(matches!(refusal, s1gate::Error::RevisionHeld { .. }));
    assert_eq!(
        refusal.to_string(),
        format!(
            "Checkpoint `{LAYA}` holds revision {FIRST}; pulling {SECOND} over it needs --force"
        )
    );
    assert_eq!(
        store
            .provenance(LAYA)
            .expect("the store is readable")
            .expect("still recorded")
            .resolved_revision,
        FIRST,
        "the refused pull changed nothing"
    );
    assert_eq!(
        stored(&root, "rl_agent_config.json"),
        source.bodies(FIRST)["rl_agent_config.json"]
    );

    let replaced = pull::pull(
        &store,
        &hub,
        &PullRequest {
            revision: Some("next".to_string()),
            force: true,
            ..request()
        },
    )
    .expect("--force replaces the Checkpoint");

    assert_eq!(replaced.provenance.resolved_revision, SECOND);
    assert_eq!(
        replaced.provenance.requested_revision.as_deref(),
        Some("next")
    );
    assert_eq!(
        stored(&root, "rl_agent_config.json"),
        source.bodies(SECOND)["rl_agent_config.json"],
        "the replaced Checkpoint holds the new revision's files"
    );
    assert_eq!(
        replaced.pulled.len(),
        ALLOWLIST.len(),
        "every file is pulled again"
    );
}

#[test]
fn pull_records_no_published_checksum_when_the_model_source_publishes_none() {
    let source = FakeSource::new(LAYA);
    source.publish(
        FIRST,
        published(FIRST)
            .into_iter()
            .map(|file| match file.path.as_str() {
                "model.safetensors" => file.without_checksum(),
                _ => file,
            })
            .collect(),
    );
    source.point("main", FIRST);
    let root = TempDir::new("no-checksum");
    let store = Store::at(root.path());

    let outcome = pull::pull(&store, &Hub::at(source.base_url()), &request()).expect("pull");

    let record = outcome
        .provenance
        .file("model.safetensors")
        .expect("recorded");
    assert_eq!(record.published, None);
    assert_eq!(
        record.size,
        source.bodies(FIRST)["model.safetensors"].len() as u64
    );
    assert_eq!(
        record.sha256,
        support::sha256(&source.bodies(FIRST)["model.safetensors"]),
        "the local checksum is recorded whether or not the Model Source publishes one"
    );
}

#[test]
fn pull_verifies_a_file_the_model_source_serves_without_redirecting() {
    let source = FakeSource::new(LAYA);
    source.publish(
        FIRST,
        corrupting(
            published(FIRST)
                .into_iter()
                .map(|file| file.served_directly())
                .collect(),
            "rl_agent_config.json",
            &"2".repeat(40),
        ),
    );
    source.point("main", FIRST);
    let root = TempDir::new("direct");
    let store = Store::at(root.path());

    let error = pull::pull(&store, &Hub::at(source.base_url()), &request())
        .expect_err("the published checksum is checked however the file arrives");

    assert!(matches!(error, s1gate::Error::ChecksumMismatch { .. }));
    assert!(
        error
            .to_string()
            .starts_with("`rl_agent_config.json` hashes to "),
        "unexpected message: {error}"
    );
    assert!(
        source
            .requests()
            .iter()
            .all(|request| !request.target.contains("/blob/")),
        "the Model Source served the bytes itself, so no presigned transfer was followed"
    );
}

fn corrupting(files: Vec<ServedFile>, path: &str, etag: &str) -> Vec<ServedFile> {
    files
        .into_iter()
        .map(|file| match file.path == path {
            true => file.advertising(etag),
            false => file,
        })
        .collect()
}

fn stored(root: &TempDir, path: &str) -> Vec<u8> {
    std::fs::read(root.path().join(LAYA).join(path))
        .unwrap_or_else(|error| panic!("{path} is stored: {error}"))
}

fn request() -> PullRequest {
    PullRequest {
        source: LAYA.to_string(),
        revision: None,
        force: false,
    }
}

/// The allowlist plus files Pull must leave alone.
fn published(commit: &str) -> Vec<ServedFile> {
    let mut files = support::allowlist_files(commit);
    files.extend(support::unrelated_files());
    files
}
