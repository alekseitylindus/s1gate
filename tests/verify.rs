//! Verification of a Checkpoint the Model Store holds, against its Provenance and its own
//! configuration, without a Model Source to reach.

mod support;

use std::fs;
use std::path::Path;

use support::checkpoint;
use support::{TempDir, git_blob_sha1, sha256, tree};

use s1gate::Error;
use s1gate::model_source::{self, ModelSource};
use s1gate::provenance::{Algorithm, Provenance, PublishedChecksum};
use s1gate::store::Store;
use s1gate::verify;

/// The curated Model Source the fixture Checkpoint came from.
fn source() -> &'static ModelSource {
    model_source::lookup(checkpoint::NAME).expect("laya is curated")
}

#[test]
fn an_intact_checkpoint_verifies_from_local_files() {
    let root = TempDir::new("verify-intact");
    let directory = checkpoint::write(root.path(), checkpoint::header());
    let before = tree(&directory);

    let report =
        verify::verify(&Store::at(root.path()), source()).expect("the Checkpoint verifies");

    assert_eq!(report.provenance.resolved_revision, checkpoint::REVISION);
    assert_eq!(report.files, 5);
    assert_eq!(tree(&directory), before, "verification writes nothing");
}

#[test]
fn a_checkpoint_of_head_layers_and_several_encoder_layers_verifies() {
    let root = TempDir::new("verify-layered");
    checkpoint::write_with(
        root.path(),
        checkpoint::LAYERED_ENCODER_CONFIG,
        checkpoint::LAYERED_AGENT_CONFIG,
        checkpoint::layered_header(),
    );

    let report =
        verify::verify(&Store::at(root.path()), source()).expect("the Checkpoint verifies");

    assert_eq!(report.files, 5);
}

#[test]
fn verification_rejects_a_norm_on_the_encoder_layer_that_has_none() {
    let root = TempDir::new("verify-layer-zero-norm");
    let mut header = checkpoint::layered_header();
    header.insert(
        "encoder.layers.0.attn_norm.weight".to_string(),
        checkpoint::tensor("F16", &[4]),
    );
    checkpoint::write_with(
        root.path(),
        checkpoint::LAYERED_ENCODER_CONFIG,
        checkpoint::LAYERED_AGENT_CONFIG,
        header,
    );

    let error = verify::verify(&Store::at(root.path()), source())
        .expect_err("layer 0 has an identity norm");

    assert!(matches!(error, Error::UnexpectedParameter { .. }));
    assert!(
        error
            .to_string()
            .contains("`encoder.layers.0.attn_norm.weight`"),
        "{error}"
    );
}

#[test]
fn verification_rejects_a_missing_head_layer_parameter() {
    let root = TempDir::new("verify-missing-head-parameter");
    let mut header = checkpoint::layered_header();
    header.remove("head.layers.0.self_attn.out_proj.bias");
    checkpoint::write_with(
        root.path(),
        checkpoint::LAYERED_ENCODER_CONFIG,
        checkpoint::LAYERED_AGENT_CONFIG,
        header,
    );

    let error =
        verify::verify(&Store::at(root.path()), source()).expect_err("the parameter is gone");

    assert!(matches!(error, Error::MissingParameter { .. }));
    assert_eq!(
        error.to_string(),
        "missing parameter `head.layers.0.self_attn.out_proj.bias`"
    );
}

#[test]
fn verification_reports_a_changed_file() {
    let root = TempDir::new("verify-changed-file");
    let directory = checkpoint::write(root.path(), checkpoint::header());
    let path = directory.join("encoder/config.json");
    let size = fs::metadata(&path).expect("the config exists").len() as usize;
    fs::write(&path, vec![b'x'; size]).expect("change the local file");

    let error =
        verify::verify(&Store::at(root.path()), source()).expect_err("the changed file fails");

    assert!(matches!(error, Error::StoredChecksumMismatch { .. }));
    assert!(error.to_string().contains("encoder/config.json"), "{error}");
}

#[test]
fn verification_reports_a_missing_file() {
    let root = TempDir::new("verify-missing-file");
    let directory = checkpoint::write(root.path(), checkpoint::header());
    fs::remove_file(directory.join("model.safetensors")).expect("remove a file");

    let error =
        verify::verify(&Store::at(root.path()), source()).expect_err("the missing file fails");

    assert!(matches!(error, Error::MissingStoredFile { .. }));
    assert_eq!(
        error.to_string(),
        "Checkpoint file `model.safetensors` is missing"
    );
}

#[test]
fn verification_reports_a_size_change() {
    let root = TempDir::new("verify-size-change");
    let directory = checkpoint::write(root.path(), checkpoint::header());
    fs::write(directory.join("encoder/config.json"), b"short").expect("change the local file size");

    let error =
        verify::verify(&Store::at(root.path()), source()).expect_err("the size change fails");

    assert!(matches!(error, Error::StoredSizeMismatch { .. }));
    assert!(error.to_string().contains("encoder/config.json"), "{error}");
}

#[test]
fn verification_accepts_the_checksums_the_model_source_publishes() {
    let root = TempDir::new("verify-published");
    let directory = checkpoint::write(root.path(), checkpoint::header());
    // The Model Source publishes a sha256 for the file it holds in LFS and a git blob object id for
    // the four it tracks in git (ADR-0010).
    rewrite_provenance(root.path(), |provenance| {
        for record in &mut provenance.files {
            let body = fs::read(directory.join(&record.path)).expect("a Checkpoint file");
            record.published = Some(if record.path == "model.safetensors" {
                PublishedChecksum {
                    algorithm: Algorithm::Sha256,
                    checksum: sha256(&body),
                }
            } else {
                PublishedChecksum {
                    algorithm: Algorithm::GitBlobSha1,
                    checksum: git_blob_sha1(&body),
                }
            });
        }
    });

    let report =
        verify::verify(&Store::at(root.path()), source()).expect("the published values agree");

    assert_eq!(report.files, 5);
}

#[test]
fn verification_reports_a_published_checksum_that_disagrees() {
    let root = TempDir::new("verify-published-sha256");
    checkpoint::write(root.path(), checkpoint::header());
    rewrite_provenance(root.path(), |provenance| {
        provenance.files[0].published = Some(PublishedChecksum {
            algorithm: Algorithm::Sha256,
            checksum: "0".repeat(64),
        });
    });

    let error = verify::verify(&Store::at(root.path()), source())
        .expect_err("the published checksum fails");

    assert!(matches!(error, Error::ChecksumMismatch { .. }));
    assert!(
        error.to_string().contains("Model Source publishes"),
        "{error}"
    );
    assert!(error.to_string().contains("model.safetensors"), "{error}");
}

#[test]
fn verification_reports_a_published_blob_id_that_disagrees() {
    let root = TempDir::new("verify-published-blob");
    checkpoint::write(root.path(), checkpoint::header());
    rewrite_provenance(root.path(), |provenance| {
        provenance.files[1].published = Some(PublishedChecksum {
            algorithm: Algorithm::GitBlobSha1,
            checksum: "0".repeat(40),
        });
    });

    let error =
        verify::verify(&Store::at(root.path()), source()).expect_err("the published blob id fails");

    assert!(matches!(error, Error::ChecksumMismatch { .. }));
    assert!(
        error.to_string().contains("rl_agent_config.json"),
        "{error}"
    );
}

#[test]
fn verification_rejects_a_parameter_the_manifest_does_not_ask_for() {
    let root = TempDir::new("verify-unexpected-parameter");
    let mut header = checkpoint::header();
    header.insert(
        "unexpected.weight".to_string(),
        checkpoint::tensor("F16", &[1]),
    );
    checkpoint::write(root.path(), header);

    let error =
        verify::verify(&Store::at(root.path()), source()).expect_err("the extra parameter fails");

    assert!(matches!(error, Error::UnexpectedParameter { .. }));
    assert!(error.to_string().contains("`unexpected.weight`"), "{error}");
}

#[test]
fn verification_rejects_a_parameter_of_the_wrong_shape() {
    let root = TempDir::new("verify-wrong-shape");
    let mut header = checkpoint::header();
    header.insert("temperature".to_string(), checkpoint::tensor("F16", &[3]));
    checkpoint::write(root.path(), header);

    let error =
        verify::verify(&Store::at(root.path()), source()).expect_err("the wrong shape fails");

    assert!(matches!(error, Error::ParameterMismatch { .. }));
    assert!(error.to_string().contains("`temperature`"), "{error}");
    assert!(error.to_string().contains("expected F32"), "{error}");
}

#[test]
fn verification_rejects_a_dtype_safetensors_does_not_define() {
    let root = TempDir::new("verify-unsupported-dtype");
    let mut header = checkpoint::header();
    header.insert("temperature".to_string(), checkpoint::tensor("F4", &[3]));
    checkpoint::write(root.path(), header);

    let error = verify::verify(&Store::at(root.path()), source()).expect_err("the dtype fails");

    assert!(matches!(error, Error::InvalidCheckpoint { .. }));
    assert!(
        error.to_string().contains("uses an unsupported dtype `F4`"),
        "{error}"
    );
}

#[test]
fn verification_rejects_a_shape_that_cannot_be_a_length() {
    let root = TempDir::new("verify-huge-shape");
    let directory = checkpoint::write(root.path(), checkpoint::header());
    // Written by hand: a tensor of this shape has no length, and no builder here could lay it out.
    let header = serde_json::json!({
        "big.weight": {"dtype": "F16", "shape": [u64::MAX, 2], "data_offsets": [0, 0]},
    });
    let bytes = checkpoint::safetensors_without_data(&header);
    fs::write(directory.join("model.safetensors"), &bytes).expect("corrupt the weights");
    record_file(root.path(), "model.safetensors", &bytes);

    let error = verify::verify(&Store::at(root.path()), source()).expect_err("the shape fails");

    assert!(matches!(error, Error::InvalidCheckpoint { .. }));
    assert!(
        error
            .to_string()
            .contains("`big.weight` shape is too large"),
        "{error}"
    );
}

#[test]
fn verification_rejects_a_malformed_safetensors_header() {
    let root = TempDir::new("verify-bad-header");
    let directory = checkpoint::write(root.path(), checkpoint::header());
    let path = directory.join("model.safetensors");
    let mut bytes = fs::read(&path).expect("the weights");
    bytes[8] = b'!';
    fs::write(&path, &bytes).expect("corrupt the header");
    record_file(root.path(), "model.safetensors", &bytes);

    let error =
        verify::verify(&Store::at(root.path()), source()).expect_err("the malformed header fails");

    assert!(matches!(error, Error::InvalidCheckpoint { .. }));
    assert!(error.to_string().contains("header"), "{error}");
}

#[test]
fn verification_rejects_a_record_that_names_another_model_source() {
    let root = TempDir::new("verify-foreign-source");
    checkpoint::write(root.path(), checkpoint::header());
    rewrite_provenance(root.path(), |provenance| {
        provenance.source = "someone/else".to_string();
    });

    let error =
        verify::verify(&Store::at(root.path()), source()).expect_err("a foreign record fails");

    assert!(matches!(error, Error::InvalidCheckpoint { .. }));
    assert!(
        error
            .to_string()
            .contains("Provenance names `someone/else`"),
        "{error}"
    );
}

#[test]
fn verification_rejects_a_record_with_duplicate_files() {
    let root = TempDir::new("verify-duplicate-records");
    checkpoint::write(root.path(), checkpoint::header());
    rewrite_provenance(root.path(), |provenance| {
        let path = provenance.files[0].path.clone();
        provenance.files[1].path = path;
    });

    let error = verify::verify(&Store::at(root.path()), source()).expect_err("the duplicate fails");

    assert!(matches!(error, Error::InvalidCheckpoint { .. }));
    assert!(
        error.to_string().contains("duplicate file records"),
        "{error}"
    );
}

#[test]
fn verification_rejects_a_record_missing_an_allowlisted_file() {
    let root = TempDir::new("verify-foreign-record");
    checkpoint::write(root.path(), checkpoint::header());
    rewrite_provenance(root.path(), |provenance| {
        provenance.files[0].path = "README.md".to_string();
    });

    let error = verify::verify(&Store::at(root.path()), source()).expect_err("the record fails");

    assert!(matches!(error, Error::InvalidCheckpoint { .. }));
    assert!(
        error
            .to_string()
            .contains("Provenance has no record for `model.safetensors`"),
        "{error}"
    );
}

#[test]
fn verification_rejects_a_record_with_the_wrong_number_of_files() {
    let root = TempDir::new("verify-record-count");
    checkpoint::write(root.path(), checkpoint::header());
    rewrite_provenance(root.path(), |provenance| {
        provenance.files.pop();
    });

    let error = verify::verify(&Store::at(root.path()), source()).expect_err("the count fails");

    assert!(matches!(error, Error::InvalidCheckpoint { .. }));
    assert!(
        error
            .to_string()
            .contains("Provenance records 4 files, expected 5"),
        "{error}"
    );
}

#[test]
fn verification_rejects_a_record_that_cannot_be_read() {
    let root = TempDir::new("verify-truncated-record");
    let directory = checkpoint::write(root.path(), checkpoint::header());
    fs::write(directory.join("provenance.json"), b"{\"source\":").expect("truncate the record");

    let error =
        verify::verify(&Store::at(root.path()), source()).expect_err("the truncated record fails");

    assert!(matches!(error, Error::Json { .. }));
    assert!(error.to_string().contains("provenance.json"), "{error}");
}

#[test]
fn verification_reports_a_checkpoint_the_store_does_not_hold() {
    let root = TempDir::new("verify-absent");

    let error = verify::verify(&Store::at(root.path()), source()).expect_err("nothing is stored");

    assert!(matches!(error, Error::MissingCheckpoint { .. }));
}

/// Rewrite the fixture Checkpoint's Provenance record, so a test can record what no Pull would.
fn rewrite_provenance(root: &Path, edit: impl FnOnce(&mut Provenance)) {
    let store = Store::at(root);
    let mut provenance = store
        .provenance(checkpoint::NAME)
        .expect("read Provenance")
        .expect("a Provenance record");
    edit(&mut provenance);
    store
        .record_provenance(checkpoint::NAME, &provenance)
        .expect("rewrite Provenance");
}

/// Record `body` as what the Checkpoint file `path` holds, so a test can corrupt the file itself.
fn record_file(root: &Path, path: &str, body: &[u8]) {
    rewrite_provenance(root, |provenance| {
        let record = provenance
            .files
            .iter_mut()
            .find(|record| record.path == path)
            .expect("the file record");
        record.size = body.len() as u64;
        record.sha256 = sha256(body);
    });
}
