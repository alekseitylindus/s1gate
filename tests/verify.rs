use std::collections::BTreeMap;
use std::fs;

use s1gate::provenance::{Algorithm, FileRecord, Provenance, PublishedChecksum};
use s1gate::store::Store;
use s1gate::verify;

const NAME: &str = "convaiinnovations/laya";
const REVISION: &str = "1c5edc17a7acd8701df6fc341c0d179f1c62c982";

#[test]
fn an_intact_checkpoint_verifies_from_local_files() {
    let root = temp_dir("intact");
    write_checkpoint(&root, valid_header());

    let report = verify::verify(&Store::at(&root), NAME).expect("the Checkpoint verifies");

    assert_eq!(report.provenance.resolved_revision, REVISION);
    assert_eq!(report.files, 5);
}

#[test]
fn verification_reports_a_changed_file() {
    let root = temp_dir("changed-file");
    write_checkpoint(&root, valid_header());
    let path = root.join(NAME).join("encoder/config.json");
    let size = fs::metadata(&path).expect("the config exists").len() as usize;
    fs::write(&path, vec![b'x'; size]).expect("change the local file");

    let error = verify::verify(&Store::at(&root), NAME).expect_err("the changed file fails");

    assert!(error.to_string().contains("encoder/config.json"));
    assert!(error.to_string().contains("sha256"));
}

#[test]
fn verification_reports_a_missing_file() {
    let root = temp_dir("missing-file");
    write_checkpoint(&root, valid_header());
    fs::remove_file(root.join(NAME).join("model.safetensors")).expect("remove a file");

    let error = verify::verify(&Store::at(&root), NAME).expect_err("the missing file fails");

    assert!(error.to_string().contains("model.safetensors"));
    assert!(error.to_string().contains("missing"));
}

#[test]
fn verification_reports_a_size_change() {
    let root = temp_dir("size-change");
    write_checkpoint(&root, valid_header());
    let path = root.join(NAME).join("encoder/config.json");
    fs::write(&path, b"short").expect("change the local file size");

    let error = verify::verify(&Store::at(&root), NAME).expect_err("the size change fails");

    assert!(error.to_string().contains("bytes"));
    assert!(error.to_string().contains("encoder/config.json"));
}

#[test]
fn verification_reports_disagreement_with_a_published_checksum() {
    let root = temp_dir("published-checksum");
    write_checkpoint(&root, valid_header());
    let store = Store::at(&root);
    let mut provenance = store
        .provenance(NAME)
        .expect("read Provenance")
        .expect("a Provenance record");
    provenance.files[0].published = Some(PublishedChecksum {
        algorithm: Algorithm::Sha256,
        checksum: "0".repeat(64),
    });
    store
        .record_provenance(NAME, &provenance)
        .expect("rewrite Provenance");

    let error = verify::verify(&store, NAME).expect_err("the published checksum fails");

    assert!(error.to_string().contains("Model Source publishes"));
    assert!(error.to_string().contains("model.safetensors"));
}

#[test]
fn verification_rejects_an_unexpected_parameter_before_loading_weights() {
    let root = temp_dir("extra-parameter");
    let mut header = valid_header();
    header.insert("unexpected.weight".to_string(), tensor("F16", &[1]));
    write_checkpoint(&root, header);

    let error = verify::verify(&Store::at(&root), NAME).expect_err("the extra parameter fails");

    assert!(error.to_string().contains("unexpected.weight"));
    assert!(error.to_string().contains("unexpected parameter"));
}

#[test]
fn verification_rejects_a_missing_parameter() {
    let root = temp_dir("missing-parameter");
    let mut header = valid_header();
    header.remove("scorer.3.bias");
    write_checkpoint(&root, header);

    let error = verify::verify(&Store::at(&root), NAME).expect_err("the missing parameter fails");

    assert!(
        error
            .to_string()
            .contains("missing parameter `scorer.3.bias`")
    );
}

#[test]
fn verification_rejects_a_mis_shaped_parameter() {
    let root = temp_dir("wrong-shape");
    let mut header = valid_header();
    header.insert("temperature".to_string(), tensor("F16", &[3]));
    write_checkpoint(&root, header);

    let error = verify::verify(&Store::at(&root), NAME).expect_err("the wrong shape fails");

    assert!(error.to_string().contains("parameter `temperature`"));
    assert!(error.to_string().contains("expected F32"));
}

#[test]
fn verification_rejects_a_malformed_safetensors_header() {
    let root = temp_dir("bad-header");
    write_checkpoint(&root, valid_header());
    let path = root.join(NAME).join("model.safetensors");
    let mut bytes = fs::read(&path).expect("weights");
    bytes[8] = b'!';
    fs::write(&path, &bytes).expect("corrupt the header");
    refresh_record(&root, "model.safetensors", &bytes);

    let error = verify::verify(&Store::at(&root), NAME).expect_err("the malformed header fails");

    assert!(error.to_string().contains("invalid Checkpoint file"));
    assert!(error.to_string().contains("header"));
}

#[test]
fn store_lists_checkpoints_two_levels_deep() {
    let root = temp_dir("listing");
    let other = root.join("other/source");
    fs::create_dir_all(&other).expect("checkpoint directory");

    assert_eq!(
        Store::at(&root).checkpoint_names().unwrap(),
        vec!["other/source"]
    );
}

fn write_checkpoint(root: &std::path::Path, header: BTreeMap<String, serde_json::Value>) {
    let directory = root.join(NAME);
    fs::create_dir_all(directory.join("encoder")).expect("encoder directory");
    fs::create_dir_all(directory.join("tokenizer")).expect("tokenizer directory");
    let files = [
        ("model.safetensors", safetensors(header)),
        (
            "rl_agent_config.json",
            br#"{"head_layers":0,"act_costs":{"escalate":0.5}}"#.to_vec(),
        ),
        (
            "encoder/config.json",
            br#"{"hidden_size":4,"intermediate_size":3,"vocab_size":5,"num_hidden_layers":1}"#
                .to_vec(),
        ),
        ("tokenizer/tokenizer.json", b"{}".to_vec()),
        ("tokenizer/tokenizer_config.json", b"{}".to_vec()),
    ];
    let records = files
        .iter()
        .map(|(path, body)| FileRecord {
            path: (*path).to_string(),
            size: body.len() as u64,
            sha256: sha256(body),
            published: None,
        })
        .collect();
    for (path, body) in files {
        fs::write(directory.join(path), body).expect("Checkpoint file");
    }
    fs::write(
        directory.join("provenance.json"),
        Provenance {
            source: NAME.to_string(),
            requested_revision: None,
            resolved_revision: REVISION.to_string(),
            files: records,
        }
        .to_json(),
    )
    .expect("Provenance file");
}

fn refresh_record(root: &std::path::Path, path: &str, body: &[u8]) {
    let store = Store::at(root);
    let mut provenance = store
        .provenance(NAME)
        .expect("read Provenance")
        .expect("a Provenance record");
    let record = provenance
        .files
        .iter_mut()
        .find(|record| record.path == path)
        .expect("the file record");
    record.size = body.len() as u64;
    record.sha256 = sha256(body);
    store
        .record_provenance(NAME, &provenance)
        .expect("rewrite Provenance");
}

fn valid_header() -> BTreeMap<String, serde_json::Value> {
    [
        ("temperature", tensor("F32", &[3])),
        ("encoder.embeddings.norm.weight", tensor("F16", &[4])),
        (
            "encoder.embeddings.tok_embeddings.weight",
            tensor("F16", &[5, 4]),
        ),
        ("encoder.final_norm.weight", tensor("F16", &[4])),
        ("encoder.layers.0.attn.Wo.weight", tensor("F16", &[4, 4])),
        ("encoder.layers.0.attn.Wqkv.weight", tensor("F16", &[12, 4])),
        ("encoder.layers.0.mlp.Wi.weight", tensor("F16", &[6, 4])),
        ("encoder.layers.0.mlp.Wo.weight", tensor("F16", &[4, 3])),
        ("encoder.layers.0.mlp_norm.weight", tensor("F16", &[4])),
        ("type_emb.weight", tensor("F16", &[3, 4])),
        ("scorer.0.weight", tensor("F16", &[4])),
        ("scorer.0.bias", tensor("F16", &[4])),
        ("scorer.1.weight", tensor("F16", &[4, 4])),
        ("scorer.1.bias", tensor("F16", &[4])),
        ("scorer.3.weight", tensor("F16", &[1, 4])),
        ("scorer.3.bias", tensor("F16", &[1])),
        ("act_head.0.weight", tensor("F16", &[256, 8])),
        ("act_head.0.bias", tensor("F16", &[256])),
        ("act_head.2.weight", tensor("F16", &[2, 256])),
        ("act_head.2.bias", tensor("F16", &[2])),
    ]
    .into_iter()
    .map(|(name, value)| (name.to_string(), value))
    .collect()
}

fn tensor(dtype: &str, shape: &[u64]) -> serde_json::Value {
    serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [0, 0]})
}

fn safetensors(header: BTreeMap<String, serde_json::Value>) -> Vec<u8> {
    let mut header = header;
    let mut data = Vec::new();
    for value in header.values_mut() {
        let dtype = value["dtype"].as_str().expect("tensor dtype");
        let size = value["shape"]
            .as_array()
            .expect("tensor shape")
            .iter()
            .map(|size| size.as_u64().expect("shape dimension"))
            .product::<u64>()
            * match dtype {
                "F32" => 4,
                _ => 2,
            };
        let start = data.len() as u64;
        data.resize(data.len() + size as usize, 0);
        value["data_offsets"] = serde_json::json!([start, data.len() as u64]);
    }
    let header = serde_json::to_vec(&header).expect("header JSON");
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(data);
    bytes
}

fn sha256(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut text = String::new();
    for byte in Sha256::digest(bytes) {
        text.push_str(&format!("{byte:02x}"));
    }
    text
}

fn temp_dir(case: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("s1gate-verify-{case}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).expect("temporary directory");
    path
}
