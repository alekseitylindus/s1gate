//! A stored Checkpoint for tests that read one without a Model Source: the five allowlisted files,
//! the two configurations the Parameter Manifest is derived from, and the Provenance record.
//!
//! The safetensors header is spelled out here rather than asked of the code under test, so a
//! manifest that names or shapes a parameter differently fails the test that uses it.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;

use s1gate::provenance::{FileRecord, Provenance};

use super::sha256;

/// The Model Source every fixture Checkpoint came from, and the revision it records.
pub const NAME: &str = "convaiinnovations/laya";
pub const REVISION: &str = "1c5edc17a7acd8701df6fc341c0d179f1c62c982";

/// The configuration of a Checkpoint with one encoder layer — whose layer 0 carries an identity
/// norm — and no decision-head layer.
pub const ENCODER_CONFIG: &str =
    r#"{"hidden_size":4,"intermediate_size":3,"vocab_size":5,"num_hidden_layers":1}"#;
pub const AGENT_CONFIG: &str = r#"{"head_layers":0,"act_costs":{"escalate":0.5}}"#;

/// The configuration of a Checkpoint with two encoder layers and one decision-head layer, whose
/// header is `layered_header`.
pub const LAYERED_ENCODER_CONFIG: &str =
    r#"{"hidden_size":4,"intermediate_size":3,"vocab_size":5,"num_hidden_layers":2}"#;
pub const LAYERED_AGENT_CONFIG: &str = r#"{"head_layers":1,"act_costs":{"escalate":0.5}}"#;

/// The dimensions the two configurations above describe.
const HIDDEN: u64 = 4;
const INTERMEDIATE: u64 = 3;
const ACTIONS: u64 = 2;

/// Write a Checkpoint of `NAME`, with `ENCODER_CONFIG` and `AGENT_CONFIG`, under the store rooted
/// at `root`; return its directory.
pub fn write(root: &Path, header: BTreeMap<String, Value>) -> PathBuf {
    write_with(root, ENCODER_CONFIG, AGENT_CONFIG, header)
}

/// Write a Checkpoint of `NAME` with `encoder` and `agent` as its two configuration files.
pub fn write_with(
    root: &Path,
    encoder: &str,
    agent: &str,
    header: BTreeMap<String, Value>,
) -> PathBuf {
    let directory = root.join(NAME);
    fs::create_dir_all(directory.join("encoder")).expect("the Checkpoint directories");
    fs::create_dir_all(directory.join("tokenizer")).expect("the Checkpoint directories");
    let files = [
        ("model.safetensors", safetensors(header)),
        ("rl_agent_config.json", agent.as_bytes().to_vec()),
        ("encoder/config.json", encoder.as_bytes().to_vec()),
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
        fs::write(directory.join(path), body).expect("a Checkpoint file");
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
    .expect("the Provenance record");
    directory
}

/// The header of a Checkpoint with `ENCODER_CONFIG` and `AGENT_CONFIG`.
pub fn header() -> BTreeMap<String, Value> {
    [
        ("temperature", tensor("F32", &[3])),
        ("encoder.embeddings.norm.weight", tensor("F16", &[HIDDEN])),
        (
            "encoder.embeddings.tok_embeddings.weight",
            tensor("F16", &[5, HIDDEN]),
        ),
        ("encoder.final_norm.weight", tensor("F16", &[HIDDEN])),
        (
            "encoder.layers.0.attn.Wo.weight",
            tensor("F16", &[HIDDEN, HIDDEN]),
        ),
        (
            "encoder.layers.0.attn.Wqkv.weight",
            tensor("F16", &[3 * HIDDEN, HIDDEN]),
        ),
        (
            "encoder.layers.0.mlp.Wi.weight",
            tensor("F16", &[2 * INTERMEDIATE, HIDDEN]),
        ),
        (
            "encoder.layers.0.mlp.Wo.weight",
            tensor("F16", &[HIDDEN, INTERMEDIATE]),
        ),
        ("encoder.layers.0.mlp_norm.weight", tensor("F16", &[HIDDEN])),
        ("type_emb.weight", tensor("F16", &[3, HIDDEN])),
        ("scorer.0.weight", tensor("F16", &[HIDDEN])),
        ("scorer.0.bias", tensor("F16", &[HIDDEN])),
        ("scorer.1.weight", tensor("F16", &[HIDDEN, HIDDEN])),
        ("scorer.1.bias", tensor("F16", &[HIDDEN])),
        ("scorer.3.weight", tensor("F16", &[1, HIDDEN])),
        ("scorer.3.bias", tensor("F16", &[1])),
        ("act_head.0.weight", tensor("F16", &[256, HIDDEN + 4])),
        ("act_head.0.bias", tensor("F16", &[256])),
        ("act_head.2.weight", tensor("F16", &[ACTIONS, 256])),
        ("act_head.2.bias", tensor("F16", &[ACTIONS])),
    ]
    .into_iter()
    .map(|(name, value)| (name.to_string(), value))
    .collect()
}

/// The header of a Checkpoint with `LAYERED_ENCODER_CONFIG` and `LAYERED_AGENT_CONFIG`: `header`,
/// plus a second encoder layer and `head.layers.0`.
pub fn layered_header() -> BTreeMap<String, Value> {
    let mut header = header();
    for (name, value) in encoder_layer(1).into_iter().chain(head_layer(0)) {
        header.insert(name, value);
    }
    header
}

/// Every parameter encoder layer `index` carries, the norm included: layer 0 has none.
fn encoder_layer(index: u64) -> Vec<(String, Value)> {
    let prefix = format!("encoder.layers.{index}");
    vec![
        (
            format!("{prefix}.attn_norm.weight"),
            tensor("F16", &[HIDDEN]),
        ),
        (
            format!("{prefix}.attn.Wo.weight"),
            tensor("F16", &[HIDDEN, HIDDEN]),
        ),
        (
            format!("{prefix}.attn.Wqkv.weight"),
            tensor("F16", &[3 * HIDDEN, HIDDEN]),
        ),
        (
            format!("{prefix}.mlp.Wi.weight"),
            tensor("F16", &[2 * INTERMEDIATE, HIDDEN]),
        ),
        (
            format!("{prefix}.mlp.Wo.weight"),
            tensor("F16", &[HIDDEN, INTERMEDIATE]),
        ),
        (
            format!("{prefix}.mlp_norm.weight"),
            tensor("F16", &[HIDDEN]),
        ),
    ]
}

/// The twelve parameters decision-head layer `index` carries.
fn head_layer(index: u64) -> Vec<(String, Value)> {
    let prefix = format!("head.layers.{index}");
    let width = 4 * HIDDEN;
    vec![
        (
            format!("{prefix}.linear1.weight"),
            tensor("F16", &[width, HIDDEN]),
        ),
        (format!("{prefix}.linear1.bias"), tensor("F16", &[width])),
        (
            format!("{prefix}.linear2.weight"),
            tensor("F16", &[HIDDEN, width]),
        ),
        (format!("{prefix}.linear2.bias"), tensor("F16", &[HIDDEN])),
        (format!("{prefix}.norm1.weight"), tensor("F16", &[HIDDEN])),
        (format!("{prefix}.norm1.bias"), tensor("F16", &[HIDDEN])),
        (format!("{prefix}.norm2.weight"), tensor("F16", &[HIDDEN])),
        (format!("{prefix}.norm2.bias"), tensor("F16", &[HIDDEN])),
        (
            format!("{prefix}.self_attn.in_proj_weight"),
            tensor("F16", &[3 * HIDDEN, HIDDEN]),
        ),
        (
            format!("{prefix}.self_attn.in_proj_bias"),
            tensor("F16", &[3 * HIDDEN]),
        ),
        (
            format!("{prefix}.self_attn.out_proj.weight"),
            tensor("F16", &[HIDDEN, HIDDEN]),
        ),
        (
            format!("{prefix}.self_attn.out_proj.bias"),
            tensor("F16", &[HIDDEN]),
        ),
    ]
}

/// One safetensors header entry, with room for `safetensors` to fill its data offsets in.
pub fn tensor(dtype: &str, shape: &[u64]) -> Value {
    serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [0, 0]})
}

/// The bytes of a safetensors file holding `header`, its tensors zeroed and laid out in order.
pub fn safetensors(header: BTreeMap<String, Value>) -> Vec<u8> {
    let mut header = header;
    let mut data = Vec::new();
    for value in header.values_mut() {
        let dtype = value["dtype"].as_str().expect("a tensor dtype");
        let size = value["shape"]
            .as_array()
            .expect("a tensor shape")
            .iter()
            .map(|size| size.as_u64().expect("a shape dimension"))
            .product::<u64>()
            * match dtype {
                "F32" => 4,
                _ => 2,
            };
        let start = data.len() as u64;
        data.resize(data.len() + size as usize, 0);
        value["data_offsets"] = serde_json::json!([start, data.len() as u64]);
    }
    let header = serde_json::to_vec(&header).expect("a header");
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(data);
    bytes
}

/// The bytes of a safetensors file holding `header` and no tensor data, for a header no builder
/// here could materialize.
pub fn safetensors_without_data(header: &Value) -> Vec<u8> {
    let header = header.to_string();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header.as_bytes());
    bytes
}
