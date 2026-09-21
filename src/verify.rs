//! Offline verification of a stored Checkpoint.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::model_source;
use crate::provenance::{Algorithm, Provenance};
use crate::pull::digest::Digest;
use crate::store::Store;

const CHUNK: usize = 64 * 1024;

/// The result of verifying one Checkpoint.
#[derive(Debug, Clone)]
pub struct Report {
    pub directory: PathBuf,
    pub provenance: Provenance,
    pub files: usize,
}

/// Verify one stored Checkpoint without contacting its Model Source.
pub fn verify(store: &Store, name: &str) -> Result<Report> {
    let directory = store.checkpoint_dir(name)?;
    let source = model_source::lookup(name)?;
    let provenance = store
        .provenance(name)?
        .ok_or_else(|| Error::MissingCheckpoint {
            name: name.to_string(),
        })?;

    if provenance.source != source.repo {
        return Err(Error::InvalidCheckpoint {
            path: directory.join(Provenance::FILE_NAME),
            message: format!(
                "Provenance names `{}`, expected `{}`",
                provenance.source, source.repo
            ),
        });
    }
    verify_records(&directory, source.files, &provenance)?;

    let encoder = read_json::<EncoderConfig>(&directory.join("encoder/config.json"))?;
    let agent = read_json::<AgentConfig>(&directory.join("rl_agent_config.json"))?;
    let manifest = Manifest::from_configs(&encoder, &agent)?;
    verify_safetensors(&directory.join("model.safetensors"), &manifest)?;

    Ok(Report {
        directory,
        files: provenance.files.len(),
        provenance,
    })
}

fn verify_records(
    directory: &Path,
    expected_paths: &[&str],
    provenance: &Provenance,
) -> Result<()> {
    if provenance.files.len() != expected_paths.len() {
        return Err(Error::InvalidCheckpoint {
            path: directory.join(Provenance::FILE_NAME),
            message: format!(
                "Provenance records {} files, expected {}",
                provenance.files.len(),
                expected_paths.len()
            ),
        });
    }
    let unique: BTreeSet<&str> = provenance
        .files
        .iter()
        .map(|record| record.path.as_str())
        .collect();
    if unique.len() != provenance.files.len() {
        return Err(Error::InvalidCheckpoint {
            path: directory.join(Provenance::FILE_NAME),
            message: "Provenance contains duplicate file records".to_string(),
        });
    }
    for path in expected_paths {
        let record = provenance
            .file(path)
            .ok_or_else(|| Error::InvalidCheckpoint {
                path: directory.join(Provenance::FILE_NAME),
                message: format!("Provenance has no record for `{path}`"),
            })?;
        verify_file(directory, record)?;
    }
    if let Some(record) = provenance
        .files
        .iter()
        .find(|record| !expected_paths.contains(&record.path.as_str()))
    {
        return Err(Error::InvalidCheckpoint {
            path: directory.join(Provenance::FILE_NAME),
            message: format!("Provenance records unexpected file `{}`", record.path),
        });
    }
    Ok(())
}

fn verify_file(directory: &Path, record: &crate::provenance::FileRecord) -> Result<()> {
    let path = directory.join(&record.path);
    let metadata = match std::fs::metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Err(Error::MissingStoredFile { path });
        }
        Err(error) => return Err(Error::io("inspect", &path, error)),
    };
    if metadata.len() != record.size {
        return Err(Error::StoredSizeMismatch {
            path,
            expected: record.size,
            actual: metadata.len(),
        });
    }

    let mut file = File::open(&path).map_err(|error| Error::io("open", &path, error))?;
    let mut digest = Digest::new(Some(record.size));
    let mut chunk = vec![0u8; CHUNK];
    loop {
        let read = file
            .read(&mut chunk)
            .map_err(|error| Error::io("read", &path, error))?;
        if read == 0 {
            break;
        }
        digest.update(&chunk[..read]);
    }

    let actual = digest.sha256();
    if !actual.eq_ignore_ascii_case(&record.sha256) {
        return Err(Error::StoredChecksumMismatch {
            path,
            expected: record.sha256.clone(),
            actual,
        });
    }
    if let Some(published) = &record.published {
        let actual = match published.algorithm {
            Algorithm::Sha256 => digest.sha256(),
            Algorithm::GitBlobSha1 => digest
                .git_blob_sha1()
                .expect("a local file always has a known size"),
        };
        if !actual.eq_ignore_ascii_case(&published.checksum) {
            return Err(Error::ChecksumMismatch {
                path: record.path.clone(),
                expected: published.checksum.clone(),
                actual,
            });
        }
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
struct EncoderConfig {
    hidden_size: usize,
    intermediate_size: usize,
    vocab_size: usize,
    num_hidden_layers: usize,
}

#[derive(Debug, Deserialize)]
struct AgentConfig {
    head_layers: usize,
    act_costs: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Parameter {
    dtype: &'static str,
    shape: Vec<u64>,
}

#[derive(Debug, Default)]
struct Manifest {
    parameters: BTreeMap<String, Parameter>,
}

impl Manifest {
    fn from_configs(encoder: &EncoderConfig, agent: &AgentConfig) -> Result<Manifest> {
        let hidden = dimension(encoder.hidden_size)?;
        let intermediate = dimension(encoder.intermediate_size)?;
        let vocab = dimension(encoder.vocab_size)?;
        let layers = encoder.num_hidden_layers;
        let doubled_intermediate = intermediate
            .checked_mul(2)
            .ok_or_else(|| invalid_config("intermediate_size is too large"))?;
        let head_width = hidden
            .checked_mul(4)
            .ok_or_else(|| invalid_config("hidden_size is too large"))?;
        let actions = dimension(agent.act_costs.len().saturating_add(1))?;

        let mut manifest = Manifest::default();
        manifest.add("temperature", "F32", vec![3]);
        manifest.add("encoder.embeddings.norm.weight", "F16", vec![hidden]);
        manifest.add(
            "encoder.embeddings.tok_embeddings.weight",
            "F16",
            vec![vocab, hidden],
        );
        manifest.add("encoder.final_norm.weight", "F16", vec![hidden]);
        for layer in 0..layers {
            let prefix = format!("encoder.layers.{layer}");
            if layer > 0 {
                manifest.add(&format!("{prefix}.attn_norm.weight"), "F16", vec![hidden]);
            }
            manifest.add(
                &format!("{prefix}.attn.Wo.weight"),
                "F16",
                vec![hidden, hidden],
            );
            manifest.add(
                &format!("{prefix}.attn.Wqkv.weight"),
                "F16",
                vec![hidden * 3, hidden],
            );
            manifest.add(
                &format!("{prefix}.mlp.Wi.weight"),
                "F16",
                vec![doubled_intermediate, hidden],
            );
            manifest.add(
                &format!("{prefix}.mlp.Wo.weight"),
                "F16",
                vec![hidden, intermediate],
            );
            manifest.add(&format!("{prefix}.mlp_norm.weight"), "F16", vec![hidden]);
        }
        for layer in 0..agent.head_layers {
            let prefix = format!("head.layers.{layer}");
            manifest.add(
                &format!("{prefix}.linear1.weight"),
                "F16",
                vec![head_width, hidden],
            );
            manifest.add(&format!("{prefix}.linear1.bias"), "F16", vec![head_width]);
            manifest.add(
                &format!("{prefix}.linear2.weight"),
                "F16",
                vec![hidden, head_width],
            );
            manifest.add(&format!("{prefix}.linear2.bias"), "F16", vec![hidden]);
            for norm in ["norm1", "norm2"] {
                manifest.add(&format!("{prefix}.{norm}.weight"), "F16", vec![hidden]);
                manifest.add(&format!("{prefix}.{norm}.bias"), "F16", vec![hidden]);
            }
            manifest.add(
                &format!("{prefix}.self_attn.in_proj_weight"),
                "F16",
                vec![hidden * 3, hidden],
            );
            manifest.add(
                &format!("{prefix}.self_attn.in_proj_bias"),
                "F16",
                vec![hidden * 3],
            );
            manifest.add(
                &format!("{prefix}.self_attn.out_proj.weight"),
                "F16",
                vec![hidden, hidden],
            );
            manifest.add(
                &format!("{prefix}.self_attn.out_proj.bias"),
                "F16",
                vec![hidden],
            );
        }
        manifest.add("type_emb.weight", "F16", vec![3, hidden]);
        manifest.add("scorer.0.weight", "F16", vec![hidden]);
        manifest.add("scorer.0.bias", "F16", vec![hidden]);
        manifest.add("scorer.1.weight", "F16", vec![hidden, hidden]);
        manifest.add("scorer.1.bias", "F16", vec![hidden]);
        manifest.add("scorer.3.weight", "F16", vec![1, hidden]);
        manifest.add("scorer.3.bias", "F16", vec![1]);
        manifest.add("act_head.0.weight", "F16", vec![256, hidden + 4]);
        manifest.add("act_head.0.bias", "F16", vec![256]);
        manifest.add("act_head.2.weight", "F16", vec![actions, 256]);
        manifest.add("act_head.2.bias", "F16", vec![actions]);
        Ok(manifest)
    }

    fn add(&mut self, name: &str, dtype: &'static str, shape: Vec<u64>) {
        self.parameters
            .insert(name.to_string(), Parameter { dtype, shape });
    }
}

fn dimension(value: usize) -> Result<u64> {
    u64::try_from(value).map_err(|_| invalid_config("a dimension is too large"))
}

fn invalid_config(message: &str) -> Error {
    Error::InvalidCheckpoint {
        path: PathBuf::from("configuration"),
        message: message.to_string(),
    }
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let bytes = std::fs::read(path).map_err(|error| Error::io("read", path, error))?;
    serde_json::from_slice(&bytes).map_err(|error| Error::json(path.display().to_string(), error))
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct HeaderEntry {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

fn verify_safetensors(path: &Path, manifest: &Manifest) -> Result<()> {
    let length = std::fs::metadata(path)
        .map_err(|error| Error::io("inspect", path, error))?
        .len();
    let mut file = File::open(path).map_err(|error| Error::io("open", path, error))?;
    let mut size = [0u8; 8];
    file.read_exact(&mut size).map_err(|error| {
        invalid_safetensors(path, format!("cannot read header length: {error}"))
    })?;
    let header_len = u64::from_le_bytes(size);
    let available = length.saturating_sub(8);
    if header_len > available {
        return Err(invalid_safetensors(
            path,
            format!("header is {header_len} bytes, but only {available} are available"),
        ));
    }
    let header_len_usize = usize::try_from(header_len)
        .map_err(|_| invalid_safetensors(path, "header is too large"))?;
    let mut bytes = vec![0u8; header_len_usize];
    file.read_exact(&mut bytes)
        .map_err(|error| invalid_safetensors(path, format!("cannot read header: {error}")))?;
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|error| invalid_safetensors(path, format!("header is not JSON: {error}")))?;
    let object = value
        .as_object()
        .ok_or_else(|| invalid_safetensors(path, "header is not an object"))?;
    let data_len = length - 8 - header_len;
    let mut entries = BTreeMap::new();
    let mut ranges = Vec::new();
    for (name, value) in object {
        if name == "__metadata__" {
            if !value.is_object() {
                return Err(invalid_safetensors(path, "__metadata__ is not an object"));
            }
            continue;
        }
        let entry: HeaderEntry = serde_json::from_value(value.clone()).map_err(|error| {
            invalid_safetensors(path, format!("parameter `{name}` is invalid: {error}"))
        })?;
        if entry.data_offsets[0] > entry.data_offsets[1] || entry.data_offsets[1] > data_len {
            return Err(invalid_safetensors(
                path,
                format!("parameter `{name}` has invalid data offsets"),
            ));
        }
        let expected_bytes = tensor_bytes(&entry.dtype, &entry.shape).ok_or_else(|| {
            invalid_safetensors(
                path,
                format!(
                    "parameter `{name}` uses an unsupported dtype `{}`",
                    entry.dtype
                ),
            )
        })?;
        if entry.data_offsets[1] - entry.data_offsets[0] != expected_bytes {
            return Err(invalid_safetensors(
                path,
                format!("parameter `{name}` data length does not match its dtype and shape"),
            ));
        }
        ranges.push((entry.data_offsets[0], entry.data_offsets[1], name.clone()));
        entries.insert(name.clone(), entry);
    }
    ranges.sort_by_key(|range| range.0);
    for pair in ranges.windows(2) {
        if pair[1].0 < pair[0].1 {
            return Err(invalid_safetensors(
                path,
                format!("parameters `{}` and `{}` overlap", pair[0].2, pair[1].2),
            ));
        }
    }

    for (name, expected) in &manifest.parameters {
        let Some(actual) = entries.get(name) else {
            return Err(Error::MissingParameter { name: name.clone() });
        };
        if actual.dtype != expected.dtype || actual.shape != expected.shape {
            return Err(Error::ParameterMismatch {
                name: name.clone(),
                expected_dtype: expected.dtype.to_string(),
                actual_dtype: actual.dtype.clone(),
                expected_shape: expected.shape.clone(),
                actual_shape: actual.shape.clone(),
            });
        }
    }
    if let Some(name) = entries
        .keys()
        .find(|name| !manifest.parameters.contains_key(*name))
    {
        return Err(Error::UnexpectedParameter { name: name.clone() });
    }
    Ok(())
}

fn tensor_bytes(dtype: &str, shape: &[u64]) -> Option<u64> {
    let bytes_per_element = match dtype {
        "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" => 1,
        "U16" | "I16" | "F16" | "BF16" => 2,
        "U32" | "I32" | "F32" => 4,
        "U64" | "I64" | "F64" => 8,
        _ => return None,
    };
    shape
        .iter()
        .try_fold(1u64, |elements, size| elements.checked_mul(*size))
        .and_then(|elements| elements.checked_mul(bytes_per_element))
}

fn invalid_safetensors(path: &Path, message: impl Into<String>) -> Error {
    Error::InvalidCheckpoint {
        path: path.to_path_buf(),
        message: message.into(),
    }
}
