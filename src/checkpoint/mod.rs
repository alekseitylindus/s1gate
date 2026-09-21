//! Offline verification of a stored Checkpoint: every file against the Provenance that records it,
//! and the weights header against the Parameter Manifest the Checkpoint's own configuration
//! describes. Reaches no network and no Model Source (ADR-0003).

mod files;
mod manifest;
mod safetensors;

use std::path::Path;

use crate::error::{Error, Result};
use crate::model_source::ModelSource;
use crate::provenance::Provenance;
use crate::store::Store;

/// The allowlisted Checkpoint files verification reads by name. Pull hashes all five; verification
/// needs the two configurations the Parameter Manifest is derived from and the weight header.
const ENCODER_CONFIG: &str = "encoder/config.json";
const AGENT_CONFIG: &str = "rl_agent_config.json";
const WEIGHTS: &str = "model.safetensors";

/// The result of verifying one Checkpoint.
#[derive(Debug, Clone)]
pub struct Report {
    pub provenance: Provenance,
    /// The files the Provenance records, every one of them verified.
    pub files: usize,
}

/// Verify the stored Checkpoint of `source` without contacting the Model Source (ADR-0003).
pub fn verify(store: &Store, source: &ModelSource) -> Result<Report> {
    let directory = store.checkpoint_dir(source.repo)?;
    let provenance = store
        .provenance(source.repo)?
        .ok_or_else(|| Error::MissingCheckpoint {
            name: source.repo.to_string(),
        })?;

    if provenance.source != source.repo {
        return Err(invalid(
            &directory.join(Provenance::FILE_NAME),
            format!(
                "Provenance names `{}`, expected `{}`",
                provenance.source, source.repo
            ),
        ));
    }
    files::verify_records(&directory, source.files, &provenance)?;

    let manifest = manifest::read(&directory)?;
    safetensors::verify(&directory.join(WEIGHTS), &manifest)?;

    Ok(Report {
        files: provenance.files.len(),
        provenance,
    })
}

/// Something the Checkpoint holds is not what it should be: `path` names the file at fault, which
/// is the Provenance record when the record itself is unusable and a Checkpoint file otherwise.
fn invalid(path: &Path, message: impl Into<String>) -> Error {
    Error::InvalidCheckpoint {
        path: path.to_path_buf(),
        message: message.into(),
    }
}
