//! Offline verification of a stored Checkpoint: every file against the Provenance that records it,
//! and the weights header against the Parameter Manifest the Checkpoint's own configuration
//! describes. Reaches no network and no Model Source (ADR-0003).

mod files;
mod manifest;
mod safetensors;

use crate::error::Result;
use crate::model_source;
use crate::provenance::Provenance;
use crate::store::Checkpoint;

/// The result of verifying one Checkpoint.
#[derive(Debug, Clone)]
pub struct Report {
    /// The record every stored file was verified against.
    pub provenance: Provenance,
    /// The files the Provenance records, every one of them verified.
    pub files: usize,
}

/// Verify a Checkpoint the Model Store holds without contacting the Model Source (ADR-0003).
///
/// # Errors
///
/// [`Error::InvalidCheckpoint`] when the Provenance record, a configuration or the weights header
/// is not what it must be; [`Error::MissingStoredFile`], [`Error::StoredSizeMismatch`],
/// [`Error::StoredChecksumMismatch`] and [`Error::ChecksumMismatch`] when a stored file is not the
/// file the record describes; [`Error::MissingParameter`], [`Error::UnexpectedParameter`] and
/// [`Error::ParameterMismatch`] when the weights header disagrees with the Parameter Manifest; and
/// [`Error::Io`] or [`Error::Json`] when a Checkpoint file cannot be read.
pub fn verify(checkpoint: &Checkpoint) -> Result<Report> {
    let directory = checkpoint.directory();
    let provenance = checkpoint.provenance();
    files::verify_records(checkpoint)?;

    let manifest = manifest::read(directory)?;
    safetensors::verify(&directory.join(model_source::WEIGHTS_FILE), &manifest)?;

    Ok(Report {
        files: provenance.files.len(),
        provenance: provenance.clone(),
    })
}
