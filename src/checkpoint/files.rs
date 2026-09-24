//! The files of a Checkpoint, checked against the Provenance record that describes them.

use std::collections::BTreeSet;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use crate::digest::Digest;
use crate::error::{Error, Result};
use crate::provenance::{FileRecord, Provenance};
use crate::store::Checkpoint;

const CHUNK: usize = 64 * 1024;

/// The Provenance record and the Checkpoint allowlist must name exactly the same files. The count
/// and the distinctness of the records, together with every allowlisted path being found below,
/// are what prove that; a record path outside the allowlist is unreachable from here.
pub(super) fn verify_records(checkpoint: &Checkpoint) -> Result<()> {
    let directory = checkpoint.directory();
    let provenance = checkpoint.provenance();
    let expected_paths = checkpoint.source().files;
    if provenance.files.len() != expected_paths.len() {
        return Err(Error::invalid_checkpoint(
            directory.join(Provenance::FILE_NAME),
            format!(
                "Provenance records {} files, expected {}",
                provenance.files.len(),
                expected_paths.len()
            ),
        ));
    }
    let unique: BTreeSet<&str> = provenance
        .files
        .iter()
        .map(|record| record.path.as_str())
        .collect();
    if unique.len() != provenance.files.len() {
        return Err(Error::invalid_checkpoint(
            directory.join(Provenance::FILE_NAME),
            "Provenance contains duplicate file records".to_string(),
        ));
    }
    for path in expected_paths {
        let record = provenance.file(path).ok_or_else(|| {
            Error::invalid_checkpoint(
                directory.join(Provenance::FILE_NAME),
                format!("Provenance has no record for `{path}`"),
            )
        })?;
        verify_file(directory, record)?;
    }
    Ok(())
}

/// Verify one file the Provenance records: its size, the sha256 of its bytes, and the checksum the
/// Model Source published for it. Every failure names the file the way the record does — the path
/// relative to the Checkpoint, as Pull's failures do — so one name identifies it whichever check
/// failed.
fn verify_file(directory: &Path, record: &FileRecord) -> Result<()> {
    let path = directory.join(&record.path);
    let metadata = match std::fs::metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Err(Error::MissingStoredFile {
                path: record.path.clone(),
            });
        }
        Err(error) => return Err(Error::io("inspect", &path, error)),
    };
    if metadata.len() != record.size {
        return Err(Error::StoredSizeMismatch {
            path: record.path.clone(),
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
            path: record.path.clone(),
            expected: record.sha256.clone(),
            actual,
        });
    }
    if let Some(published) = &record.published {
        published.check(&digest, &record.path)?;
    }
    Ok(())
}
