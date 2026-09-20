//! Pull: stream a Checkpoint's files from a Model Source into the Model Store and record its
//! Provenance.
//!
//! This is the only operation in s1gate that reaches the network (ADR-0003), which is why the HTTP
//! client lives inside this module and nowhere else.
//!
//! A Pull writes each file to `<path>.part` and renames it only after the bytes match the checksum
//! the Model Source publishes, so a file under its final name is always the file the record
//! describes. The Provenance record is written last, so its presence marks a complete Checkpoint.

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::model_source::{self, ModelSource};
use crate::provenance::{Algorithm, FileRecord, Provenance, PublishedChecksum};
use crate::store::{PART_SUFFIX, Store};

pub mod digest;
pub mod hub;

use digest::Digest;
pub use hub::{Hub, RemoteFile};

/// How many bytes of a Checkpoint file Pull holds in memory at once.
const CHUNK: usize = 64 * 1024;

/// One Pull, as the caller asks for it.
#[derive(Debug, Clone)]
pub struct PullRequest {
    /// The name the Checkpoint is stored under.
    pub name: String,
    /// The Model Source repository to pull from.
    pub source: String,
    /// The revision to pull, or `None` for the Model Source's default branch.
    pub revision: Option<String>,
    /// Replace a Checkpoint already stored under this name.
    pub force: bool,
}

/// What a Pull stored.
#[derive(Debug, Clone)]
pub struct Outcome {
    pub name: String,
    /// The Checkpoint directory.
    pub directory: PathBuf,
    pub provenance: Provenance,
    /// The files this Pull streamed; empty when the name already held the resolved revision.
    pub pulled: Vec<String>,
}

impl Outcome {
    /// Whether this Pull streamed nothing because the Checkpoint was already there.
    pub fn unchanged(&self) -> bool {
        self.pulled.is_empty()
    }
}

/// Pull `request` from `hub` into `store`.
pub fn pull(store: &Store, hub: &Hub, request: &PullRequest) -> Result<Outcome> {
    let source = model_source::lookup(&request.source)?;
    let directory = store.checkpoint_dir(&request.name)?;
    let held = store.provenance(&request.name)?;
    let resolved = hub.resolve(source.repo, request.revision.as_deref())?;

    let reusing = match &held {
        Some(held) if held.resolved_revision == resolved => !request.force,
        Some(held) => {
            if !request.force {
                return Err(Error::RevisionHeld {
                    name: request.name.clone(),
                    held: held.resolved_revision.clone(),
                    requested: resolved,
                });
            }
            false
        }
        None => false,
    };
    if !reusing {
        store.discard_checkpoint(&request.name)?;
    }
    std::fs::create_dir_all(&directory).map_err(|error| Error::io("create", &directory, error))?;

    let mut files = Vec::with_capacity(source.files.len());
    let mut pulled = Vec::new();
    for path in source.files {
        match stored(&directory, path, held.as_ref(), reusing) {
            Some(record) => files.push(record),
            None => {
                eprintln!("s1gate: pulling {path}");
                let record = stream_file(hub, source, &resolved, &directory, path)?;
                eprintln!("s1gate: stored {path} ({} bytes)", record.size);
                files.push(record);
                pulled.push((*path).to_string());
            }
        }
    }

    let provenance = Provenance {
        source: source.repo.to_string(),
        requested_revision: request.revision.clone(),
        resolved_revision: resolved,
        files,
    };
    store.record_provenance(&request.name, &provenance)?;
    Ok(Outcome {
        name: request.name.clone(),
        directory,
        provenance,
        pulled,
    })
}

/// The record of `path` when the Checkpoint already holds it at the recorded size, so Pull can skip
/// streaming it again. Loading checks presence and size; `verify` is what re-hashes.
fn stored(
    directory: &Path,
    path: &str,
    held: Option<&Provenance>,
    reusing: bool,
) -> Option<FileRecord> {
    if !reusing {
        return None;
    }
    let record = held?.file(path)?;
    let size = std::fs::metadata(directory.join(path)).ok()?.len();
    (size == record.size).then(|| record.clone())
}

/// Stream one allowlisted file of a Checkpoint into its place in the Checkpoint directory.
fn stream_file(
    hub: &Hub,
    source: &ModelSource,
    commit: &str,
    directory: &Path,
    path: &str,
) -> Result<FileRecord> {
    let target = directory.join(path);
    let partial = directory.join(format!("{path}{PART_SUFFIX}"));
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent).map_err(|error| Error::io("create", parent, error))?;
    }

    let mut remote = hub.open(source.repo, commit, path)?;
    let mut digest = Digest::new(remote.size());
    let mut file = File::create(&partial).map_err(|error| Error::io("create", &partial, error))?;
    let mut chunk = vec![0u8; CHUNK];
    loop {
        let read = remote
            .read(&mut chunk)
            .map_err(|error| Error::io("read", &partial, error))?;
        if read == 0 {
            break;
        }
        digest.update(&chunk[..read]);
        file.write_all(&chunk[..read])
            .map_err(|error| Error::io("write", &partial, error))?;
    }
    file.flush()
        .map_err(|error| Error::io("write", &partial, error))?;
    drop(file);

    if let Some(announced) = remote.size()
        && digest.len() != announced
    {
        return Err(Error::SizeMismatch {
            path: path.to_string(),
            expected: announced,
            actual: digest.len(),
        });
    }
    if let Some(published) = remote.published() {
        verify(published, &digest, path)?;
    }

    std::fs::rename(&partial, &target).map_err(|error| Error::io("rename", &target, error))?;
    Ok(FileRecord {
        path: path.to_string(),
        size: digest.len(),
        sha256: digest.sha256(),
        published: remote.published().cloned(),
    })
}

/// Check the streamed bytes against the checksum the Model Source publishes. A git blob digest
/// cannot be computed when the Model Source announced no size, and then there is nothing to check.
fn verify(published: &PublishedChecksum, digest: &Digest, path: &str) -> Result<()> {
    let actual = match published.algorithm {
        Algorithm::Sha256 => digest.sha256(),
        Algorithm::GitBlobSha1 => match digest.git_blob_sha1() {
            Some(actual) => actual,
            None => return Ok(()),
        },
    };
    if actual.eq_ignore_ascii_case(&published.checksum) {
        Ok(())
    } else {
        Err(Error::ChecksumMismatch {
            path: path.to_string(),
            expected: published.checksum.clone(),
            actual,
        })
    }
}
