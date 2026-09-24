//! The Provenance record: the facts that identify a pulled Checkpoint and verify every stored file.

use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::digest::Digest;
use crate::error::{Error, Result};

/// The digest a Model Source publishes for a file. It says which hash the recorded value is, so a
/// reader can recompute it from the stored bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Algorithm {
    /// A sha256 over the file bytes.
    Sha256,
    /// A git blob object id: sha1 over `blob <size>\0` followed by the file bytes.
    GitBlobSha1,
}

/// The checksum a Model Source publishes for one file, as it publishes it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublishedChecksum {
    /// The hash the checksum is, which says how a reader recomputes it from the stored bytes.
    pub algorithm: Algorithm,
    /// The checksum itself, as that hash's hexadecimal text.
    pub checksum: String,
}

impl PublishedChecksum {
    /// Check the bytes hashed into `digest`, naming `path` if they disagree with what the Model
    /// Source published for it.
    ///
    /// A git blob object id can be recomputed only from a digest whose size was known before the
    /// first byte was hashed, so a Model Source that announced no size leaves that file checked by
    /// its size and its recorded sha256 alone (ADR-0010).
    ///
    /// # Errors
    ///
    /// [`Error::ChecksumMismatch`] when the recomputed digest is not this checksum.
    pub fn check(&self, digest: &Digest, path: &str) -> Result<()> {
        let actual = match self.algorithm {
            Algorithm::Sha256 => digest.sha256(),
            Algorithm::GitBlobSha1 => match digest.git_blob_sha1() {
                Some(actual) => actual,
                None => return Ok(()),
            },
        };
        if actual.eq_ignore_ascii_case(&self.checksum) {
            return Ok(());
        }
        Err(Error::ChecksumMismatch {
            path: path.to_string(),
            expected: self.checksum.clone(),
            actual,
        })
    }
}

/// One file of a Checkpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileRecord {
    /// Path relative to the Checkpoint directory, as the Model Source publishes it.
    pub path: String,
    /// The size in bytes of the stored file.
    pub size: u64,
    /// The sha256 of the stored bytes, computed while Pull streamed them.
    pub sha256: String,
    /// Absent when the Model Source publishes no checksum for this file.
    pub published: Option<PublishedChecksum>,
}

/// What a Pull records about the Checkpoint it stored.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    /// The Model Source repository the Checkpoint came from.
    pub source: String,
    /// The ref the caller asked for, absent when Pull resolved the default branch itself.
    pub requested_revision: Option<String>,
    /// The commit the files were pulled from.
    pub resolved_revision: String,
    /// One record per allowlisted Checkpoint file, as Pull stored them.
    pub files: Vec<FileRecord>,
}

impl Provenance {
    /// The record's file name inside a Checkpoint directory. Its presence marks a complete
    /// Checkpoint: Pull writes it only after every file is stored.
    pub const FILE_NAME: &'static str = "provenance.json";

    /// The record of `path`, or `None` when the record holds no file by that name.
    pub fn file(&self, path: &str) -> Option<&FileRecord> {
        self.files.iter().find(|file| file.path == path)
    }

    /// This record as the JSON text `provenance.json` holds.
    ///
    /// # Panics
    ///
    /// If the record could not be serialized, which its plain fields do not allow.
    pub fn to_json(&self) -> String {
        let mut json = serde_json::to_string_pretty(self).expect("Provenance serializes");
        json.push('\n');
        json
    }

    /// Read a Provenance record from `json`.
    ///
    /// # Errors
    ///
    /// The `serde_json` error for text that is not a Provenance record: not JSON at all, or a
    /// field that is missing or is not of its type.
    pub fn from_json(json: &str) -> serde_json::Result<Self> {
        json.parse()
    }
}

impl FromStr for Provenance {
    type Err = serde_json::Error;

    fn from_str(json: &str) -> Result<Self, Self::Err> {
        serde_json::from_str(json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Provenance {
        Provenance {
            source: "convaiinnovations/laya".to_string(),
            requested_revision: Some("main".to_string()),
            resolved_revision: "1c5edc17a7acd8701df6fc341c0d179f1c62c982".to_string(),
            files: vec![FileRecord {
                path: "tokenizer/tokenizer_config.json".to_string(),
                size: 308,
                sha256: "50044de60daaa73df97d262e15a40d4faf0160e7d742df64b377877a1320dd12"
                    .to_string(),
                published: Some(PublishedChecksum {
                    algorithm: Algorithm::GitBlobSha1,
                    checksum: "9fd800115c5c92353220aa66addfce67a9135f32".to_string(),
                }),
            }],
        }
    }

    #[test]
    fn provenance_round_trips_through_its_record() {
        let provenance = sample();
        let json = provenance.to_json();
        assert_eq!(Provenance::from_json(&json).unwrap(), provenance);
    }

    #[test]
    fn the_record_states_the_algorithm_of_each_published_checksum() {
        let json = sample().to_json();
        assert!(
            json.contains(r#""algorithm": "git-blob-sha1""#),
            "unexpected record:\n{json}"
        );
        assert!(json.contains(r#""requested_revision": "main""#));
    }
}
