//! Hashing a Checkpoint file while Pull streams it, so Pull never holds a whole file in memory.

use sha1::Sha1;
use sha2::{Digest as _, Sha256};

/// Hashes the bytes Pull streams for one Checkpoint file.
///
/// The sha256 is always computed — it is what the Provenance record stores. The git blob object id
/// is computed only when the file's size is known before the first byte, because that digest is
/// prefixed with the size.
#[derive(Debug, Clone)]
pub struct Digest {
    sha256: Sha256,
    git_blob: Option<Sha1>,
    len: u64,
}

impl Digest {
    /// `blob_size` is the announced size of the file, when the Model Source announced one.
    pub fn new(blob_size: Option<u64>) -> Digest {
        let git_blob = blob_size.map(|size| {
            let mut sha1 = Sha1::new();
            sha1.update(format!("blob {size}\0").as_bytes());
            sha1
        });
        Digest {
            sha256: Sha256::new(),
            git_blob,
            len: 0,
        }
    }

    pub fn update(&mut self, bytes: &[u8]) {
        self.sha256.update(bytes);
        if let Some(sha1) = &mut self.git_blob {
            sha1.update(bytes);
        }
        self.len += bytes.len() as u64;
    }

    /// The number of bytes hashed so far.
    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn sha256(&self) -> String {
        hex(&self.sha256.clone().finalize())
    }

    /// The git blob object id, or `None` when the size was unknown when hashing began.
    pub fn git_blob_sha1(&self) -> Option<String> {
        self.git_blob.clone().map(|sha1| hex(&sha1.finalize()))
    }
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;

    let mut text = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(text, "{byte:02x}");
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashes_bytes_delivered_in_any_chunking() {
        // Literals from `sha256sum` and `git hash-object --stdin`.
        let one_shot = digest_of(&[b"hello world\n"], Some(12));
        assert_eq!(
            one_shot.sha256(),
            "a948904f2f0f479b8f8197694b30184b0d2ed1c1cd2a1ec0fb85d299a192a447"
        );
        assert_eq!(
            one_shot.git_blob_sha1().as_deref(),
            Some("3b18e512dba79e4c8300dd08aeb37f8e728b8dad")
        );

        let split = digest_of(&[b"hello", b" ", b"world", b"\n"], Some(12));
        assert_eq!(split.sha256(), one_shot.sha256());
        assert_eq!(split.git_blob_sha1(), one_shot.git_blob_sha1());
        assert_eq!(split.len(), 12);
    }

    #[test]
    fn a_git_blob_digest_needs_the_size_before_the_first_byte() {
        assert_eq!(
            digest_of(&[b"abc"], None).git_blob_sha1(),
            None,
            "without the announced size the object id is not computable"
        );
        assert_eq!(
            digest_of(&[b"abc"], Some(3)).git_blob_sha1().as_deref(),
            Some("f2ba8f84ab5c1bce84a7b441cb1959cfc7093b7f")
        );
        assert_eq!(
            digest_of(&[], Some(0)).git_blob_sha1().as_deref(),
            Some("e69de29bb2d1d6434b8b29ae775ad8c2e48c5391")
        );
    }

    #[test]
    fn the_sha256_of_nothing_is_known() {
        let digest = Digest::new(None);
        assert!(digest.is_empty());
        assert_eq!(
            digest.sha256(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    fn digest_of(chunks: &[&[u8]], blob_size: Option<u64>) -> Digest {
        let mut digest = Digest::new(blob_size);
        for chunk in chunks {
            digest.update(chunk);
        }
        digest
    }
}
