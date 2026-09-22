//! The Model Source's HTTP endpoint — the only socket s1gate opens.
//!
//! Pull follows the presigned redirect itself instead of letting the client do it, because the
//! first response is where the Model Source states the commit it served and the checksum it
//! publishes for the file.

use std::fmt::{self, Write as _};
use std::io::Read;
use std::time::Duration;

use serde::Deserialize;
use ureq::Agent;

use crate::error::{Error, Result};
use crate::provenance::{Algorithm, PublishedChecksum};

/// The host the curated Model Sources live on.
const PUBLIC_HUB: &str = "https://huggingface.co";

/// How many redirects one Pull may follow before it is treated as a loop.
const MAX_REDIRECTS: usize = 5;

/// The Model Source's HTTP endpoint.
pub struct Hub {
    base: String,
    agent: Agent,
}

/// A Checkpoint file opened for streaming, with what the Model Source says about it.
pub struct RemoteFile {
    size: Option<u64>,
    published: Option<PublishedChecksum>,
    reader: ureq::BodyReader<'static>,
}

impl RemoteFile {
    /// The announced size: the LFS size when the file is in LFS, else the body's content length.
    pub fn size(&self) -> Option<u64> {
        self.size
    }

    /// The checksum the Model Source publishes, when it publishes one.
    pub fn published(&self) -> Option<&PublishedChecksum> {
        self.published.as_ref()
    }
}

impl Read for RemoteFile {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        self.reader.read(buffer)
    }
}

impl Hub {
    /// The Hub the curated Model Sources live on.
    pub fn public() -> Self {
        Self::at(PUBLIC_HUB)
    }

    /// A Hub at `base`, which is how tests point Pull at a local Model Source.
    pub fn at(base: impl Into<String>) -> Self {
        let base = base.into();
        let base = base.trim_end_matches('/').to_string();
        let config = Agent::config_builder()
            // Redirects are followed below, so the first response stays readable.
            .max_redirects(0)
            .http_status_as_error(false)
            .timeout_resolve(Some(Duration::from_secs(30)))
            .timeout_connect(Some(Duration::from_secs(30)))
            .timeout_recv_response(Some(Duration::from_secs(60)))
            .build();
        Self {
            base,
            agent: config.new_agent(),
        }
    }

    /// The commit `revision` names; `None` asks for the default branch.
    ///
    /// # Errors
    ///
    /// [`Error::RevisionNotFound`] when the Model Source has no such revision, and
    /// [`Error::UnresolvedRevision`] when it answers without naming a commit.
    /// [`Error::Status`] and [`Error::Transport`] when the request does not complete, and
    /// [`Error::Json`] when the answer is not a model.
    pub fn resolve(&self, repo: &str, revision: Option<&str>) -> Result<String> {
        let url = match revision {
            Some(revision) => format!(
                "{}/api/models/{repo}/revision/{}",
                self.base,
                encode_segment(revision)
            ),
            None => format!("{}/api/models/{repo}", self.base),
        };
        let mut response = self
            .agent
            .get(&url)
            .call()
            .map_err(|error| transport(&url, error))?;
        match response.status().as_u16() {
            200 => {}
            404 => {
                return Err(match revision {
                    Some(revision) => Error::RevisionNotFound {
                        source: repo.to_string(),
                        revision: revision.to_string(),
                    },
                    None => Error::Status {
                        url,
                        status: 404,
                        note: "the Model Source does not exist",
                    },
                });
            }
            status => return Err(status_error(&url, status)),
        }
        let body = response
            .body_mut()
            .read_to_string()
            .map_err(|error| transport(&url, error))?;
        let model: ResolvedModel =
            serde_json::from_str(&body).map_err(|error| Error::json(&url, error))?;
        model.sha.ok_or_else(|| Error::UnresolvedRevision {
            source: repo.to_string(),
            revision: revision.unwrap_or("the default branch").to_string(),
        })
    }

    /// Open `path` of `commit` for streaming.
    ///
    /// # Errors
    ///
    /// [`Error::Status`] and [`Error::Transport`] when the Model Source cannot serve the request,
    /// [`Error::UnexpectedCommit`] when the first response says the file comes from another commit,
    /// and [`Error::MissingFile`] when the Model Source does not publish `path`.
    pub fn open(&self, repo: &str, commit: &str, path: &str) -> Result<RemoteFile> {
        let mut url = format!(
            "{}/{repo}/resolve/{}/{path}",
            self.base,
            encode_segment(commit)
        );
        let mut published = None;
        let mut announced_size = None;
        for hop in 0..=MAX_REDIRECTS {
            let first = hop == 0;
            let response = self
                .agent
                .get(&url)
                .call()
                .map_err(|error| transport(&url, error))?;
            let status = response.status().as_u16();
            if first {
                // The first response is where the Model Source states the commit it served and the
                // checksum it publishes, whether it serves the file itself or redirects to a
                // presigned URL.
                if let Some(served) = header(&response, "x-repo-commit", &url, status)?
                    && served != commit
                {
                    return Err(Error::UnexpectedCommit {
                        path: path.to_string(),
                        resolved: commit.to_string(),
                        served: served.to_string(),
                    });
                }
                published = header(&response, "x-linked-etag", &url, status)?
                    .as_deref()
                    .and_then(published_checksum);
                announced_size = match header(&response, "x-linked-size", &url, status)? {
                    None => None,
                    Some(size) => Some(size.parse().map_err(|_| Error::Status {
                        url: url.clone(),
                        status,
                        note: "the announced file size is not a number",
                    })?),
                };
            }
            if status / 100 == 3 {
                let location =
                    header(&response, "location", &url, status)?.ok_or_else(|| Error::Status {
                        url: url.clone(),
                        status,
                        note: "the redirect names no location",
                    })?;
                url = join(&self.base, &location);
                continue;
            }
            if status == 404 && first {
                return Err(Error::MissingFile {
                    source: repo.to_string(),
                    path: path.to_string(),
                });
            }
            if status != 200 {
                return Err(status_error(&url, status));
            }
            let size = announced_size.or_else(|| response.body().content_length());
            return Ok(RemoteFile {
                size,
                published,
                reader: response.into_body().into_reader(),
            });
        }
        Err(Error::Status {
            url,
            status: 310,
            note: "too many redirects",
        })
    }
}

#[derive(Deserialize)]
struct ResolvedModel {
    sha: Option<String>,
}

/// The Model Source's published checksum for a file, as carried in `x-linked-etag`: LFS files are
/// published as a sha256, git-tracked files as a git blob object id. `None` when the etag is not a
/// digest of either kind, which is how the Model Source states that it publishes none.
fn published_checksum(etag: &str) -> Option<PublishedChecksum> {
    PublishedChecksum::try_from(etag).ok()
}

/// Why an `x-linked-etag` is not a checksum s1gate can recompute, and so is not a checksum Pull
/// checks the streamed bytes against. Any other etag is; a Model Source that publishes nothing for
/// a file simply sends no `x-linked-etag`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnusableEtag {
    /// The tag is neither 64 nor 40 hexadecimal digits, so it is neither a sha256 nor a git blob
    /// object id.
    NotADigest,
}

impl fmt::Display for UnusableEtag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let note = match self {
            Self::NotADigest => "the etag is neither 64 nor 40 hexadecimal digits",
        };
        f.write_str(note)
    }
}

impl std::error::Error for UnusableEtag {}

impl TryFrom<&str> for PublishedChecksum {
    type Error = UnusableEtag;

    fn try_from(etag: &str) -> Result<Self, Self::Error> {
        let etag = etag.trim();
        let etag = etag.strip_prefix("W/").unwrap_or(etag);
        let etag = etag.trim_matches('"');
        if !etag.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(UnusableEtag::NotADigest);
        }
        let algorithm = match etag.len() {
            64 => Algorithm::Sha256,
            40 => Algorithm::GitBlobSha1,
            _ => return Err(UnusableEtag::NotADigest),
        };
        Ok(Self {
            algorithm,
            checksum: etag.to_ascii_lowercase(),
        })
    }
}

/// The value of the response header `name`: `Ok(None)` when the response carries no such header.
///
/// # Errors
///
/// [`Error::Status`] when the header is there but its bytes are not text, which is otherwise
/// indistinguishable from the header being absent.
fn header(
    response: &ureq::http::Response<ureq::Body>,
    name: &str,
    url: &str,
    status: u16,
) -> Result<Option<String>> {
    let Some(value) = response.headers().get(name) else {
        return Ok(None);
    };
    let value = value.to_str().map_err(|_| Error::Status {
        url: url.to_string(),
        status,
        note: "a response header is not text",
    })?;
    Ok(Some(value.to_string()))
}

fn status_error(url: &str, status: u16) -> Error {
    let note = match status {
        401 | 403 => "the Model Source is gated or private, and s1gate does not authenticate",
        429 => "the Model Source is rate limiting",
        _ => "",
    };
    Error::Status {
        url: url.to_string(),
        status,
        note,
    }
}

fn transport(url: &str, error: ureq::Error) -> Error {
    Error::Transport {
        url: url.to_string(),
        message: error.to_string(),
    }
}

/// Resolve a `location` header against the Hub: presigned URLs are absolute, the Model Source's own
/// redirects are paths.
fn join(base: &str, location: &str) -> String {
    if location.starts_with("http://") || location.starts_with("https://") {
        return location.to_string();
    }
    // The origin is the first three `/`-separated pieces of `base`, which spell `<scheme>`, the
    // empty piece between the two slashes and `<authority>`; a base naming no path still has them.
    let mut pieces = base.splitn(4, '/');
    let origin = match (pieces.next(), pieces.next(), pieces.next()) {
        (Some(scheme), Some(separator), Some(authority)) => {
            format!("{scheme}/{separator}/{authority}")
        }
        _ => base.to_string(),
    };
    if location.starts_with('/') {
        format!("{origin}{location}")
    } else {
        format!("{origin}/{location}")
    }
}

/// Percent-encode one path segment, so a revision like `feature/x` addresses one segment.
fn encode_segment(segment: &str) -> String {
    let mut encoded = String::with_capacity(segment.len());
    for byte in segment.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                encoded.push(char::from(byte));
            }
            _ => {
                let _ = write!(encoded, "%{byte:02X}");
            }
        }
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_published_checksum_is_classified_by_its_length() {
        let sha256 = published_checksum(
            "\"891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c\"",
        )
        .expect("an LFS etag is a checksum");
        assert_eq!(sha256.algorithm, Algorithm::Sha256);
        assert_eq!(
            sha256.checksum,
            "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c"
        );

        let blob = published_checksum("W/\"3e4fcbf12cf36164ce18a1398aa9f35f58375ae0\"")
            .expect("a git etag is a checksum");
        assert_eq!(blob.algorithm, Algorithm::GitBlobSha1);
        assert_eq!(blob.checksum, "3e4fcbf12cf36164ce18a1398aa9f35f58375ae0");

        assert_eq!(published_checksum("not-a-checksum"), None);
        assert_eq!(published_checksum(""), None);
    }

    #[test]
    fn a_revision_addresses_one_path_segment() {
        assert_eq!(encode_segment("main"), "main");
        assert_eq!(encode_segment("1c5edc17"), "1c5edc17");
        assert_eq!(encode_segment("feature/x"), "feature%2Fx");
    }

    #[test]
    fn a_location_is_resolved_against_the_hub() {
        assert_eq!(
            join("https://huggingface.co", "/api/resolve-cache/models/x/y"),
            "https://huggingface.co/api/resolve-cache/models/x/y"
        );
        assert_eq!(
            join("http://127.0.0.1:8080", "https://cdn.example/x"),
            "https://cdn.example/x"
        );
    }
}
