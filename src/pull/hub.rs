//! The Model Source's HTTP endpoint — the only socket s1gate opens.
//!
//! Pull follows the presigned redirect itself instead of letting the client do it, because the
//! first response is where the Model Source states the commit it served and the checksum it
//! publishes for the file.

use std::fmt::Write as _;
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
    pub fn public() -> Hub {
        Hub::at(PUBLIC_HUB)
    }

    /// A Hub at `base`, which is how tests point Pull at a local Model Source.
    pub fn at(base: impl Into<String>) -> Hub {
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
        Hub {
            base,
            agent: config.new_agent(),
        }
    }

    pub fn base(&self) -> &str {
        &self.base
    }

    /// The commit `revision` names; `None` asks for the default branch.
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
    pub fn open(&self, repo: &str, commit: &str, path: &str) -> Result<RemoteFile> {
        let mut url = format!("{}/{repo}/resolve/{commit}/{path}", self.base);
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
                if let Some(served) = header(&response, "x-repo-commit")
                    && served != commit
                {
                    return Err(Error::UnexpectedCommit {
                        path: path.to_string(),
                        resolved: commit.to_string(),
                        served: served.to_string(),
                    });
                }
                published = header(&response, "x-linked-etag")
                    .as_deref()
                    .and_then(published_checksum);
                announced_size =
                    header(&response, "x-linked-size").and_then(|size| size.parse().ok());
            }
            if status / 100 == 3 {
                let location = header(&response, "location").ok_or_else(|| Error::Status {
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
/// published as a sha256, git-tracked files as a git blob object id.
fn published_checksum(etag: &str) -> Option<PublishedChecksum> {
    let etag = etag.trim();
    let etag = etag.strip_prefix("W/").unwrap_or(etag);
    let etag = etag.trim_matches('"');
    let algorithm = match etag.len() {
        64 if etag.bytes().all(|byte| byte.is_ascii_hexdigit()) => Algorithm::Sha256,
        40 if etag.bytes().all(|byte| byte.is_ascii_hexdigit()) => Algorithm::GitBlobSha1,
        _ => return None,
    };
    Some(PublishedChecksum {
        algorithm,
        checksum: etag.to_ascii_lowercase(),
    })
}

fn header(response: &ureq::http::Response<ureq::Body>, name: &str) -> Option<String> {
    let value = response.headers().get(name)?.to_str().ok()?;
    Some(value.to_string())
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
    let origin: String = base.split('/').take(3).collect::<Vec<_>>().join("/");
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
                encoded.push(byte as char);
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
