//! Every failure s1gate reports, and the exit code it maps to.

use std::fmt;
use std::io;
use std::path::{Path, PathBuf};

/// A failure that reaches the operator.
pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug)]
pub enum Error {
    /// The Model Source is outside the curated set.
    UnsupportedSource { requested: String },
    /// The Checkpoint name is not usable as a single directory name.
    InvalidName { name: String },
    /// Neither `XDG_DATA_HOME` nor `HOME` is set, so the Model Store has no location.
    NoStoreRoot,
    /// The Model Source has no such revision.
    RevisionNotFound { source: String, revision: String },
    /// The Model Source answered without the revision it was asked to resolve.
    UnresolvedRevision { source: String, revision: String },
    /// The Model Source does not publish a file the Checkpoint allowlist requires.
    MissingFile { source: String, path: String },
    /// A file came from a commit other than the resolved one.
    UnexpectedCommit {
        path: String,
        resolved: String,
        served: String,
    },
    /// An HTTP response s1gate cannot use.
    Status {
        url: String,
        status: u16,
        note: &'static str,
    },
    /// The request to the Model Source did not complete.
    Transport { url: String, message: String },
    /// The JSON supplied to `infer` is not one valid System One Call.
    InvalidCall { message: String },
    /// The requested Checkpoint is not in the local Model Store.
    MissingCheckpoint { name: String },
    /// The input contract exists before native inference does.
    InferenceUnavailable,
    /// The Pull did not carry the number of bytes the Model Source announced.
    SizeMismatch {
        path: String,
        expected: u64,
        actual: u64,
    },
    /// The bytes Pull streamed do not match the checksum the Model Source publishes.
    ChecksumMismatch {
        path: String,
        expected: String,
        actual: String,
    },
    /// The name already holds a Checkpoint of another revision.
    RevisionHeld {
        name: String,
        held: String,
        requested: String,
    },
    Io {
        op: &'static str,
        path: PathBuf,
        source: io::Error,
    },
    Json {
        location: String,
        source: serde_json::Error,
    },
}

impl Error {
    pub fn io(op: &'static str, path: impl AsRef<Path>, source: io::Error) -> Self {
        Error::Io {
            op,
            path: path.as_ref().to_path_buf(),
            source,
        }
    }

    pub fn json(location: impl Into<String>, source: serde_json::Error) -> Self {
        Error::Json {
            location: location.into(),
            source,
        }
    }

    pub fn invalid_call(message: impl Into<String>) -> Self {
        Error::InvalidCall {
            message: message.into(),
        }
    }

    /// 0 success, 1 runtime error, 2 usage error.
    pub fn exit_code(&self) -> u8 {
        match self {
            Error::UnsupportedSource { .. }
            | Error::InvalidName { .. }
            | Error::InvalidCall { .. } => 2,
            _ => 1,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::UnsupportedSource { requested } => {
                write!(f, "unsupported Model Source `{requested}` (supported: ")?;
                let mut first = true;
                for source in crate::model_source::supported() {
                    if !first {
                        write!(f, ", ")?;
                    }
                    write!(f, "{source}")?;
                    first = false;
                }
                write!(f, ")")
            }
            Error::InvalidName { name } => write!(
                f,
                "invalid Checkpoint name `{name}`: it must be a Model Source, `<owner>/<name>`"
            ),
            Error::NoStoreRoot => write!(
                f,
                "cannot locate the Model Store: neither XDG_DATA_HOME nor HOME is set"
            ),
            Error::RevisionNotFound { source, revision } => {
                write!(f, "{source} has no revision `{revision}`")
            }
            Error::UnresolvedRevision { source, revision } => write!(
                f,
                "{source} did not report a commit for revision `{revision}`"
            ),
            Error::MissingFile { source, path } => {
                write!(f, "{source} does not publish `{path}`")
            }
            Error::UnexpectedCommit {
                path,
                resolved,
                served,
            } => write!(
                f,
                "`{path}` was served from revision {served}, not the resolved revision {resolved}"
            ),
            Error::Status { url, status, note } => {
                write!(f, "{url} answered HTTP {status}")?;
                if !note.is_empty() {
                    write!(f, " ({note})")?;
                }
                Ok(())
            }
            Error::Transport { url, message } => write!(f, "cannot reach {url}: {message}"),
            Error::InvalidCall { message } => write!(f, "invalid System One Call: {message}"),
            Error::MissingCheckpoint { name } => write!(
                f,
                "Checkpoint `{name}` is not in the Model Store; run `s1gate pull {name}` first"
            ),
            Error::InferenceUnavailable => {
                write!(f, "native inference is not available yet")
            }
            Error::SizeMismatch {
                path,
                expected,
                actual,
            } => write!(
                f,
                "`{path}` is {actual} bytes, but the Model Source announced {expected}"
            ),
            Error::ChecksumMismatch {
                path,
                expected,
                actual,
            } => write!(
                f,
                "`{path}` hashes to {actual}, but the Model Source publishes {expected}"
            ),
            Error::RevisionHeld {
                name,
                held,
                requested,
            } => write!(
                f,
                "Checkpoint `{name}` holds revision {held}; pulling {requested} over it needs --force"
            ),
            Error::Io { op, path, source } => write!(f, "cannot {op} {}: {source}", path.display()),
            Error::Json { location, source } => {
                write!(f, "cannot read {location}: {source}")
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io { source, .. } => Some(source),
            Error::Json { source, .. } => Some(source),
            _ => None,
        }
    }
}
