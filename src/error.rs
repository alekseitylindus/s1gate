//! Every failure s1gate reports, and the exit code it maps to.

use std::fmt;
use std::io;
use std::path::PathBuf;

/// A failure that reaches the operator.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Every failure s1gate reports.
#[derive(Debug)]
pub enum Error {
    /// The Model Source is outside the curated set.
    UnsupportedSource {
        /// The Model Source that was asked for.
        requested: String,
    },
    /// The Model Identifier is not supported by `infer`.
    UnsupportedModelIdentifier {
        /// The Model Identifier that was rejected.
        requested: String,
    },
    /// The Checkpoint name is not usable as a single directory name.
    InvalidName {
        /// The Checkpoint name that was rejected.
        name: String,
    },
    /// Neither `XDG_DATA_HOME` nor `HOME` is set, so the Model Store has no location.
    NoStoreRoot,
    /// The Model Source has no such revision.
    RevisionNotFound {
        /// The Model Source that was asked.
        source: String,
        /// The revision it has no such entry for.
        revision: String,
    },
    /// The Model Source answered without the revision it was asked to resolve.
    UnresolvedRevision {
        /// The Model Source that was asked.
        source: String,
        /// The revision it was asked to resolve, or the default branch when none was asked.
        revision: String,
    },
    /// The Model Source does not publish a file the Checkpoint allowlist requires.
    MissingFile {
        /// The Model Source that was asked.
        source: String,
        /// The required file it does not publish.
        path: String,
    },
    /// A file came from a commit other than the resolved one.
    UnexpectedCommit {
        /// The file the Model Source served.
        path: String,
        /// The revision the Model Source resolved the request to.
        resolved: String,
        /// The revision the file was served from.
        served: String,
    },
    /// An HTTP response s1gate cannot use.
    Status {
        /// The URL that answered.
        url: String,
        /// The HTTP status it answered with.
        status: u16,
        /// What s1gate knows about that response, empty when it knows nothing.
        note: &'static str,
    },
    /// The request to the Model Source did not complete.
    Transport {
        /// The URL the request went to.
        url: String,
        /// The underlying transport failure.
        message: String,
    },
    /// The JSON supplied to `infer` is not one valid System One Call.
    InvalidCall {
        /// What is wrong with the call.
        message: String,
    },
    /// The requested Checkpoint is not in the local Model Store.
    MissingCheckpoint {
        /// The name of the Checkpoint that is not stored.
        name: String,
    },
    /// Native inference failed after the call and Checkpoint were validated.
    Inference {
        /// The failure that stopped inference, or writing its result.
        message: String,
    },
    /// The remote `TypeSafe` Backend could not judge the call.
    TypeSafe {
        /// The HTTP status, when `TypeSafe` answered.
        status: Option<u16>,
        /// A short diagnostic without request data or credentials.
        message: String,
    },
    /// The Pull did not carry the number of bytes the Model Source announced.
    SizeMismatch {
        /// The file the Pull was writing.
        path: String,
        /// The number of bytes the Model Source announced.
        expected: u64,
        /// The number of bytes the Pull carried.
        actual: u64,
    },
    /// The bytes of a file do not match the checksum the Model Source publishes.
    ChecksumMismatch {
        /// The Checkpoint file whose bytes were checked.
        path: String,
        /// The Published Checksum for that file.
        expected: String,
        /// The checksum the bytes hash to.
        actual: String,
    },
    /// The name already holds a Checkpoint of another revision.
    RevisionHeld {
        /// The Checkpoint name that is already taken.
        name: String,
        /// The revision the local Checkpoint holds.
        held: String,
        /// The revision the Pull asked for.
        requested: String,
    },
    // The three failures of verifying one Checkpoint file. Each `path` is the path the Provenance
    // record holds, relative to the Checkpoint directory, which is how Pull names the same file.
    /// A file recorded in Provenance is absent from the local Checkpoint.
    MissingStoredFile {
        /// The recorded path, relative to the Checkpoint directory.
        path: String,
    },
    /// A local Checkpoint file no longer has its recorded size.
    StoredSizeMismatch {
        /// The recorded path, relative to the Checkpoint directory.
        path: String,
        /// The number of bytes Provenance records.
        expected: u64,
        /// The number of bytes the file has now.
        actual: u64,
    },
    /// A local Checkpoint file no longer has its recorded checksum.
    StoredChecksumMismatch {
        /// The recorded path, relative to the Checkpoint directory.
        path: String,
        /// The checksum Provenance records.
        expected: String,
        /// The checksum the file hashes to now.
        actual: String,
    },
    /// A Checkpoint's configuration or safetensors structure is invalid.
    InvalidCheckpoint {
        /// The file at fault.
        path: PathBuf,
        /// What is wrong with it.
        message: String,
    },
    /// A required parameter is absent from the safetensors header.
    MissingParameter {
        /// The parameter the Parameter Manifest requires.
        name: String,
    },
    /// An unrecognized parameter is present in the safetensors header.
    UnexpectedParameter {
        /// The parameter the Parameter Manifest does not list.
        name: String,
    },
    /// A safetensors tensor has the wrong type or shape.
    ParameterMismatch {
        /// The parameter the Parameter Manifest and the header disagree on.
        name: String,
        /// The dtype the Parameter Manifest expects.
        expected_dtype: String,
        /// The dtype the safetensors header holds.
        actual_dtype: String,
        /// The shape the Parameter Manifest expects.
        expected_shape: Vec<u64>,
        /// The shape the safetensors header holds.
        actual_shape: Vec<u64>,
    },
    /// A bare `verify` found Checkpoints that failed, after verifying the rest of the store.
    VerificationFailed {
        /// The number of Checkpoints that failed verification.
        failed: usize,
        /// The number of Checkpoints checked.
        total: usize,
    },
    /// A filesystem operation s1gate performed failed.
    Io {
        /// The verb the failure is reported under: `read`, `write`, `create`, `rename`, `remove`,
        /// `inspect` or `open`.
        op: &'static str,
        /// What the operation was performed on.
        path: PathBuf,
        /// The underlying failure.
        source: io::Error,
    },
    /// A file or response s1gate read does not hold valid JSON.
    Json {
        /// What was being read.
        location: String,
        /// The parse failure.
        source: serde_json::Error,
    },
}

impl Error {
    /// The `op` s1gate performed on `path` failed with `source`.
    #[must_use]
    pub fn io(op: &'static str, path: impl Into<PathBuf>, source: io::Error) -> Self {
        Self::Io {
            op,
            path: path.into(),
            source,
        }
    }

    /// `location` does not hold JSON: `source`.
    #[must_use]
    pub fn json(location: impl Into<String>, source: serde_json::Error) -> Self {
        Self::Json {
            location: location.into(),
            source,
        }
    }

    /// `message` says why the input is not one valid System One Call.
    #[must_use]
    pub fn invalid_call(message: impl Into<String>) -> Self {
        Self::InvalidCall {
            message: message.into(),
        }
    }

    /// 0 success, 1 runtime error, 2 usage error.
    #[must_use]
    pub fn exit_code(&self) -> u8 {
        match self {
            Self::UnsupportedSource { .. }
            | Self::UnsupportedModelIdentifier { .. }
            | Self::InvalidName { .. }
            | Self::InvalidCall { .. } => 2,
            Self::TypeSafe {
                status: Some(422), ..
            } => 2,
            Self::NoStoreRoot
            | Self::RevisionNotFound { .. }
            | Self::UnresolvedRevision { .. }
            | Self::MissingFile { .. }
            | Self::UnexpectedCommit { .. }
            | Self::Status { .. }
            | Self::Transport { .. }
            | Self::MissingCheckpoint { .. }
            | Self::Inference { .. }
            | Self::TypeSafe { .. }
            | Self::SizeMismatch { .. }
            | Self::ChecksumMismatch { .. }
            | Self::RevisionHeld { .. }
            | Self::MissingStoredFile { .. }
            | Self::StoredSizeMismatch { .. }
            | Self::StoredChecksumMismatch { .. }
            | Self::InvalidCheckpoint { .. }
            | Self::MissingParameter { .. }
            | Self::UnexpectedParameter { .. }
            | Self::ParameterMismatch { .. }
            | Self::VerificationFailed { .. }
            | Self::Io { .. }
            | Self::Json { .. } => 1,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSource { requested } => {
                write_unsupported_name(f, "Model Source", requested)
            }
            Self::UnsupportedModelIdentifier { requested } => {
                write_unsupported_name(f, "Model Identifier", requested)
            }
            Self::InvalidName { name } => write!(
                f,
                "invalid Checkpoint name `{name}`: it must be a Model Source, `<owner>/<name>`"
            ),
            Self::NoStoreRoot => write!(
                f,
                "cannot locate the Model Store: neither XDG_DATA_HOME nor HOME is set"
            ),
            Self::RevisionNotFound { source, revision } => {
                write!(f, "{source} has no revision `{revision}`")
            }
            Self::UnresolvedRevision { source, revision } => write!(
                f,
                "{source} did not report a commit for revision `{revision}`"
            ),
            Self::MissingFile { source, path } => {
                write!(f, "{source} does not publish `{path}`")
            }
            Self::UnexpectedCommit {
                path,
                resolved,
                served,
            } => write!(
                f,
                "`{path}` was served from revision {served}, not the resolved revision {resolved}"
            ),
            Self::Status { url, status, note } => {
                write!(f, "{url} answered HTTP {status}")?;
                if !note.is_empty() {
                    write!(f, " ({note})")?;
                }
                Ok(())
            }
            Self::Transport { url, message } => write!(f, "cannot reach {url}: {message}"),
            Self::InvalidCall { message } => write!(f, "invalid System One Call: {message}"),
            Self::MissingCheckpoint { name } => write!(
                f,
                "Checkpoint `{name}` is not in the Model Store; run `s1gate pull {name}` first"
            ),
            Self::Inference { message } => write!(f, "native inference failed: {message}"),
            Self::TypeSafe {
                status: Some(status),
                message,
            } => write!(f, "TypeSafe answered HTTP {status}: {message}"),
            Self::TypeSafe {
                status: None,
                message,
            } => write!(f, "TypeSafe inference failed: {message}"),
            Self::SizeMismatch {
                path,
                expected,
                actual,
            } => write!(
                f,
                "`{path}` is {actual} bytes, but the Model Source announced {expected}"
            ),
            Self::ChecksumMismatch {
                path,
                expected,
                actual,
            } => write!(
                f,
                "`{path}` hashes to {actual}, but the Model Source publishes {expected}"
            ),
            Self::MissingStoredFile { path } => {
                write!(f, "Checkpoint file `{path}` is missing")
            }
            Self::StoredSizeMismatch {
                path,
                expected,
                actual,
            } => write!(
                f,
                "Checkpoint file `{path}` is {actual} bytes, but Provenance records {expected}"
            ),
            Self::StoredChecksumMismatch {
                path,
                expected,
                actual,
            } => write!(
                f,
                "Checkpoint file `{path}` sha256 is {actual}, but Provenance records {expected}"
            ),
            Self::InvalidCheckpoint { path, message } => {
                write!(f, "invalid Checkpoint file `{}`: {message}", path.display())
            }
            Self::MissingParameter { name } => {
                write!(f, "missing parameter `{name}`")
            }
            Self::UnexpectedParameter { name } => {
                write!(f, "unexpected parameter `{name}`")
            }
            Self::ParameterMismatch {
                name,
                expected_dtype,
                actual_dtype,
                expected_shape,
                actual_shape,
            } => write!(
                f,
                "parameter `{name}` has dtype {actual_dtype} and shape {actual_shape:?}, expected {expected_dtype} and {expected_shape:?}"
            ),
            Self::VerificationFailed { failed, total } => {
                write!(f, "{failed} of {total} Checkpoints failed verification")
            }
            Self::RevisionHeld {
                name,
                held,
                requested,
            } => write!(
                f,
                "Checkpoint `{name}` holds revision {held}; pulling {requested} over it needs --force"
            ),
            Self::Io { op, path, source } => write!(f, "cannot {op} {}: {source}", path.display()),
            Self::Json { location, source } => {
                write!(f, "cannot read {location}: {source}")
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Json { source, .. } => Some(source),
            Self::UnsupportedSource { .. }
            | Self::UnsupportedModelIdentifier { .. }
            | Self::InvalidName { .. }
            | Self::NoStoreRoot
            | Self::RevisionNotFound { .. }
            | Self::UnresolvedRevision { .. }
            | Self::MissingFile { .. }
            | Self::UnexpectedCommit { .. }
            | Self::Status { .. }
            | Self::Transport { .. }
            | Self::InvalidCall { .. }
            | Self::MissingCheckpoint { .. }
            | Self::Inference { .. }
            | Self::TypeSafe { .. }
            | Self::SizeMismatch { .. }
            | Self::ChecksumMismatch { .. }
            | Self::RevisionHeld { .. }
            | Self::MissingStoredFile { .. }
            | Self::StoredSizeMismatch { .. }
            | Self::StoredChecksumMismatch { .. }
            | Self::InvalidCheckpoint { .. }
            | Self::MissingParameter { .. }
            | Self::UnexpectedParameter { .. }
            | Self::ParameterMismatch { .. }
            | Self::VerificationFailed { .. } => None,
        }
    }
}

fn write_unsupported_name(f: &mut fmt::Formatter<'_>, kind: &str, requested: &str) -> fmt::Result {
    write!(f, "unsupported {kind} `{requested}` (supported: ")?;
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
