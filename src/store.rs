//! The Model Store: the Checkpoints on disk, each held under the full name of its Model Source.

use std::path::{Path, PathBuf};

use crate::checkpoint::Report;
use crate::error::{Error, Result};
use crate::model_source::ModelSource;
use crate::provenance::Provenance;

/// The store root below `$XDG_DATA_HOME`; the Checkpoint of `owner/name` lives at
/// `<root>/<owner>/<name>` (ADR-0011).
const STORE_PATH: &str = "s1gate/models";
/// The store root below `$HOME` when `XDG_DATA_HOME` says nothing.
const HOME_STORE_PATH: &str = ".local/share/s1gate/models";

/// The suffix a file carries in the Model Store until it is complete: a Checkpoint file being
/// streamed, and the Provenance record being written.
pub const PART_SUFFIX: &str = ".part";

/// The on-disk collection of pulled Checkpoints.
#[derive(Debug, Clone)]
pub struct Store {
    root: PathBuf,
}

impl Store {
    /// A store rooted at `root`; the caller owns the location.
    pub fn at(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The store at the location the environment names.
    ///
    /// # Errors
    ///
    /// Neither `XDG_DATA_HOME` nor `HOME` names a location: [`Error::NoStoreRoot`].
    pub fn from_env() -> Result<Self> {
        let xdg_data_home = std::env::var_os("XDG_DATA_HOME").map(PathBuf::from);
        let home = std::env::var_os("HOME").map(PathBuf::from);
        Ok(Self::at(store_root(
            xdg_data_home.as_deref(),
            home.as_deref(),
        )?))
    }

    /// Where the Checkpoint of the Model Source `name` lives.
    ///
    /// # Errors
    ///
    /// `name` is not one `<owner>/<name>` pair of directory names: [`Error::InvalidName`].
    pub(crate) fn checkpoint_dir(&self, name: &str) -> Result<PathBuf> {
        let (owner, checkpoint) = segments(name)?;
        Ok(self.root.join(owner).join(checkpoint))
    }

    /// The Checkpoint of `source`, or `None` when this store holds none.
    ///
    /// # Errors
    ///
    /// `source` names no Checkpoint directory ([`Error::InvalidName`]), the record cannot be read
    /// ([`Error::Io`] or [`Error::Json`]), or the record names another Model Source
    /// ([`Error::InvalidCheckpoint`]) — a Checkpoint is held under the Model Source it came from
    /// (ADR-0011), so a record that says otherwise is not this Checkpoint.
    pub fn checkpoint(&self, source: &'static ModelSource) -> Result<Option<Checkpoint>> {
        let directory = self.checkpoint_dir(source.repo)?;
        let Some(provenance) = self.provenance(source.repo)? else {
            return Ok(None);
        };
        if provenance.source != source.repo {
            return Err(Error::invalid_checkpoint(
                directory.join(Provenance::FILE_NAME),
                format!(
                    "Provenance names `{}`, expected `{}`",
                    provenance.source, source.repo
                ),
            ));
        }
        Ok(Some(Checkpoint {
            directory,
            provenance,
            source,
        }))
    }

    /// The Provenance `name` records, or `None` when it holds no Checkpoint.
    ///
    /// # Errors
    ///
    /// `name` is not one `<owner>/<name>` pair of directory names ([`Error::InvalidName`]), the
    /// record cannot be read, or it is not a Provenance.
    pub fn provenance(&self, name: &str) -> Result<Option<Provenance>> {
        let path = self.checkpoint_dir(name)?.join(Provenance::FILE_NAME);
        let json = match std::fs::read_to_string(&path) {
            Ok(json) => json,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(Error::io("read", &path, error)),
        };
        Provenance::from_json(&json)
            .map(Some)
            .map_err(|error| Error::json(path.display().to_string(), error))
    }

    /// List the Checkpoints this store holds: the directory names, two levels below the root, that
    /// record a Provenance. A missing store root holds no Checkpoint, and neither does a directory
    /// a Pull left incomplete — the record is written last. Entries that are not directories, and
    /// names that are not UTF-8, are not Checkpoints and are skipped.
    ///
    /// # Errors
    ///
    /// An entry of the store root cannot be read, or an entry that names a directory cannot be
    /// inspected.
    pub fn checkpoint_names(&self) -> Result<Vec<String>> {
        let owners = match std::fs::read_dir(&self.root) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(Error::io("read", &self.root, error)),
        };
        let mut names = Vec::new();
        for owner in owners {
            let owner = owner.map_err(|error| Error::io("read", &self.root, error))?;
            if !is_directory(&owner)? {
                continue;
            }
            let owner_file_name = owner.file_name();
            let Some(owner_name) = owner_file_name.to_str() else {
                continue;
            };
            let checkpoints = std::fs::read_dir(owner.path())
                .map_err(|error| Error::io("read", owner.path(), error))?;
            for checkpoint in checkpoints {
                let checkpoint =
                    checkpoint.map_err(|error| Error::io("read", owner.path(), error))?;
                if !is_directory(&checkpoint)? {
                    continue;
                }
                let checkpoint_file_name = checkpoint.file_name();
                let Some(checkpoint_name) = checkpoint_file_name.to_str() else {
                    continue;
                };
                if !holds_provenance(&checkpoint.path())? {
                    continue;
                }
                names.push(format!("{owner_name}/{checkpoint_name}"));
            }
        }
        names.sort();
        Ok(names)
    }

    /// Record `provenance` as the Checkpoint of `name`, replacing any earlier record. Written
    /// through `provenance.json.part` and renamed, so a record on disk is always complete.
    ///
    /// # Errors
    ///
    /// `name` is not one `<owner>/<name>` pair of directory names ([`Error::InvalidName`]), or the
    /// Checkpoint directory cannot be created, written, or renamed into place.
    pub fn record_provenance(&self, name: &str, provenance: &Provenance) -> Result<()> {
        let directory = self.checkpoint_dir(name)?;
        std::fs::create_dir_all(&directory)
            .map_err(|error| Error::io("create", &directory, error))?;
        let path = directory.join(Provenance::FILE_NAME);
        let partial = directory.join(format!("{}{PART_SUFFIX}", Provenance::FILE_NAME));
        std::fs::write(&partial, provenance.to_json())
            .map_err(|error| Error::io("write", &partial, error))?;
        std::fs::rename(&partial, &path).map_err(|error| Error::io("rename", &path, error))
    }

    /// Discard whatever `name` holds, so a Pull can store a different Checkpoint under it.
    ///
    /// # Errors
    ///
    /// `name` is not one `<owner>/<name>` pair of directory names ([`Error::InvalidName`]), or the
    /// Checkpoint directory cannot be removed.
    pub fn discard_checkpoint(&self, name: &str) -> Result<()> {
        let directory = self.checkpoint_dir(name)?;
        match std::fs::remove_dir_all(&directory) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(Error::io("remove", &directory, error)),
        }
    }
}

/// One Checkpoint the Model Store holds: its directory, the Provenance record whose presence marks
/// it complete, and the Model Source whose allowlist its files must be (ADR-0011).
///
/// This is what "the Model Store holds a usable Checkpoint" means, so a caller asks the handle
/// instead of re-deriving the rule from the record and the allowlist.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    directory: PathBuf,
    provenance: Provenance,
    source: &'static ModelSource,
}

impl Checkpoint {
    /// The Checkpoint's directory in the Model Store.
    pub fn directory(&self) -> &Path {
        &self.directory
    }

    /// The record that marks this Checkpoint complete.
    pub fn provenance(&self) -> &Provenance {
        &self.provenance
    }

    /// Whether every file the Model Source's allowlist names is a file here. This is what a Model
    /// Identifier's availability rests on, and it hashes nothing: a Checkpoint whose bytes changed
    /// is present all the same, and fails [`Self::verify`] rather than this.
    ///
    /// # Errors
    ///
    /// An allowlisted path cannot be inspected, which is a fault rather than an absent file.
    pub fn files_present(&self) -> Result<bool> {
        for path in self.source.files {
            let file = self.directory.join(path);
            match std::fs::metadata(&file) {
                Ok(metadata) if metadata.is_file() => {}
                // A directory, and a link to nothing, is not the file the allowlist names.
                Ok(_) => return Ok(false),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
                Err(error) => return Err(Error::io("inspect", &file, error)),
            }
        }
        Ok(true)
    }

    /// Verify every stored file against this record, and the weights header against the Parameter
    /// Manifest the Checkpoint's own configuration describes (ADR-0005).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidCheckpoint`] when a configuration or the weights header is not what it must
    /// be; [`Error::MissingStoredFile`], [`Error::StoredSizeMismatch`],
    /// [`Error::StoredChecksumMismatch`] and [`Error::ChecksumMismatch`] when a stored file is not
    /// the file the record describes; [`Error::MissingParameter`], [`Error::UnexpectedParameter`]
    /// and [`Error::ParameterMismatch`] when the weights header disagrees with the Parameter
    /// Manifest; [`Error::Io`] or [`Error::Json`] when a Checkpoint file cannot be read.
    pub fn verify(&self) -> Result<Report> {
        crate::checkpoint::verify(self)
    }

    /// The Model Source this Checkpoint is held under, which is the source of its allowlist.
    pub(crate) fn source(&self) -> &'static ModelSource {
        self.source
    }
}

/// Check that `name` is a Model Source name the Model Store can hold a Checkpoint under: one
/// `<owner>/<name>` pair of directory names, neither a level of its own.
///
/// # Errors
///
/// [`Error::InvalidName`] when it is not.
pub(crate) fn valid_name(name: &str) -> Result<()> {
    segments(name).map(|_| ())
}

/// `$XDG_DATA_HOME/s1gate/models`, defaulting to `~/.local/share/s1gate/models` (ADR-0011). A
/// relative `XDG_DATA_HOME` is not a valid path and is ignored, as the XDG Base Directory
/// specification requires.
///
/// # Errors
///
/// Neither `xdg_data_home` nor `home` names a location: [`Error::NoStoreRoot`].
pub fn store_root(xdg_data_home: Option<&Path>, home: Option<&Path>) -> Result<PathBuf> {
    if let Some(data_home) = xdg_data_home.filter(|path| path.is_absolute()) {
        return Ok(data_home.join(STORE_PATH));
    }
    match home {
        Some(home) => Ok(home.join(HOME_STORE_PATH)),
        None => Err(Error::NoStoreRoot),
    }
}

/// Split the full name of a Model Source, `owner/name`, into the two segments that address its
/// Checkpoint directory below the store root.
fn segments(name: &str) -> Result<(&str, &str)> {
    if let Some((owner, checkpoint)) = name.split_once('/')
        && !name.contains(['\\', '\0'])
        && names_a_directory(owner)
        && names_a_directory(checkpoint)
        && !checkpoint.contains('/')
    {
        return Ok((owner, checkpoint));
    }
    Err(Error::InvalidName {
        name: name.to_string(),
    })
}

/// Whether one segment names a directory below the store root rather than a level above it.
fn names_a_directory(segment: &str) -> bool {
    !segment.is_empty() && segment != "." && segment != ".."
}

/// Whether a store entry names a directory. A symlink is followed, as the rest of the store reads
/// through one; a dangling link names nothing.
fn is_directory(entry: &std::fs::DirEntry) -> Result<bool> {
    match std::fs::metadata(entry.path()) {
        Ok(metadata) => Ok(metadata.is_dir()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(Error::io("inspect", entry.path(), error)),
    }
}

/// Whether `directory` records a Provenance. An entry that cannot be inspected is an error rather
/// than a Checkpoint that is not one, so an unreadable store is never reported as an empty one.
fn holds_provenance(directory: &Path) -> Result<bool> {
    let record = directory.join(Provenance::FILE_NAME);
    match std::fs::metadata(&record) {
        Ok(metadata) => Ok(metadata.is_file()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(Error::io("inspect", &record, error)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provenance::FileRecord;

    #[test]
    fn the_store_root_follows_xdg_then_home() {
        assert_eq!(
            store_root(Some(Path::new("/data")), Some(Path::new("/home/op"))).unwrap(),
            PathBuf::from("/data/s1gate/models")
        );
        assert_eq!(
            store_root(None, Some(Path::new("/home/op"))).unwrap(),
            PathBuf::from("/home/op/.local/share/s1gate/models")
        );
    }

    #[test]
    fn a_relative_xdg_data_home_is_ignored() {
        assert_eq!(
            store_root(Some(Path::new("data")), Some(Path::new("/home/op"))).unwrap(),
            PathBuf::from("/home/op/.local/share/s1gate/models")
        );
    }

    #[test]
    fn without_a_home_the_store_has_no_location() {
        assert!(matches!(
            store_root(None, None).expect_err("no location"),
            Error::NoStoreRoot
        ));
    }

    #[test]
    fn a_name_that_is_not_one_model_source_is_rejected() {
        for name in [
            "",
            ".",
            "..",
            "laya",
            "/laya",
            "laya/",
            "/",
            "../laya",
            "a/../b",
            "convaiinnovations/..",
            "a/b/c",
            "a\\b/c",
            "lay\0a/x",
        ] {
            assert!(
                matches!(
                    Store::at("/store").checkpoint_dir(name),
                    Err(Error::InvalidName { .. })
                ),
                "`{name}` should be rejected"
            );
        }
        assert_eq!(
            Store::at("/store")
                .checkpoint_dir("convaiinnovations/laya")
                .unwrap(),
            PathBuf::from("/store/convaiinnovations/laya"),
            "a Model Source addresses its Checkpoint below the store root"
        );
    }

    #[test]
    fn a_checkpoint_round_trips_through_the_store() {
        let root = temp_dir("round-trip");
        let store = Store::at(&root);
        assert_eq!(store.provenance("convaiinnovations/laya").unwrap(), None);

        let provenance = Provenance {
            source: "convaiinnovations/laya".to_string(),
            requested_revision: None,
            resolved_revision: "1c5edc1".to_string(),
            files: vec![FileRecord {
                path: "rl_agent_config.json".to_string(),
                size: 745,
                sha256: "0".repeat(64),
                published: None,
            }],
        };
        store
            .record_provenance("convaiinnovations/laya", &provenance)
            .unwrap();
        assert_eq!(
            store.provenance("convaiinnovations/laya").unwrap(),
            Some(provenance)
        );
        assert_eq!(
            store.checkpoint_dir("convaiinnovations/laya").unwrap(),
            root.join("convaiinnovations").join("laya"),
            "the Checkpoint lives under its Model Source"
        );
        assert!(
            !store
                .checkpoint_dir("convaiinnovations/laya")
                .unwrap()
                .join("provenance.json.part")
                .exists(),
            "the record is renamed into place"
        );

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn discarding_a_checkpoint_removes_its_files_and_record() {
        let root = temp_dir("discard");
        let store = Store::at(&root);
        let directory = store.checkpoint_dir("convaiinnovations/laya").unwrap();
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("model.safetensors.part"), b"half").unwrap();

        store.discard_checkpoint("convaiinnovations/laya").unwrap();
        assert!(!directory.exists(), "the name is free again");
        store.discard_checkpoint("convaiinnovations/laya").unwrap();

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_checkpoint_is_present_only_with_every_allowlisted_file() {
        let root = temp_dir("presence");
        let store = Store::at(&root);
        let source = &crate::model_source::LAYA;

        assert!(
            store.checkpoint(source).unwrap().is_none(),
            "a store that records nothing holds no Checkpoint"
        );

        store.record_provenance(source.repo, &provenance()).unwrap();
        let checkpoint = store
            .checkpoint(source)
            .unwrap()
            .expect("the record is stored");
        let directory = store.checkpoint_dir(source.repo).unwrap();
        assert_eq!(checkpoint.directory(), directory);
        assert_eq!(checkpoint.provenance().source, source.repo);
        assert!(
            !checkpoint.files_present().unwrap(),
            "no allowlisted file is stored yet"
        );

        // A directory where an allowlisted file belongs is not that file.
        std::fs::create_dir_all(directory.join(source.files[0])).unwrap();
        for path in source.files.iter().skip(1) {
            let file = directory.join(*path);
            if let Some(parent) = file.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(&file, b"x").unwrap();
        }
        assert!(
            !checkpoint.files_present().unwrap(),
            "a directory is not the file the allowlist names"
        );

        std::fs::remove_dir_all(directory.join(source.files[0])).unwrap();
        std::fs::write(directory.join(source.files[0]), b"x").unwrap();
        assert!(
            checkpoint.files_present().unwrap(),
            "every allowlisted file is there"
        );

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn checkpoints_are_listed_two_levels_below_the_store_root() {
        let root = temp_dir("listing");
        let store = Store::at(&root);
        store
            .record_provenance("convaiinnovations/laya", &provenance())
            .unwrap();
        // A directory that records no Provenance is not a Checkpoint, at either level, and neither
        // is a file.
        std::fs::create_dir_all(root.join("scratch/notes")).unwrap();
        std::fs::create_dir_all(root.join("convaiinnovations/incomplete")).unwrap();
        std::fs::write(root.join("convaiinnovations/README"), b"x").unwrap();

        assert_eq!(
            store.checkpoint_names().unwrap(),
            vec!["convaiinnovations/laya".to_string()]
        );

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn checkpoints_are_listed_in_name_order() {
        let root = temp_dir("order");
        let store = Store::at(&root);
        for name in ["z/zeta", "a/zulu", "a/alpha"] {
            store.record_provenance(name, &provenance()).unwrap();
        }

        assert_eq!(
            store.checkpoint_names().unwrap(),
            vec![
                "a/alpha".to_string(),
                "a/zulu".to_string(),
                "z/zeta".to_string()
            ]
        );

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_missing_store_root_holds_no_checkpoint() {
        let root = temp_dir("missing").join("nowhere");

        assert!(Store::at(&root).checkpoint_names().unwrap().is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn a_symlinked_checkpoint_is_listed() {
        let root = temp_dir("symlink");
        let store = Store::at(&root);
        // A Checkpoint kept on another volume, one level below the store root so the walk cannot
        // reach it under its own name.
        std::fs::create_dir_all(root.join("elsewhere")).unwrap();
        std::fs::create_dir_all(root.join("convaiinnovations")).unwrap();
        std::os::unix::fs::symlink(root.join("elsewhere"), root.join("convaiinnovations/laya"))
            .unwrap();
        store
            .record_provenance("convaiinnovations/laya", &provenance())
            .unwrap();

        assert_eq!(
            store.checkpoint_names().unwrap(),
            vec!["convaiinnovations/laya".to_string()]
        );
        assert!(
            root.join("elsewhere").join("provenance.json").is_file(),
            "the record is written through the link"
        );

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn a_dangling_symlink_names_no_checkpoint() {
        let root = temp_dir("dangling");
        let store = Store::at(&root);
        std::fs::create_dir_all(root.join("convaiinnovations")).unwrap();
        std::os::unix::fs::symlink(root.join("gone"), root.join("convaiinnovations/laya")).unwrap();

        assert!(store.checkpoint_names().unwrap().is_empty());

        std::fs::remove_dir_all(&root).unwrap();
    }

    fn provenance() -> Provenance {
        Provenance {
            source: "convaiinnovations/laya".to_string(),
            requested_revision: None,
            resolved_revision: "1c5edc17a7acd8701df6fc341c0d179f1c62c982".to_string(),
            files: Vec::new(),
        }
    }

    fn temp_dir(case: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("s1gate-store-{case}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        root
    }
}
