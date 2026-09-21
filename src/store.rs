//! The Model Store: the Checkpoints on disk, each held under the full name of its Model Source.

use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
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
    pub fn at(root: impl Into<PathBuf>) -> Store {
        Store { root: root.into() }
    }

    /// The store at the location the environment names.
    pub fn from_env() -> Result<Store> {
        let xdg_data_home = std::env::var_os("XDG_DATA_HOME").map(PathBuf::from);
        let home = std::env::var_os("HOME").map(PathBuf::from);
        Ok(Store::at(store_root(
            xdg_data_home.as_deref(),
            home.as_deref(),
        )?))
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Where the Checkpoint of the Model Source `name` lives.
    pub fn checkpoint_dir(&self, name: &str) -> Result<PathBuf> {
        let (owner, checkpoint) = segments(name)?;
        Ok(self.root.join(owner).join(checkpoint))
    }

    /// The Provenance `name` records, or `None` when it holds no Checkpoint.
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

    /// Record `provenance` as the Checkpoint of `name`, replacing any earlier record. Written
    /// through `provenance.json.part` and renamed, so a record on disk is always complete.
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
    pub fn discard_checkpoint(&self, name: &str) -> Result<()> {
        let directory = self.checkpoint_dir(name)?;
        match std::fs::remove_dir_all(&directory) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(Error::io("remove", &directory, error)),
        }
    }
}

/// `$XDG_DATA_HOME/s1gate/models`, defaulting to `~/.local/share/s1gate/models` (ADR-0011). A
/// relative `XDG_DATA_HOME` is not a valid path and is ignored, as the XDG Base Directory
/// specification requires.
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
        store.record_provenance("convaiinnovations/laya", &provenance).unwrap();
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

    fn temp_dir(case: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("s1gate-store-{case}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        root
    }
}
