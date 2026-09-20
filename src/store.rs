//! The Model Store: the Checkpoints on disk, each held under the name it was pulled with.

use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::provenance::Provenance;

/// The store root below `$XDG_DATA_HOME`; each Checkpoint lives at `<root>/<name>` (ADR-0006).
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

    /// Where the Checkpoint pulled under `name` lives.
    pub fn checkpoint_dir(&self, name: &str) -> Result<PathBuf> {
        validate_name(name)?;
        Ok(self.root.join(name))
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

/// `$XDG_DATA_HOME/s1gate/models`, defaulting to `~/.local/share/s1gate/models` (ADR-0006). A
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

/// Reject a name that would not address exactly one Checkpoint directory.
pub fn validate_name(name: &str) -> Result<()> {
    let usable =
        !name.is_empty() && name != "." && name != ".." && !name.contains(['/', '\\', '\0']);
    if usable {
        Ok(())
    } else {
        Err(Error::InvalidName {
            name: name.to_string(),
        })
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
    fn names_addressing_more_than_one_directory_are_rejected() {
        for name in ["", ".", "..", "a/b", "../laya", "a\\b", "lay\0a"] {
            assert!(
                matches!(validate_name(name), Err(Error::InvalidName { .. })),
                "`{name}` should be rejected"
            );
        }
        assert!(validate_name("laya").is_ok());
        assert!(validate_name("laya-1c5edc1").is_ok());
    }

    #[test]
    fn a_checkpoint_round_trips_through_the_store() {
        let root = temp_dir("round-trip");
        let store = Store::at(&root);
        assert_eq!(store.provenance("laya").unwrap(), None);

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
        store.record_provenance("laya", &provenance).unwrap();
        assert_eq!(store.provenance("laya").unwrap(), Some(provenance));
        assert_eq!(
            store.checkpoint_dir("laya").unwrap(),
            root.join("laya"),
            "the Checkpoint lives under the name it was pulled with"
        );
        assert!(
            !store
                .checkpoint_dir("laya")
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
        let directory = store.checkpoint_dir("laya").unwrap();
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("model.safetensors.part"), b"half").unwrap();

        store.discard_checkpoint("laya").unwrap();
        assert!(!directory.exists(), "the name is free again");
        store.discard_checkpoint("laya").unwrap();

        std::fs::remove_dir_all(&root).unwrap();
    }

    fn temp_dir(case: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("s1gate-store-{case}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        root
    }
}
