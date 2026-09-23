//! The `models` command: list Model Identifiers available to `infer` from local files and config.

use std::path::{Path, PathBuf};

use crate::model_source;
use crate::store::Store;

pub fn run() -> crate::error::Result<()> {
    if let Ok(store) = Store::from_env() {
        for source in model_source::SOURCES {
            if store
                .provenance(source.repo)
                .ok()
                .flatten()
                .is_some_and(|provenance| provenance.source == source.repo)
            {
                let directory = store.checkpoint_dir(source.repo)?;
                if source.files.iter().all(|file| {
                    std::fs::metadata(directory.join(file)).is_ok_and(|metadata| metadata.is_file())
                }) {
                    println!("{}", source.repo);
                }
            }
        }
    }

    if typesafe_key_is_configured(
        std::env::var_os("TYPESAFE_API_KEY"),
        std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .as_deref(),
        std::env::var_os("HOME").map(PathBuf::from).as_deref(),
    ) {
        println!("jev-latest");
    }

    Ok(())
}

fn typesafe_key_is_configured(
    env_api_key: Option<std::ffi::OsString>,
    xdg_config_home: Option<&Path>,
    home: Option<&Path>,
) -> bool {
    if let Some(key) = env_api_key {
        return key.to_str().is_some_and(|key| !key.trim().is_empty());
    }

    let config_path = xdg_config_home
        .filter(|path| !path.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .or_else(|| home.map(|path| path.join(".config")))
        .map(|config_home| config_home.join("s1gate/config.toml"));
    let Some(config_path) = config_path else {
        return false;
    };
    let Ok(contents) = std::fs::read_to_string(config_path) else {
        return false;
    };
    let Ok(config) = toml::from_str::<toml::Value>(&contents) else {
        return false;
    };

    config
        .get("typesafe")
        .and_then(|typesafe| typesafe.get("api_key"))
        .and_then(toml::Value::as_str)
        .is_some_and(|key| !key.trim().is_empty())
}
