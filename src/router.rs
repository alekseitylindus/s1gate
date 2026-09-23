//! Resolve a System One Call's Model Identifier to its local or remote Backend.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};

use serde_json::{Value, json};

use crate::call::Call;
use crate::error::{Error, Result};
use crate::laya::Loaded;
use crate::model_source;
use crate::store::Store;

/// What the Model Router resolved one System One Call to.
pub enum Judged {
    /// The selected Backend judged the call, and this is the response.
    Answer(Value),
    /// The remote Backend refused the call, in `TypeSafe`'s own terms.
    Refusal(Refusal),
}

/// The remote Backend's final refusal of a System One Call, as `TypeSafe` answered it.
///
/// An HTTP client receives the status, body, and retry delay unchanged. The CLI prints `message`
/// instead: a short diagnostic that carries neither the credential nor request data.
pub struct Refusal {
    /// The HTTP status `TypeSafe` answered with.
    pub status: u16,
    /// The body `TypeSafe` answered with, as it wrote it.
    pub body: Vec<u8>,
    /// The `Retry-After` header of that answer, when it carried one.
    pub retry_after: Option<String>,
    /// A short diagnostic without request data or credentials.
    pub message: String,
}

/// Routes validated System One Calls without reading or writing a transport stream.
pub struct ModelRouter {
    endpoint: String,
    env_api_key: Option<OsString>,
    xdg_config_home: Option<PathBuf>,
    home: Option<PathBuf>,
    local: Local,
}

enum Local {
    /// Select the local Backend from the Model Store for each Call.
    FromStore,
    /// The local Backend the server loaded at startup, judged one Call at a time. `None` when no
    /// Checkpoint was present, so no local Model Identifier is served.
    Loaded(Option<Mutex<Loaded>>),
}

impl ModelRouter {
    /// Describe the local Backends loaded when the HTTP server started.
    pub(crate) fn loaded_local_models(&self) -> Value {
        let models = if matches!(self.local, Local::Loaded(Some(_))) {
            vec![json!({
                "name": model_source::LAYA.repo,
                "description": "Laya, a local Backend for choice, score, and noul Questions.",
                "release_date": "2026-09-18"
            })]
        } else {
            Vec::new()
        };
        json!({"models": models})
    }

    /// Use the process configuration for the Model Store and TypeSafe Backend.
    pub fn from_env() -> Self {
        Self::with_settings(
            crate::typesafe::ENDPOINT,
            std::env::var_os("TYPESAFE_API_KEY"),
            std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from),
            std::env::var_os("HOME").map(PathBuf::from),
        )
    }

    /// Load present local Checkpoints once for the HTTP server.
    pub(crate) fn for_server() -> Result<Self> {
        let store = Store::from_env()?;
        let local = if store.provenance(model_source::LAYA.repo)?.is_some() {
            Some(Mutex::new(Loaded::load(&store, &model_source::LAYA)?))
        } else {
            None
        };
        let mut router = Self::from_env();
        router.local = Local::Loaded(local);
        Ok(router)
    }

    pub(crate) fn with_settings(
        endpoint: &str,
        env_api_key: Option<OsString>,
        xdg_config_home: Option<PathBuf>,
        home: Option<PathBuf>,
    ) -> Self {
        Self {
            endpoint: endpoint.to_string(),
            env_api_key,
            xdg_config_home,
            home,
            local: Local::FromStore,
        }
    }

    /// Parse and judge one JSON System One Call.
    ///
    /// The remote Backend's own answer is kept as it stands, so every transport reports a refusal
    /// in its own terms instead of through a shared error.
    ///
    /// # Errors
    ///
    /// Returns a usage error for invalid calls or unsupported identifiers, and a runtime error
    /// when the selected Backend cannot judge the call at all. A remote Backend's refusal is not
    /// an error here: it is `Judged::Refusal`.
    pub fn judge(&self, body: &[u8]) -> Result<Judged> {
        let call = Call::from_bytes(body)?;
        if call.model.starts_with("jev-") {
            let settings = typesafe_settings(
                self.env_api_key.clone(),
                self.xdg_config_home.as_deref(),
                self.home.as_deref(),
                &self.endpoint,
            )?;
            return crate::typesafe::run_at(
                &call,
                body,
                Some(&settings.api_key),
                &settings.endpoint,
            );
        }
        let source = model_source::lookup_identifier(&call.model)?;
        match &self.local {
            Local::FromStore => {
                let store = Store::from_env()?;
                if store.provenance(source.repo)?.is_none() {
                    return Err(Error::MissingCheckpoint {
                        name: source.repo.to_string(),
                    });
                }
                crate::laya::run(&store, source, &call).map(Judged::Answer)
            }
            // One Call at a time: a Call that panicked must not stop every later local Call.
            Local::Loaded(Some(loaded)) => loaded
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .run(&call)
                .map(Judged::Answer),
            Local::Loaded(None) => Err(Error::MissingCheckpoint {
                name: source.repo.to_string(),
            }),
        }
    }
}

struct TypeSafeSettings {
    api_key: String,
    endpoint: String,
}

fn typesafe_settings(
    env_api_key: Option<OsString>,
    xdg_config_home: Option<&Path>,
    home: Option<&Path>,
    default_endpoint: &str,
) -> Result<TypeSafeSettings> {
    let env_api_key = env_api_key
        .map(|key| {
            let key = key.to_str().ok_or_else(|| Error::TypeSafe {
                status: None,
                message: "TYPESAFE_API_KEY is not valid Unicode".to_string(),
            })?;
            if key.trim().is_empty() {
                return Err(Error::TypeSafe {
                    status: None,
                    message: "TYPESAFE_API_KEY is empty; set it to a non-empty key".to_string(),
                });
            }
            Ok(key.to_string())
        })
        .transpose()?;

    let config_path = xdg_config_home
        .filter(|path| !path.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .or_else(|| home.map(|path| path.join(".config")))
        .map(|config_home| config_home.join("s1gate/config.toml"));
    let config = if let Some(path) = config_path {
        match std::fs::read_to_string(&path) {
            Ok(contents) => match toml::from_str::<toml::Value>(&contents) {
                Ok(config) => Some((path, config)),
                Err(_) if env_api_key.is_some() => None,
                Err(_) => {
                    return Err(Error::TypeSafe {
                        status: None,
                        message: format!(
                            "{} is not valid TOML; fix it or set TYPESAFE_API_KEY",
                            path.display()
                        ),
                    });
                }
            },
            Err(_) if env_api_key.is_some() => None,
            Err(error) => {
                return Err(Error::TypeSafe {
                    status: None,
                    message: format!(
                        "cannot read {} ({error}); set TYPESAFE_API_KEY or add typesafe.api_key to this file",
                        path.display()
                    ),
                });
            }
        }
    } else {
        None
    };

    let configured_api_key = config.as_ref().and_then(|(_, config)| {
        config
            .get("typesafe")
            .and_then(|typesafe| typesafe.get("api_key"))
            .and_then(toml::Value::as_str)
            .filter(|key| !key.trim().is_empty())
            .map(str::to_string)
    });
    let api_key = env_api_key.or(configured_api_key).ok_or_else(|| Error::TypeSafe {
        status: None,
        message: config.as_ref().map_or_else(
            || "cannot locate configuration; set TYPESAFE_API_KEY or HOME".to_string(),
            |(path, _)| format!(
                "{} must contain a non-empty string at typesafe.api_key; set TYPESAFE_API_KEY or fix the file",
                path.display()
            ),
        ),
    })?;

    let endpoint = if let Some((path, config)) = &config {
        match config
            .get("typesafe")
            .and_then(|typesafe| typesafe.get("endpoint"))
        {
            None => default_endpoint.to_string(),
            Some(toml::Value::String(endpoint)) if !endpoint.trim().is_empty() => endpoint.clone(),
            Some(_) => {
                return Err(Error::TypeSafe {
                    status: None,
                    message: format!(
                        "{} must contain a non-empty string at typesafe.endpoint",
                        path.display()
                    ),
                });
            }
        }
    } else {
        default_endpoint.to_string()
    };

    Ok(TypeSafeSettings { api_key, endpoint })
}
