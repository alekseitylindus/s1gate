//! Resolve a System One Call's Model Identifier to its local or remote Backend.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};

use serde_json::{Value, json};

use crate::call::Call;
use crate::error::{Error, Result};
use crate::laya::Loaded;
use crate::model_source::{self, ModelSource};
use crate::store::Store;

/// What the Model Router resolved one System One Call to.
pub enum Judged {
    /// The selected Backend judged the call, and this is the response.
    Answer(Value),
    /// The remote Backend refused the call, in `TypeSafe`'s own terms.
    Refusal(Refusal),
}

/// What the Model Router found available to a System One Call.
pub enum Models {
    /// One entry per Model Identifier available, each carrying the `name`, `description`, and
    /// `release_date` of `TypeSafe`'s model-list shape, and whatever else `TypeSafe` sent with it.
    List(Vec<Value>),
    /// The remote Backend refused the model-list request, in `TypeSafe`'s own terms.
    Refusal(Refusal),
}

/// One Model Identifier this process can judge a System One Call with, and the entry a model list
/// publishes for it.
#[derive(Debug, Clone, PartialEq)]
pub struct Available {
    /// The Model Identifier a caller sends as `model`.
    pub identifier: String,
    /// The entry a model list publishes for it: a local Backend describes itself, and a remote one
    /// carries what `TypeSafe` sent. `None` for an identifier that is available by configuration
    /// alone, which no model list has described.
    pub listing: Option<Value>,
}

/// The remote Backend's final refusal of a request, as `TypeSafe` answered it.
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
    /// The Model Store this process resolves local Checkpoints from. `None` when the environment
    /// names no location: a remote Call still judges, and a local one says so.
    store: Option<Store>,
    local: Local,
}

enum Local {
    /// Select the local Backend from the Model Store for each Call.
    FromStore,
    /// The local Backend the server loaded at startup, judged one Call at a time. `None` when no
    /// Checkpoint was present, so no local Model Identifier is served. Boxed so a variant that is
    /// empty for every process that resolves its Checkpoint per Call is not the enum's size.
    Loaded(Option<Box<Mutex<Loaded>>>),
}

impl ModelRouter {
    /// The Model Identifiers this process can judge a System One Call with, without reaching the
    /// network: the local Checkpoints it can load, followed by the remote alias when a `TypeSafe`
    /// credential is configured.
    ///
    /// Availability is decided from the Pull record and the presence of the Checkpoint's files,
    /// never by hashing them, so an available Checkpoint can still fail to load (ADR-0018).
    ///
    /// # Errors
    ///
    /// Returns a `TypeSafe` runtime error when the configured credential is unusable, and
    /// [`Error::InvalidCheckpoint`] or [`Error::Io`] when the Model Store cannot be read. A process
    /// with no credential configured is not a failure: its local identifiers stand alone.
    pub fn available(&self) -> Result<Vec<Available>> {
        let mut available = self.local_available()?;
        if matches!(self.typesafe_credential()?, Credential::Key(_)) {
            available.push(Available {
                identifier: crate::typesafe::ALIAS.to_string(),
                listing: None,
            });
        }
        Ok(available)
    }

    /// The local Model Identifiers this process can judge a Call with: the Checkpoints the server
    /// loaded at startup, or, in a process that resolves its Checkpoint per Call, the stored
    /// Checkpoints the Model Store holds complete. A process the environment gives no Model Store
    /// location has none.
    fn local_available(&self) -> Result<Vec<Available>> {
        match &self.local {
            // A Checkpoint that arrived after startup is not available to the server: a Call naming
            // it fails rather than reloading weights.
            Local::Loaded(loaded) => Ok(loaded
                .as_ref()
                .map(|_| local_entry(&model_source::LAYA))
                .into_iter()
                .collect()),
            Local::FromStore => {
                let Some(store) = &self.store else {
                    return Ok(Vec::new());
                };
                let mut available = Vec::new();
                for source in model_source::SOURCES {
                    let Some(checkpoint) = store.checkpoint(source)? else {
                        continue;
                    };
                    if checkpoint.files_present()? {
                        available.push(local_entry(source));
                    }
                }
                Ok(available)
            }
        }
    }

    /// The Model Identifiers a model list publishes: the local Backends' own entries, followed by
    /// the models `TypeSafe` currently serves when a credential is configured (ADR-0016).
    ///
    /// # Errors
    ///
    /// Returns a `TypeSafe` runtime error when the configured credential is unusable, when
    /// `TypeSafe` cannot be reached, and when it answers with anything but its documented model
    /// list. A process with no configured credential is not a failure: its local list stands alone.
    pub fn models(&self) -> Result<Models> {
        let mut models: Vec<Value> = self
            .local_available()?
            .into_iter()
            .filter_map(|available| available.listing)
            .collect();
        let Credential::Key(settings) = self.typesafe_credential()? else {
            return Ok(Models::List(models));
        };
        match crate::typesafe::models_at(&settings.api_key, &settings.endpoint)? {
            Models::List(remote) => {
                models.extend(remote);
                Ok(Models::List(models))
            }
            // A refusal is not a partial list: the caller receives what `TypeSafe` answered.
            Models::Refusal(refusal) => Ok(Models::Refusal(refusal)),
        }
    }

    /// The `TypeSafe` credential this process's configuration provides.
    fn typesafe_credential(&self) -> Result<Credential> {
        typesafe_credential(
            self.env_api_key.clone(),
            self.xdg_config_home.as_deref(),
            self.home.as_deref(),
            &self.endpoint,
        )
    }

    /// Use the process configuration for the Model Store and `TypeSafe` Backend. A process the
    /// environment gives no Model Store location still judges remote Calls.
    pub fn from_env() -> Self {
        Self::from_env_with(Store::from_env().ok())
    }

    /// The process configuration, over `store` as the Model Store.
    fn from_env_with(store: Option<Store>) -> Self {
        Self::with_settings(
            crate::typesafe::ENDPOINT,
            std::env::var_os("TYPESAFE_API_KEY"),
            std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from),
            std::env::var_os("HOME").map(PathBuf::from),
            store,
        )
    }

    /// Load present local Checkpoints once for the HTTP server.
    pub(crate) fn for_server() -> Result<Self> {
        let store = Store::from_env()?;
        let local = match store.checkpoint(&model_source::LAYA)? {
            Some(checkpoint) => Some(Box::new(Mutex::new(Loaded::load(&checkpoint)?))),
            None => None,
        };
        let mut router = Self::from_env_with(Some(store));
        router.local = Local::Loaded(local);
        Ok(router)
    }

    pub(crate) fn with_settings(
        endpoint: &str,
        env_api_key: Option<OsString>,
        xdg_config_home: Option<PathBuf>,
        home: Option<PathBuf>,
        store: Option<Store>,
    ) -> Self {
        Self {
            endpoint: endpoint.to_string(),
            env_api_key,
            xdg_config_home,
            home,
            store,
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
            return match self.typesafe_credential()? {
                Credential::Key(settings) => {
                    crate::typesafe::run_at(&call, body, &settings.api_key, &settings.endpoint)
                }
                // A Call that needs the remote Backend fails without a credential; discovery,
                // which does not need one, reports what it has instead.
                Credential::Missing { message } => Err(Error::TypeSafe {
                    status: None,
                    message,
                }),
            };
        }
        let source = model_source::lookup_identifier(&call.model)?;
        match &self.local {
            Local::FromStore => {
                let store = self.store.as_ref().ok_or(Error::NoStoreRoot)?;
                let checkpoint = store
                    .checkpoint(source)?
                    .ok_or_else(|| Error::missing_checkpoint(source.repo))?;
                Loaded::load(&checkpoint)?.run(&call).map(Judged::Answer)
            }
            // One Call at a time: a Call that panicked must not stop every later local Call.
            Local::Loaded(Some(loaded)) => loaded
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .run(&call)
                .map(Judged::Answer),
            Local::Loaded(None) => Err(Error::missing_checkpoint(source.repo)),
        }
    }
}

/// The entry a model list publishes for the local Backend of `source`: what the Backend judges and
/// the date its Checkpoint was published (ADR-0016).
fn local_entry(source: &ModelSource) -> Available {
    Available {
        identifier: source.repo.to_string(),
        listing: Some(json!({
            "name": source.repo,
            "description": "Laya, a local Backend for choice, score, and noul Questions.",
            "release_date": "2026-09-18"
        })),
    }
}

struct TypeSafeSettings {
    api_key: String,
    endpoint: String,
}

/// The `TypeSafe` credential the process configuration provides.
enum Credential {
    /// A usable key, with the endpoint to reach it at.
    Key(TypeSafeSettings),
    /// No usable key is configured. `message` tells a caller that needs one where to put it; a
    /// caller that does not, such as model discovery, goes on without it.
    Missing { message: String },
}

fn typesafe_credential(
    env_api_key: Option<OsString>,
    xdg_config_home: Option<&Path>,
    home: Option<&Path>,
    default_endpoint: &str,
) -> Result<Credential> {
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

    // The file supplies a key and an endpoint alike, but it never gates the environment's key: a
    // process with only `TYPESAFE_API_KEY` set reaches the default endpoint, as it always has.
    let config = match config_path {
        None => None,
        Some(path) => match std::fs::read_to_string(&path) {
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
            // A file that is not there is no credential configured, which only a caller that needs
            // one reports. A file that is there but unreadable is this process's defect.
            Err(error) if env_api_key.is_none() && error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Credential::Missing {
                    message: format!(
                        "cannot read {} ({error}); set TYPESAFE_API_KEY or add typesafe.api_key to this file",
                        path.display()
                    ),
                });
            }
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
        },
    };

    let Some((path, config)) = config else {
        let Some(api_key) = env_api_key else {
            return Ok(Credential::Missing {
                message: "cannot locate configuration; set TYPESAFE_API_KEY or HOME".to_string(),
            });
        };
        return Ok(Credential::Key(TypeSafeSettings {
            api_key,
            endpoint: default_endpoint.to_string(),
        }));
    };

    let file_api_key = config
        .get("typesafe")
        .and_then(|typesafe| typesafe.get("api_key"))
        .and_then(toml::Value::as_str)
        .filter(|key| !key.trim().is_empty())
        .map(str::to_string);
    let Some(api_key) = env_api_key.or(file_api_key) else {
        return Ok(Credential::Missing {
            message: format!(
                "{} must contain a non-empty string at typesafe.api_key; set TYPESAFE_API_KEY or fix the file",
                path.display()
            ),
        });
    };

    let endpoint = match config
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
    };

    Ok(Credential::Key(TypeSafeSettings { api_key, endpoint }))
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::path::PathBuf;
    use std::thread;

    use super::{Error, ModelRouter, Models};
    use crate::model_source::{self, ModelSource};
    use crate::store::Store;

    /// What `TypeSafe` answers a model-list request with.
    const MODELS: &str = r#"{"models":[{"name":"jev-latest","description":"The most recent stable release.","release_date":"2025-11-04"}]}"#;

    /// An endpoint nothing listens on: the tests below that never ask `TypeSafe` anything send no
    /// request to it.
    const ENDPOINT: &str = "http://127.0.0.1:1/v1/systemone";

    #[test]
    fn availability_is_the_complete_checkpoint_and_the_configured_alias() {
        let root = temp_dir("availability");
        let store = Store::at(&root);
        let source = &model_source::LAYA;
        let available = |key: Option<&str>| {
            ModelRouter::with_settings(
                ENDPOINT,
                key.map(OsString::from),
                None,
                None,
                Some(store.clone()),
            )
            .available()
            .expect("the process configuration is usable")
        };

        // A store that records nothing holds no Checkpoint, and no key is configured: a list of
        // Model Identifiers with nothing in it is the answer, not a failure.
        assert!(available(None).is_empty());

        store.record_provenance(source.repo, &provenance()).unwrap();
        assert!(
            available(None).is_empty(),
            "a Checkpoint whose files are not there is not available"
        );

        write_files(&root, source);
        let listed = available(None);
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].identifier, source.repo);
        assert_eq!(
            listed[0]
                .listing
                .as_ref()
                .expect("the local Backend's entry")["name"],
            source.repo
        );

        // A configured key adds the remote alias, which no model list has described yet.
        assert_eq!(
            available(Some("test-key"))
                .into_iter()
                .map(|available| available.identifier)
                .collect::<Vec<_>>(),
            vec![source.repo.to_string(), "jev-latest".to_string()]
        );

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_configuration_that_cannot_be_read_fails_availability() {
        for (key, diagnostic) in [
            ("", "TYPESAFE_API_KEY is empty"),
            ("  ", "TYPESAFE_API_KEY is empty"),
        ] {
            let router =
                ModelRouter::with_settings(ENDPOINT, Some(OsString::from(key)), None, None, None);

            let error = router
                .available()
                .expect_err("a key that cannot be used is a failure");

            assert!(matches!(error, Error::TypeSafe { .. }), "{error:?}");
            assert!(error.to_string().contains(diagnostic), "{error}");
        }
    }

    /// A Model Store holding a Checkpoint of `source` whose files are in place: availability asks
    /// whether they are there, not what they hold.
    fn write_files(root: &std::path::Path, source: &ModelSource) {
        for path in source.files {
            let file = root.join(source.repo).join(path);
            std::fs::create_dir_all(file.parent().expect("a file has a directory")).unwrap();
            std::fs::write(&file, b"x").unwrap();
        }
    }

    fn provenance() -> crate::provenance::Provenance {
        crate::provenance::Provenance {
            source: model_source::LAYA.repo.to_string(),
            requested_revision: None,
            resolved_revision: "1c5edc17a7acd8701df6fc341c0d179f1c62c982".to_string(),
            files: Vec::new(),
        }
    }

    fn temp_dir(case: &str) -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("s1gate-router-{case}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    #[test]
    fn an_environment_key_lists_remote_models_without_a_configuration_file() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("a local endpoint");
        let endpoint = format!(
            "http://{}/v1/systemone",
            listener.local_addr().expect("address")
        );
        let stand_in = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("the model list request");
            let mut request = Vec::new();
            let mut buffer = [0; 1024];
            while !request.windows(4).any(|part| part == b"\r\n\r\n") {
                let count = stream.read(&mut buffer).expect("read the request");
                assert!(count > 0, "the request closed before its headers");
                request.extend_from_slice(&buffer[..count]);
            }
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{MODELS}",
                MODELS.len()
            )
            .expect("send the model list");
            String::from_utf8(request).expect("UTF-8 request")
        });

        // The key is the whole credential: no configuration file is there and none is reachable.
        let router = ModelRouter::with_settings(
            &endpoint,
            Some(OsString::from("test-key")),
            None,
            None,
            None,
        );
        let Models::List(models) = router.models().expect("the model list is discovered") else {
            panic!("the stand-in answers with a list");
        };

        assert_eq!(models.len(), 1);
        assert_eq!(models[0]["name"], "jev-latest");
        assert_eq!(models[0]["release_date"], "2025-11-04");
        let request = stand_in.join().expect("the stand-in thread");
        let request = request.to_ascii_lowercase();
        assert!(request.starts_with("get /v1/models http/1.1"), "{request}");
        assert!(
            request.contains("authorization: bearer test-key"),
            "{request}"
        );
    }
}
