//! The HTTP transport core: a request in, a response out, over plain data.
//!
//! What serving System One Calls means lives here — the routes, the status each failure answers,
//! the `detail` body s1gate writes, the reason phrases, and the size limits — so a transport reads
//! a socket and writes one, and nothing else. The Model Router neither reads nor writes a stream
//! (ADR-0017).

use serde_json::{Value, json};

use crate::error::Error;
use crate::router::{Judged, ModelRouter, Models, Refusal};

/// The most bytes of a request body s1gate reads.
pub(crate) const MAX_BODY: usize = 8 * 1024 * 1024;
/// The most bytes of request headers s1gate reads.
pub(crate) const MAX_HEADERS: usize = 16 * 1024;

/// One request, as a transport read it. No header reaches the core: an incoming `Authorization`
/// never replaces the process's own credential, and nothing else is read.
pub(crate) struct Request {
    pub(crate) method: String,
    pub(crate) path: String,
    pub(crate) body: Vec<u8>,
}

/// One response for a transport to write.
pub(crate) struct Response {
    pub(crate) status: u16,
    /// The remote Backend's own `Retry-After`, when it sent one.
    pub(crate) retry_after: Option<String>,
    pub(crate) body: Vec<u8>,
}

impl Response {
    /// This server's own answer, as a `TypeSafe`-shaped `detail` body.
    pub(crate) fn error(status: u16, message: &str) -> Self {
        Self::json(status, detail(message))
    }

    pub(crate) fn json(status: u16, value: Value) -> Self {
        Self {
            status,
            retry_after: None,
            body: serde_json::to_vec(&value).expect("JSON Value serializes"),
        }
    }

    /// The remote Backend's answer: its status, its body, and its retry delay as it sent them.
    pub(crate) fn forward(refusal: Refusal) -> Self {
        Self {
            status: refusal.status,
            // A header cannot carry a line break, whatever the remote Backend writes.
            retry_after: refusal
                .retry_after
                .filter(|value| !value.chars().any(char::is_control)),
            body: refusal.body,
        }
    }
}

/// The answer to one request.
pub(crate) fn handle(router: &ModelRouter, request: Request) -> Response {
    let Request { method, path, body } = request;
    match (method.as_str(), path.as_str()) {
        ("GET", "/v1/models") => list_models(router),
        ("POST", "/v1/systemone") => judge_call(router, &body),
        (_, path) if path != "/v1/systemone" && path != "/v1/models" => {
            Response::error(404, "not found")
        }
        _ => Response::error(405, "method not allowed"),
    }
}

/// The list of Model Identifiers available to a System One Call, local and remote.
fn list_models(router: &ModelRouter) -> Response {
    match router.models() {
        Ok(Models::List(models)) => Response::json(200, json!({"models": models})),
        Ok(Models::Refusal(refusal)) => Response::forward(refusal),
        // The remote Backend never answered with a model list of its own to forward.
        Err(error) => Response::error(502, &error.to_string()),
    }
}

/// The answer to one System One Call: the Backend's response, or what stopped it.
fn judge_call(router: &ModelRouter, body: &[u8]) -> Response {
    match router.judge(body) {
        Ok(Judged::Answer(answer)) => Response::json(200, answer),
        Ok(Judged::Refusal(refusal)) => Response::forward(refusal),
        // A Call s1gate will not judge at all: the caller sent something it does not support.
        Err(error @ (Error::InvalidCall { .. } | Error::UnsupportedModelIdentifier { .. })) => {
            Response::error(422, &error.to_string())
        }
        Err(Error::MissingCheckpoint { name }) => Response::error(
            503,
            &format!("missing Checkpoint `{name}`; run s1gate pull {name}"),
        ),
        // The remote Backend never answered with a response of its own to forward.
        Err(error @ Error::TypeSafe { .. }) => Response::error(502, &error.to_string()),
        // Anything else is s1gate's own failure, which is what a 500 says.
        Err(error) => Response::error(500, &error.to_string()),
    }
}

/// The reason phrase of `status`, which a client may ignore either way.
pub(crate) fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        409 => "Conflict",
        413 => "Content Too Large",
        422 => "Unprocessable Entity",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        529 => "Site is Overloaded",
        _ => "Unknown",
    }
}

fn detail(message: &str) -> Value {
    json!({"detail": [{"loc": ["body"], "msg": message, "type": "value_error"}]})
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Store;

    /// A valid local System One Call.
    const CALL: &str =
        r#"{"model":"convaiinnovations/laya","state":"x","questions":{"risk":{"type":"noul"}}}"#;
    /// An endpoint nothing listens on: no test here asks `TypeSafe` anything.
    const ENDPOINT: &str = "http://127.0.0.1:1/v1/systemone";

    #[test]
    fn an_unknown_path_is_not_found_and_a_known_one_rejects_the_wrong_method() {
        let router = router(None);

        let response = handle(&router, request("GET", "/v2/answers", b""));
        assert_eq!(response.status, 404);

        let response = handle(&router, request("POST", "/v1/models", b""));
        assert_eq!(response.status, 405, "the path exists, the method does not");
    }

    #[test]
    fn an_invalid_call_is_unprocessable_with_the_detail_body() {
        let router = router(None);

        let response = handle(
            &router,
            request("POST", "/v1/systemone", br#"{"model":"x"}"#),
        );

        assert_eq!(response.status, 422);
        let body: Value = serde_json::from_slice(&response.body).expect("the detail body");
        assert_eq!(body["detail"][0]["loc"], json!(["body"]));
        assert!(
            body["detail"][0]["msg"]
                .as_str()
                .is_some_and(|message| !message.is_empty()),
            "{body}"
        );
    }

    #[test]
    fn an_unsupported_model_identifier_is_unprocessable() {
        let router = router(None);
        let call = CALL.replace("convaiinnovations/laya", "someone/else");

        let response = handle(&router, request("POST", "/v1/systemone", call.as_bytes()));

        assert_eq!(response.status, 422);
    }

    #[test]
    fn a_missing_checkpoint_is_unavailable_and_names_its_pull() {
        let root = temp_dir();
        let router = router(Some(Store::at(&root)));

        let response = handle(&router, request("POST", "/v1/systemone", CALL.as_bytes()));

        assert_eq!(response.status, 503);
        let body: Value = serde_json::from_slice(&response.body).expect("the detail body");
        assert!(
            body["detail"][0]["msg"]
                .as_str()
                .is_some_and(|message| message.contains("run s1gate pull convaiinnovations/laya")),
            "{body}"
        );

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_remote_call_without_a_credential_is_bad_gateway() {
        let router = router(None);
        let call = CALL.replace("convaiinnovations/laya", "jev-latest");

        let response = handle(&router, request("POST", "/v1/systemone", call.as_bytes()));

        assert_eq!(
            response.status, 502,
            "there is no answer of TypeSafe's to forward"
        );
    }

    #[test]
    fn a_forwarded_refusal_keeps_what_typesafe_answered() {
        let response = Response::forward(Refusal {
            status: 429,
            body: br#"{"detail":"slow down"}"#.to_vec(),
            retry_after: Some("30".to_string()),
            message: "rate limit remained after retries".to_string(),
        });

        assert_eq!(response.status, 429);
        assert_eq!(response.body, br#"{"detail":"slow down"}"#);
        assert_eq!(response.retry_after.as_deref(), Some("30"));
    }

    #[test]
    fn a_retry_after_carrying_a_line_break_is_dropped() {
        let response = Response::forward(Refusal {
            status: 429,
            body: Vec::new(),
            retry_after: Some("30\r\nX-Injected: 1".to_string()),
            message: String::new(),
        });

        assert_eq!(
            response.retry_after, None,
            "a value that would write a second header is not forwarded"
        );
    }

    fn request(method: &str, path: &str, body: &[u8]) -> Request {
        Request {
            method: method.to_string(),
            path: path.to_string(),
            body: body.to_vec(),
        }
    }

    /// A router with no configured credential, over `store` as the Model Store.
    fn router(store: Option<Store>) -> ModelRouter {
        ModelRouter::with_settings(ENDPOINT, None, None, None, store)
    }

    fn temp_dir() -> std::path::PathBuf {
        let root = std::env::temp_dir().join(format!("s1gate-http-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        root
    }
}
