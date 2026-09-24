//! The remote `TypeSafe` Backend selected by `jev-*` Model Identifiers.

use std::io::Read;
use std::thread;
use std::time::{Duration, SystemTime};

use serde_json::{Map, Value};
use ureq::Agent;

use crate::call::Call;
use crate::error::{Error, Result};
use crate::router::{Judged, Models, Refusal};

pub(crate) const ENDPOINT: &str = "https://api.typesafe.ai/v1/systemone";

/// The remote Model Identifier s1gate lists, and accepts, without asking `TypeSafe` for its model
/// list: the alias `TypeSafe` serves for its most recent stable release (ADR-0016, ADR-0018).
pub(crate) const ALIAS: &str = "jev-latest";
const MAX_ATTEMPTS: usize = 3;

/// Judge `call` through `TypeSafe` and return what it answered.
///
/// # Errors
///
/// Returns a `TypeSafe` runtime error when `TypeSafe` cannot be reached and when it answers 200
/// with anything but one complete response. An answer that is not 200 comes back as a `Refusal` for
/// the caller to report as its transport requires.
pub(crate) fn run_at(call: &Call, body: &[u8], api_key: &str, endpoint: &str) -> Result<Judged> {
    let agent = agent();
    let reply = send(|| {
        agent
            .post(endpoint)
            .header("Authorization", &format!("Bearer {api_key}"))
            .header("Content-Type", "application/json")
            .send(body.to_vec())
            .map_err(|_| ())
    })?;
    if reply.status != 200 {
        let fallback = match reply.status {
            401 | 403 => "check TYPESAFE_API_KEY",
            422 => "TypeSafe rejected the System One Call",
            429 => "rate limit remained after retries",
            529 => "service remained overloaded after retries",
            _ => "TypeSafe could not complete the request",
        };
        let message =
            api_error_message(&reply.body, api_key, body).unwrap_or_else(|| fallback.to_string());
        return Ok(Judged::Refusal(reply.refusal(message)));
    }
    let response: Value = serde_json::from_str(&reply.body).map_err(|_| Error::TypeSafe {
        status: Some(reply.status),
        message: "TypeSafe returned invalid JSON".to_string(),
    })?;
    validate_response(call, response).map(Judged::Answer)
}

/// List the models `TypeSafe` reports for `api_key` through the model-list endpoint beside
/// `endpoint`, which addresses the System One API.
///
/// # Errors
///
/// Returns a `TypeSafe` runtime error when `TypeSafe` cannot be reached and when it answers 200
/// with anything but the documented model list. An answer that is not 200 comes back as a
/// `Refusal` for the caller to report as its transport requires.
pub(crate) fn models_at(api_key: &str, endpoint: &str) -> Result<Models> {
    let agent = agent();
    let list_endpoint = models_endpoint(endpoint);
    let reply = send(|| {
        agent
            .get(&list_endpoint)
            .header("Authorization", &format!("Bearer {api_key}"))
            .call()
            .map_err(|_| ())
    })?;
    if reply.status != 200 {
        // The list request carries no Call to redact, and an HTTP client receives the body itself.
        let message = "TypeSafe did not list its models".to_string();
        return Ok(Models::Refusal(reply.refusal(message)));
    }
    validate_models(&reply.body).map(Models::List)
}

/// The model-list endpoint beside `endpoint`: `TypeSafe` publishes the list under the same base as
/// the System One API, at `models` instead of the System One path.
fn models_endpoint(endpoint: &str) -> String {
    let endpoint = endpoint.trim_end_matches('/');
    match endpoint.rsplit_once('/') {
        // A path to replace. A base that ends in `/` is the host itself, which has no path to
        // replace, so the list hangs off the host.
        Some((base, _)) if !base.is_empty() && !base.ends_with('/') => format!("{base}/models"),
        _ => format!("{endpoint}/models"),
    }
}

/// The client every `TypeSafe` request uses: no redirects, and every status returned rather than
/// raised, so a refusal reaches the caller in `TypeSafe`'s own terms.
fn agent() -> Agent {
    Agent::config_builder()
        .max_redirects(0)
        .http_status_as_error(false)
        .timeout_resolve(Some(Duration::from_secs(10)))
        .timeout_connect(Some(Duration::from_secs(10)))
        .timeout_recv_response(Some(Duration::from_secs(60)))
        .build()
        .new_agent()
}

/// One reply `TypeSafe` gave to a request, read in full.
struct Reply {
    status: u16,
    retry_after: Option<String>,
    body: String,
}

impl Reply {
    /// This reply as the refusal it is, with `message` standing in for the body a CLI cannot print.
    fn refusal(self, message: String) -> Refusal {
        Refusal {
            status: self.status,
            message,
            retry_after: self.retry_after,
            body: self.body.into_bytes(),
        }
    }
}

/// Send one request through the bounded retry policy and read its final reply.
///
/// `request` runs once per attempt, so a retry sends the same request again. A transport failure is
/// a `TypeSafe` runtime error: `TypeSafe` answered nothing for a caller to forward.
///
/// # Errors
///
/// Returns a `TypeSafe` runtime error when `TypeSafe` cannot be reached, and when its answer cannot
/// be read. Any status is an answer, including a refusal.
fn send(
    mut request: impl FnMut() -> std::result::Result<ureq::http::Response<ureq::Body>, ()>,
) -> Result<Reply> {
    for attempt in 0..MAX_ATTEMPTS {
        let mut response = request().map_err(|()| Error::TypeSafe {
            status: None,
            message: "could not reach the TypeSafe API".to_string(),
        })?;
        let status = response.status().as_u16();
        let retry_after = retry_after_header(&response);
        if (status == 429 || status == 529) && attempt + 1 < MAX_ATTEMPTS {
            let delay = retry_after
                .as_deref()
                .and_then(|value| retry_after_delay(value, SystemTime::now()))
                .unwrap_or_else(|| Duration::from_millis(200 * (1 << attempt)));
            thread::sleep(delay);
            continue;
        }
        let mut body = String::new();
        response
            .body_mut()
            .as_reader()
            .read_to_string(&mut body)
            .map_err(|_| Error::TypeSafe {
                status: Some(status),
                message: "could not read the TypeSafe response".to_string(),
            })?;
        return Ok(Reply {
            status,
            retry_after,
            body,
        });
    }
    unreachable!("the bounded retry loop always returns or continues to its last attempt")
}

/// The `Retry-After` header of `response`, as `TypeSafe` wrote it, for the retry policy to honor
/// and an HTTP client to receive.
fn retry_after_header(response: &ureq::http::Response<ureq::Body>) -> Option<String> {
    response
        .headers()
        .get("retry-after")
        .and_then(|value| value.to_str().ok())
        .map(str::to_string)
}

fn retry_after_delay(value: &str, now: SystemTime) -> Option<Duration> {
    value
        .parse::<u64>()
        .ok()
        .map(Duration::from_secs)
        .or_else(|| {
            httpdate::parse_http_date(value)
                .ok()
                .map(|deadline| deadline.duration_since(now).unwrap_or_default())
        })
}

fn api_error_message(response_body: &str, api_key: &str, call_body: &[u8]) -> Option<String> {
    let response: Value = serde_json::from_str(response_body).ok()?;
    let error = response.get("error").unwrap_or(&response);
    let message = error
        .as_str()
        .or_else(|| error.get("message").and_then(Value::as_str))
        .or_else(|| error.get("detail").and_then(Value::as_str))?;
    let call_body = String::from_utf8_lossy(call_body);
    let message = message
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .replace(api_key, "[redacted]")
        .replace(call_body.as_ref(), "[redacted]");
    let call: Value = serde_json::from_slice(call_body.as_bytes()).ok()?;
    if contains_call_text(&call, &message) {
        return None;
    }
    let message: String = message.chars().take(160).collect();
    let message = message.trim();
    (!message.is_empty()).then(|| message.to_string())
}

fn contains_call_text(call: &Value, message: &str) -> bool {
    match call {
        Value::Object(object) => object.iter().any(|(key, value)| {
            (key.len() >= 4 && message.contains(key)) || contains_call_text(value, message)
        }),
        Value::Array(values) => values
            .iter()
            .any(|value| contains_call_text(value, message)),
        Value::String(value) => value.len() >= 4 && message.contains(value),
        _ => false,
    }
}

/// The error for a 200 that is not the complete answer it claims to be.
fn incomplete(message: &str) -> Error {
    Error::TypeSafe {
        status: Some(200),
        message: message.to_string(),
    }
}

/// The model-list entries `TypeSafe` published, each carrying the fields it documents and anything
/// else it chose to send, as it wrote them.
///
/// # Errors
///
/// Returns a `TypeSafe` runtime error when the body is not JSON, when it carries no `models` array,
/// and when an entry lacks a `name`, a `description`, or a `release_date`.
fn validate_models(response: &str) -> Result<Vec<Value>> {
    let invalid = || incomplete("TypeSafe returned an incomplete model list");
    let response: Value = serde_json::from_str(response).map_err(|_| invalid())?;
    let models = response
        .get("models")
        .and_then(Value::as_array)
        .ok_or_else(invalid)?;
    let documented = models.iter().all(|model| {
        model
            .get("name")
            .and_then(Value::as_str)
            .is_some_and(|name| !name.is_empty())
            && model.get("description").is_some_and(Value::is_string)
            && model.get("release_date").is_some_and(Value::is_string)
    });
    if !documented {
        return Err(invalid());
    }
    Ok(models.clone())
}

fn validate_response(call: &Call, response: Value) -> Result<Value> {
    let invalid = || incomplete("TypeSafe returned an incomplete response");
    let Some(object) = response.as_object() else {
        return Err(invalid());
    };
    let Some(model) = object.get("model").and_then(Value::as_str) else {
        return Err(invalid());
    };
    if model.is_empty() {
        return Err(invalid());
    }
    let Some(answers) = object.get("answers").and_then(Value::as_object) else {
        return Err(invalid());
    };
    if answers.len() != call.questions.iter().count()
        || call
            .questions
            .iter()
            .any(|(id, _)| !answers.get(id).is_some_and(Value::is_object))
    {
        return Err(invalid());
    }
    let Some(usage) = object.get("usage").and_then(Value::as_object) else {
        return Err(invalid());
    };
    if !usage.get("input_tokens").is_some_and(Value::is_u64)
        || !usage.get("output_tokens").is_some_and(Value::is_u64)
    {
        return Err(invalid());
    }

    let mut result = Map::new();
    result.insert("model".to_string(), Value::String(model.to_string()));
    result.insert("answers".to_string(), Value::Object(answers.clone()));
    result.insert("usage".to_string(), Value::Object(usage.clone()));
    Ok(Value::Object(result))
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, SystemTime};

    use super::{models_endpoint, retry_after_delay};

    #[test]
    fn the_model_list_sits_beside_the_system_one_endpoint() {
        assert_eq!(
            models_endpoint("https://api.typesafe.ai/v1/systemone"),
            "https://api.typesafe.ai/v1/models"
        );
        assert_eq!(
            models_endpoint("http://127.0.0.1:8080/v1/systemone"),
            "http://127.0.0.1:8080/v1/models"
        );
        assert_eq!(
            models_endpoint("https://proxy.example/systemone"),
            "https://proxy.example/models"
        );
        assert_eq!(
            models_endpoint("https://api.typesafe.ai/v1/systemone/"),
            "https://api.typesafe.ai/v1/models",
            "a trailing slash does not become a path segment"
        );
        assert_eq!(
            models_endpoint("https://api.typesafe.ai"),
            "https://api.typesafe.ai/models",
            "a host with no path has no System One path to replace"
        );
    }

    #[test]
    fn retry_after_accepts_seconds_and_http_dates() {
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000);
        let deadline = now + Duration::from_secs(3);
        let date = httpdate::fmt_http_date(deadline);

        assert_eq!(retry_after_delay("5", now), Some(Duration::from_secs(5)));
        assert_eq!(retry_after_delay(&date, now), Some(Duration::from_secs(3)));
        assert_eq!(
            retry_after_delay("not a date", now),
            None,
            "an invalid header falls back to exponential backoff"
        );
    }
}
