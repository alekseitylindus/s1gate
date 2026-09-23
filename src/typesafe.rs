//! The remote `TypeSafe` Backend selected by `jev-*` Model Identifiers.

use std::io::Read;
use std::thread;
use std::time::{Duration, SystemTime};

use serde_json::{Map, Value};
use ureq::Agent;

use crate::call::Call;
use crate::error::{Error, Result};
use crate::router::{Judged, Refusal};

pub(crate) const ENDPOINT: &str = "https://api.typesafe.ai/v1/systemone";
const MAX_ATTEMPTS: usize = 3;

/// Judge `call` through `TypeSafe` and return what it answered.
///
/// # Errors
///
/// Returns a `TypeSafe` runtime error when no usable credential is configured, when `TypeSafe`
/// cannot be reached, and when it answers 200 with anything but one complete response. An answer
/// that is not 200 comes back as a `Refusal` for the caller to report as its transport requires.
pub(crate) fn run_at(
    call: &Call,
    body: &[u8],
    api_key: Option<&str>,
    endpoint: &str,
) -> Result<Judged> {
    let api_key = api_key
        .filter(|key| !key.trim().is_empty())
        .ok_or_else(|| Error::TypeSafe {
            status: None,
            message: "set a non-empty TYPESAFE_API_KEY".to_string(),
        })?;
    let agent = Agent::config_builder()
        .max_redirects(0)
        .http_status_as_error(false)
        .timeout_resolve(Some(Duration::from_secs(10)))
        .timeout_connect(Some(Duration::from_secs(10)))
        .timeout_recv_response(Some(Duration::from_secs(60)))
        .build()
        .new_agent();

    for attempt in 0..MAX_ATTEMPTS {
        let mut response = agent
            .post(endpoint)
            .header("Authorization", &format!("Bearer {api_key}"))
            .header("Content-Type", "application/json")
            .send(body.to_vec())
            .map_err(|_| Error::TypeSafe {
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
        let mut response_body = String::new();
        response
            .body_mut()
            .as_reader()
            .read_to_string(&mut response_body)
            .map_err(|_| Error::TypeSafe {
                status: Some(status),
                message: "could not read the TypeSafe response".to_string(),
            })?;
        if status != 200 {
            let fallback = match status {
                401 | 403 => "check TYPESAFE_API_KEY",
                422 => "TypeSafe rejected the System One Call",
                429 => "rate limit remained after retries",
                529 => "service remained overloaded after retries",
                _ => "TypeSafe could not complete the request",
            };
            return Ok(Judged::Refusal(Refusal {
                status,
                message: api_error_message(&response_body, api_key, body)
                    .unwrap_or_else(|| fallback.to_string()),
                retry_after,
                body: response_body.into_bytes(),
            }));
        }
        let response: Value =
            serde_json::from_str(&response_body).map_err(|_| Error::TypeSafe {
                status: Some(status),
                message: "TypeSafe returned invalid JSON".to_string(),
            })?;
        return validate_response(call, response).map(Judged::Answer);
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

fn validate_response(call: &Call, response: Value) -> Result<Value> {
    let invalid = || Error::TypeSafe {
        status: Some(200),
        message: "TypeSafe returned an incomplete response".to_string(),
    };
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

    use super::retry_after_delay;

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
