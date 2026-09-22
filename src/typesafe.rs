//! The remote `TypeSafe` Backend selected by `jev-latest`.

use std::io::Read;
use std::thread;
use std::time::Duration;

use serde_json::{Map, Value};
use ureq::Agent;

use crate::call::Call;
use crate::error::{Error, Result};

pub(crate) const ENDPOINT: &str = "https://api.typesafe.ai/v1/systemone";
const MAX_ATTEMPTS: usize = 3;

/// Judge `call` through `TypeSafe` and return its validated response.
///
/// # Errors
///
/// Returns a `TypeSafe` runtime error for missing credentials, transport or service failures, and
/// malformed responses. HTTP 422 is a usage error.
pub(crate) fn run_at(
    call: &Call,
    body: &[u8],
    api_key: Option<&str>,
    endpoint: &str,
) -> Result<Value> {
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
        if (status == 429 || status == 529) && attempt + 1 < MAX_ATTEMPTS {
            let retry_after = response
                .headers()
                .get("retry-after")
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.parse::<u64>().ok())
                .map(Duration::from_secs);
            let delay = retry_after
                .unwrap_or_else(|| Duration::from_millis(200 * (1 << attempt)))
                .min(Duration::from_secs(2));
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
            let message = match status {
                401 | 403 => "check TYPESAFE_API_KEY",
                422 => "TypeSafe rejected the System One Call",
                429 => "rate limit remained after retries",
                529 => "service remained overloaded after retries",
                _ => "TypeSafe could not complete the request",
            };
            return Err(Error::TypeSafe {
                status: Some(status),
                message: message.to_string(),
            });
        }
        let response: Value =
            serde_json::from_str(&response_body).map_err(|_| Error::TypeSafe {
                status: Some(status),
                message: "TypeSafe returned invalid JSON".to_string(),
            })?;
        return validate_response(call, response);
    }
    unreachable!("the bounded retry loop always returns or continues to its last attempt")
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
