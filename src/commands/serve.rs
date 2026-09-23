//! The `serve` command: one loaded local Backend and a localhost HTTP listener.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use clap::Args as ClapArgs;
use serde_json::{Value, json};

use crate::error::{Error, Result};
use crate::router::{Judged, ModelRouter, Models, Refusal};

const MAX_BODY: usize = 8 * 1024 * 1024;
const MAX_HEADERS: usize = 16 * 1024;

#[derive(ClapArgs)]
pub struct Args {
    /// Address to listen on
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    /// Port to listen on
    #[arg(long, default_value_t = 8080)]
    port: u16,
}

pub fn run(args: Args) -> Result<()> {
    let router = Arc::new(ModelRouter::for_server()?);
    let listener =
        TcpListener::bind((args.host.as_str(), args.port)).map_err(|error| Error::Inference {
            message: format!("binding {}:{}: {error}", args.host, args.port),
        })?;
    eprintln!(
        "s1gate: listening on {}",
        listener.local_addr().map_err(|error| Error::Inference {
            message: format!("reading listener address: {error}"),
        })?
    );
    for stream in listener.incoming() {
        let mut stream = stream.map_err(|error| Error::Inference {
            message: format!("accepting HTTP connection: {error}"),
        })?;
        stream
            .set_read_timeout(Some(Duration::from_secs(30)))
            .map_err(|error| Error::Inference {
                message: format!("setting HTTP read timeout: {error}"),
            })?;
        // One thread per connection: a remote Call waits for nothing but its own Backend, and local
        // Calls contend only with each other, for the loaded Checkpoint they share.
        let router = Arc::clone(&router);
        let serving = thread::Builder::new().spawn(move || {
            if let Err(error) = serve_connection(&mut stream, &router) {
                eprintln!("s1gate: HTTP connection: {error}");
            }
        });
        if let Err(error) = serving {
            eprintln!("s1gate: HTTP connection: {error}");
        }
    }
    Ok(())
}

fn serve_connection(stream: &mut TcpStream, router: &ModelRouter) -> std::io::Result<()> {
    let response = match read_request(stream) {
        Ok((method, path, _)) if method == "GET" && path == "/v1/models" => list_models(router),
        Ok((method, path, body)) if method == "POST" && path == "/v1/systemone" => {
            judge_call(router, &body)
        }
        Ok((_, path, _)) if path != "/v1/systemone" && path != "/v1/models" => {
            Response::error(404, "not found")
        }
        Ok(_) => Response::error(405, "method not allowed"),
        Err(message) => Response::error(400, &message),
    };
    write_response(stream, &response)
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
        Err(error) if error.exit_code() == 2 => Response::error(422, &error.to_string()),
        Err(Error::MissingCheckpoint { name }) => Response::error(
            503,
            &format!("missing Checkpoint `{name}`; run s1gate pull {name}"),
        ),
        // The remote Backend never answered with a response of its own to forward.
        Err(error @ Error::TypeSafe { .. }) => Response::error(502, &error.to_string()),
        Err(error) => Response::error(500, &error.to_string()),
    }
}

/// One response the server writes back.
struct Response {
    status: u16,
    /// The remote Backend's own `Retry-After`, when it sent one.
    retry_after: Option<String>,
    body: Vec<u8>,
}

impl Response {
    /// This server's own answer, as a `TypeSafe`-shaped `detail` body.
    fn error(status: u16, message: &str) -> Self {
        Self::json(status, detail(message))
    }

    fn json(status: u16, value: Value) -> Self {
        Self {
            status,
            retry_after: None,
            body: serde_json::to_vec(&value).expect("JSON Value serializes"),
        }
    }

    /// The remote Backend's answer: its status, its body, and its retry delay as it sent them.
    fn forward(refusal: Refusal) -> Self {
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

fn write_response(stream: &mut TcpStream, response: &Response) -> std::io::Result<()> {
    let retry_after = response
        .retry_after
        .as_ref()
        .map_or(String::new(), |value| format!("Retry-After: {value}\r\n"));
    write!(
        stream,
        "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{retry_after}Connection: close\r\n\r\n",
        response.status,
        reason(response.status),
        response.body.len()
    )?;
    stream.write_all(&response.body)
}

/// The reason phrase of `status`, which a client may ignore either way.
fn reason(status: u16) -> &'static str {
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

fn read_request(stream: &mut TcpStream) -> std::result::Result<(String, String, Vec<u8>), String> {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    let mut remaining = MAX_HEADERS;
    remaining -= read_line(&mut reader, &mut line, remaining)?;
    let mut parts = line.split_whitespace();
    let method = parts.next().ok_or("missing HTTP method")?.to_string();
    let path = parts.next().ok_or("missing HTTP path")?.to_string();
    if !matches!(parts.next(), Some("HTTP/1.1" | "HTTP/1.0")) || parts.next().is_some() {
        return Err("unsupported HTTP version".to_string());
    }
    let mut content_length = None;
    loop {
        remaining -= read_line(&mut reader, &mut line, remaining)?;
        if line == "\r\n" || line == "\n" {
            break;
        }
        if line.is_empty() {
            return Err("incomplete HTTP headers".to_string());
        }
        if let Some((name, _)) = line.split_once(':')
            && name.eq_ignore_ascii_case("transfer-encoding")
        {
            return Err("Transfer-Encoding is unsupported".to_string());
        }
        if let Some((name, value)) = line.split_once(':')
            && name.eq_ignore_ascii_case("content-length")
        {
            if content_length.is_some() {
                return Err("duplicate Content-Length".to_string());
            }
            content_length = Some(
                value
                    .trim()
                    .parse::<usize>()
                    .map_err(|_| "invalid Content-Length")?,
            );
        }
    }
    let length = content_length
        .or_else(|| (method == "GET").then_some(0))
        .ok_or("missing Content-Length")?;
    if length > MAX_BODY {
        return Err("HTTP body is too large".to_string());
    }
    let mut body = vec![0; length];
    reader
        .read_exact(&mut body)
        .map_err(|error| error.to_string())?;
    Ok((method, path, body))
}

fn read_line(
    reader: &mut impl BufRead,
    line: &mut String,
    remaining: usize,
) -> std::result::Result<usize, String> {
    line.clear();
    let count = reader
        .take(remaining as u64 + 1)
        .read_line(line)
        .map_err(|error| error.to_string())?;
    if count > remaining {
        return Err("HTTP headers are too large".to_string());
    }
    if count > 0 && !line.ends_with('\n') {
        return Err("incomplete HTTP headers".to_string());
    }
    Ok(count)
}
