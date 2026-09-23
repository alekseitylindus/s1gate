//! The `serve` command: one loaded local Backend and a localhost HTTP listener.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Duration;

use clap::Args as ClapArgs;
use serde_json::{Value, json};

use crate::error::{Error, Result};
use crate::router::ModelRouter;

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
    let router = ModelRouter::for_server()?;
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
        if let Err(error) = serve_connection(&mut stream, &router) {
            eprintln!("s1gate: HTTP connection: {error}");
        }
    }
    Ok(())
}

fn serve_connection(stream: &mut TcpStream, router: &ModelRouter) -> std::io::Result<()> {
    let request = read_request(stream);
    let (status, body) = match request {
        Ok((method, path, body)) if method == "POST" && path == "/v1/systemone" => {
            match router.route(&body) {
                Ok(value) => (200, value),
                Err(error) if error.exit_code() == 2 => (422, detail(&error.to_string())),
                Err(Error::MissingCheckpoint { name }) => (
                    503,
                    detail(&format!(
                        "missing Checkpoint `{name}`; run s1gate pull {name}"
                    )),
                ),
                Err(error) => (500, detail(&error.to_string())),
            }
        }
        Ok((_, path, _)) if path != "/v1/systemone" => (404, detail("not found")),
        Ok(_) => (405, detail("method not allowed")),
        Err(message) => (400, detail(&message)),
    };
    let body = serde_json::to_vec(&body).expect("JSON Value serializes");
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        422 => "Unprocessable Entity",
        503 => "Service Unavailable",
        _ => "Internal Server Error",
    };
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(&body)
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
    let length = content_length.ok_or("missing Content-Length")?;
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
