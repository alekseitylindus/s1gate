//! The `serve` command: one loaded local Backend and a localhost HTTP listener.
//!
//! The listener is the transport: it reads a request off a socket, asks the HTTP core what to
//! answer, and writes it back. What is answered lives in [`crate::http`].

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use clap::Args as ClapArgs;

use crate::error::{Error, Result};
use crate::http::{self, MAX_BODY, MAX_HEADERS, Request, Response};
use crate::router::ModelRouter;

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

/// Read one request off `stream`, answer it, and write the answer back.
fn serve_connection(stream: &mut TcpStream, router: &ModelRouter) -> std::io::Result<()> {
    let response = match read_request(stream) {
        Ok(request) => http::handle(router, request),
        // A request s1gate cannot read is the caller's, and is answered rather than dropped.
        Err(message) => Response::error(400, &message),
    };
    write_response(stream, &response)
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
        http::reason(response.status),
        response.body.len()
    )?;
    stream.write_all(&response.body)
}

fn read_request(stream: &mut TcpStream) -> std::result::Result<Request, String> {
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
    Ok(Request { method, path, body })
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
