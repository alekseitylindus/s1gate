//! The `infer` command: judge one System One Call against a local or remote Backend.

use std::io::{self, Read, Write};

use crate::call::Call;
use crate::error::{Error, Result};
use crate::laya;
use crate::model_source;
use crate::store::Store;

pub fn run() -> Result<()> {
    run_with_endpoint(
        io::stdin().lock(),
        io::stdout().lock(),
        crate::typesafe::ENDPOINT,
        None,
    )
}

fn run_with_endpoint(
    mut input: impl Read,
    mut stdout: impl Write,
    endpoint: &str,
    api_key: Option<&str>,
) -> Result<()> {
    let (call, body) = read_call(&mut input)?;
    let result = if call.model == "jev-latest" {
        let api_key = api_key
            .map(str::to_string)
            .or_else(|| std::env::var("TYPESAFE_API_KEY").ok());
        crate::typesafe::run_at(&call, &body, api_key.as_deref(), endpoint)?
    } else {
        let source = model_source::lookup_identifier(&call.model)?;
        let name = source.repo;
        let store = Store::from_env()?;
        if store.provenance(name)?.is_none() {
            return Err(Error::MissingCheckpoint {
                name: name.to_string(),
            });
        }
        laya::run(&store, source, &call)?
    };
    serde_json::to_writer(&mut stdout, &result).map_err(|error| Error::Inference {
        message: format!("writing result: {error}"),
    })?;
    stdout
        .write_all(b"\n")
        .map_err(|error| Error::io("write", "<stdout>", error))?;
    Ok(())
}

/// Read the System One Call from stdin, the only place `infer` takes input: a reader that fails is
/// this command's failure, while bytes that are not a Call are the caller's.
fn read_call(input: &mut impl Read) -> Result<(Call, Vec<u8>)> {
    let mut bytes = Vec::new();
    input
        .read_to_end(&mut bytes)
        .map_err(|error| Error::io("read", "<stdin>", error))?;
    let call = Call::from_bytes(&bytes)?;
    Ok((call, bytes))
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::net::TcpListener;
    use std::thread;

    use serde_json::Value;

    use super::*;

    #[test]
    fn jev_posts_the_validated_call_and_returns_the_remote_response() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("a local endpoint");
        let endpoint = format!(
            "http://{}/v1/systemone",
            listener.local_addr().expect("address")
        );
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("the infer request");
            let request = read_http_request(&mut stream);
            let response = r#"{"model":"jev-1.13.0","answers":{"route":{"type":"choice","choice":"z","probabilities":{"z":1.0,"a":0.0},"confidence":1.0},"urgency":{"type":"score","score":1.0,"legend":{"0":"low","1":"high"},"probabilities":{"0":0.0,"1":1.0},"confidence":1.0},"risk":{"type":"noul","noul":0.95}},"usage":{"input_tokens":12,"output_tokens":3}}"#;
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response.len(),
                response
            )
            .expect("send the TypeSafe response");
            request
        });

        let call = r#"{"model":"jev-latest","state":{"message":"evidence","meta":[1,true]},"questions":{"route":{"type":"choice","instructions":{"prompt":"where?"},"criteria":{"z":{"desc":"last"},"a":null}},"urgency":{"type":"score","instructions":["how urgent?"],"criteria":["low",{"level":"high"}]},"risk":{"type":"noul","instructions":"is it risky?","criteria":{"true":{"reason":"yes"}}}}}"#;
        let mut stdout = Vec::new();
        run_with_endpoint(Cursor::new(call), &mut stdout, &endpoint, Some("test-key"))
            .expect("Jev inference succeeds without a local Model Store");

        let request = server.join().expect("the server thread");
        assert!(
            request.starts_with("POST /v1/systemone HTTP/1.1\r\n"),
            "{request}"
        );
        assert!(
            request
                .to_ascii_lowercase()
                .contains("authorization: bearer test-key\r\n")
        );
        assert!(
            request
                .to_ascii_lowercase()
                .contains("content-type: application/json\r\n")
        );
        let request_body = request.split_once("\r\n\r\n").expect("HTTP body").1;
        let request_json: Value = serde_json::from_str(request_body).expect("JSON request");
        let input_json: Value = serde_json::from_str(call).expect("JSON call");
        assert_eq!(request_json, input_json);
        assert!(
            request_body.find("\"z\"").expect("z Option")
                < request_body.find("\"a\"").expect("a Option"),
            "the caller's Criteria order is retained"
        );
        assert_eq!(
            request_json["questions"]["route"]["criteria"]["z"]["desc"],
            "last"
        );
        let response: Value = serde_json::from_slice(&stdout).expect("stdout JSON");
        let mut response_keys: Vec<_> = response
            .as_object()
            .expect("a response object")
            .keys()
            .map(String::as_str)
            .collect();
        response_keys.sort_unstable();
        assert_eq!(response_keys, ["answers", "model", "usage"]);
        assert_eq!(response["model"], "jev-1.13.0");
        assert_eq!(response["answers"]["route"]["choice"], "z");
        assert_eq!(response["answers"]["urgency"]["type"], "score");
        assert_eq!(response["answers"]["risk"]["noul"], 0.95);
        assert_eq!(response["usage"]["input_tokens"], 12);
        assert_eq!(response["usage"]["output_tokens"], 3);
        assert!(response.get("transcript").is_none());
    }

    #[test]
    fn invalid_calls_are_rejected_before_the_remote_endpoint_is_contacted() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("a local endpoint");
        listener
            .set_nonblocking(true)
            .expect("nonblocking listener");
        let endpoint = format!(
            "http://{}/v1/systemone",
            listener.local_addr().expect("address")
        );
        let mut stdout = Vec::new();

        let error = run_with_endpoint(
            Cursor::new(r#"{"model":"jev-latest","state":3,"questions":{}}"#),
            &mut stdout,
            &endpoint,
            Some("test-key"),
        )
        .expect_err("the invalid call is rejected");

        assert_eq!(error.exit_code(), 2);
        assert!(stdout.is_empty());
        assert!(listener.accept().is_err(), "no remote request was made");
    }

    #[test]
    fn an_empty_api_key_fails_without_contacting_the_remote_endpoint() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("a local endpoint");
        listener
            .set_nonblocking(true)
            .expect("nonblocking listener");
        let endpoint = format!(
            "http://{}/v1/systemone",
            listener.local_addr().expect("address")
        );
        let mut stdout = Vec::new();

        let error = run_with_endpoint(
            Cursor::new(r#"{"model":"jev-latest","state":"x","questions":{"risk":{"type":"noul","instructions":"risky?"}}}"#),
            &mut stdout,
            &endpoint,
            Some(" "),
        )
        .expect_err("empty credentials are unusable");

        assert_eq!(error.exit_code(), 1);
        assert!(error.to_string().contains("TYPESAFE_API_KEY"));
        assert!(stdout.is_empty());
        assert!(listener.accept().is_err(), "no remote request was made");
    }

    #[test]
    fn unsupported_model_identifiers_are_rejected_before_the_remote_endpoint_is_contacted() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("a local endpoint");
        listener
            .set_nonblocking(true)
            .expect("nonblocking listener");
        let endpoint = format!(
            "http://{}/v1/systemone",
            listener.local_addr().expect("address")
        );
        let mut stdout = Vec::new();

        let error = run_with_endpoint(
            Cursor::new(r#"{"model":"jev-preview","state":"x","questions":{"risk":{"type":"noul","instructions":"risky?"}}}"#),
            &mut stdout,
            &endpoint,
            Some("test-key"),
        )
        .expect_err("unsupported model identifiers stay rejected");

        assert_eq!(error.exit_code(), 2);
        assert!(error.to_string().contains("jev-preview"));
        assert!(stdout.is_empty());
        assert!(listener.accept().is_err(), "no remote request was made");
    }

    #[test]
    fn a_typesafe_422_is_a_usage_error_without_success_json() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("a local endpoint");
        let endpoint = format!(
            "http://{}/v1/systemone",
            listener.local_addr().expect("address")
        );
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("the infer request");
            read_http_request(&mut stream);
            write_response(
                &mut stream,
                "422 Unprocessable Entity",
                "{\"message\":\"bad call\"}",
            );
        });
        let mut stdout = Vec::new();

        let error = run_with_endpoint(
            Cursor::new(r#"{"model":"jev-latest","state":"x","questions":{"risk":{"type":"noul","instructions":"risky?"}}}"#),
            &mut stdout,
            &endpoint,
            Some("test-key"),
        )
        .expect_err("TypeSafe rejected the call");

        server.join().expect("the server thread");
        assert_eq!(error.exit_code(), 2);
        assert!(error.to_string().contains("HTTP 422"));
        assert!(stdout.is_empty());
    }

    #[test]
    fn incomplete_remote_responses_are_runtime_errors_without_success_json() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("a local endpoint");
        let endpoint = format!(
            "http://{}/v1/systemone",
            listener.local_addr().expect("address")
        );
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("the infer request");
            read_http_request(&mut stream);
            write_response(
                &mut stream,
                "200 OK",
                r#"{"model":"jev-1.13.0","answers":{"risk":{"type":"noul","noul":0.9}}}"#,
            );
        });
        let mut stdout = Vec::new();

        let error = run_with_endpoint(
            Cursor::new(r#"{"model":"jev-latest","state":"x","questions":{"risk":{"type":"noul","instructions":"risky?"}}}"#),
            &mut stdout,
            &endpoint,
            Some("test-key"),
        )
        .expect_err("usage is required in the response");

        server.join().expect("the server thread");
        assert_eq!(error.exit_code(), 1);
        assert!(error.to_string().contains("incomplete response"));
        assert!(stdout.is_empty());
    }

    #[test]
    fn rate_limits_are_retried_with_a_finite_attempt_count() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("a local endpoint");
        let endpoint = format!(
            "http://{}/v1/systemone",
            listener.local_addr().expect("address")
        );
        let server = thread::spawn(move || {
            for _ in 0..3 {
                let (mut stream, _) = listener.accept().expect("a retry");
                read_http_request(&mut stream);
                write_response(
                    &mut stream,
                    "429 Too Many Requests",
                    "{\"message\":\"please retry\"}",
                );
            }
        });
        let mut stdout = Vec::new();

        let error = run_with_endpoint(
            Cursor::new(r#"{"model":"jev-latest","state":"x","questions":{"risk":{"type":"noul","instructions":"risky?"}}}"#),
            &mut stdout,
            &endpoint,
            Some("test-key"),
        )
        .expect_err("the rate limit remains after retries");

        server.join().expect("the server thread");
        assert_eq!(error.exit_code(), 1);
        assert!(error.to_string().contains("HTTP 429"));
        assert!(stdout.is_empty());
    }

    fn read_http_request(stream: &mut std::net::TcpStream) -> String {
        use std::io::Read;

        let mut bytes = Vec::new();
        let mut buffer = [0; 4096];
        loop {
            let count = stream.read(&mut buffer).expect("read HTTP request");
            if count == 0 {
                break;
            }
            bytes.extend_from_slice(&buffer[..count]);
            if let Some(header_end) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
                let header = String::from_utf8_lossy(&bytes[..header_end]);
                let length = header
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().expect("content length"))
                    })
                    .expect("Content-Length");
                if bytes.len() >= header_end + 4 + length {
                    break;
                }
            }
        }
        String::from_utf8(bytes).expect("UTF-8 HTTP request")
    }

    fn write_response(stream: &mut std::net::TcpStream, status: &str, body: &str) {
        use std::io::Write;

        write!(
            stream,
            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\nRetry-After: 0\r\n\r\n{body}",
            body.len()
        )
        .expect("send the TypeSafe response");
    }
}
