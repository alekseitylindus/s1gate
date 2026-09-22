//! The `infer` command: judge one System One Call against a local or remote Backend.

use std::ffi::OsString;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use crate::call::Call;
use crate::error::{Error, Result};
use crate::laya;
use crate::model_source;
use crate::store::Store;

pub fn run() -> Result<()> {
    run_with_settings(
        io::stdin().lock(),
        io::stdout().lock(),
        crate::typesafe::ENDPOINT,
        std::env::var_os("TYPESAFE_API_KEY"),
        std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from),
        std::env::var_os("HOME").map(PathBuf::from),
    )
}

#[cfg(test)]
fn run_with_endpoint(
    input: impl Read,
    stdout: impl Write,
    endpoint: &str,
    api_key: Option<&str>,
) -> Result<()> {
    run_with_settings(
        input,
        stdout,
        endpoint,
        api_key.map(OsString::from),
        None,
        None,
    )
}

fn run_with_settings(
    mut input: impl Read,
    mut stdout: impl Write,
    endpoint: &str,
    env_api_key: Option<OsString>,
    xdg_config_home: Option<PathBuf>,
    home: Option<PathBuf>,
) -> Result<()> {
    let (call, body) = read_call(&mut input)?;
    let result = if call.model == "jev-latest" {
        let api_key = typesafe_api_key(env_api_key, xdg_config_home.as_deref(), home.as_deref())?;
        crate::typesafe::run_at(&call, &body, Some(&api_key), endpoint)?
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

fn typesafe_api_key(
    env_api_key: Option<OsString>,
    xdg_config_home: Option<&Path>,
    home: Option<&Path>,
) -> Result<String> {
    if let Some(key) = env_api_key {
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
        return Ok(key.to_string());
    }

    let config_home = xdg_config_home
        .filter(|path| !path.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .or_else(|| home.map(|path| path.join(".config")))
        .ok_or_else(|| Error::TypeSafe {
            status: None,
            message: "cannot locate configuration; set TYPESAFE_API_KEY or HOME".to_string(),
        })?;
    let path = config_home.join("s1gate/config.toml");
    let contents = std::fs::read_to_string(&path).map_err(|error| Error::TypeSafe {
        status: None,
        message: format!(
            "cannot read {} ({error}); set TYPESAFE_API_KEY or add typesafe.api_key to this file",
            path.display()
        ),
    })?;
    let config: toml::Value = toml::from_str(&contents).map_err(|_| Error::TypeSafe {
        status: None,
        message: format!(
            "{} is not valid TOML; fix it or set TYPESAFE_API_KEY",
            path.display()
        ),
    })?;
    let api_key = config
        .get("typesafe")
        .and_then(|typesafe| typesafe.get("api_key"))
        .and_then(toml::Value::as_str)
        .filter(|key| !key.trim().is_empty())
        .ok_or_else(|| Error::TypeSafe {
            status: None,
            message: format!(
                "{} must contain a non-empty string at typesafe.api_key; set TYPESAFE_API_KEY or fix the file",
                path.display()
            ),
        })?;
    Ok(api_key.to_string())
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
    use std::ffi::OsString;
    use std::fs;
    use std::io::Cursor;
    use std::net::TcpListener;
    use std::path::{Path, PathBuf};
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
    fn jev_reads_the_default_and_xdg_config_locations() {
        for (xdg_name, home_name, key) in [
            (None, "default-home", "default-key"),
            (Some("xdg-home"), "unused-home", "xdg-key"),
        ] {
            let root = temp_dir("config-location");
            let config_home = xdg_name
                .map(|name| root.join(name))
                .unwrap_or_else(|| root.join(home_name).join(".config"));
            write_config(&config_home, &format!("api_key = \"{key}\""));

            let request = run_jev_with_config(
                None,
                xdg_name.map(|name| root.join(name)),
                Some(root.join(home_name)),
            );
            assert!(
                request
                    .to_ascii_lowercase()
                    .contains(&format!("authorization: bearer {key}\r\n"))
            );
            fs::remove_dir_all(root).expect("remove test config");
        }
    }

    #[test]
    fn environment_key_overrides_config_and_empty_environment_key_does_not_fall_back() {
        let root = temp_dir("config-precedence");
        let config_home = root.join("config");
        write_config(&config_home, "this is malformed TOML");

        let request = run_jev_with_config(
            Some(OsString::from("environment-key")),
            Some(config_home.clone()),
            None,
        );
        assert!(
            request
                .to_ascii_lowercase()
                .contains("authorization: bearer environment-key\r\n")
        );

        let (endpoint, listener) = local_endpoint();
        let mut stdout = Vec::new();
        let error = run_with_settings(
            Cursor::new(valid_jev_call()),
            &mut stdout,
            &endpoint,
            Some(OsString::from("  ")),
            Some(config_home),
            None,
        )
        .expect_err("an explicitly empty environment key fails");
        assert!(error.to_string().contains("TYPESAFE_API_KEY is empty"));
        assert!(stdout.is_empty());
        assert_no_request(listener);
        fs::remove_dir_all(root).expect("remove test config");
    }

    #[test]
    fn unusable_config_fails_before_contacting_typesafe() {
        let root = temp_dir("config-errors");
        let cases = [
            ("missing", None, "cannot read"),
            ("malformed", Some("not = [ TOML"), "not valid TOML"),
            (
                "missing-key",
                Some("[other]\nvalue = \"x\""),
                "typesafe.api_key",
            ),
        ];
        for (name, contents, diagnostic) in cases {
            let config_home = root.join(name);
            if let Some(contents) = contents {
                write_config(&config_home, contents);
            }
            let (endpoint, listener) = local_endpoint();
            let mut stdout = Vec::new();
            let error = run_with_settings(
                Cursor::new(valid_jev_call()),
                &mut stdout,
                &endpoint,
                None,
                Some(config_home),
                None,
            )
            .expect_err("the missing or invalid config prevents a Jev call");
            assert!(error.to_string().contains(diagnostic));
            assert!(stdout.is_empty());
            assert_no_request(listener);
        }

        let unreadable_config_home = root.join("unreadable");
        fs::create_dir_all(unreadable_config_home.join("s1gate/config.toml"))
            .expect("create directory where config file should be");
        let (endpoint, listener) = local_endpoint();
        let mut stdout = Vec::new();
        let error = run_with_settings(
            Cursor::new(valid_jev_call()),
            &mut stdout,
            &endpoint,
            None,
            Some(unreadable_config_home),
            None,
        )
        .expect_err("a directory cannot be read as config");
        assert!(error.to_string().contains("cannot read"));
        assert!(stdout.is_empty());
        assert_no_request(listener);
        fs::remove_dir_all(root).expect("remove test config");
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

    fn run_jev_with_config(
        env_api_key: Option<OsString>,
        xdg_config_home: Option<PathBuf>,
        home: Option<PathBuf>,
    ) -> String {
        let (endpoint, listener) = local_endpoint();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("the infer request");
            let request = read_http_request(&mut stream);
            write_response(
                &mut stream,
                "200 OK",
                r#"{"model":"jev-1.13.0","answers":{"risk":{"type":"noul","noul":0.9}},"usage":{"input_tokens":1,"output_tokens":1}}"#,
            );
            request
        });
        let mut stdout = Vec::new();
        run_with_settings(
            Cursor::new(valid_jev_call()),
            &mut stdout,
            &endpoint,
            env_api_key,
            xdg_config_home,
            home,
        )
        .expect("Jev inference succeeds");
        serde_json::from_slice::<Value>(&stdout).expect("success JSON");
        server.join().expect("the server thread")
    }

    fn valid_jev_call() -> &'static str {
        r#"{"model":"jev-latest","state":"x","questions":{"risk":{"type":"noul","instructions":"risky?"}}}"#
    }

    fn write_config(config_home: &Path, body: &str) {
        let directory = config_home.join("s1gate");
        fs::create_dir_all(&directory).expect("create config directory");
        fs::write(
            directory.join("config.toml"),
            format!("[typesafe]\n{body}\n"),
        )
        .expect("write config");
    }

    fn local_endpoint() -> (String, TcpListener) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("a local endpoint");
        let endpoint = format!(
            "http://{}/v1/systemone",
            listener.local_addr().expect("address")
        );
        (endpoint, listener)
    }

    fn assert_no_request(listener: TcpListener) {
        listener
            .set_nonblocking(true)
            .expect("nonblocking listener");
        assert!(listener.accept().is_err(), "no remote request was made");
    }

    fn temp_dir(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "s1gate-infer-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock after epoch")
                .as_nanos()
        ));
        fs::create_dir_all(&root).expect("create temporary directory");
        root
    }
}
