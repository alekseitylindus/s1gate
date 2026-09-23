//! HTTP acceptance tests exercise the running binary and a temporary Model Store.

mod support;

use std::io::{BufRead, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::Value;
use support::{TempDir, fixture};

const CALL: &str = r#"{"model":"convaiinnovations/laya","state":"a customer was charged twice","questions":{"risk":{"type":"noul","instructions":"Is this risky?"},"route":{"type":"choice","instructions":"Which team?","criteria":{"billing":"Billing","support":"Support"}},"urgency":{"type":"score","instructions":"How urgent?","criteria":["low","high"]}}}"#;

struct Server {
    child: Child,
    port: u16,
}

impl Server {
    fn start(data_home: &TempDir) -> Self {
        Self::start_with_config(data_home, None)
    }

    fn start_with_config(data_home: &TempDir, api_key: Option<&str>) -> Self {
        let port = free_port();
        let mut command = Command::new(env!("CARGO_BIN_EXE_s1gate"));
        command
            .args(["serve", "--port", &port.to_string()])
            .env("XDG_DATA_HOME", data_home.path())
            .env("XDG_CONFIG_HOME", data_home.path())
            .env_remove("TYPESAFE_API_KEY")
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        if let Some(api_key) = api_key {
            command.env("TYPESAFE_API_KEY", api_key);
        }
        let mut child = command.spawn().expect("start s1gate serve");
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if TcpStream::connect(("127.0.0.1", port)).is_ok() {
                return Self { child, port };
            }
            if let Some(status) = child.try_wait().expect("check server") {
                panic!("server exited at startup: {status}");
            }
            assert!(Instant::now() < deadline, "server did not start");
            thread::sleep(Duration::from_millis(20));
        }
    }

    fn post(&self, body: &str, authorization: Option<&str>) -> (u16, Value) {
        post(self.port, body, authorization)
    }
}

fn post(port: u16, body: &str, authorization: Option<&str>) -> (u16, Value) {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect to server");
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .expect("read timeout");
    let header = authorization
        .map(|value| format!("Authorization: {value}\r\n"))
        .unwrap_or_default();
    write!(
            stream,
            "POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n{header}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
        .expect("send call");
    let mut response = String::new();
    stream.read_to_string(&mut response).expect("read response");
    let (head, body) = response.split_once("\r\n\r\n").expect("HTTP response");
    let status = head
        .split_whitespace()
        .nth(1)
        .expect("status")
        .parse()
        .expect("numeric status");
    assert!(
        head.to_ascii_lowercase()
            .contains("content-type: application/json")
    );
    (status, serde_json::from_str(body).expect("JSON response"))
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .expect("free port")
        .local_addr()
        .expect("address")
        .port()
}

#[test]
fn present_checkpoint_judges_all_three_question_types_over_http() {
    let data_home = TempDir::new("serve-valid");
    fixture::write_inferable(&data_home.path().join("s1gate/models"));
    let server = Server::start(&data_home);

    let (status, body) = server.post(CALL, None);
    assert_eq!(status, 200);
    assert_eq!(body["model"], fixture::NAME);
    assert_eq!(body["answers"]["risk"]["type"], "noul");
    assert_eq!(body["answers"]["route"]["type"], "choice");
    assert_eq!(body["answers"]["urgency"]["type"], "score");
    assert!(body["usage"]["input_tokens"].is_number());
}

#[test]
fn empty_store_starts_and_invalid_calls_get_typesafe_detail() {
    let data_home = TempDir::new("serve-empty");
    let server = Server::start(&data_home);
    let (status, body) = server.post("{}", Some("Bearer caller-token"));
    assert_eq!(status, 422);
    assert!(body["detail"].is_array(), "{body}");

    let unknown = CALL.replace(fixture::NAME, "someone/else");
    let (status, body) = server.post(&unknown, None);
    assert_eq!(status, 422);
    assert!(body["detail"].is_array(), "{body}");
}

#[test]
fn a_loaded_checkpoint_is_reused_after_its_files_are_removed() {
    let data_home = TempDir::new("serve-reuse");
    let checkpoint = fixture::write_inferable(&data_home.path().join("s1gate/models"));
    let server = Server::start(&data_home);
    let first = server.post(CALL, None);
    assert_eq!(first.0, 200);

    std::fs::remove_dir_all(checkpoint).expect("remove on-disk Checkpoint");
    let second = server.post(CALL, None);
    assert_eq!(second, first);
}

#[test]
fn invalid_present_checkpoint_prevents_startup() {
    let data_home = TempDir::new("serve-invalid-checkpoint");
    let checkpoint = fixture::write_inferable(&data_home.path().join("s1gate/models"));
    std::fs::write(checkpoint.join("encoder/config.json"), "not JSON").expect("corrupt Checkpoint");
    let output = Command::new(env!("CARGO_BIN_EXE_s1gate"))
        .arg("serve")
        .env("XDG_DATA_HOME", data_home.path())
        .output()
        .expect("run server");
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("encoder/config.json"));
}

#[test]
fn binding_an_occupied_port_fails_visibly() {
    let occupied = TcpListener::bind("127.0.0.1:0").expect("occupied port");
    let port = occupied.local_addr().expect("address").port();
    let data_home = TempDir::new("serve-bind-error");
    let output = Command::new(env!("CARGO_BIN_EXE_s1gate"))
        .args(["serve", "--host", "127.0.0.1", "--port", &port.to_string()])
        .env("XDG_DATA_HOME", data_home.path())
        .output()
        .expect("run server");
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("binding 127.0.0.1"));
}

#[test]
fn concurrent_calls_all_receive_answers() {
    let data_home = TempDir::new("serve-concurrent");
    fixture::write_inferable(&data_home.path().join("s1gate/models"));
    let server = Server::start(&data_home);
    let barrier = Arc::new(Barrier::new(4));
    let calls: Vec<_> = (0..4)
        .map(|_| {
            let barrier = Arc::clone(&barrier);
            let port = server.port;
            thread::spawn(move || {
                barrier.wait();
                post(port, CALL, None)
            })
        })
        .collect();
    for call in calls {
        let (status, body) = call.join().expect("call thread");
        assert_eq!(status, 200, "{body}");
        assert_eq!(body["model"], fixture::NAME);
    }
}

#[test]
fn incoming_authorization_does_not_replace_the_configured_typesafe_key() {
    let upstream = TcpListener::bind("127.0.0.1:0").expect("upstream port");
    let endpoint = format!(
        "http://{}/v1/systemone",
        upstream.local_addr().expect("address")
    );
    let upstream = thread::spawn(move || {
        let mut headers = Vec::new();
        for _ in 0..2 {
            let (mut stream, _) = upstream.accept().expect("Jev call");
            let mut reader = std::io::BufReader::new(&mut stream);
            let mut line = String::new();
            loop {
                line.clear();
                reader.read_line(&mut line).expect("request header");
                if line == "\r\n" {
                    break;
                }
                if line.to_ascii_lowercase().starts_with("authorization:") {
                    headers.push(line.trim().to_string());
                }
            }
            let body = r#"{"model":"jev-1.0","answers":{"risk":{"type":"noul","noul":0.5}},"usage":{"input_tokens":3,"output_tokens":1}}"#;
            write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).expect("upstream response");
        }
        headers
    });
    let data_home = TempDir::new("serve-typesafe-auth");
    let config_dir = data_home.path().join("s1gate");
    std::fs::create_dir_all(&config_dir).expect("config directory");
    std::fs::write(
        config_dir.join("config.toml"),
        format!("[typesafe]\nendpoint = \"{endpoint}\"\n"),
    )
    .expect("endpoint config");
    let server = Server::start_with_config(&data_home, Some("configured-secret"));
    let call = r#"{"model":"jev-latest","state":"evidence","questions":{"risk":{"type":"noul"}}}"#;
    assert_eq!(server.post(call, None).0, 200);
    assert_eq!(server.post(call, Some("Bearer caller-secret")).0, 200);
    assert_eq!(
        upstream.join().expect("upstream thread"),
        [
            "authorization: Bearer configured-secret",
            "authorization: Bearer configured-secret",
        ]
    );
}
