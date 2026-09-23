//! HTTP acceptance tests exercise the running binary and a temporary Model Store.

mod support;

use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::Value;
use support::{TempDir, fixture};

const CALL: &str = r#"{"model":"convaiinnovations/laya","state":"a customer was charged twice","questions":{"risk":{"type":"noul","instructions":"Is this risky?"},"route":{"type":"choice","instructions":"Which team?","criteria":{"billing":"Billing","support":"Support"}},"urgency":{"type":"score","instructions":"How urgent?","criteria":["low","high"]}}}"#;

/// A remote System One Call, written with the alias a caller sends.
const JEV_CALL: &str =
    r#"{"model":"jev-latest","state":"evidence","questions":{"risk":{"type":"noul"}}}"#;

/// The versioned Model Identifier `TypeSafe` resolves that alias to.
const JEV_MODEL: &str = "jev-1.13.0";

/// What `TypeSafe` answers a judged Call with.
const JEV_ANSWER: &str = r#"{"model":"jev-1.13.0","answers":{"risk":{"type":"noul","noul":0.75}},"usage":{"input_tokens":11,"output_tokens":2}}"#;

/// What `TypeSafe` answers a model-list request with: the aliases it serves, each with the
/// description and release date it publishes for it.
const JEV_MODELS: &str = r#"{"models":[{"name":"jev-latest","description":"The most recent stable release.","release_date":"2025-11-04"},{"name":"jev-preview","description":"The most recent release, preview included.","release_date":"2026-01-15"}]}"#;

struct Server {
    child: Child,
    port: u16,
}

impl Server {
    fn start(data_home: &TempDir) -> Self {
        Self::start_with_config(data_home, None)
    }

    fn start_with_config(data_home: &TempDir, api_key: Option<&str>) -> Self {
        let mut command = Command::new(env!("CARGO_BIN_EXE_s1gate"));
        command
            // An ephemeral port, reported by the server once it is bound, so a test uses the port
            // this server actually holds rather than one another test could take in between.
            .args(["serve", "--port", "0"])
            .env("XDG_DATA_HOME", data_home.path())
            .env("XDG_CONFIG_HOME", data_home.path())
            .env_remove("TYPESAFE_API_KEY")
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        if let Some(api_key) = api_key {
            command.env("TYPESAFE_API_KEY", api_key);
        }
        let mut child = command.spawn().expect("start s1gate serve");
        let mut stderr = BufReader::new(child.stderr.take().expect("the server's stderr")).lines();
        let port = loop {
            match stderr.next() {
                Some(Ok(line)) => {
                    if let Some(port) = reported_port(&line) {
                        break port;
                    }
                }
                Some(Err(error)) => panic!("reading the server's stderr: {error}"),
                None => panic!(
                    "server exited at startup: {}",
                    child.wait().expect("the server exits")
                ),
            }
        };
        // Keep draining the pipe: a server that writes a diagnostic must not block on a full one.
        thread::spawn(move || stderr.for_each(drop));
        Self { child, port }
    }

    fn post(&self, body: &str, authorization: Option<&str>) -> (u16, Value) {
        post(self.port, body, authorization)
    }

    fn post_raw(&self, body: &str, authorization: Option<&str>) -> (u16, String, String) {
        post_raw(self.port, body, authorization)
    }

    fn get_models(&self) -> (u16, Value) {
        let (status, _, body) = self.get_models_raw();
        (status, serde_json::from_str(&body).expect("JSON response"))
    }

    /// One `GET /v1/models`: the status, the response headers, and the response body as answered.
    fn get_models_raw(&self) -> (u16, String, String) {
        let mut stream = TcpStream::connect(("127.0.0.1", self.port)).expect("connect to server");
        write!(
            stream,
            "GET /v1/models HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
        )
        .expect("send model list request");
        let mut response = String::new();
        stream.read_to_string(&mut response).expect("read response");
        let (head, body) = response.split_once("\r\n\r\n").expect("HTTP response");
        assert!(
            head.to_ascii_lowercase()
                .contains("content-type: application/json")
        );
        let status = head.split_whitespace().nth(1).unwrap().parse().unwrap();
        (status, head.to_string(), body.to_string())
    }
}

fn post(port: u16, body: &str, authorization: Option<&str>) -> (u16, Value) {
    let (status, _, body) = post_raw(port, body, authorization);
    (status, serde_json::from_str(&body).expect("JSON response"))
}

/// One `POST /v1/systemone`: the status, the response headers, and the response body as answered.
fn post_raw(port: u16, body: &str, authorization: Option<&str>) -> (u16, String, String) {
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
    (status, head.to_string(), body.to_string())
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// The port the server reports it bound, from the startup line it writes to stderr.
fn reported_port(line: &str) -> Option<u16> {
    let address = line.split_once("listening on ")?.1.trim();
    address.rsplit_once(':')?.1.parse().ok()
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
fn empty_local_server_lists_no_models_without_a_typesafe_key() {
    let data_home = TempDir::new("serve-empty-models");
    let server = Server::start(&data_home);
    assert_eq!(
        server.get_models(),
        (200, serde_json::json!({"models": []}))
    );

    fixture::write_inferable(&data_home.path().join("s1gate/models"));
    assert_eq!(
        server.get_models(),
        (200, serde_json::json!({"models": []}))
    );
}

#[test]
fn loaded_laya_is_listed_after_its_checkpoint_is_removed() {
    let data_home = TempDir::new("serve-loaded-models");
    let checkpoint = fixture::write_inferable(&data_home.path().join("s1gate/models"));
    let server = Server::start(&data_home);
    std::fs::remove_dir_all(checkpoint).expect("remove on-disk Checkpoint");

    let (status, body) = server.get_models();
    assert_eq!(status, 200);
    let models = body["models"].as_array().expect("models array");
    assert_eq!(models.len(), 1);
    assert_eq!(models[0]["name"], "convaiinnovations/laya");
    assert_eq!(models[0]["release_date"], "2026-09-18");
    assert!(
        models[0]["description"]
            .as_str()
            .is_some_and(|s| !s.is_empty())
    );
}

#[test]
fn remote_models_are_merged_with_loaded_local_models() {
    let stand_in = StandIn::start(Replies::InOrder(vec![reply("200 OK", JEV_MODELS)]));
    let data_home = TempDir::new("serve-models-merged");
    fixture::write_inferable(&data_home.path().join("s1gate/models"));
    let server = remote_server(&data_home, &stand_in.endpoint());

    let (status, body) = server.get_models();
    assert_eq!(status, 200, "{body}");
    let models = body["models"].as_array().expect("models array");
    assert_eq!(models.len(), 3, "{body}");
    assert_eq!(models[0]["name"], "convaiinnovations/laya");
    assert_eq!(models[1]["name"], "jev-latest");
    assert_eq!(models[1]["description"], "The most recent stable release.");
    assert_eq!(models[1]["release_date"], "2025-11-04");
    assert_eq!(models[2]["name"], "jev-preview");
    assert_eq!(models[2]["description"], "The most recent release, preview included.");
    assert_eq!(models[2]["release_date"], "2026-01-15");

    let request = stand_in.requests().pop().expect("the model list request");
    let request = request.to_ascii_lowercase();
    assert!(request.starts_with("get /v1/models http/1.1"), "{request}");
    assert!(
        request.contains("authorization: bearer configured-secret"),
        "{request}"
    );
}

#[test]
fn remote_models_are_listed_by_a_server_without_a_local_checkpoint() {
    let stand_in = StandIn::start(Replies::InOrder(vec![reply("200 OK", JEV_MODELS)]));
    let data_home = TempDir::new("serve-models-remote-only");
    let server = remote_server(&data_home, &stand_in.endpoint());

    let (status, body) = server.get_models();
    assert_eq!(status, 200, "{body}");
    let models = body["models"].as_array().expect("models array");
    assert_eq!(models.len(), 2, "{body}");
    assert_eq!(models[0]["name"], "jev-latest");
    assert_eq!(models[1]["name"], "jev-preview");
}

#[test]
fn a_missing_typesafe_key_lists_local_models_without_asking_typesafe() {
    let stand_in = StandIn::start(Replies::InOrder(vec![reply("200 OK", JEV_MODELS)]));
    let data_home = TempDir::new("serve-models-no-key");
    fixture::write_inferable(&data_home.path().join("s1gate/models"));
    configure_endpoint(&data_home, &stand_in.endpoint());
    let server = Server::start(&data_home);

    let (status, body) = server.get_models();
    assert_eq!(status, 200, "{body}");
    let models = body["models"].as_array().expect("models array");
    assert_eq!(models.len(), 1, "{body}");
    assert_eq!(models[0]["name"], "convaiinnovations/laya");
    assert_eq!(models[0]["release_date"], "2026-09-18");
    assert!(
        stand_in.requests().is_empty(),
        "the local-only list asked TypeSafe"
    );
}

#[test]
fn a_remote_model_list_failure_is_forwarded_instead_of_a_partial_local_list() {
    let refusal = r#"{"error":{"message":"bad key","code":"unauthorized"}}"#;
    let stand_in = StandIn::start(Replies::InOrder(vec![reply("401 Unauthorized", refusal)]));
    let data_home = TempDir::new("serve-models-failure");
    fixture::write_inferable(&data_home.path().join("s1gate/models"));
    let server = remote_server(&data_home, &stand_in.endpoint());

    let (status, _, body) = server.get_models_raw();
    assert_eq!(status, 401);
    assert_eq!(
        body, refusal,
        "the remote body is forwarded as TypeSafe wrote it"
    );
    assert_eq!(stand_in.requests().len(), 1);
}

#[test]
fn a_rate_limited_model_list_is_retried_and_its_last_retry_after_is_kept() {
    let refusal = r#"{"error":{"message":"quota exhausted","code":"rate_limited"}}"#;
    let stand_in = StandIn::start(Replies::InOrder(vec![
        reply("429 Too Many Requests", refusal).retry("0"),
        reply("429 Too Many Requests", refusal).retry("0"),
        reply("429 Too Many Requests", refusal).retry("7"),
    ]));
    let data_home = TempDir::new("serve-models-retry");
    let server = remote_server(&data_home, &stand_in.endpoint());

    let (status, head, body) = server.get_models_raw();
    assert_eq!(status, 429);
    assert_eq!(body, refusal);
    assert!(
        head.to_ascii_lowercase().contains("retry-after: 7"),
        "{head}"
    );
    assert_eq!(
        stand_in.requests().len(),
        3,
        "two retries, then the answer that stood"
    );
}

#[test]
fn a_remote_model_list_that_is_not_the_documented_shape_is_an_error() {
    for body in [
        r#"{"models":[{"name":"jev-latest"}]}"#,
        r#"{"models":[{"name":"","description":"x","release_date":"2025-11-04"}]}"#,
        r#"{"data":[]}"#,
        "not JSON",
    ] {
        let stand_in = StandIn::start(Replies::InOrder(vec![reply("200 OK", body)]));
        let data_home = TempDir::new("serve-models-incomplete");
        fixture::write_inferable(&data_home.path().join("s1gate/models"));
        let server = remote_server(&data_home, &stand_in.endpoint());

        let (status, answer) = server.get_models();
        assert_eq!(status, 502, "{body} answered {answer}");
        assert!(answer["detail"].is_array(), "{answer}");
    }
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
fn remote_alias_and_versioned_calls_reach_typesafe_with_the_servers_key() {
    let stand_in = StandIn::start(Replies::InOrder(vec![reply("200 OK", JEV_ANSWER)]));
    let data_home = TempDir::new("serve-typesafe-routing");
    let server = remote_server(&data_home, &stand_in.endpoint());

    let (status, body) = server.post(JEV_CALL, Some("Bearer caller-secret"));
    assert_eq!(status, 200);
    assert_eq!(body["model"], JEV_MODEL);
    assert_eq!(body["answers"]["risk"]["type"], "noul");
    assert_eq!(body["usage"]["input_tokens"], 11);

    let versioned = JEV_CALL.replace("jev-latest", JEV_MODEL);
    let (status, body) = server.post(&versioned, None);
    assert_eq!(status, 200);
    assert_eq!(body["model"], JEV_MODEL);

    let requests = stand_in.requests();
    assert_eq!(requests.len(), 2);
    for (request, model) in requests.iter().zip(["jev-latest", JEV_MODEL]) {
        let request = request.to_ascii_lowercase();
        assert!(
            request.contains("authorization: bearer configured-secret"),
            "{request}"
        );
        assert!(!request.contains("caller-secret"), "{request}");
        assert!(
            request.contains(&format!("\"model\":\"{model}\"")),
            "{request}"
        );
    }
}

#[test]
fn rate_limited_remote_calls_are_retried_and_their_last_refusal_is_forwarded() {
    let refusal =
        r#"{"error":{"message":"quota exhausted","code":"rate_limited"},"data":{"retry_after":7}}"#;
    for (code, status) in [
        (429, "429 Too Many Requests"),
        (529, "529 Site is Overloaded"),
    ] {
        let stand_in = StandIn::start(Replies::InOrder(vec![
            reply(status, r#"{"error":{"message":"please retry"}}"#).retry("0"),
            reply(status, r#"{"error":{"message":"please retry"}}"#).retry("0"),
            reply(status, refusal).retry("7"),
        ]));
        let data_home = TempDir::new("serve-typesafe-retry");
        let server = remote_server(&data_home, &stand_in.endpoint());

        let (answered, head, body) = server.post_raw(JEV_CALL, None);
        assert_eq!(answered, code);
        assert_eq!(
            serde_json::from_str::<Value>(&body).expect("forwarded JSON body"),
            serde_json::from_str::<Value>(refusal).expect("TypeSafe JSON body")
        );
        assert!(
            head.to_ascii_lowercase().contains("retry-after: 7"),
            "{head}"
        );
        assert_eq!(
            stand_in.requests().len(),
            3,
            "two retries, then the answer that stood"
        );
    }
}

#[test]
fn a_remote_refusal_that_is_not_a_rate_limit_is_forwarded_without_a_retry() {
    for (code, status, body) in [
        (
            401,
            "401 Unauthorized",
            r#"{"error":{"message":"bad key"}}"#,
        ),
        (
            422,
            "422 Unprocessable Entity",
            r#"{"detail":[{"msg":"questions must not be empty","loc":["body","questions"],"type":"value_error"}]}"#,
        ),
    ] {
        let stand_in = StandIn::start(Replies::InOrder(vec![reply(status, body)]));
        let data_home = TempDir::new("serve-typesafe-forward");
        let server = remote_server(&data_home, &stand_in.endpoint());

        let (answered, _, forwarded) = server.post_raw(JEV_CALL, None);
        assert_eq!(answered, code);
        assert_eq!(
            forwarded, body,
            "the remote body is forwarded as TypeSafe wrote it"
        );
        assert_eq!(stand_in.requests().len(), 1);
    }
}

#[test]
fn remote_calls_do_not_wait_for_each_other() {
    let stand_in = StandIn::start(Replies::Together {
        count: 2,
        reply: reply("200 OK", JEV_ANSWER),
    });
    let data_home = TempDir::new("serve-typesafe-concurrent");
    let server = remote_server(&data_home, &stand_in.endpoint());

    let barrier = Arc::new(Barrier::new(2));
    let calls: Vec<_> = (0..2)
        .map(|_| {
            let barrier = Arc::clone(&barrier);
            let port = server.port;
            thread::spawn(move || {
                barrier.wait();
                post(port, JEV_CALL, None)
            })
        })
        .collect();
    for call in calls {
        let (status, body) = call.join().expect("remote call thread");
        assert_eq!(status, 200, "{body}");
        assert_eq!(body["model"], JEV_MODEL);
    }
    assert_eq!(stand_in.arrived(), 2);
    assert!(
        !stand_in.gave_up(),
        "the remote Calls reached TypeSafe one after the other"
    );
}

#[test]
fn a_local_call_is_judged_while_a_remote_call_is_in_flight() {
    let data_home = TempDir::new("serve-typesafe-hold");
    fixture::write_inferable(&data_home.path().join("s1gate/models"));
    let stand_in = StandIn::start(Replies::OnRelease(reply("200 OK", JEV_ANSWER)));
    let server = remote_server(&data_home, &stand_in.endpoint());

    let port = server.port;
    let remote = thread::spawn(move || post(port, JEV_CALL, None));
    stand_in.await_request();

    let (status, body) = server.post(CALL, None);
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["model"], fixture::NAME);

    stand_in.release();
    assert_eq!(remote.join().expect("remote call thread").0, 200);
    assert!(
        !stand_in.gave_up(),
        "the local Call waited for the held remote Call"
    );
}

#[test]
fn a_remote_backend_that_cannot_be_reached_answers_bad_gateway() {
    let data_home = TempDir::new("serve-typesafe-unreachable");
    let server = remote_server(&data_home, &closed_endpoint());

    let (status, body) = server.post(JEV_CALL, None);
    assert_eq!(status, 502);
    assert!(body["detail"].is_array(), "{body}");
}

/// Start a server whose remote Backend is `endpoint`, under `data_home` and its key.
fn remote_server(data_home: &TempDir, endpoint: &str) -> Server {
    configure_endpoint(data_home, endpoint);
    Server::start_with_config(data_home, Some("configured-secret"))
}

/// Point the server's `TypeSafe` Backend at `endpoint` through the configuration file it reads.
fn configure_endpoint(data_home: &TempDir, endpoint: &str) {
    let config_dir = data_home.path().join("s1gate");
    std::fs::create_dir_all(&config_dir).expect("config directory");
    std::fs::write(
        config_dir.join("config.toml"),
        format!("[typesafe]\nendpoint = \"{endpoint}\"\n"),
    )
    .expect("endpoint config");
}

/// An endpoint on a loopback port with nothing listening on it.
fn closed_endpoint() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("a free port");
    let endpoint = format!(
        "http://{}/v1/systemone",
        listener.local_addr().expect("address")
    );
    drop(listener);
    endpoint
}

/// A local `TypeSafe` stand-in: a listener that records every request and answers it as the test
/// scripts, so HTTP forwarding is observable without reaching the real API.
struct StandIn {
    endpoint: String,
    received: mpsc::Receiver<String>,
    arrived: Arc<AtomicUsize>,
    gave_up: Arc<AtomicBool>,
    release: mpsc::Sender<()>,
    stop: Arc<AtomicBool>,
}

/// The replies a `StandIn` writes, in the order and at the time its script says.
enum Replies {
    /// Reply to each request in turn with the next scripted reply, repeating the last one.
    InOrder(Vec<Reply>),
    /// Hold every request until `count` have arrived, then reply to all of them at once.
    Together { count: usize, reply: Reply },
    /// Hold every request until the test releases the stand-in, then reply to all of them.
    OnRelease(Reply),
}

/// One reply a `StandIn` writes.
#[derive(Clone, Copy)]
struct Reply {
    status: &'static str,
    body: &'static str,
    retry_after: Option<&'static str>,
}

fn reply(status: &'static str, body: &'static str) -> Reply {
    Reply {
        status,
        body,
        retry_after: None,
    }
}

impl Reply {
    /// The same reply with a `Retry-After` header.
    fn retry(self, retry_after: &'static str) -> Self {
        Self {
            retry_after: Some(retry_after),
            ..self
        }
    }
}

impl StandIn {
    fn start(replies: Replies) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("stand-in port");
        listener
            .set_nonblocking(true)
            .expect("nonblocking stand-in");
        let endpoint = format!(
            "http://{}/v1/systemone",
            listener.local_addr().expect("stand-in address")
        );
        let (requests, received) = mpsc::channel();
        let (release, released) = mpsc::channel();
        let recording = Recording {
            requests,
            arrived: Arc::new(AtomicUsize::new(0)),
            gave_up: Arc::new(AtomicBool::new(false)),
            released,
            stop: Arc::new(AtomicBool::new(false)),
        };
        let stand_in = Self {
            endpoint,
            received,
            arrived: Arc::clone(&recording.arrived),
            gave_up: Arc::clone(&recording.gave_up),
            release,
            stop: Arc::clone(&recording.stop),
        };
        thread::spawn(move || serve_requests(&listener, &replies, &recording));
        stand_in
    }

    fn endpoint(&self) -> String {
        self.endpoint.clone()
    }

    /// Every request the stand-in received, forgetting them.
    fn requests(&self) -> Vec<String> {
        self.received.try_iter().collect()
    }

    /// How many requests have arrived.
    fn arrived(&self) -> usize {
        self.arrived.load(Ordering::SeqCst)
    }

    /// Whether the stand-in answered a request it was still holding, without being told to and
    /// without every request it was waiting for having arrived.
    fn gave_up(&self) -> bool {
        self.gave_up.load(Ordering::SeqCst)
    }

    /// Wait until at least one request is held, so the test can rely on a Call being in flight.
    fn await_request(&self) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while self.arrived() == 0 {
            assert!(Instant::now() < deadline, "no remote Call arrived");
            thread::sleep(Duration::from_millis(5));
        }
    }

    /// Let every held request be answered.
    fn release(&self) {
        let _ = self.release.send(());
    }
}

impl Drop for StandIn {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
    }
}

/// What a `StandIn` records while it answers.
struct Recording {
    requests: mpsc::Sender<String>,
    arrived: Arc<AtomicUsize>,
    gave_up: Arc<AtomicBool>,
    released: mpsc::Receiver<()>,
    stop: Arc<AtomicBool>,
}

impl Recording {
    /// Read one request in full, record it, and count it as arrived. A request is always read
    /// before the stand-in holds or answers it, so a held Call has provably reached the stand-in.
    /// Returns how many requests have arrived, this one included.
    fn record(&self, stream: &mut TcpStream) -> usize {
        let request = read_http_request(stream);
        let _ = self.requests.send(request);
        self.arrived.fetch_add(1, Ordering::SeqCst) + 1
    }
}

/// How long a `StandIn` holds requests that cannot be answered yet.
const HOLD: Duration = Duration::from_secs(5);

fn serve_requests(listener: &TcpListener, replies: &Replies, recording: &Recording) {
    while !recording.stop.load(Ordering::SeqCst) {
        match replies {
            Replies::InOrder(script) => {
                let Some(mut stream) = accept(listener) else {
                    continue;
                };
                let arrived = recording.record(&mut stream);
                write_reply(&mut stream, script[(arrived - 1).min(script.len() - 1)]);
            }
            Replies::Together { count, reply } => {
                let mut held = Vec::new();
                while held.len() < *count {
                    match accept_until(listener, HOLD) {
                        Some(mut stream) => {
                            recording.record(&mut stream);
                            held.push(stream);
                        }
                        None => {
                            recording.gave_up.store(true, Ordering::SeqCst);
                            break;
                        }
                    }
                }
                for stream in &mut held {
                    write_reply(stream, *reply);
                }
            }
            Replies::OnRelease(reply) => {
                let Some(mut stream) = accept(listener) else {
                    continue;
                };
                recording.record(&mut stream);
                if recording.released.recv_timeout(HOLD).is_err() {
                    recording.gave_up.store(true, Ordering::SeqCst);
                }
                write_reply(&mut stream, *reply);
            }
        }
    }
}

/// One connection, or none while the stand-in's stop flag is polled.
fn accept(listener: &TcpListener) -> Option<TcpStream> {
    accept_until(listener, Duration::from_millis(50))
}

/// One connection, or none within `timeout`.
fn accept_until(listener: &TcpListener, timeout: Duration) -> Option<TcpStream> {
    let deadline = Instant::now() + timeout;
    loop {
        match listener.accept() {
            Ok((stream, _)) => {
                // A socket accepted by a nonblocking listener inherits that flag on BSD, where a
                // partial read would then fail before the rest of the request arrives.
                stream
                    .set_nonblocking(false)
                    .expect("blocking stand-in connection");
                return Some(stream);
            }
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                if Instant::now() >= deadline {
                    return None;
                }
                thread::sleep(Duration::from_millis(2));
            }
            Err(error) => panic!("stand-in accept: {error}"),
        }
    }
}

fn read_http_request(stream: &mut TcpStream) -> String {
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .expect("stand-in read timeout");
    let mut request = Vec::new();
    let mut buffer = [0; 4096];
    loop {
        let count = stream
            .read(&mut buffer)
            .expect("the stand-in reads a request");
        assert!(count > 0, "a request closed before its body");
        request.extend_from_slice(&buffer[..count]);
        let Some(end) = request.windows(4).position(|part| part == b"\r\n\r\n") else {
            continue;
        };
        let head = String::from_utf8_lossy(&request[..end]);
        // A request that names no length, such as a GET, carries no body.
        let length = head
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().expect("content length"))
            })
            .unwrap_or(0);
        if request.len() >= end + 4 + length {
            return String::from_utf8(request).expect("UTF-8 request");
        }
    }
}

/// Reply to one request. A Call the server has already given up on is not the stand-in's failure.
fn write_reply(stream: &mut TcpStream, reply: Reply) {
    let retry_after = reply
        .retry_after
        .map_or(String::new(), |value| format!("Retry-After: {value}\r\n"));
    let _ = write!(
        stream,
        "HTTP/1.1 {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{retry_after}Connection: close\r\n\r\n{}",
        reply.status,
        reply.body.len(),
        reply.body
    );
}
