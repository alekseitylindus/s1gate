//! A local Model Source for tests: an HTTP server that answers the way the Model Source does —
//! revision lookups, the resolve redirect, LFS and git blob etags — so Pull's tests exercise the
//! real Pull path without contacting `convaiinnovations/laya` (ADR-0009).
//!
//! This module is shared by several test binaries, each of which uses a part of it.
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::thread;

use parking_lot::Mutex;
use sha1::Sha1;
use sha2::{Digest as _, Sha256};

pub mod fixture;

pub struct FakeSource {
    port: u16,
    state: Arc<Mutex<State>>,
}

struct State {
    repo: String,
    default_branch: String,
    refs: BTreeMap<String, String>,
    commits: BTreeMap<String, Vec<ServedFile>>,
    requests: Vec<Request>,
}

/// One file the Model Source publishes.
#[derive(Clone, Debug)]
pub struct ServedFile {
    pub path: String,
    body: Vec<u8>,
    etag: Option<String>,
    announce_size: bool,
    truncate_at: Option<usize>,
    served_commit: Option<String>,
    absent: bool,
    direct: bool,
}

#[derive(Clone, Debug)]
pub struct Request {
    pub target: String,
    pub headers: Vec<(String, String)>,
}

impl Request {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }
}

impl FakeSource {
    pub fn new(repo: &str) -> FakeSource {
        let listener = TcpListener::bind("127.0.0.1:0").expect("a local port");
        let port = listener.local_addr().expect("a bound address").port();
        let state = Arc::new(Mutex::new(State {
            repo: repo.to_string(),
            default_branch: "main".to_string(),
            refs: BTreeMap::new(),
            commits: BTreeMap::new(),
            requests: Vec::new(),
        }));
        let server_state = Arc::clone(&state);
        thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { break };
                let state = Arc::clone(&server_state);
                thread::spawn(move || serve(stream, state));
            }
        });
        FakeSource { port, state }
    }

    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    /// Publish `files` as commit `commit`, replacing that commit's contents.
    pub fn publish(&self, commit: &str, files: Vec<ServedFile>) {
        self.state.lock().commits.insert(commit.to_string(), files);
    }

    /// Point revision `name` at `commit`.
    pub fn point(&self, name: &str, commit: &str) {
        self.state
            .lock()
            .refs
            .insert(name.to_string(), commit.to_string());
    }

    /// Make `commit` the head of the default branch.
    pub fn set_default_branch(&self, name: &str) {
        self.state.lock().default_branch = name.to_string();
    }

    pub fn requests(&self) -> Vec<Request> {
        self.state.lock().requests.clone()
    }

    /// The requests that would carry a Checkpoint file's bytes.
    pub fn file_requests(&self) -> Vec<Request> {
        self.requests()
            .into_iter()
            .filter(|request| resolve_path(&request.target).is_some())
            .collect()
    }

    pub fn bodies(&self, commit: &str) -> BTreeMap<String, Vec<u8>> {
        self.state
            .lock()
            .commits
            .get(commit)
            .map(|files| {
                files
                    .iter()
                    .map(|file| (file.path.clone(), file.body.clone()))
                    .collect()
            })
            .unwrap_or_default()
    }
}

impl ServedFile {
    /// A file the Model Source tracks in git, so it publishes a git blob object id.
    pub fn json(path: &str, body: &str) -> ServedFile {
        ServedFile {
            path: path.to_string(),
            etag: Some(git_blob_sha1(body.as_bytes())),
            body: body.as_bytes().to_vec(),
            announce_size: false,
            truncate_at: None,
            served_commit: None,
            absent: false,
            direct: false,
        }
    }

    /// A file the Model Source holds in LFS, so it publishes a sha256 and a linked size.
    pub fn lfs(path: &str, body: impl Into<Vec<u8>>) -> ServedFile {
        let body = body.into();
        ServedFile {
            path: path.to_string(),
            etag: Some(sha256(&body)),
            body,
            announce_size: true,
            truncate_at: None,
            served_commit: None,
            absent: false,
            direct: false,
        }
    }

    /// Advertise `etag` instead of the checksum of the bytes — a corrupt upload.
    pub fn advertising(mut self, etag: &str) -> ServedFile {
        self.etag = Some(etag.to_string());
        self
    }

    /// Advertise no checksum at all.
    pub fn without_checksum(mut self) -> ServedFile {
        self.etag = None;
        self
    }

    /// Answer with only the first `bytes` of the body.
    pub fn truncated_at(mut self, bytes: usize) -> ServedFile {
        self.truncate_at = Some(bytes);
        self
    }

    /// Do not publish this file at all.
    pub fn absent(mut self) -> ServedFile {
        self.absent = true;
        self
    }

    /// Serve the file itself instead of redirecting to a presigned URL.
    pub fn served_directly(mut self) -> ServedFile {
        self.direct = true;
        self
    }

    /// Announce that the file was served from another commit.
    pub fn served_from(mut self, commit: &str) -> ServedFile {
        self.served_commit = Some(commit.to_string());
        self
    }
}

/// The allowlist files, all present, as a Model Source would publish them for `commit`.
pub fn allowlist_files(commit: &str) -> Vec<ServedFile> {
    vec![
        ServedFile::lfs("model.safetensors", format!("weights of {commit}")),
        ServedFile::json(
            "rl_agent_config.json",
            &format!("{{\"checkpoint\": \"{commit}\"}}"),
        ),
        ServedFile::json(
            "encoder/config.json",
            &format!("{{\"encoder\": \"{commit}\"}}"),
        ),
        ServedFile::json("tokenizer/tokenizer.json", &format!("[\"{commit}\"]")),
        ServedFile::json(
            "tokenizer/tokenizer_config.json",
            &format!("{{\"tokenizer\": \"{commit}\"}}"),
        ),
    ]
}

/// Files a Model Source publishes that the Checkpoint allowlist does not name.
pub fn unrelated_files() -> Vec<ServedFile> {
    vec![
        ServedFile::json("README.md", "# laya\n"),
        ServedFile::json("multilingual/config.json", "{\"multilingual\": true}"),
    ]
}

/// The five Checkpoint paths, in the order Pull streams them.
pub const ALLOWLIST: [&str; 5] = [
    "model.safetensors",
    "rl_agent_config.json",
    "encoder/config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
];

pub fn git_blob_sha1(bytes: &[u8]) -> String {
    let mut sha1 = Sha1::new();
    sha1.update(format!("blob {}\0", bytes.len()).as_bytes());
    sha1.update(bytes);
    hex(&sha1.finalize())
}

pub fn sha256(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;

    let mut text = String::new();
    for byte in bytes {
        let _ = write!(text, "{byte:02x}");
    }
    text
}

/// A temporary Model Store root, removed when the test ends.
pub struct TempDir(PathBuf);

impl TempDir {
    pub fn new(case: &str) -> TempDir {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let unique = NEXT.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("s1gate-{case}-{}-{unique}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("a temporary directory");
        TempDir(path)
    }

    pub fn path(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Every file below `root`, relative and sorted.
pub fn tree(root: &std::path::Path) -> Vec<String> {
    let mut found = Vec::new();
    collect(root, root, &mut found);
    found.sort();
    found
}

fn collect(root: &std::path::Path, directory: &std::path::Path, found: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect(root, &path, found);
        } else {
            let relative = path.strip_prefix(root).expect("below the root");
            found.push(relative.to_string_lossy().into_owned());
        }
    }
}

/// The `{repo}/resolve/{commit}/{path}` requests, as `(commit, path)`.
pub fn resolve_path(target: &str) -> Option<(String, String)> {
    let (path, _) = target.split_once('?').unwrap_or((target, ""));
    let segments: Vec<String> = path
        .split('/')
        .filter(|segment| !segment.is_empty())
        .map(decode)
        .collect();
    let resolve_at = segments.iter().position(|segment| segment == "resolve")?;
    let commit = segments.get(resolve_at + 1)?.clone();
    let file = segments.get(resolve_at + 2..)?.join("/");
    Some((commit, file))
}

fn serve(mut stream: TcpStream, state: Arc<Mutex<State>>) {
    let Some(request) = read_request(&mut stream) else {
        return;
    };
    let reply = {
        let mut state = state.lock();
        state.requests.push(request.clone());
        route(&state, &request.target)
    };
    let _ = write_reply(&mut stream, &reply);
}

fn read_request(stream: &mut TcpStream) -> Option<Request> {
    let mut head = Vec::new();
    let mut byte = [0u8; 1];
    while !head.ends_with(b"\r\n\r\n") {
        match stream.read(&mut byte) {
            Ok(0) | Err(_) => return None,
            Ok(_) => head.push(byte[0]),
        }
    }
    let head = String::from_utf8_lossy(&head).into_owned();
    let mut lines = head.lines();
    let target = lines.next()?.split_whitespace().nth(1)?.to_string();
    let headers = lines
        .filter_map(|line| line.split_once(':'))
        .map(|(key, value)| (key.trim().to_string(), value.trim().to_string()))
        .collect();
    Some(Request { target, headers })
}

struct Reply {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
    truncate_at: Option<usize>,
}

impl Reply {
    fn new(status: u16) -> Reply {
        Reply {
            status,
            headers: Vec::new(),
            body: Vec::new(),
            truncate_at: None,
        }
    }

    fn json(status: u16, body: impl Into<String>) -> Reply {
        let body = body.into().into_bytes();
        let mut reply = Reply::new(status);
        reply.headers.push((
            "content-type".to_string(),
            "application/json; charset=utf-8".to_string(),
        ));
        reply.body = body;
        reply
    }

    fn header(mut self, name: &str, value: &str) -> Reply {
        self.headers.push((name.to_string(), value.to_string()));
        self
    }
}

fn route(state: &State, target: &str) -> Reply {
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    let segments: Vec<String> = path
        .split('/')
        .filter(|segment| !segment.is_empty())
        .map(decode)
        .collect();
    let resolve_at = segments.iter().position(|segment| segment == "resolve");

    if segments.first().map(String::as_str) == Some("api")
        && segments.get(1).map(String::as_str) == Some("models")
    {
        // The Model Source's default branch, and a requested revision, as the API reports them.
        let rest = &segments[2..];
        let (repo, revision) = match rest.iter().position(|segment| segment == "revision") {
            Some(at) => (rest[..at].join("/"), Some(rest[at + 1..].join("/"))),
            None => (rest.join("/"), None),
        };
        if repo != state.repo {
            return Reply::json(
                404,
                format!("{{\"error\": \"Repository not found: {repo}\"}}"),
            );
        }
        let name = revision
            .clone()
            .unwrap_or_else(|| state.default_branch.clone());
        return match state.refs.get(&name) {
            Some(commit) => Reply::json(200, format!("{{\"sha\": \"{commit}\"}}")),
            None if revision.is_some() => {
                Reply::json(404, format!("{{\"error\": \"Invalid rev id: {name}\"}}"))
            }
            None => Reply::json(404, "{\"error\": \"no default branch\"}"),
        };
    }

    // The presigned transfer of one file.
    if let Some(at) = resolve_at {
        let repo = segments[..at].join("/");
        let commit = segments[at + 1].clone();
        let file = segments[at + 2..].join("/");
        if repo != state.repo {
            return Reply::json(
                404,
                format!("{{\"error\": \"Repository not found: {repo}\"}}"),
            );
        }
        return match find(state, &commit, &file) {
            Some(served) if !served.absent => {
                let mut reply = Reply::new(if served.direct { 200 } else { 307 });
                reply.headers.push((
                    "x-repo-commit".to_string(),
                    served
                        .served_commit
                        .clone()
                        .unwrap_or_else(|| commit.clone()),
                ));
                if let Some(etag) = &served.etag {
                    reply
                        .headers
                        .push(("x-linked-etag".to_string(), format!("\"{etag}\"")));
                }
                if served.announce_size {
                    reply
                        .headers
                        .push(("x-linked-size".to_string(), served.body.len().to_string()));
                }
                if served.direct {
                    reply.body = served.body.clone();
                    reply.truncate_at = served.truncate_at;
                } else {
                    reply
                        .headers
                        .push(("location".to_string(), format!("/blob/{file}?c={commit}")));
                }
                reply
            }
            _ => Reply::json(404, format!("{{\"error\": \"Entry not found: {file}\"}}")),
        };
    }

    // The redirect target.
    if segments.first().map(String::as_str) == Some("blob") {
        let file = segments[1..].join("/");
        let commit = query_value(query, "c").unwrap_or_default();
        return match find(state, &commit, &file) {
            Some(served) => {
                let mut reply = Reply::new(200);
                reply.body = served.body.clone();
                reply.truncate_at = served.truncate_at;
                reply
            }
            None => Reply::json(404, "{\"error\": \"Entry not found\"}"),
        };
    }

    Reply::json(404, "{\"error\": \"Not found\"}")
}

fn find<'a>(state: &'a State, commit: &str, path: &str) -> Option<&'a ServedFile> {
    state
        .commits
        .get(commit)?
        .iter()
        .find(|file| file.path == path)
}

fn query_value(query: &str, key: &str) -> Option<String> {
    query.split('&').find_map(|pair| {
        let (name, value) = pair.split_once('=')?;
        (name == key).then(|| decode(value))
    })
}

fn write_reply(stream: &mut TcpStream, reply: &Reply) -> std::io::Result<()> {
    let reason = match reply.status {
        200 => "OK",
        307 => "Temporary Redirect",
        404 => "Not Found",
        _ => "Unknown",
    };
    let mut head = format!(
        "HTTP/1.1 {} {}\r\ncontent-length: {}\r\nconnection: close\r\n",
        reply.status,
        reason,
        reply.body.len()
    );
    for (name, value) in &reply.headers {
        head.push_str(&format!("{name}: {value}\r\n"));
    }
    head.push_str("\r\n");
    stream.write_all(head.as_bytes())?;
    match reply.truncate_at {
        Some(bytes) => stream.write_all(&reply.body[..bytes.min(reply.body.len())]),
        None => stream.write_all(&reply.body),
    }?;
    stream.flush()
}

fn decode(segment: &str) -> String {
    let bytes = segment.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%'
            && index + 2 < bytes.len()
            && let Ok(byte) = u8::from_str_radix(&segment[index + 1..index + 3], 16)
        {
            decoded.push(byte);
            index += 3;
            continue;
        }
        decoded.push(bytes[index]);
        index += 1;
    }
    String::from_utf8_lossy(&decoded).into_owned()
}
