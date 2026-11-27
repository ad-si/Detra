//! HTTP server with an OpenAI-compatible chat API, so that existing chat front ends
//! can talk to the engine.
//!
//! - `GET /v1/models`
//! - `POST /v1/chat/completions`, optionally streamed as server-sent events
//! - `GET /health`
//!
//! A response is a pure function of the model file and the request body: there are no
//! timestamps or random ids in it. Unless the request says otherwise, decoding is
//! greedy and the seed is 0, like on the command line.
//!
//! One request is generated at a time. Snapshots of earlier sessions are kept (after
//! the last message, after the generation prompt and after the reply), so a
//! conversation that continues or a reply that is regenerated only pays for its new
//! tokens; this changes the speed, never the result.

use crate::chat::{chat_tokens, Role};
use crate::engine::Session;
use crate::fixed::{parse_q16, ONE_Q16};
use crate::generate::{feed_prompt, sample_tokens, Finish, GenParams};
use crate::json::{self, quote, Value};
use crate::model::Model;
use crate::sample::SampleParams;
use crate::sha256::{hex, Sha256};
use crate::tokenizer::Tokenizer;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

const MAX_HEADER_BYTES: usize = 64 << 10;
const MAX_BODY_BYTES: usize = 32 << 20;
const MAX_CONNECTIONS: usize = 64;
const MAX_STOP_STRINGS: usize = 16;
const IO_TIMEOUT: Duration = Duration::from_secs(30);
/// Number of sessions kept for continuing conversations.
const KEPT_SESSIONS: usize = 8;

pub struct Config {
    pub model: Arc<Model>,
    /// SHA-256 of the model file.
    pub model_id: [u8; 32],
    /// Name under which the model is listed.
    pub name: String,
    /// Number of threads. Affects speed only.
    pub threads: usize,
    /// Web origin that may use the server from a browser, or `*` for any. Pages
    /// served from this machine are always allowed.
    pub allow_origin: Option<String>,
    /// Write one line per request to standard error.
    pub log: bool,
    /// Append every chat request and its reply here, as one JSON object per line.
    pub record: Option<Box<dyn Write + Send + Sync>>,
}

/// Serve requests on `listener` forever.
pub fn serve(listener: TcpListener, mut config: Config) {
    let record = config.record.take().map(Mutex::new);
    let server = Server {
        record,
        tokenizer: Tokenizer::new(&config.model),
        config: &config,
        kept: Mutex::new(Vec::new()),
        busy: Mutex::new(()),
        connections: AtomicUsize::new(0),
    };
    std::thread::scope(|scope| {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { continue };
            let server = &server;
            if server.connections.fetch_add(1, Ordering::SeqCst) >= MAX_CONNECTIONS {
                server.connections.fetch_sub(1, Ordering::SeqCst);
                continue;
            }
            scope.spawn(move || {
                server.connection(stream);
                server.connections.fetch_sub(1, Ordering::SeqCst);
            });
        }
    });
}

struct Server<'a> {
    config: &'a Config,
    tokenizer: Tokenizer<'a>,
    /// Sessions with the tokens they have consumed, least recently used first.
    kept: Mutex<Vec<(Vec<u32>, Session)>>,
    record: Option<Mutex<Box<dyn Write + Send + Sync>>>,
    /// Held while generating: the engine already uses every core it was given.
    busy: Mutex<()>,
    connections: AtomicUsize,
}

struct Request {
    method: String,
    path: String,
    /// HTTP/1.1 or later: the client understands chunked responses.
    chunked_ok: bool,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Request {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }
}

/// An error response: HTTP status and message.
type Failure = (u16, String);

fn bad_request<T>(message: impl Into<String>) -> Result<T, Failure> {
    Err((400, message.into()))
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        411 => "Length Required",
        413 => "Content Too Large",
        431 => "Request Header Fields Too Large",
        _ => "Internal Server Error",
    }
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn read_request(stream: &mut TcpStream) -> Result<Request, Failure> {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let head_end = loop {
        // Only the new bytes and the three before them can complete the separator.
        let from = buf.len().saturating_sub(3);
        let n = stream.read(&mut chunk).map_err(|e| (400, format!("cannot read the request: {e}")))?;
        if n == 0 {
            return bad_request("incomplete request");
        }
        buf.extend_from_slice(&chunk[..n]);
        if let Some(at) = find(&buf[from..], b"\r\n\r\n") {
            break from + at;
        }
        if buf.len() > MAX_HEADER_BYTES {
            return Err((431, "the request header is too large".into()));
        }
    };
    let Ok(head) = core::str::from_utf8(&buf[..head_end]) else {
        return bad_request("the request header is not valid UTF-8");
    };
    let mut lines = head.split("\r\n");
    let mut parts = lines.next().unwrap_or("").split(' ');
    let (Some(method), Some(target), Some(version), None) = (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return bad_request("malformed request line");
    };
    if !version.starts_with("HTTP/1.") {
        return bad_request("unsupported HTTP version");
    }
    let mut headers = Vec::new();
    for line in lines {
        let Some((name, value)) = line.split_once(':') else {
            return bad_request("malformed header line");
        };
        headers.push((name.trim().to_string(), value.trim().to_string()));
    }
    let mut request = Request {
        method: method.to_string(),
        path: target.split('?').next().unwrap_or("").to_string(),
        chunked_ok: version != "HTTP/1.0",
        headers,
        body: buf[head_end + 4..].to_vec(),
    };
    if request.header("transfer-encoding").is_some() {
        return Err((411, "chunked request bodies are not supported; send Content-Length".into()));
    }
    let length = match request.header("content-length") {
        Some(v) => v.parse::<usize>().map_err(|_| (400, "invalid Content-Length".to_string()))?,
        None => 0,
    };
    if length > MAX_BODY_BYTES {
        return Err((413, format!("the request body is larger than {MAX_BODY_BYTES} bytes")));
    }
    if request.body.len() > length {
        return bad_request("more data than Content-Length announces");
    }
    if request.body.len() < length {
        if request.header("expect").is_some_and(|v| v.eq_ignore_ascii_case("100-continue")) {
            let _ = stream.write_all(b"HTTP/1.1 100 Continue\r\n\r\n");
        }
        let have = request.body.len();
        request.body.resize(length, 0);
        stream
            .read_exact(&mut request.body[have..])
            .map_err(|e| (400, format!("cannot read the request body: {e}")))?;
    }
    Ok(request)
}

/// Whether a web origin is a page served from this machine.
fn is_local_origin(origin: &str) -> bool {
    let Some(host) = origin.strip_prefix("http://").or_else(|| origin.strip_prefix("https://")) else {
        return false;
    };
    let host = match host.rsplit_once(':') {
        Some((name, port)) if !port.is_empty() && port.bytes().all(|b| b.is_ascii_digit()) => name,
        _ => host,
    };
    matches!(host, "localhost" | "127.0.0.1" | "[::1]")
}

fn error_body(message: &str) -> String {
    format!(
        "{{\"error\":{{\"message\":{},\"type\":\"invalid_request_error\",\"param\":null,\"code\":null}}}}",
        quote(message)
    )
}

/// A response body sent piece by piece.
struct Stream<'s> {
    stream: &'s mut TcpStream,
    chunked: bool,
    failed: bool,
}

impl Stream<'_> {
    /// Send one server-sent event. After a failure (the client has gone) nothing more
    /// is sent.
    fn event(&mut self, data: &str) {
        if self.failed {
            return;
        }
        let event = format!("data: {data}\n\n");
        let message = if self.chunked { format!("{:x}\r\n{event}\r\n", event.len()) } else { event };
        self.failed = self.stream.write_all(message.as_bytes()).and_then(|()| self.stream.flush()).is_err();
    }

    fn finish(&mut self) {
        if self.chunked && !self.failed {
            self.failed = self.stream.write_all(b"0\r\n\r\n").is_err();
        }
    }
}

/// The longest prefix of `bytes` as text whose characters are already certain, with
/// invalid sequences replaced by U+FFFD. Unless `complete`, an unfinished character
/// at the end is left out, so the result for a longer input starts with the result
/// for a shorter one.
fn text_prefix(mut bytes: &[u8], complete: bool) -> String {
    let mut out = String::new();
    loop {
        match core::str::from_utf8(bytes) {
            Ok(text) => {
                out.push_str(text);
                return out;
            }
            Err(e) => {
                let (valid, rest) = bytes.split_at(e.valid_up_to());
                out.push_str(core::str::from_utf8(valid).unwrap_or(""));
                match e.error_len() {
                    Some(n) => {
                        out.push('\u{FFFD}');
                        bytes = &rest[n..];
                    }
                    None => {
                        if complete {
                            out.push('\u{FFFD}');
                        }
                        return out;
                    }
                }
            }
        }
    }
}

struct ChatRequest {
    messages: Vec<(Role, String)>,
    max_tokens: usize,
    sample: SampleParams,
    stream: bool,
    include_usage: bool,
    stop: Vec<String>,
}

/// A non-negative JSON integer.
fn integer(value: &Value, name: &str) -> Result<u64, Failure> {
    match value {
        Value::Number(n) => n.parse().map_err(|_| (400, format!("`{name}` must be a non-negative integer"))),
        _ => bad_request(format!("`{name}` must be a non-negative integer")),
    }
}

/// A non-negative JSON decimal in Q16. Digits after the ninth decimal place are
/// dropped; exponents are not supported.
fn decimal_q16(value: &Value, name: &str) -> Result<i64, Failure> {
    let parsed = match value {
        Value::Number(n) => {
            let (int, frac) = n.split_once('.').unwrap_or((n, ""));
            let digits = frac.bytes().all(|b| b.is_ascii_digit());
            let frac = &frac[..frac.len().min(if digits { 9 } else { 0 })];
            parse_q16(&format!("{int}.{frac}")).filter(|_| digits)
        }
        _ => None,
    };
    parsed.ok_or((400, format!("`{name}` must be a non-negative decimal number without exponent")))
}

fn flag(value: &Value, name: &str) -> Result<bool, Failure> {
    match value {
        Value::Bool(b) => Ok(*b),
        _ => bad_request(format!("`{name}` must be true or false")),
    }
}

/// The text of a message: a string, or a list of text parts.
fn content_text(content: Option<&Value>) -> Result<String, Failure> {
    match content {
        None => Ok(String::new()),
        Some(Value::String(text)) => Ok(text.clone()),
        Some(Value::Array(parts)) => {
            let mut text = String::new();
            for part in parts {
                match (part.get("type").and_then(Value::as_str), part.get("text").and_then(Value::as_str)) {
                    (Some("text"), Some(t)) => text.push_str(t),
                    _ => return bad_request("only text content is supported"),
                }
            }
            Ok(text)
        }
        Some(_) => bad_request("message `content` must be a string or a list of text parts"),
    }
}

fn messages_of(list: &Value) -> Result<Vec<(Role, String)>, Failure> {
    let Value::Array(list) = list else {
        return bad_request("`messages` must be a list");
    };
    let mut messages = Vec::new();
    for message in list {
        let role = match message.get("role").and_then(Value::as_str) {
            Some("system" | "developer") => Role::System,
            Some("user") => Role::User,
            Some("assistant") => Role::Assistant,
            Some(other) => return bad_request(format!("unsupported message role `{other}`")),
            None => return bad_request("every message needs a `role`"),
        };
        messages.push((role, content_text(message.get("content"))?));
    }
    Ok(messages)
}

/// A conversation given as a JSON list of messages in the chat API's format.
pub fn parse_messages(json: &[u8]) -> crate::Result<Vec<(Role, String)>> {
    let list = json::parse(json)?;
    messages_of(&list).map_err(|(_, message)| crate::Error(message))
}

fn parse_chat(body: &[u8]) -> Result<ChatRequest, Failure> {
    let v = json::parse(body).map_err(|e| (400, e.0))?;
    if !matches!(v, Value::Object(_)) {
        return bad_request("the request body must be a JSON object");
    }
    let Some(list) = v.get("messages") else {
        return bad_request("`messages` must be a list");
    };
    let messages = messages_of(list)?;
    if matches!(v.get("tools"), Some(Value::Array(tools)) if !tools.is_empty()) {
        return bad_request("tools are not supported");
    }
    if let Some(n) = v.get("n") {
        if integer(n, "n")? != 1 {
            return bad_request("`n` must be 1");
        }
    }
    let mut sample = SampleParams::GREEDY;
    if let Some(t) = v.get("temperature") {
        sample.temperature_q16 = decimal_q16(t, "temperature")?;
    }
    if let Some(p) = v.get("top_p") {
        let p = decimal_q16(p, "top_p")?;
        if p > ONE_Q16 {
            return bad_request("`top_p` must be at most 1");
        }
        // Anything below the resolution keeps the most likely token only.
        sample.top_p_q16 = p.max(1);
    }
    if let Some(k) = v.get("top_k") {
        sample.top_k = usize::try_from(integer(k, "top_k")?).unwrap_or(usize::MAX);
    }
    if let Some(seed) = v.get("seed") {
        // A negative seed is taken as its two's complement.
        sample.seed = match seed {
            Value::Number(n) if n.starts_with('-') => n.parse::<i64>().ok().map(|s| s as u64),
            Value::Number(n) => n.parse::<u64>().ok(),
            _ => None,
        }
        .ok_or((400, "`seed` must be an integer".to_string()))?;
    }
    let mut max_tokens = usize::MAX;
    for name in ["max_tokens", "max_completion_tokens"] {
        if let Some(n) = v.get(name) {
            max_tokens = usize::try_from(integer(n, name)?).unwrap_or(usize::MAX);
        }
    }
    let stop = match v.get("stop") {
        None => Vec::new(),
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(items)) => {
            let strings: Option<Vec<String>> = items.iter().map(|s| s.as_str().map(str::to_string)).collect();
            strings.ok_or((400, "`stop` must be a string or a list of strings".to_string()))?
        }
        Some(_) => return bad_request("`stop` must be a string or a list of strings"),
    };
    if stop.len() > MAX_STOP_STRINGS {
        return bad_request(format!("at most {MAX_STOP_STRINGS} stop strings are supported"));
    }
    Ok(ChatRequest {
        messages,
        max_tokens,
        sample,
        stream: v.get("stream").map(|s| flag(s, "stream")).transpose()?.unwrap_or(false),
        include_usage: v
            .get("stream_options")
            .and_then(|o| o.get("include_usage"))
            .map(|u| flag(u, "stream_options.include_usage"))
            .transpose()?
            .unwrap_or(false),
        stop: stop.into_iter().filter(|s| !s.is_empty()).collect(),
    })
}

/// What a request did, for the log.
#[derive(Default)]
struct Summary {
    status: u16,
    note: String,
    /// For a chat request: the members describing the reply in the record.
    reply: Option<String>,
}

impl Server<'_> {
    fn connection(&self, mut stream: TcpStream) {
        let _ = stream.set_read_timeout(Some(IO_TIMEOUT));
        let _ = stream.set_write_timeout(Some(IO_TIMEOUT));
        let _ = stream.set_nodelay(true);
        let request = match read_request(&mut stream) {
            Ok(request) => request,
            Err((status, message)) => {
                let _ = self.respond(&mut stream, status, "", "application/json", &error_body(&message));
                return;
            }
        };
        let cors = self.cors_headers(&request);
        let summary = match self.route(&request, &mut stream, &cors) {
            Ok(summary) => summary,
            Err((status, message)) => {
                let _ = self.respond(&mut stream, status, &cors, "application/json", &error_body(&message));
                let reply = format!("\"error\":{{\"status\":{status},\"message\":{}}}", quote(&message));
                Summary { status, note: message, reply: Some(reply) }
            }
        };
        if self.config.log {
            eprintln!("{} {} {} {}", request.method, request.path, summary.status, summary.note);
        }
        if request.method == "POST" && request.path == "/v1/chat/completions" {
            if let (Some(record), Some(reply)) = (&self.record, &summary.reply) {
                self.record(record, &request, reply);
            }
        }
    }

    /// Append one line: the time, the model, the request as received and the reply.
    fn record(&self, record: &Mutex<Box<dyn Write + Send + Sync>>, request: &Request, reply: &str) {
        let time = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
        let body = match json::parse(&request.body) {
            Ok(value) => value.to_json(),
            Err(_) => quote(&String::from_utf8_lossy(&request.body)),
        };
        let line = format!(
            "{{\"time\":{time},\"model\":\"{}\",\"request\":{body},{reply}}}\n",
            hex(&self.config.model_id)
        );
        let mut file = record.lock().unwrap_or_else(|e| e.into_inner());
        if let Err(e) = file.write_all(line.as_bytes()).and_then(|()| file.flush()) {
            eprintln!("error: cannot record the request: {e}");
        }
    }

    /// Headers that let a browser page from an allowed origin read the response.
    fn cors_headers(&self, request: &Request) -> String {
        let Some(origin) = request.header("origin") else {
            return String::new();
        };
        let allowed = match self.config.allow_origin.as_deref() {
            Some("*") => "*",
            Some(o) if o == origin => origin,
            _ if is_local_origin(origin) => origin,
            _ => return String::new(),
        };
        let asked = request.header("access-control-request-headers").unwrap_or("authorization, content-type");
        format!(
            "Access-Control-Allow-Origin: {allowed}\r\nVary: Origin\r\n\
             Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\
             Access-Control-Allow-Headers: {asked}\r\nAccess-Control-Max-Age: 600\r\n"
        )
    }

    fn respond(
        &self,
        stream: &mut TcpStream,
        status: u16,
        extra_headers: &str,
        content_type: &str,
        body: &str,
    ) -> std::io::Result<()> {
        let head = format!(
            "HTTP/1.1 {status} {}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\
             Connection: close\r\n{extra_headers}\r\n",
            reason(status),
            body.len()
        );
        stream.write_all([head.as_bytes(), body.as_bytes()].concat().as_slice())?;
        stream.flush()
    }

    fn route(&self, request: &Request, stream: &mut TcpStream, cors: &str) -> Result<Summary, Failure> {
        let json = "application/json";
        let mut reply = |status: u16, body: &str| {
            let _ = self.respond(stream, status, cors, json, body);
            Ok(Summary { status, note: String::new(), reply: None })
        };
        match (request.method.as_str(), request.path.as_str()) {
            ("OPTIONS", _) => reply(204, ""),
            ("GET", "/health") => reply(200, "{\"status\":\"ok\"}"),
            ("GET", "/v1/models") => reply(
                200,
                &format!(
                    "{{\"object\":\"list\",\"data\":[{{\"id\":{},\"object\":\"model\",\"created\":0,\"owned_by\":\"detra\"}}]}}",
                    quote(&self.config.name)
                ),
            ),
            ("POST", "/v1/chat/completions") => self.chat(request, stream, cors),
            (_, "/health" | "/v1/models" | "/v1/chat/completions") => Err((405, "method not allowed".into())),
            _ => Err((404, format!("unknown path {}", request.path))),
        }
    }

    /// Take the kept session that has consumed the longest prefix of `tokens`, or
    /// start a new one. A session that has consumed all of them is only of use if it
    /// still holds the logits that the first new token is chosen from.
    fn session_for(&self, tokens: &[u32], want_logits: bool) -> Session {
        let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        let usable = |(fed, session): &(Vec<u32>, Session)| {
            tokens.starts_with(fed) && (fed.len() < tokens.len() || !want_logits || !session.logits().is_empty())
        };
        let best = (0..kept.len()).filter(|&i| usable(&kept[i])).max_by_key(|&i| kept[i].0.len());
        match best {
            Some(i) => kept.remove(i).1,
            None => Session::new(Arc::clone(&self.config.model), self.config.threads),
        }
    }

    fn keep(&self, tokens: &[u32], session: Session) {
        let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        kept.retain(|(fed, _)| fed != tokens);
        if kept.len() >= KEPT_SESSIONS {
            kept.remove(0);
        }
        kept.push((tokens.to_vec(), session));
    }

    fn chat(&self, request: &Request, stream: &mut TcpStream, cors: &str) -> Result<Summary, Failure> {
        let model = &self.config.model;
        let chat = parse_chat(&request.body)?;
        let messages: Vec<(Role, &[u8])> = chat.messages.iter().map(|(role, text)| (*role, text.as_bytes())).collect();
        let mut tokens = chat_tokens(model, &self.tokenizer, &messages).map_err(|e| (400, e.0))?;
        let prompt_len = tokens.len();
        let params = GenParams { max_new_tokens: chat.max_tokens, sample: chat.sample, threads: self.config.threads };

        // The id names the request: the model, the prompt and everything that
        // influences the reply.
        let mut hasher = Sha256::new();
        hasher.update(&self.config.model_id);
        hasher.update_u32s(&tokens);
        for v in [chat.sample.temperature_q16 as u64, chat.sample.top_k as u64, chat.sample.top_p_q16 as u64] {
            hasher.update(&v.to_le_bytes());
        }
        hasher.update(&chat.sample.seed.to_le_bytes());
        hasher.update(&(chat.max_tokens.min(model.seq_len) as u64).to_le_bytes());
        for stop in &chat.stop {
            hasher.update(&(stop.len() as u64).to_le_bytes());
            hasher.update(stop.as_bytes());
        }
        let id = format!("chatcmpl-{}", &hex(&hasher.finalize())[..24]);
        let fingerprint = format!("detra-{}", &hex(&self.config.model_id)[..16]);
        let common = format!(
            "\"id\":\"{id}\",\"created\":0,\"model\":{},\"system_fingerprint\":\"{fingerprint}\"",
            quote(&self.config.name)
        );
        let chunk = |delta: &str, finish: &str| {
            format!(
                "{{{common},\"object\":\"chat.completion.chunk\",\"choices\":[{{\"index\":0,\"delta\":{delta},\"finish_reason\":{finish}}}]}}"
            )
        };

        let mut out = Stream { stream, chunked: request.chunked_ok, failed: false };
        if chat.stream {
            let encoding = if out.chunked { "Transfer-Encoding: chunked\r\n" } else { "" };
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n\
                 Connection: close\r\n{encoding}{cors}\r\n"
            );
            out.failed = out.stream.write_all(head.as_bytes()).is_err();
            out.event(&chunk("{\"role\":\"assistant\",\"content\":\"\"}", "null"));
        }

        let _busy = self.busy.lock().unwrap_or_else(|e| e.into_inner());
        let want_logits = chat.max_tokens > 0 && prompt_len < model.seq_len;
        let mut session = self.session_for(&tokens, want_logits);
        let reused = session.pos();
        let internal = |e: crate::Error| (500, e.0);
        // Earlier replies appear in the history without the generation prompt, so the
        // next turn shares the conversation up to here but not beyond.
        let generation = model.chat.as_ref().map_or(0, |c| c.generation.len());
        let messages_end = prompt_len - generation;
        if session.pos() < messages_end {
            feed_prompt(&mut session, &tokens[..messages_end], false, |_, _| {}).map_err(internal)?;
            self.keep(&tokens[..messages_end], session.fork());
        }
        feed_prompt(&mut session, &tokens, want_logits, |_, _| {}).map_err(internal)?;
        self.keep(&tokens, session.fork());

        // Text is sent once it can no longer turn out to be part of a stop string or
        // of an unfinished character.
        let holdback = chat.stop.iter().map(String::len).max().unwrap_or(1) - 1;
        let mut sent = 0;
        let mut stopped: Option<String> = None;
        let finish = sample_tokens(&mut session, &mut tokens, &params, |_, tokens| {
            let text = text_prefix(&self.tokenizer.decode(&tokens[prompt_len..]), false);
            if let Some(at) = chat.stop.iter().filter_map(|s| text.find(s.as_str())).min() {
                stopped = Some(text[..at].to_string());
                return false;
            }
            if chat.stream {
                let mut upto = text.len().saturating_sub(holdback);
                while !text.is_char_boundary(upto) {
                    upto -= 1;
                }
                if upto > sent {
                    out.event(&chunk(&format!("{{\"content\":{}}}", quote(&text[sent..upto])), "null"));
                    sent = upto;
                }
            }
            !out.failed
        })
        .map_err(internal)?;
        let text = match stopped {
            Some(text) => text,
            // A reply cut off by a limit may end inside a character; that character
            // is left out rather than shown as invalid.
            None => text_prefix(&self.tokenizer.decode(&tokens[prompt_len..]), finish == Finish::Stop),
        };
        let completion = tokens.len() - prompt_len;
        self.keep(&tokens, session);

        let finish_reason = match finish {
            Finish::Length | Finish::Context => "\"length\"",
            Finish::Stop | Finish::Cancelled => "\"stop\"",
        };
        let usage = format!(
            "\"usage\":{{\"prompt_tokens\":{prompt_len},\"completion_tokens\":{completion},\"total_tokens\":{}}}",
            prompt_len + completion
        );
        if chat.stream {
            if text.len() > sent {
                out.event(&chunk(&format!("{{\"content\":{}}}", quote(&text[sent..])), "null"));
            }
            out.event(&chunk("{}", finish_reason));
            if chat.include_usage {
                out.event(&format!("{{{common},\"object\":\"chat.completion.chunk\",\"choices\":[],{usage}}}"));
            }
            out.event("[DONE]");
            out.finish();
        } else {
            let body = format!(
                "{{{common},\"object\":\"chat.completion\",\"choices\":[{{\"index\":0,\"message\":{{\"role\":\"assistant\",\"content\":{}}},\"finish_reason\":{finish_reason}}}],{usage}}}",
                quote(&text)
            );
            let _ = self.respond(out.stream, 200, cors, "application/json", &body);
        }
        let gone = if out.failed { ", client disconnected" } else { "" };
        let reply = format!(
            "\"id\":\"{id}\",\"reply\":{},\"finish_reason\":{finish_reason},{usage},\"client_disconnected\":{}",
            quote(&text),
            out.failed
        );
        Ok(Summary {
            status: 200,
            note: format!("prompt {prompt_len} tokens ({reused} reused), completion {completion} tokens{gone}"),
            reply: Some(reply),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_prefix_is_stable() {
        let bytes = "a\u{e9}\u{1F642}b".as_bytes();
        let full = text_prefix(bytes, true);
        assert_eq!(full, "a\u{e9}\u{1F642}b");
        for cut in 0..=bytes.len() {
            let part = text_prefix(&bytes[..cut], false);
            assert!(full.starts_with(&part), "prefix of {cut} bytes");
            assert!(part.len() + 3 >= cut);
        }
        assert_eq!(text_prefix(&[b'a', 0xFF, b'b', 0xE2, 0x82], false), "a\u{FFFD}b");
        assert_eq!(text_prefix(&[b'a', 0xFF, b'b', 0xE2, 0x82], true), "a\u{FFFD}b\u{FFFD}");
        assert_eq!(text_prefix(&[0xE2, 0x28], false), "\u{FFFD}(");
    }

    #[test]
    fn recognises_local_origins() {
        for origin in ["http://localhost", "http://localhost:3000", "https://127.0.0.1:8443", "http://[::1]:80"] {
            assert!(is_local_origin(origin), "{origin}");
        }
        for origin in ["http://example.com", "http://localhost.example.com", "localhost", "http://localhost:x", "null"] {
            assert!(!is_local_origin(origin), "{origin}");
        }
    }

    #[test]
    fn parses_chat_requests() {
        let chat = parse_chat(
            br#"{"model":"x","messages":[{"role":"system","content":"s"},{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}],
                "temperature":0.30000000000000004,"top_p":0.9,"top_k":40,"seed":-1,"max_tokens":7,"stream":true,
                "stream_options":{"include_usage":true},"stop":["x",""],"unknown":[1,2]}"#,
        )
        .unwrap();
        assert_eq!(chat.messages, [(Role::System, "s".to_string()), (Role::User, "ab".to_string())]);
        assert_eq!(chat.sample, SampleParams { temperature_q16: 19_660, top_k: 40, top_p_q16: 58_982, seed: u64::MAX });
        assert_eq!((chat.max_tokens, chat.stream, chat.include_usage), (7, true, true));
        assert_eq!(chat.stop, ["x"]);
        let chat = parse_chat(br#"{"messages":[{"role":"user","content":null}],"top_p":0,"max_completion_tokens":null}"#).unwrap();
        assert_eq!((chat.sample.top_p_q16, chat.max_tokens, chat.stream), (1, usize::MAX, false));
        assert_eq!(chat.sample.temperature_q16, 0);
        for bad in [
            r#"[]"#,
            r#"{"messages":"hi"}"#,
            r#"{"messages":[{"content":"x"}]}"#,
            r#"{"messages":[{"role":"tool","content":"x"}]}"#,
            r#"{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"x"}}]}]}"#,
            r#"{"messages":[],"temperature":-1}"#,
            r#"{"messages":[],"temperature":1e-3}"#,
            r#"{"messages":[],"temperature":"hot"}"#,
            r#"{"messages":[],"top_p":1.5}"#,
            r#"{"messages":[],"max_tokens":-1}"#,
            r#"{"messages":[],"seed":1.5}"#,
            r#"{"messages":[],"n":2}"#,
            r#"{"messages":[],"stream":"yes"}"#,
            r#"{"messages":[],"stop":[1]}"#,
            r#"{"messages":[],"tools":[{"type":"function"}]}"#,
            r#"{"messages":[]"#,
        ] {
            assert_eq!(parse_chat(bad.as_bytes()).err().map(|e| e.0), Some(400), "{bad}");
        }
    }
}
