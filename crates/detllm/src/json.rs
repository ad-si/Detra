//! A small JSON reader and writer for the HTTP server. Numbers are kept as their
//! source text so that nothing here needs floating point.

use crate::{err, Result};

const MAX_DEPTH: usize = 64;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Value {
    Null,
    Bool(bool),
    /// The number exactly as written, e.g. `-1`, `0.70` or `1e-3`.
    Number(String),
    String(String),
    Array(Vec<Value>),
    /// Members in source order. Of members with the same name, the last one counts.
    Object(Vec<(String, Value)>),
}

impl Value {
    /// Member `name` of an object; `None` if absent, `null` or `self` is not an object.
    pub fn get(&self, name: &str) -> Option<&Value> {
        match self {
            Value::Object(members) => {
                members.iter().rev().find(|(k, _)| k == name).map(|(_, v)| v).filter(|v| **v != Value::Null)
            }
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }
}

/// Parse a complete JSON document.
pub fn parse(text: &[u8]) -> Result<Value> {
    let Ok(text) = core::str::from_utf8(text) else {
        return err("the JSON is not valid UTF-8");
    };
    let mut p = Parser { text, bytes: text.as_bytes(), at: 0 };
    let value = p.value(0)?;
    p.space();
    if p.at != p.bytes.len() {
        return p.fail("unexpected data after the JSON value");
    }
    Ok(value)
}

struct Parser<'a> {
    text: &'a str,
    bytes: &'a [u8],
    at: usize,
}

impl Parser<'_> {
    fn fail<T>(&self, what: &str) -> Result<T> {
        err(format!("invalid JSON at byte {}: {what}", self.at))
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.at).copied()
    }

    fn space(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.at += 1;
        }
    }

    fn literal(&mut self, word: &str, value: Value) -> Result<Value> {
        if self.bytes[self.at..].starts_with(word.as_bytes()) {
            self.at += word.len();
            Ok(value)
        } else {
            self.fail("unknown literal")
        }
    }

    fn value(&mut self, depth: usize) -> Result<Value> {
        if depth > MAX_DEPTH {
            return self.fail("nested too deeply");
        }
        self.space();
        match self.peek() {
            Some(b'n') => self.literal("null", Value::Null),
            Some(b't') => self.literal("true", Value::Bool(true)),
            Some(b'f') => self.literal("false", Value::Bool(false)),
            Some(b'"') => Ok(Value::String(self.string()?)),
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(b'[') => {
                self.at += 1;
                let mut items = Vec::new();
                self.space();
                if self.peek() == Some(b']') {
                    self.at += 1;
                    return Ok(Value::Array(items));
                }
                loop {
                    items.push(self.value(depth + 1)?);
                    self.space();
                    match self.peek() {
                        Some(b',') => self.at += 1,
                        Some(b']') => {
                            self.at += 1;
                            return Ok(Value::Array(items));
                        }
                        _ => return self.fail("expected `,` or `]`"),
                    }
                }
            }
            Some(b'{') => {
                self.at += 1;
                let mut members = Vec::new();
                self.space();
                if self.peek() == Some(b'}') {
                    self.at += 1;
                    return Ok(Value::Object(members));
                }
                loop {
                    self.space();
                    if self.peek() != Some(b'"') {
                        return self.fail("expected a member name");
                    }
                    let name = self.string()?;
                    self.space();
                    if self.peek() != Some(b':') {
                        return self.fail("expected `:`");
                    }
                    self.at += 1;
                    members.push((name, self.value(depth + 1)?));
                    self.space();
                    match self.peek() {
                        Some(b',') => self.at += 1,
                        Some(b'}') => {
                            self.at += 1;
                            return Ok(Value::Object(members));
                        }
                        _ => return self.fail("expected `,` or `}`"),
                    }
                }
            }
            _ => self.fail("expected a value"),
        }
    }

    fn digits(&mut self) -> Result<()> {
        let start = self.at;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.at += 1;
        }
        if self.at == start {
            return self.fail("expected a digit");
        }
        Ok(())
    }

    fn number(&mut self) -> Result<Value> {
        let start = self.at;
        if self.peek() == Some(b'-') {
            self.at += 1;
        }
        if self.peek() == Some(b'0') {
            self.at += 1;
        } else {
            self.digits()?;
        }
        if self.peek() == Some(b'.') {
            self.at += 1;
            self.digits()?;
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.at += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.at += 1;
            }
            self.digits()?;
        }
        Ok(Value::Number(self.text[start..self.at].to_string()))
    }

    fn hex4(&mut self) -> Result<u32> {
        let Some(digits) = self.bytes.get(self.at..self.at + 4) else {
            return self.fail("truncated \\u escape");
        };
        let mut v = 0;
        for &d in digits {
            let Some(d) = (d as char).to_digit(16) else {
                return self.fail("invalid \\u escape");
            };
            v = v * 16 + d;
        }
        self.at += 4;
        Ok(v)
    }

    /// A string; `self.at` is at the opening quote. An escaped surrogate without its
    /// partner becomes U+FFFD.
    fn string(&mut self) -> Result<String> {
        self.at += 1;
        let mut out = String::new();
        loop {
            let start = self.at;
            while !matches!(self.peek(), None | Some(b'"' | b'\\' | 0..=0x1F)) {
                self.at += 1;
            }
            // The run ends at an ASCII byte or at the end, so it is whole characters.
            out.push_str(&self.text[start..self.at]);
            match self.peek() {
                Some(b'"') => {
                    self.at += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.at += 1;
                    let escape = self.peek();
                    self.at += 1;
                    match escape {
                        Some(b'"') => out.push('"'),
                        Some(b'\\') => out.push('\\'),
                        Some(b'/') => out.push('/'),
                        Some(b'b') => out.push('\u{8}'),
                        Some(b'f') => out.push('\u{c}'),
                        Some(b'n') => out.push('\n'),
                        Some(b'r') => out.push('\r'),
                        Some(b't') => out.push('\t'),
                        Some(b'u') => {
                            let mut code = self.hex4()?;
                            if (0xD800..0xDC00).contains(&code) && self.bytes[self.at..].starts_with(b"\\u") {
                                let rewind = self.at;
                                self.at += 2;
                                let low = self.hex4()?;
                                if (0xDC00..0xE000).contains(&low) {
                                    code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
                                } else {
                                    self.at = rewind;
                                }
                            }
                            out.push(char::from_u32(code).unwrap_or('\u{FFFD}'));
                        }
                        _ => {
                            self.at -= 1;
                            return self.fail("invalid escape");
                        }
                    }
                }
                Some(_) => return self.fail("control character in a string"),
                None => return self.fail("unterminated string"),
            }
        }
    }
}

/// `text` as a JSON string literal, quotes included.
pub fn quote(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_documents() {
        let v = parse(br#" {"a": [1, -2.50, 1e-3, true, false, null], "b": {"c": "x"}, "a": "last"} "#).unwrap();
        assert_eq!(v.get("a"), Some(&Value::String("last".into())));
        assert_eq!(v.get("b").and_then(|b| b.get("c")).and_then(Value::as_str), Some("x"));
        assert_eq!(v.get("missing"), None);
        let Value::Object(members) = &v else { panic!() };
        let numbers = ["1", "-2.50", "1e-3"].map(|n| Value::Number(n.into()));
        let rest = [Value::Bool(true), Value::Bool(false), Value::Null];
        assert_eq!(members[0].1, Value::Array(numbers.into_iter().chain(rest).collect()));
        assert_eq!(parse(b"[]").unwrap(), Value::Array(vec![]));
        assert_eq!(parse(b"{}").unwrap(), Value::Object(vec![]));
        assert_eq!(parse(br#"{"a": null}"#).unwrap().get("a"), None);
    }

    #[test]
    fn parses_strings() {
        let s = |text: &str| parse(text.as_bytes()).unwrap();
        assert_eq!(s(r#""a\"\\\/\b\f\n\r\t\u00e9\u20AC""#), Value::String("a\"\\/\u{8}\u{c}\n\r\t\u{e9}\u{20ac}".into()));
        assert_eq!(s(r#""\ud83d\ude42 \ud83d x \ude42 \ud83d\u0041""#), Value::String("\u{1F642} \u{FFFD} x \u{FFFD} \u{FFFD}A".into()));
        assert_eq!(s("\"\u{1F642}\u{e9}\""), Value::String("\u{1F642}\u{e9}".into()));
    }

    #[test]
    fn rejects_invalid_documents() {
        for bad in [
            "", "{", "[1,]", "{\"a\"}", "{\"a\":}", "{a:1}", "01", "1.", "-", "1e", "nul", "\"a", "\"\\x\"", "\"\\u12\"",
            "\"a\nb\"", "1 2", "[1 2]", "{\"a\":1,}", "tru", "+1", ".5",
        ] {
            assert!(parse(bad.as_bytes()).is_err(), "{bad:?} should be rejected");
        }
        assert!(parse(&[b'"', 0xFF, b'"']).is_err());
        assert!(parse("[".repeat(100).as_bytes()).is_err());
        let deep = format!("{}{}", "[".repeat(60), "]".repeat(60));
        assert!(parse(deep.as_bytes()).is_ok());
    }

    #[test]
    fn quotes_strings() {
        let text = "a\"b\\c\n\r\t\u{1}\u{1F642}\u{e9}";
        assert_eq!(quote(text), "\"a\\\"b\\\\c\\n\\r\\t\\u0001\u{1F642}\u{e9}\"");
        assert_eq!(parse(quote(text).as_bytes()).unwrap(), Value::String(text.into()));
    }
}
