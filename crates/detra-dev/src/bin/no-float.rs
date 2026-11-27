//! Fail if Rust sources mention a floating-point type or contain a float literal.
//!
//! Usage: no-float <dir>...
//!
//! The sources are tokenized well enough to skip comments, string and character
//! literals (including raw and multi-line strings), so text such as "HTTP/1.1" or
//! "127.0.0.1" inside a string does not count.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn is_ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

fn is_ident(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Length of the UTF-8 character starting with byte `b`.
fn char_len(b: u8) -> usize {
    match b {
        0xF0..=0xFF => 4,
        0xE0..=0xEF => 3,
        0xC0..=0xDF => 2,
        _ => 1,
    }
}

/// End of the character literal starting at the `'` at `i`, if there is one there
/// rather than a lifetime.
fn char_literal(t: &[u8], i: usize) -> Option<usize> {
    let closes = |j: usize| (t.get(j) == Some(&b'\'')).then_some(j + 1);
    let body = i + 1;
    match *t.get(body)? {
        b'\\' => {
            let e = body + 1;
            let hex = |j: usize| t.get(j).is_some_and(u8::is_ascii_hexdigit);
            let mut candidates = Vec::new();
            if t.get(e) == Some(&b'x') && hex(e + 1) && hex(e + 2) {
                candidates.push(e + 3);
            }
            if t.get(e) == Some(&b'u') && t.get(e + 1) == Some(&b'{') {
                let mut j = e + 2;
                while hex(j) {
                    j += 1;
                }
                if j > e + 2 && t.get(j) == Some(&b'}') {
                    candidates.push(j + 1);
                }
            }
            if let Some(&b) = t.get(e) {
                candidates.push(e + char_len(b));
            }
            candidates.into_iter().find_map(closes)
        }
        b'\'' => None,
        b => closes(body + char_len(b)),
    }
}

struct Number {
    end: usize,
    float: bool,
}

/// The numeric literal starting at the digit at `i`.
fn number(t: &[u8], i: usize) -> Number {
    let at = |j: usize| t.get(j).copied().unwrap_or(0);
    let digits = |mut j: usize, ok: fn(u8) -> bool| {
        while ok(at(j)) || at(j) == b'_' {
            j += 1;
        }
        j
    };
    if at(i) == b'0' && matches!(at(i + 1), b'x' | b'o' | b'b') && (at(i + 2).is_ascii_hexdigit() || at(i + 2) == b'_') {
        let mut j = digits(i + 2, |b| b.is_ascii_hexdigit());
        for suffix in ["i8", "i16", "i32", "i64", "i128", "isize", "u8", "u16", "u32", "u64", "u128", "usize"] {
            if t[j..].starts_with(suffix.as_bytes()) {
                j += suffix.len();
                break;
            }
        }
        return Number { end: j, float: false };
    }
    let mut j = digits(i, |b| b.is_ascii_digit());
    let mut float = false;
    if at(j) == b'.' && at(j + 1).is_ascii_digit() {
        j = digits(j + 1, |b| b.is_ascii_digit());
        float = true;
    }
    if matches!(at(j), b'e' | b'E') {
        let k = if matches!(at(j + 1), b'+' | b'-') { j + 2 } else { j + 1 };
        if at(k).is_ascii_digit() || at(k) == b'_' {
            j = digits(k, |b| b.is_ascii_digit());
            float = true;
        }
    }
    if at(j).is_ascii_alphabetic() {
        let start = j;
        while is_ident(at(j)) {
            j += 1;
        }
        float |= matches!(&t[start..j], b"f32" | b"f64");
    }
    if at(j) == b'.' && !(at(j + 1) == b'.' || is_ident(at(j + 1))) {
        j += 1;
        float = true;
    }
    Number { end: j, float }
}

/// (line, token) for every floating-point type or literal in Rust source `t`.
fn findings(t: &[u8]) -> Vec<(usize, String)> {
    let (n, mut i, mut line) = (t.len(), 0, 1);
    let mut out = Vec::new();
    // The previous token: after a `.` a number is a tuple field (`pair.0.1`).
    let mut after_dot = false;
    while i < n {
        let c = t[i];
        if t[i..].starts_with(b"//") {
            i = t[i..].iter().position(|&b| b == b'\n').map_or(n, |p| i + p);
            continue;
        }
        if t[i..].starts_with(b"/*") {
            let mut depth = 1;
            i += 2;
            while i < n && depth > 0 {
                if t[i..].starts_with(b"/*") {
                    depth += 1;
                    i += 2;
                } else if t[i..].starts_with(b"*/") {
                    depth -= 1;
                    i += 2;
                } else {
                    line += (t[i] == b'\n') as usize;
                    i += 1;
                }
            }
            continue;
        }
        if c == b'b' || c == b'r' {
            let mut j = i + (c == b'b') as usize;
            if t.get(j) == Some(&b'r') {
                j += 1;
                let hashes = t[j..].iter().take_while(|&&b| b == b'#').count();
                if t.get(j + hashes) == Some(&b'"') {
                    let mut close = vec![b'"'];
                    close.extend(std::iter::repeat_n(b'#', hashes));
                    let body = j + hashes + 1;
                    let end = t[body..].windows(close.len()).position(|w| w == close).map_or(n, |p| body + p + close.len());
                    line += t[i..end].iter().filter(|&&b| b == b'\n').count();
                    i = end;
                    after_dot = false;
                    continue;
                }
            }
        }
        if c == b'"' || t[i..].starts_with(b"b\"") {
            i += if c == b'"' { 1 } else { 2 };
            while i < n && t[i] != b'"' {
                if t[i] == b'\\' {
                    i += 1;
                }
                line += (i < n && t[i] == b'\n') as usize;
                i += 1;
            }
            i += 1;
            after_dot = false;
            continue;
        }
        if c == b'\'' {
            i = char_literal(t, i).unwrap_or(i + 1);
            after_dot = false;
            continue;
        }
        if is_ident_start(c) {
            let start = i;
            while i < n && is_ident(t[i]) {
                i += 1;
            }
            let word = &t[start..i];
            if word == b"f32" || word == b"f64" {
                out.push((line, String::from_utf8_lossy(word).into_owned()));
            }
            after_dot = false;
            continue;
        }
        if c.is_ascii_digit() {
            let m = number(t, i);
            if !after_dot && m.float {
                out.push((line, String::from_utf8_lossy(&t[i..m.end]).into_owned()));
            }
            i = m.end;
            after_dot = false;
            continue;
        }
        line += (c == b'\n') as usize;
        if !c.is_ascii_whitespace() {
            after_dot = c == b'.';
        }
        i += 1;
    }
    out
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            rust_files(&path, out)?;
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    let roots: Vec<String> = std::env::args().skip(1).collect();
    let roots = if roots.is_empty() { vec!["crates/detra".to_string()] } else { roots };
    let mut status = ExitCode::SUCCESS;
    for root in roots {
        let mut files = Vec::new();
        if let Err(e) = rust_files(Path::new(&root), &mut files) {
            eprintln!("{root}: {e}");
            return ExitCode::FAILURE;
        }
        files.sort();
        for path in files {
            let text = match std::fs::read(&path) {
                Ok(text) => text,
                Err(e) => {
                    eprintln!("{}: {e}", path.display());
                    return ExitCode::FAILURE;
                }
            };
            for (line, token) in findings(&text) {
                eprintln!("{}:{line}: floating point: {token}", path.display());
                status = ExitCode::FAILURE;
            }
        }
    }
    status
}

#[cfg(test)]
mod tests {
    use super::findings;

    fn tokens(src: &str) -> Vec<String> {
        findings(src.as_bytes()).into_iter().map(|(_, t)| t).collect()
    }

    #[test]
    fn finds_float_types_and_literals() {
        assert_eq!(tokens("let x: f32 = 1.5;"), ["f32", "1.5"]);
        assert_eq!(tokens("let y = 2e10 + 3E-2 + 1f64 + 7.;"), ["2e10", "3E-2", "1f64", "7."]);
        assert_eq!(tokens("x as f64"), ["f64"]);
        let found = findings(b"// 1.5\nlet a = 1;\nlet b = 0.5;");
        assert_eq!(found, [(3, "0.5".to_string())]);
    }

    #[test]
    fn ignores_integers_strings_comments_and_fields() {
        let src = r##"
            let a = 0x1f32 + 0b1010u8 + 1_000 + 3i64 + 1usize;
            let b = pair.0.1 + t.0;
            let c = 1..2; let d = 1.max(2); let e = 1..=2;
            let s = "1.5 f32 HTTP/1.1"; let r = r#"2.5 "f64""#; let bs = b"3.5";
            let ch = '.'; let q = '\''; let u = '\u{2e}'; let x = '\x2e';
            fn f<'a>(v: &'a str) -> &'a str { v }
            /* 1.5 /* nested f64 */ still a comment 2.5 */
            let ident_f32 = 1; let e10 = 2;
        "##;
        assert!(tokens(src).is_empty(), "{:?}", tokens(src));
    }

    #[test]
    fn counts_lines_through_strings_and_comments() {
        let src = "let s = \"a\nb\";\n/* x\ny */\nlet r = r\"c\nd\";\nlet f = 1.0;";
        assert_eq!(findings(src.as_bytes()), [(7, "1.0".to_string())]);
    }
}
