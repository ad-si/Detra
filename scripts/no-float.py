"""Fail if Rust sources mention a floating-point type or contain a float literal.

Usage: uv run --no-project scripts/no-float.py <dir>...

The sources are tokenized well enough to skip comments, string and character
literals (including raw and multi-line strings), so text such as "HTTP/1.1" or
"127.0.0.1" inside a string does not count.
"""

import re
import sys
from pathlib import Path

IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
NUMBER = re.compile(
    r"0[xob][0-9A-Fa-f_]+(?:[iu](?:8|16|32|64|128|size))?"
    r"|(?P<int>[0-9][0-9_]*)(?P<frac>\.[0-9][0-9_]*)?(?P<exp>[eE][+-]?[0-9_]+)?(?P<suffix>[A-Za-z][A-Za-z0-9_]*)?"
    r"(?P<dot>\.(?![.A-Za-z_0-9]))?"
)
CHAR = re.compile(r"'(?:\\(?:x[0-9A-Fa-f]{2}|u\{[0-9A-Fa-f]+\}|.)|[^\\'])'", re.S)
RAW = re.compile(r'b?r(#*)"')


def findings(text):
    """(line, token) for every floating-point type or literal in Rust source `text`."""
    i, n, line = 0, len(text), 1
    out = []
    previous = ""
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            end = text.find("\n", i)
            i = n if end < 0 else end
            continue
        if text.startswith("/*", i):
            depth, i = 1, i + 2
            while i < n and depth:
                if text.startswith("/*", i):
                    depth, i = depth + 1, i + 2
                elif text.startswith("*/", i):
                    depth, i = depth - 1, i + 2
                else:
                    line += text[i] == "\n"
                    i += 1
            continue
        raw = RAW.match(text, i) if c in "br" else None
        if raw:
            close = '"' + raw.group(1)
            end = text.find(close, raw.end())
            end = n if end < 0 else end + len(close)
            line += text.count("\n", i, end)
            i, previous = end, '"'
            continue
        if c == '"' or text.startswith('b"', i):
            i += 1 if c == '"' else 2
            while i < n and text[i] != '"':
                if text[i] == "\\":
                    i += 1
                line += i < n and text[i] == "\n"
                i += 1
            i, previous = i + 1, '"'
            continue
        if c == "'":
            m = CHAR.match(text, i)
            if m:
                i, previous = m.end(), "'"
                continue
            i += 1  # a lifetime
            continue
        if c.isalpha() or c == "_":
            m = IDENT.match(text, i)
            word = m.group()
            if word in ("f32", "f64"):
                out.append((line, word))
            i, previous = m.end(), word
            continue
        if c.isdigit():
            m = NUMBER.match(text, i)
            # After a `.` a number is a tuple field (`pair.0.1`), not a literal.
            if previous != "." and (m.group("frac") or m.group("exp") or m.group("dot")
                                    or (m.group("suffix") or "") in ("f32", "f64")):
                out.append((line, m.group()))
            i, previous = m.end(), m.group()
            continue
        line += c == "\n"
        if not c.isspace():
            previous = c
        i += 1
    return out


def main():
    status = 0
    for root in sys.argv[1:] or ["crates/detra"]:
        for path in sorted(Path(root).rglob("*.rs")):
            for line, token in findings(path.read_text()):
                print(f"{path}:{line}: floating point: {token}", file=sys.stderr)
                status = 1
    sys.exit(status)


if __name__ == "__main__":
    main()
