//! Tokenizers. Both kinds work on raw prompt bytes and take every table they need from
//! the model file, so they cannot depend on the OS or on a library's Unicode version.
//! No Unicode normalisation is applied: the prompt bytes are used as given.
//!
//! # Kind 0: scored merges
//!
//! 1. If `text` is empty the result is empty. Otherwise start with the regular token `" "`.
//! 2. Split `text` into units: one byte followed by up to three bytes of the form
//!    `10xxxxxx`. A unit that is a regular token becomes that token; otherwise each of
//!    its bytes becomes the byte token `byte_base + byte`.
//! 3. Repeatedly find the adjacent pair of regular tokens whose concatenated bytes form
//!    a regular token with the highest rank (leftmost pair on ties) and replace the
//!    pair with that token. Stop when no pair qualifies.
//!
//! Regular tokens are those with id `>= byte_base + 256`; of several with identical
//! bytes the lowest id is used.
//!
//! # Kind 1: byte-level BPE
//!
//! 1. Decode `text` into characters. A well-formed UTF-8 sequence is one character;
//!    any other byte is a character of its own without a class.
//! 2. Split into pieces. At each position the first matching rule wins (L = letter,
//!    M = mark, N = number, S = whitespace or CR/LF, NL = CR or LF):
//!    1. `'` followed by one of `s t m d re ve ll` (ASCII, either case);
//!    2. optionally one character that is not NL, L or N, then one or more L/M;
//!    3. a single N;
//!    4. optionally one space, then one or more characters that are not S, L, M or N,
//!       then any number of NL;
//!    5. a run of S up to and including its last NL;
//!    6. a run of S, without its last character if a non-S character follows and the
//!       run is longer than one;
//!    7. a run of S.
//! 3. Each piece starts as its bytes' tokens. Repeatedly take the adjacent pair with
//!    the lowest merge index and replace every occurrence, left to right, by the merge
//!    result. Stop when no adjacent pair has a merge.

use crate::model::{Model, Vocab, CLASS_LETTER, CLASS_MARK, CLASS_NUMBER, CLASS_SPACE};
use crate::{err, Result};
use std::collections::BTreeMap;

enum Kind<'m> {
    Scored { byte_base: u32, ranks: &'m [u32], lookup: BTreeMap<&'m [u8], u32>, space: Option<u32> },
    Bpe {
        special: &'m [bool],
        byte_tokens: &'m [u32],
        classes: &'m [[u32; 3]],
        /// (left, right) -> (merge index, result)
        merges: BTreeMap<(u32, u32), (u32, u32)>,
    },
}

pub struct Tokenizer<'m> {
    model: &'m Model,
    kind: Kind<'m>,
}

/// One decoded character: its class and the byte offset where it starts.
#[derive(Clone, Copy)]
struct Char {
    cp: u32,
    class: u32,
    start: usize,
}

/// Length and code point of the well-formed UTF-8 sequence at the start of `b`, if any.
fn utf8_char(b: &[u8]) -> Option<(usize, u32)> {
    let cont = |i: usize| b.get(i).filter(|&&c| c & 0xC0 == 0x80).map(|&c| (c & 0x3F) as u32);
    let b0 = *b.first()? as u32;
    match b0 {
        0x00..=0x7F => Some((1, b0)),
        0xC2..=0xDF => Some((2, (b0 & 0x1F) << 6 | cont(1)?)),
        0xE0..=0xEF => {
            let cp = (b0 & 0x0F) << 12 | cont(1)? << 6 | cont(2)?;
            (cp >= 0x800 && !(0xD800..=0xDFFF).contains(&cp)).then_some((3, cp))
        }
        0xF0..=0xF4 => {
            let cp = (b0 & 0x07) << 18 | cont(1)? << 12 | cont(2)? << 6 | cont(3)?;
            (0x10000..=0x10FFFF).contains(&cp).then_some((4, cp))
        }
        _ => None,
    }
}

fn class_of(classes: &[[u32; 3]], cp: u32) -> u32 {
    let i = classes.partition_point(|r| r[1] < cp);
    match classes.get(i) {
        Some(&[first, _, class]) if first <= cp => class,
        _ => 0,
    }
}

const CR: u32 = 0x0D;
const LF: u32 = 0x0A;

/// Index one past the end of the piece that starts at character `i` (rules of kind 1).
fn piece_end(chars: &[Char], i: usize) -> usize {
    let n = chars.len();
    let cp = |k: usize| chars[k].cp;
    let is_nl = |k: usize| k < n && (cp(k) == CR || cp(k) == LF);
    let is_space = |k: usize| k < n && (chars[k].class == CLASS_SPACE || is_nl(k));
    let is_lm = |k: usize| k < n && !is_nl(k) && matches!(chars[k].class, CLASS_LETTER | CLASS_MARK);
    let is_ln = |k: usize| k < n && !is_nl(k) && matches!(chars[k].class, CLASS_LETTER | CLASS_NUMBER);
    let is_number = |k: usize| k < n && !is_nl(k) && chars[k].class == CLASS_NUMBER;
    let is_other = |k: usize| k < n && !is_space(k) && !is_lm(k) && !is_number(k);
    let ascii_lower = |k: usize| if k < n && cp(k) < 0x80 { cp(k) as u8 | 0x20 } else { 0 };

    // Rule 1: contractions.
    if cp(i) == '\'' as u32 {
        let (a, b) = (ascii_lower(i + 1), ascii_lower(i + 2));
        if matches!(a, b's' | b't') {
            return i + 2;
        }
        if matches!((a, b), (b'r', b'e') | (b'v', b'e')) {
            return i + 3;
        }
        if a == b'm' {
            return i + 2;
        }
        if (a, b) == (b'l', b'l') {
            return i + 3;
        }
        if a == b'd' {
            return i + 2;
        }
    }
    // Rule 2: letters, optionally with one leading character.
    let start = if !is_nl(i) && !is_ln(i) && is_lm(i + 1) {
        Some(i + 2)
    } else if is_lm(i) {
        Some(i + 1)
    } else {
        None
    };
    if let Some(mut j) = start {
        while is_lm(j) {
            j += 1;
        }
        return j;
    }
    // Rule 3: a single number character.
    if is_number(i) {
        return i + 1;
    }
    // Rule 4: other characters, optionally after one space, with trailing newlines.
    let start = if cp(i) == ' ' as u32 && is_other(i + 1) {
        Some(i + 1)
    } else if is_other(i) {
        Some(i)
    } else {
        None
    };
    if let Some(mut j) = start {
        while is_other(j) {
            j += 1;
        }
        while is_nl(j) {
            j += 1;
        }
        return j;
    }
    // Rules 5 to 7: whitespace.
    let mut end = i;
    let mut last_nl = None;
    while is_space(end) {
        if is_nl(end) {
            last_nl = Some(end);
        }
        end += 1;
    }
    if let Some(l) = last_nl {
        return l + 1;
    }
    if end < n && end - i >= 2 {
        return end - 1;
    }
    end.max(i + 1)
}

impl<'m> Tokenizer<'m> {
    pub fn new(model: &'m Model) -> Tokenizer<'m> {
        let kind = match &model.vocab {
            Vocab::Scored { byte_base, ranks } => {
                let mut lookup = BTreeMap::new();
                let first_regular = *byte_base as usize + 256;
                for (id, token) in model.tokens.iter().enumerate().skip(first_regular) {
                    lookup.entry(token.as_slice()).or_insert(id as u32);
                }
                let space = lookup.get(b" ".as_slice()).copied();
                Kind::Scored { byte_base: *byte_base, ranks, lookup, space }
            }
            Vocab::Bpe { special, merges, byte_tokens, classes } => {
                let mut map = BTreeMap::new();
                for (index, &[left, right, result]) in merges.iter().enumerate() {
                    map.entry((left, right)).or_insert((index as u32, result));
                }
                Kind::Bpe { special, byte_tokens, classes, merges: map }
            }
        };
        Tokenizer { model, kind }
    }

    /// `text` without leading and trailing whitespace. Kind 0: the ASCII characters
    /// space, tab, LF, VT, FF and CR. Kind 1: characters of the whitespace class and
    /// CR/LF, decoded as for encoding.
    pub fn trim<'a>(&self, text: &'a [u8]) -> &'a [u8] {
        match &self.kind {
            Kind::Scored { .. } => {
                let space = |b: &u8| matches!(b, b' ' | 0x09..=0x0D);
                let start = text.iter().position(|b| !space(b)).unwrap_or(text.len());
                let end = text.iter().rposition(|b| !space(b)).map_or(start, |i| i + 1);
                &text[start..end]
            }
            Kind::Bpe { classes, .. } => {
                let (mut start, mut end, mut seen, mut i) = (text.len(), text.len(), false, 0);
                while i < text.len() {
                    let (len, space) = match utf8_char(&text[i..]) {
                        Some((len, cp)) => (len, cp == CR || cp == LF || class_of(classes, cp) == CLASS_SPACE),
                        None => (1, false),
                    };
                    if !space {
                        if !seen {
                            start = i;
                            seen = true;
                        }
                        end = i + len;
                    }
                    i += len;
                }
                &text[start..end]
            }
        }
    }

    /// Token ids of `text`, without the model's prompt prefix and suffix.
    pub fn encode(&self, text: &[u8]) -> Result<Vec<u32>> {
        match &self.kind {
            Kind::Scored { byte_base, ranks, lookup, space } => {
                self.encode_scored(text, *byte_base, ranks, lookup, *space)
            }
            Kind::Bpe { byte_tokens, classes, merges, .. } => {
                Ok(encode_bpe(text, byte_tokens, classes, merges))
            }
        }
    }

    fn encode_scored(
        &self,
        text: &[u8],
        byte_base: u32,
        ranks: &[u32],
        lookup: &BTreeMap<&[u8], u32>,
        space: Option<u32>,
    ) -> Result<Vec<u32>> {
        let m = self.model;
        let first_regular = byte_base + 256;
        let mut tokens = Vec::new();
        if text.is_empty() {
            return Ok(tokens);
        }
        match space {
            Some(id) => tokens.push(id),
            None => return err("vocabulary has no \" \" token"),
        }
        let mut i = 0;
        while i < text.len() {
            let mut j = i + 1;
            while j < text.len() && text[j] & 0xC0 == 0x80 && j - i < 4 {
                j += 1;
            }
            let unit = &text[i..j];
            match lookup.get(unit) {
                Some(&id) => tokens.push(id),
                None => tokens.extend(unit.iter().map(|&b| byte_base + b as u32)),
            }
            i = j;
        }
        let mut cat: Vec<u8> = Vec::new();
        loop {
            let mut best: Option<(u32, u32, usize)> = None;
            for i in 0..tokens.len().saturating_sub(1) {
                let (a, b) = (tokens[i], tokens[i + 1]);
                if a < first_regular || b < first_regular {
                    continue;
                }
                cat.clear();
                cat.extend_from_slice(&m.tokens[a as usize]);
                cat.extend_from_slice(&m.tokens[b as usize]);
                if let Some(&id) = lookup.get(cat.as_slice()) {
                    let rank = ranks[id as usize];
                    if best.is_none_or(|(best_rank, _, _)| rank > best_rank) {
                        best = Some((rank, id, i));
                    }
                }
            }
            let Some((_, id, i)) = best else { break };
            tokens[i] = id;
            tokens.remove(i + 1);
        }
        Ok(tokens)
    }

    /// Concatenate token bytes. Special tokens produce nothing. For kind 0, one leading
    /// space is dropped from the output.
    pub fn decode(&self, tokens: &[u32]) -> Vec<u8> {
        let m = self.model;
        let mut out = Vec::new();
        match &self.kind {
            Kind::Scored { byte_base, .. } => {
                for &t in tokens.iter().filter(|&&t| t >= *byte_base) {
                    out.extend_from_slice(&m.tokens[t as usize]);
                }
                if out.first() == Some(&b' ') {
                    out.remove(0);
                }
            }
            Kind::Bpe { special, .. } => {
                for &t in tokens.iter().filter(|&&t| !special[t as usize]) {
                    out.extend_from_slice(&m.tokens[t as usize]);
                }
            }
        }
        out
    }
}

fn encode_bpe(
    text: &[u8],
    byte_tokens: &[u32],
    classes: &[[u32; 3]],
    merges: &BTreeMap<(u32, u32), (u32, u32)>,
) -> Vec<u32> {
    let mut chars = Vec::new();
    let mut i = 0;
    while i < text.len() {
        let (len, cp, class) = match utf8_char(&text[i..]) {
            Some((len, cp)) => (len, cp, class_of(classes, cp)),
            None => (1, u32::MAX, 0),
        };
        chars.push(Char { cp, class, start: i });
        i += len;
    }
    let mut tokens = Vec::new();
    let mut piece: Vec<u32> = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let j = piece_end(&chars, i);
        let end = chars.get(j).map_or(text.len(), |c| c.start);
        piece.clear();
        piece.extend(text[chars[i].start..end].iter().map(|&b| byte_tokens[b as usize]));
        loop {
            let best = piece.windows(2).filter_map(|w| merges.get(&(w[0], w[1]))).min();
            let Some(&(index, result)) = best else { break };
            let mut merged = Vec::with_capacity(piece.len());
            let mut k = 0;
            while k < piece.len() {
                if k + 1 < piece.len() && merges.get(&(piece[k], piece[k + 1])).map(|m| m.0) == Some(index) {
                    merged.push(result);
                    k += 2;
                } else {
                    merged.push(piece[k]);
                    k += 1;
                }
            }
            piece = merged;
        }
        tokens.extend_from_slice(&piece);
        i = j;
    }
    tokens
}
