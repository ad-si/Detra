//! Byte-level tokenizer. It operates on raw prompt bytes and uses no Unicode tables,
//! so it cannot depend on the OS or on a library version.
//!
//! Encoding of a prompt `text`:
//! 1. Start with `[bos]`. If `text` is empty, stop.
//! 2. Append the id of the regular token `" "`.
//! 3. Split `text` into units: one byte followed by up to three bytes of the form
//!    `10xxxxxx`. A unit that is a regular token becomes that token; otherwise each of
//!    its bytes becomes the byte token `byte_base + byte`.
//! 4. Repeatedly find the adjacent pair of regular tokens whose concatenated bytes form
//!    a regular token with the highest rank (leftmost pair on ties) and replace the
//!    pair with that token. Stop when no pair qualifies.
//!
//! If several regular tokens have identical bytes, the lowest id is used.

use crate::model::Model;
use crate::{err, Result};
use std::collections::BTreeMap;

pub struct Tokenizer<'m> {
    model: &'m Model,
    lookup: BTreeMap<&'m [u8], u32>,
}

impl<'m> Tokenizer<'m> {
    pub fn new(model: &'m Model) -> Tokenizer<'m> {
        let mut lookup = BTreeMap::new();
        for (id, token) in model.tokens.iter().enumerate().skip(model.first_regular() as usize) {
            lookup.entry(token.as_slice()).or_insert(id as u32);
        }
        Tokenizer { model, lookup }
    }

    pub fn encode(&self, text: &[u8]) -> Result<Vec<u32>> {
        let m = self.model;
        let first_regular = m.first_regular();
        let mut tokens = vec![m.bos_id];
        if text.is_empty() {
            return Ok(tokens);
        }
        match self.lookup.get(b" ".as_slice()) {
            Some(&id) => tokens.push(id),
            None => return err("vocabulary has no \" \" token"),
        }
        let mut i = 0;
        while i < text.len() {
            let mut j = i + 1;
            while j < text.len() && text[j] & 0xC0 == 0x80 && j - i < 4 {
                j += 1;
            }
            let unit = &text[i..j];
            match self.lookup.get(unit) {
                Some(&id) => tokens.push(id),
                None => tokens.extend(unit.iter().map(|&b| m.byte_base + b as u32)),
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
                if let Some(&id) = self.lookup.get(cat.as_slice()) {
                    let rank = m.ranks[id as usize];
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

    /// Concatenate token bytes. Special tokens produce nothing, and a single leading
    /// space is dropped from a token that directly follows `bos`.
    pub fn decode(&self, tokens: &[u32]) -> Vec<u8> {
        let m = self.model;
        let mut out = Vec::new();
        let mut prev_is_bos = false;
        for &t in tokens {
            if t >= m.byte_base {
                let mut piece = m.tokens[t as usize].as_slice();
                if prev_is_bos && piece.first() == Some(&b' ') {
                    piece = &piece[1..];
                }
                out.extend_from_slice(piece);
            }
            prev_is_bos = t == m.bos_id;
        }
        out
    }
}
