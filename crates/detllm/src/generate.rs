//! Generation loop and the run digest used to compare machines.

use crate::engine::Session;
use crate::model::Model;
use crate::sample::{sample, Rng, SampleParams};
use crate::sha256::Sha256;
use crate::tokenizer::Tokenizer;
use crate::{err, Result};
use std::sync::Arc;

/// Number of prompt positions processed per forward call. Affects speed only.
const PROMPT_BLOCK: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GenParams {
    pub max_new_tokens: usize,
    pub sample: SampleParams,
    /// Number of threads. Affects speed only.
    pub threads: usize,
}

/// What a forward step produced, for observers.
pub struct Step<'a> {
    /// Final normalised hidden state (Q20).
    pub hidden: &'a [i32],
    /// Logits (Q20); empty for prompt positions whose prediction is not needed.
    pub logits: &'a [i32],
}

pub struct Output {
    /// Prompt tokens (`prefix ++ encode(text) ++ suffix`) followed by generated tokens.
    pub tokens: Vec<u32>,
    /// Range of `tokens` that encodes the user's text.
    pub text: core::ops::Range<usize>,
    /// Index of the first generated token.
    pub generated_from: usize,
}

/// Run the model on `prompt`.
///
/// `observe` is called after every forward step. Generation stops after
/// `max_new_tokens`, when the context window is full, or when a stop token is selected
/// (it is not part of the result). A prompt that does not fit in the context window is
/// an error; nothing is truncated silently.
pub fn generate(
    model: &Arc<Model>,
    prompt: &[u8],
    params: &GenParams,
    mut observe: impl FnMut(Step),
) -> Result<Output> {
    let tokenizer = Tokenizer::new(model);
    let prompt = if model.trim_prompt { tokenizer.trim(prompt) } else { prompt };
    // Every token covers at most `longest` bytes, so a prompt this long cannot fit.
    // Rejecting it here avoids tokenizing arbitrarily large inputs first.
    let longest = model.tokens.iter().map(Vec::len).max().unwrap_or(0).max(1);
    if prompt.len() / longest > model.seq_len {
        return err(format!(
            "prompt has {} bytes, which is more than the context window of {} tokens can hold",
            prompt.len(),
            model.seq_len
        ));
    }
    let text = tokenizer.encode(prompt)?;
    let mut tokens = model.prefix.clone();
    let text_range = tokens.len()..tokens.len() + text.len();
    tokens.extend(text);
    tokens.extend(&model.suffix);
    if tokens.is_empty() {
        return err("the prompt is empty and the model defines no prompt prefix");
    }
    if tokens.len() > model.seq_len {
        return err(format!(
            "prompt has {} tokens but the context window holds {}",
            tokens.len(),
            model.seq_len
        ));
    }
    let prompt_len = tokens.len();
    let dim = model.dim;
    let mut session = Session::new(Arc::clone(model), params.threads);
    let mut rng = Rng::new(params.sample.seed);
    // Logits are only needed where a token is going to be chosen.
    let mut done = params.max_new_tokens == 0 || prompt_len >= model.seq_len;

    // The prompt is fed in blocks; this gives the same result as token by token.
    let mut pos = 0;
    while pos < prompt_len {
        let end = (pos + PROMPT_BLOCK).min(prompt_len);
        let want_logits = end == prompt_len && !done;
        session.feed(&tokens[pos..end], want_logits)?;
        let last = end - pos - 1;
        for (i, hidden) in session.hidden().chunks_exact(dim).enumerate() {
            let logits = if i == last { session.logits() } else { &[] };
            observe(Step { hidden, logits });
        }
        pos = end;
    }
    while !done {
        let next = sample(session.logits(), &params.sample, &mut rng);
        if model.stop.contains(&next) {
            break;
        }
        tokens.push(next);
        done = tokens.len() - prompt_len >= params.max_new_tokens || tokens.len() >= model.seq_len;
        session.step(next, !done)?;
        observe(Step { hidden: session.hidden(), logits: session.logits() });
    }
    Ok(Output { tokens, text: text_range, generated_from: prompt_len })
}

/// SHA-256 over the hidden state and the logits of every step (little-endian `i32`),
/// followed by every token (little-endian `u32`). Two machines agree on this only if
/// they agree on every bit the engine produced, not just on the chosen tokens.
pub fn run_digest(model: &Arc<Model>, prompt: &[u8], params: &GenParams) -> Result<([u8; 32], Output)> {
    let mut hasher = Sha256::new();
    let output = generate(model, prompt, params, |step| {
        hasher.update_i32s(step.hidden);
        hasher.update_i32s(step.logits);
    })?;
    hasher.update_u32s(&output.tokens);
    Ok((hasher.finalize(), output))
}
