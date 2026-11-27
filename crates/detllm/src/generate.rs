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
    /// The tokens that encode the user's text.
    pub text: &'a [u32],
    /// The tokens generated so far; empty while the prompt is being processed.
    pub generated: &'a [u32],
}

/// Why generation ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finish {
    /// The model selected a stop token.
    Stop,
    /// `max_new_tokens` were generated.
    Length,
    /// The context window is full.
    Context,
    /// The observer asked to stop.
    Cancelled,
}

pub struct Output {
    /// Prompt tokens (`prefix ++ encode(text) ++ suffix`) followed by generated tokens.
    pub tokens: Vec<u32>,
    /// Range of `tokens` that encodes the user's text.
    pub text: core::ops::Range<usize>,
    /// Index of the first generated token.
    pub generated_from: usize,
    pub finish: Finish,
}

/// Tokens of `text`, trimmed first if the model asks for it. Text that cannot fit in
/// the context window is rejected before it is tokenized.
pub fn encode_text(model: &Model, tokenizer: &Tokenizer, text: &[u8]) -> Result<Vec<u32>> {
    let text = if model.trim_prompt { tokenizer.trim(text) } else { text };
    // Every token covers at most `longest` bytes, so a text this long cannot fit.
    // Rejecting it here avoids tokenizing arbitrarily large inputs first.
    let longest = model.tokens.iter().map(Vec::len).max().unwrap_or(0).max(1);
    if text.len() / longest > model.seq_len {
        return err(format!(
            "prompt has {} bytes, which is more than the context window of {} tokens can hold",
            text.len(),
            model.seq_len
        ));
    }
    tokenizer.encode(text)
}

/// A prompt must have at least one token and fit in the context window; nothing is
/// truncated silently.
pub fn check_prompt(model: &Model, tokens: &[u32]) -> Result<()> {
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
    Ok(())
}

/// Feed the part of `tokens` that `session` has not consumed yet, with logits for the
/// last token if `want_logits`. The session must have been fed a prefix of `tokens`;
/// the result is the same as that of a fresh session fed all of them.
///
/// `observe(hidden, logits)` is called for every position fed; `logits` is empty
/// except at the last one.
pub fn feed_prompt(
    session: &mut Session,
    tokens: &[u32],
    want_logits: bool,
    mut observe: impl FnMut(&[i32], &[i32]),
) -> Result<()> {
    let dim = session.model().dim;
    let mut pos = session.pos();
    if pos > tokens.len() {
        return err("the session is ahead of the prompt");
    }
    // The prompt is fed in blocks; this gives the same result as token by token.
    while pos < tokens.len() {
        let end = (pos + PROMPT_BLOCK).min(tokens.len());
        session.feed(&tokens[pos..end], want_logits && end == tokens.len())?;
        let last = end - pos - 1;
        for (i, hidden) in session.hidden().chunks_exact(dim).enumerate() {
            observe(hidden, if i == last { session.logits() } else { &[] });
        }
        pos = end;
    }
    Ok(())
}

/// Extend `tokens`, all of which `session` has consumed, by sampling until a stop
/// token is selected (it is not appended), `max_new_tokens` are generated or the
/// context window is full. Unless one of the limits is already reached, the session
/// must hold the logits of the last token.
///
/// `observe(session, tokens)` is called after every forward step; returning false
/// cancels the generation.
pub fn sample_tokens(
    session: &mut Session,
    tokens: &mut Vec<u32>,
    params: &GenParams,
    mut observe: impl FnMut(&Session, &[u32]) -> bool,
) -> Result<Finish> {
    let model = Arc::clone(session.model());
    let start = tokens.len();
    let limit = |tokens: &Vec<u32>| {
        if tokens.len() - start >= params.max_new_tokens {
            Some(Finish::Length)
        } else if tokens.len() >= model.seq_len {
            Some(Finish::Context)
        } else {
            None
        }
    };
    let mut rng = Rng::new(params.sample.seed);
    loop {
        if let Some(finish) = limit(tokens) {
            return Ok(finish);
        }
        if session.pos() != tokens.len() || session.logits().is_empty() {
            return err("the session does not hold the logits of the last token");
        }
        let next = sample(session.logits(), &params.sample, &mut rng);
        if model.stop.contains(&next) {
            return Ok(Finish::Stop);
        }
        tokens.push(next);
        // Logits are only needed where a token is going to be chosen.
        session.step(next, limit(tokens).is_none())?;
        if !observe(session, tokens) {
            return Ok(Finish::Cancelled);
        }
    }
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
    let text = encode_text(model, &tokenizer, prompt)?;
    let mut tokens = model.prefix.clone();
    let text_range = tokens.len()..tokens.len() + text.len();
    tokens.extend(text);
    tokens.extend(&model.suffix);
    check_prompt(model, &tokens)?;
    let prompt_len = tokens.len();
    let mut session = Session::new(Arc::clone(model), params.threads);
    let want_logits = params.max_new_tokens > 0 && prompt_len < model.seq_len;
    feed_prompt(&mut session, &tokens, want_logits, |hidden, logits| {
        observe(Step { hidden, logits, text: &tokens[text_range.clone()], generated: &[] });
    })?;
    let finish = sample_tokens(&mut session, &mut tokens, params, |session, tokens| {
        observe(Step {
            hidden: session.hidden(),
            logits: session.logits(),
            text: &tokens[text_range.clone()],
            generated: &tokens[prompt_len..],
        });
        true
    })?;
    Ok(Output { tokens, text: text_range, generated_from: prompt_len, finish })
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
