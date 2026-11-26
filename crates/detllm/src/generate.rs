//! Generation loop and the run digest used to compare machines.

use crate::engine::Session;
use crate::model::Model;
use crate::sample::{sample, Rng, SampleParams};
use crate::sha256::Sha256;
use crate::tokenizer::Tokenizer;
use crate::{err, Result};
use std::sync::Arc;

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
    let mut session = Session::new(Arc::clone(model), params.threads);
    let mut rng = Rng::new(params.sample.seed);
    let mut pos = 0;
    loop {
        let last = pos + 1 == tokens.len();
        let done = tokens.len() - prompt_len >= params.max_new_tokens || pos + 1 >= model.seq_len;
        let want_logits = last && !done;
        session.step(tokens[pos], want_logits)?;
        pos += 1;
        // The logits live in the session; borrow them again together with the hidden state.
        let logits = session.logits();
        observe(Step { hidden: session.hidden(), logits });
        if !last {
            continue;
        }
        if done {
            break;
        }
        let next = sample(logits, &params.sample, &mut rng);
        if model.stop.contains(&next) {
            break;
        }
        tokens.push(next);
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
