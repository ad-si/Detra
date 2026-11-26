//! Generation loop and the run digest used to compare machines.

use crate::engine::Session;
use crate::model::Model;
use crate::sample::{sample, Rng, SampleParams};
use crate::sha256::Sha256;
use crate::tokenizer::Tokenizer;
use crate::{err, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GenParams {
    pub max_new_tokens: usize,
    pub sample: SampleParams,
}

/// Run the model on `prompt` and return all tokens (prompt tokens, then generated ones).
///
/// `observe` receives the logits of every forward step, prompt positions included.
/// Generation stops after `max_new_tokens`, when the context window is full, or when
/// `bos` or `eos` is selected (the stop token is not part of the result). A prompt that
/// does not fit in the context window is an error; nothing is truncated silently.
pub fn generate(
    model: &Model,
    prompt: &[u8],
    params: &GenParams,
    mut observe: impl FnMut(&[i32]),
) -> Result<Vec<u32>> {
    let mut tokens = Tokenizer::new(model).encode(prompt)?;
    if tokens.len() > model.seq_len {
        return err(format!(
            "prompt has {} tokens but the context window holds {}",
            tokens.len(),
            model.seq_len
        ));
    }
    let prompt_len = tokens.len();
    let mut session = Session::new(model);
    let mut rng = Rng::new(params.sample.seed);
    let mut pos = 0;
    loop {
        let logits = session.step(tokens[pos])?;
        observe(logits);
        pos += 1;
        if pos < tokens.len() {
            continue;
        }
        if tokens.len() - prompt_len >= params.max_new_tokens || pos >= model.seq_len {
            break;
        }
        let next = sample(logits, &params.sample, &mut rng);
        if next == model.bos_id || next == model.eos_id {
            break;
        }
        tokens.push(next);
    }
    Ok(tokens)
}

/// SHA-256 over every logit of every step (little-endian `i32`), followed by every token
/// (little-endian `u32`). Two machines agree on this only if they agree on every bit the
/// engine produced, not just on the chosen tokens.
pub fn run_digest(model: &Model, prompt: &[u8], params: &GenParams) -> Result<([u8; 32], Vec<u32>)> {
    let mut hasher = Sha256::new();
    let tokens = generate(model, prompt, params, |logits| hasher.update_i32s(logits))?;
    hasher.update_u32s(&tokens);
    Ok((hasher.finalize(), tokens))
}
