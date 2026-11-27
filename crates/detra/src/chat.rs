//! Conversations: turning a list of messages into the prompt tokens of a chat model.

use crate::generate::{check_prompt, encode_text};
use crate::model::Model;
use crate::tokenizer::Tokenizer;
use crate::{err, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

/// The prompt for the next assistant message: `open ++ encode(content) ++ close` for
/// every message, then the model's generation prompt.
///
/// Message contents are plain text: they never produce the special tokens that
/// structure the conversation, whatever they contain. A system message is only
/// allowed in first place and the last message must be the user's. A conversation
/// that does not fit in the context window is an error; nothing is dropped silently.
pub fn chat_tokens(model: &Model, tokenizer: &Tokenizer, messages: &[(Role, &[u8])]) -> Result<Vec<u32>> {
    let Some(chat) = &model.chat else {
        return err("this model has no chat template");
    };
    if messages.last().map(|m| m.0) != Some(Role::User) {
        return err("the last message must be a user message");
    }
    let mut tokens = Vec::new();
    for (i, &(role, content)) in messages.iter().enumerate() {
        let turn = match role {
            Role::System if i > 0 => return err("a system message must be the first message"),
            Role::System => &chat.system,
            Role::User => &chat.user,
            Role::Assistant => &chat.assistant,
        };
        tokens.extend(&turn.open);
        tokens.extend(encode_text(model, tokenizer, content)?);
        tokens.extend(&turn.close);
        // Stop early instead of tokenizing the rest of a conversation that is too long.
        check_prompt(model, &tokens)?;
    }
    tokens.extend(&chat.generation);
    check_prompt(model, &tokens)?;
    Ok(tokens)
}
