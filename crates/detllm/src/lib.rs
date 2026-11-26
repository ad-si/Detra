//! Bit-exact LLM inference.
//!
//! The output is a pure function of (model file bytes, prompt bytes, parameters).
//! Rules that make this hold on every machine:
//!
//! - No floating point anywhere in this crate (enforced by `make lint`).
//! - Every operation that may overflow uses explicit wrapping arithmetic, so debug and
//!   release builds agree. Plain operators are only used where overflow is impossible.
//! - No dependence on pointer width, endianness, hash-map iteration order, or threads.
//! - Right shifts of signed values are arithmetic (floor); `/` truncates toward zero.
#![forbid(unsafe_code)]
#![deny(clippy::float_arithmetic, clippy::float_cmp)]

pub mod engine;
pub mod fixed;
pub mod generate;
pub mod model;
pub mod sample;
pub mod sha256;
pub mod tokenizer;

/// Error type for everything in this crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = core::result::Result<T, Error>;

pub(crate) fn err<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error(msg.into()))
}
