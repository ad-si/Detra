CKPT_URL := https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
TOK_URL := https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin
MODEL := models/stories15M.dllm
DIGEST := digest golden/prompts.txt --model $(MODEL)
GOLDEN := golden/digests.txt
RUST_IMAGE ?= rust:1-slim
DATA_DIR := $(or $(XDG_DATA_HOME),$(HOME)/.local/share)/detllm
MIRI_TEST := cargo +nightly miri test -p detllm --test synthetic

# Build inside an emulated Linux container and compare its digests with the golden file.
# $(1) = docker platform
docker_verify = docker run --rm --platform $(1) -v "$(CURDIR)":/src:ro -w /src \
	-e CARGO_TARGET_DIR=/tmp/target $(RUST_IMAGE) sh -c \
	'cargo build --release --locked -p detllm 2>&1 >/dev/null | tail -1 >&2; /tmp/target/release/detllm $(DIGEST)' \
	| diff - $(GOLDEN) && echo "$(1): bit-exact"

.PHONY: help
help: ## Show this help
	@grep -E '^[a-z0-9_-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "%-18s %s\n", $$1, $$2}'

.PHONY: build
build: ## Build the engine and the converter (release)
	cargo build --release --locked

.PHONY: test
test: ## Run unit tests and the synthetic-model determinism test
	cargo test --locked

.PHONY: lint
lint: ## Clippy, plus a check that the engine crate contains no floating point
	cargo clippy --locked --all-targets -- -D warnings
	@! grep -rnE '\bf(32|64)\b' crates/detllm/src crates/detllm/tests || { echo "floating point found in the engine crate"; exit 1; }

models/stories15M.bin:
	curl -L --fail -o $@ $(CKPT_URL)

models/tokenizer.bin:
	curl -L --fail -o $@ $(TOK_URL)

$(MODEL): models/stories15M.bin models/tokenizer.bin | build
	target/release/detllm-convert convert models/stories15M.bin models/tokenizer.bin $@

.PHONY: model
model: $(MODEL) ## Download stories15M, convert it and check all file hashes
	cd models && shasum -a 256 -c SHA256SUMS

.PHONY: golden
golden: model ## Regenerate golden/digests.txt on this machine
	target/release/detllm $(DIGEST) > $(GOLDEN)

.PHONY: verify-native
verify-native: model ## Native release build against the golden digests
	target/release/detllm $(DIGEST) | diff - $(GOLDEN) && echo "native release: bit-exact"

.PHONY: verify-debug
verify-debug: model ## Unoptimized build with overflow checks enabled
	cargo build --locked -p detllm
	target/debug/detllm $(DIGEST) | diff - $(GOLDEN) && echo "native debug: bit-exact"

.PHONY: verify-x86_64
verify-x86_64: model ## x86-64 build run under Rosetta
	cargo build --release --locked -p detllm --target x86_64-apple-darwin
	target/x86_64-apple-darwin/release/detllm $(DIGEST) | diff - $(GOLDEN) && echo "x86_64 (Rosetta): bit-exact"

.PHONY: verify-wasm
verify-wasm: model ## WebAssembly (32-bit) build run under Node's WASI
	cargo build --release --locked -p detllm --target wasm32-wasip1
	node --no-warnings scripts/wasi-run.mjs target/wasm32-wasip1/release/detllm.wasm $(DIGEST) | diff - $(GOLDEN) && echo "wasm32: bit-exact"

.PHONY: verify-miri
verify-miri: ## Synthetic-model test interpreted for big-endian 64-bit and for 32-bit targets (about 11 minutes)
	$(MIRI_TEST) --target s390x-unknown-linux-gnu
	$(MIRI_TEST) --target i686-unknown-linux-gnu

.PHONY: verify
verify: test lint verify-native verify-debug verify-x86_64 verify-wasm ## Fast checks: native, debug, x86-64, wasm32

.PHONY: verify-docker
verify-docker: model ## Full model on emulated big-endian s390x, 32-bit x86 and x86-64 Linux (needs Docker)
	$(call docker_verify,linux/s390x)
	$(call docker_verify,linux/386)
	$(call docker_verify,linux/amd64)

.PHONY: install
install: model ## Install the `detllm` CLI (cargo bin dir) and the model ($XDG_DATA_HOME/detllm)
	cargo install --locked --path crates/detllm
	mkdir -p "$(DATA_DIR)"
	cp $(MODEL) "$(DATA_DIR)/model.dllm"
	@echo 'Installed. Try: detllm generate "Once upon a time" --max-tokens 100'

.PHONY: uninstall
uninstall: ## Remove the installed CLI and model
	cargo uninstall detllm
	rm -f "$(DATA_DIR)/model.dllm"
	-rmdir "$(DATA_DIR)"

.PHONY: setup
setup: ## Install the Rust targets and components the verify targets need
	rustup target add x86_64-apple-darwin wasm32-wasip1
	rustup +nightly component add miri rust-src
