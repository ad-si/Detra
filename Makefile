# Two models are supported:
#   stories15M    15M-parameter Llama-style model. Small enough to run under emulation, so
#                 it is the one verified on every architecture.
#   qwen3.5-0.8b  Qwen3.5-0.8B (text part). The model that `make install` installs.
STORIES_URL := https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
STORIES_TOK_URL := https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin
QWEN_URL := https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/2fc06364715b967f1860aea9cf38778875588b17
QWEN_DIR := models/qwen3.5-0.8b
STORIES := models/stories15M.dllm
QWEN := models/qwen3.5-0.8b.dllm
GOLDEN_STORIES := golden/stories15M.digests
GOLDEN_QWEN := golden/qwen3.5-0.8b.digests
RUST_IMAGE ?= rust:1-slim
DATA_DIR := $(or $(XDG_DATA_HOME),$(HOME)/.local/share)/detllm
MIRI_TEST := cargo +nightly miri test -p detllm --test synthetic

# $(1) = detllm command, $(2) = model, $(3) = golden file, $(4) = label
check = $(1) digest golden/prompts.txt --model $(2) | diff - $(3) && echo "$(4): bit-exact"

# Build inside an emulated Linux container and compare its digests with the golden file.
# $(1) = docker platform
docker_verify = docker run --rm --platform $(1) -v "$(CURDIR)":/src:ro -w /src \
	-e CARGO_TARGET_DIR=/tmp/target $(RUST_IMAGE) sh -c \
	'cargo test --release --locked -p detllm 2>&1 | grep "test result" >&2; /tmp/target/release/detllm digest golden/prompts.txt --model $(STORIES)' \
	| diff - $(GOLDEN_STORIES) && echo "$(1) stories15M: bit-exact"

.PHONY: help
help: ## Show this help
	@grep -E '^[a-z0-9_-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "%-18s %s\n", $$1, $$2}'

.PHONY: build
build: ## Build the engine and the converter (release)
	cargo build --release --locked

.PHONY: test
test: ## Run unit tests and the synthetic-model determinism tests (debug and release)
	cargo test --locked
	cargo test --locked --release

.PHONY: lint
lint: ## Clippy, plus a check that the engine crate contains no floating point
	cargo clippy --locked --all-targets -- -D warnings
	scripts/no-float.sh

models/stories15M.bin:
	curl -L --fail -o $@ $(STORIES_URL)

models/tokenizer.bin:
	curl -L --fail -o $@ $(STORIES_TOK_URL)

$(STORIES): models/stories15M.bin models/tokenizer.bin | build
	target/release/detllm-convert convert models/stories15M.bin models/tokenizer.bin $@

$(QWEN_DIR)/model.safetensors:
	mkdir -p $(QWEN_DIR)
	curl -L --fail -o $@ $(QWEN_URL)/model.safetensors-00001-of-00001.safetensors

$(QWEN_DIR)/config.json $(QWEN_DIR)/tokenizer.json $(QWEN_DIR)/tokenizer_config.json $(QWEN_DIR)/chat_template.jinja:
	mkdir -p $(QWEN_DIR)
	curl -L --fail -o $@ $(QWEN_URL)/$(notdir $@)

$(QWEN): $(QWEN_DIR)/model.safetensors $(QWEN_DIR)/config.json $(QWEN_DIR)/tokenizer.json tools/convert_qwen35.py
	uv run tools/convert_qwen35.py $(QWEN_DIR) $@

.PHONY: models
models: $(STORIES) $(QWEN) ## Download both models, convert them and check all file hashes
	cd models && shasum -a 256 -c SHA256SUMS

.PHONY: golden
golden: models ## Regenerate the golden digests on this machine
	target/release/detllm digest golden/prompts.txt --model $(STORIES) > $(GOLDEN_STORIES)
	target/release/detllm digest golden/prompts.txt --model $(QWEN) > $(GOLDEN_QWEN)

.PHONY: verify-native
verify-native: models ## Native release build against the golden digests
	$(call check,target/release/detllm,$(STORIES),$(GOLDEN_STORIES),native stories15M)
	$(call check,target/release/detllm,$(QWEN),$(GOLDEN_QWEN),native qwen3.5-0.8b)
	$(call check,DETLLM_THREADS=1 target/release/detllm,$(QWEN),$(GOLDEN_QWEN),native qwen3.5-0.8b single-threaded)

.PHONY: verify-debug
verify-debug: models ## Unoptimized build with overflow checks enabled (small model)
	cargo build --locked -p detllm
	$(call check,target/debug/detllm,$(STORIES),$(GOLDEN_STORIES),native debug stories15M)

.PHONY: verify-x86_64
verify-x86_64: models ## x86-64 build run under Rosetta
	cargo test --release --locked -p detllm --target x86_64-apple-darwin
	$(call check,target/x86_64-apple-darwin/release/detllm,$(STORIES),$(GOLDEN_STORIES),x86_64 stories15M)
	$(call check,target/x86_64-apple-darwin/release/detllm,$(QWEN),$(GOLDEN_QWEN),x86_64 qwen3.5-0.8b)

.PHONY: verify-wasm
verify-wasm: models ## WebAssembly (32-bit) build run under Node's WASI (small model)
	CARGO_TARGET_WASM32_WASIP1_RUNNER="node --no-warnings $(CURDIR)/scripts/wasi-run.mjs" cargo test --release --locked -p detllm --target wasm32-wasip1
	cargo build --release --locked -p detllm --target wasm32-wasip1
	$(call check,node --no-warnings scripts/wasi-run.mjs target/wasm32-wasip1/release/detllm.wasm,$(STORIES),$(GOLDEN_STORIES),wasm32 stories15M)

.PHONY: verify-miri
verify-miri: ## Synthetic models interpreted for big-endian 64-bit and for 32-bit targets (slow)
	$(MIRI_TEST) --target s390x-unknown-linux-gnu
	$(MIRI_TEST) --target i686-unknown-linux-gnu

.PHONY: verify
verify: test lint verify-native verify-debug verify-x86_64 verify-wasm ## Fast checks: native, debug, x86-64, wasm32

.PHONY: verify-docker
verify-docker: models ## Small model on emulated big-endian s390x, 32-bit x86 and x86-64 Linux (needs Docker)
	$(call docker_verify,linux/s390x)
	$(call docker_verify,linux/386)
	$(call docker_verify,linux/amd64)

.PHONY: verify-docker-qwen
verify-docker-qwen: models ## Qwen3.5 on emulated big-endian s390x Linux (needs Docker, very slow)
	docker run --rm --platform linux/s390x -v "$(CURDIR)":/src:ro -w /src \
		-e CARGO_TARGET_DIR=/tmp/target $(RUST_IMAGE) sh -c \
		'cargo build --release --locked -p detllm 2>&1 >/dev/null | tail -1 >&2; /tmp/target/release/detllm digest golden/prompts.txt --model $(QWEN)' \
		| diff - $(GOLDEN_QWEN) && echo "linux/s390x qwen3.5-0.8b: bit-exact"

.PHONY: check-qwen
check-qwen: models ## Compare Qwen3.5 on the integer engine with the Hugging Face float implementation
	uv run tools/check_qwen35.py $(QWEN_DIR) $(QWEN)

.PHONY: install
install: build $(QWEN) ## Install the `detllm` CLI (cargo bin dir) and Qwen3.5-0.8B ($XDG_DATA_HOME/detllm)
	cargo install --locked --path crates/detllm
	mkdir -p "$(DATA_DIR)"
	cp $(QWEN) "$(DATA_DIR)/model.dllm"
	@echo 'Installed. Try: detllm generate "Why is the sky blue?" --max-tokens 100'

.PHONY: uninstall
uninstall: ## Remove the installed CLI and model
	cargo uninstall detllm
	rm -f "$(DATA_DIR)/model.dllm"
	-rmdir "$(DATA_DIR)"

.PHONY: setup
setup: ## Install the Rust targets and components the verify targets need
	rustup target add x86_64-apple-darwin wasm32-wasip1
	rustup +nightly component add miri rust-src
