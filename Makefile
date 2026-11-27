# Models:
#   qwen3.5-2b    Qwen3.5-2B (text part). The default model, installed by `make install`.
#   qwen3.5-0.8b  Qwen3.5-0.8B (text part). About 2.3x faster, lower quality.
#   Both are converted by `detra convert`, the same integer-only converter `detra pull` uses.
#   stories15M    15M-parameter Llama-style test model. Small enough to run under emulation,
#                 so it is the one verified on every architecture. Not meant for actual use.
# `install`, `serve` and `check-qwen` use $(QWEN_MODEL), e.g. `make install QWEN_MODEL=qwen3.5-0.8b`.
QWEN_MODEL ?= qwen3.5-2b
STORIES_URL := https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
STORIES_TOK_URL := https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin
QWEN_2B_URL := https://huggingface.co/Qwen/Qwen3.5-2B/resolve/15852e8c16360a2fea060d615a32b45270f8a8fc
QWEN_08B_URL := https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/2fc06364715b967f1860aea9cf38778875588b17
STORIES := models/stories15M.dllm
QWEN := models/$(QWEN_MODEL).dllm
QWEN_2B := models/qwen3.5-2b.dllm
QWEN_08B := models/qwen3.5-0.8b.dllm
GOLDEN_STORIES := golden/stories15M.digests
GOLDEN_QWEN_2B := golden/qwen3.5-2b.digests
GOLDEN_QWEN_08B := golden/qwen3.5-0.8b.digests
RUST_IMAGE ?= rust:1-slim
DATA_DIR := $(or $(XDG_DATA_HOME),$(HOME)/.local/share)/detra
MIRI_TEST := cargo +nightly miri test -p detra --test synthetic

# $(1) = detra command, $(2) = model, $(3) = golden file, $(4) = label
check = $(1) digest golden/prompts.txt --model $(2) | diff - $(3) && echo "$(4): bit-exact"

# Build inside an emulated Linux container and compare its digests with the golden file.
# $(1) = docker platform
docker_verify = docker run --rm --platform $(1) -v "$(CURDIR)":/src:ro -w /src \
	-e CARGO_TARGET_DIR=/tmp/target $(RUST_IMAGE) sh -c \
	'cargo test --release --locked -p detra 2>&1 | grep "test result" >&2; /tmp/target/release/detra digest golden/prompts.txt --model $(STORIES)' \
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
	cargo run --release --locked --quiet -p detra-dev --bin no-float -- crates/detra

models/stories15M.bin:
	curl -L --fail -o $@ $(STORIES_URL)

models/tokenizer.bin:
	curl -L --fail -o $@ $(STORIES_TOK_URL)

$(STORIES): models/stories15M.bin models/tokenizer.bin | build
	target/release/detra-convert convert models/stories15M.bin models/tokenizer.bin $@

UNICODE_VERSION := 16.0.0
UCD_URL := https://www.unicode.org/Public/$(UNICODE_VERSION)/ucd
UCD := target/ucd-$(UNICODE_VERSION)
CONVERTER := $(addprefix crates/detra/src/,qwen35.rs exact.rs unicode_classes.rs model.rs tokenizer.rs)

# Download and conversion rules for one Qwen3.5 checkpoint.
# $(1) = model name (directory under models/), $(2) = Hugging Face URL at a pinned revision
define qwen_rules
models/$(1)/model.safetensors:
	mkdir -p models/$(1)
	curl -L --fail -o $$@ $(2)/model.safetensors-00001-of-00001.safetensors

models/$(1)/config.json models/$(1)/tokenizer.json models/$(1)/tokenizer_config.json models/$(1)/chat_template.jinja:
	mkdir -p models/$(1)
	curl -L --fail -o $$@ $(2)/$$(notdir $$@)

models/$(1).dllm: models/$(1)/model.safetensors models/$(1)/config.json models/$(1)/tokenizer.json $(CONVERTER) | build
	target/release/detra convert models/$(1) $$@
endef
$(eval $(call qwen_rules,qwen3.5-2b,$(QWEN_2B_URL)))
$(eval $(call qwen_rules,qwen3.5-0.8b,$(QWEN_08B_URL)))

$(UCD)/UnicodeData.txt $(UCD)/PropList.txt:
	mkdir -p $(UCD)
	curl -L --fail -o $@ $(UCD_URL)/$(notdir $@)

.PHONY: unicode-classes
unicode-classes: $(UCD)/UnicodeData.txt $(UCD)/PropList.txt ## Regenerate the tokenizer's Unicode table (changes the model hashes if it changes)
	cd $(UCD) && shasum -a 256 -c $(CURDIR)/tools/ucd-$(UNICODE_VERSION).sha256
	cargo run --release --locked --quiet -p detra-dev --bin unicode-classes -- $^ > crates/detra/src/unicode_classes.rs
	git diff --stat --exit-code crates/detra/src/unicode_classes.rs && echo "unicode_classes.rs: unchanged"

.PHONY: models
models: $(STORIES) $(QWEN_2B) $(QWEN_08B) ## Download all models, convert them and check all file hashes
	cd models && shasum -a 256 -c SHA256SUMS

.PHONY: golden
golden: models ## Regenerate the golden digests on this machine
	target/release/detra digest golden/prompts.txt --model $(STORIES) > $(GOLDEN_STORIES)
	target/release/detra digest golden/prompts.txt --model $(QWEN_2B) > $(GOLDEN_QWEN_2B)
	target/release/detra digest golden/prompts.txt --model $(QWEN_08B) > $(GOLDEN_QWEN_08B)

.PHONY: verify-native
verify-native: models ## Native release build against the golden digests
	$(call check,target/release/detra,$(STORIES),$(GOLDEN_STORIES),native stories15M)
	$(call check,target/release/detra,$(QWEN_2B),$(GOLDEN_QWEN_2B),native qwen3.5-2b)
	$(call check,target/release/detra,$(QWEN_08B),$(GOLDEN_QWEN_08B),native qwen3.5-0.8b)
	$(call check,DETRA_THREADS=1 target/release/detra,$(QWEN_2B),$(GOLDEN_QWEN_2B),native qwen3.5-2b single-threaded)
	$(call check,DETRA_THREADS=1 target/release/detra,$(QWEN_08B),$(GOLDEN_QWEN_08B),native qwen3.5-0.8b single-threaded)

.PHONY: verify-debug
verify-debug: models ## Unoptimized build with overflow checks enabled (small model)
	cargo build --locked -p detra
	$(call check,target/debug/detra,$(STORIES),$(GOLDEN_STORIES),native debug stories15M)

.PHONY: verify-x86_64
verify-x86_64: models ## x86-64 build run under Rosetta
	cargo test --release --locked -p detra --target x86_64-apple-darwin
	$(call check,target/x86_64-apple-darwin/release/detra,$(STORIES),$(GOLDEN_STORIES),x86_64 stories15M)
	$(call check,target/x86_64-apple-darwin/release/detra,$(QWEN_2B),$(GOLDEN_QWEN_2B),x86_64 qwen3.5-2b)
	$(call check,target/x86_64-apple-darwin/release/detra,$(QWEN_08B),$(GOLDEN_QWEN_08B),x86_64 qwen3.5-0.8b)

.PHONY: verify-wasm
verify-wasm: models ## WebAssembly (32-bit) build run under Node's WASI (small model)
	CARGO_TARGET_WASM32_WASIP1_RUNNER="node --no-warnings $(CURDIR)/scripts/wasi-run.mjs" cargo test --release --locked -p detra --target wasm32-wasip1
	cargo build --release --locked -p detra --target wasm32-wasip1
	$(call check,node --no-warnings scripts/wasi-run.mjs target/wasm32-wasip1/release/detra.wasm,$(STORIES),$(GOLDEN_STORIES),wasm32 stories15M)

.PHONY: verify-miri
verify-miri: ## Synthetic models interpreted for big-endian 64-bit and for 32-bit targets (slow)
	$(MIRI_TEST) --target s390x-unknown-linux-gnu
	# Miri hands out fresh addresses for half of all allocations by default, which
	# exhausts a 32-bit address space before the tests finish.
	MIRIFLAGS="-Zmiri-address-reuse-rate=1.0" $(MIRI_TEST) --target i686-unknown-linux-gnu

.PHONY: verify
verify: test lint verify-native verify-debug verify-x86_64 verify-wasm ## Fast checks: native, debug, x86-64, wasm32

.PHONY: verify-docker
verify-docker: models ## Small model on emulated big-endian s390x, 32-bit x86 and x86-64 Linux (needs Docker)
	$(call docker_verify,linux/s390x)
	$(call docker_verify,linux/386)
	$(call docker_verify,linux/amd64)

.PHONY: verify-docker-qwen
verify-docker-qwen: models ## Both Qwen3.5 models on emulated big-endian s390x Linux (needs Docker, very slow)
	mkdir -p target
	docker run --rm --platform linux/s390x -v "$(CURDIR)":/src:ro -w /src \
		-e CARGO_TARGET_DIR=/tmp/target $(RUST_IMAGE) sh -c \
		'cargo build --release --locked -p detra 2>&1 >/dev/null | tail -1 >&2; \
		 /tmp/target/release/detra digest golden/prompts.txt --model $(QWEN_2B) && \
		 /tmp/target/release/detra digest golden/prompts.txt --model $(QWEN_08B)' \
		> target/s390x-qwen.digests
	cat $(GOLDEN_QWEN_2B) $(GOLDEN_QWEN_08B) | diff - target/s390x-qwen.digests \
		&& echo "linux/s390x qwen3.5-2b and qwen3.5-0.8b: bit-exact"

.PHONY: check-qwen
check-qwen: build $(QWEN) ## Compare a Qwen3.5 model (QWEN_MODEL, default qwen3.5-2b) on the integer engine with the Hugging Face float implementation
	uv run tools/check_qwen35.py models/$(QWEN_MODEL) $(QWEN)

.PHONY: serve
serve: build $(QWEN) ## Serve QWEN_MODEL (default qwen3.5-2b) as an OpenAI-compatible chat API on http://127.0.0.1:8080/v1
	target/release/detra serve --model $(QWEN)

.PHONY: install
install: build $(QWEN) ## Install the `detra` CLI (cargo bin dir) and QWEN_MODEL (default qwen3.5-2b, to $XDG_DATA_HOME/detra)
	cargo install --locked --path crates/detra
	mkdir -p "$(DATA_DIR)"
	cp $(QWEN) "$(DATA_DIR)/$(QWEN_MODEL).dllm"
	@echo 'Installed. Try: detra generate "Why is the sky blue?" --max-tokens 100 --model $(QWEN_MODEL)'
	@echo 'or point a chat front end at `detra serve --model $(QWEN_MODEL)` (http://127.0.0.1:8080/v1).'

.PHONY: uninstall
uninstall: ## Remove the installed CLI and models
	cargo uninstall detra
	rm -f "$(DATA_DIR)"/qwen3.5-*.dllm "$(DATA_DIR)/model.dllm"
	-rmdir "$(DATA_DIR)"

.PHONY: setup
setup: ## Install the Rust targets and components the verify targets need
	rustup target add x86_64-apple-darwin wasm32-wasip1
	rustup +nightly component add miri rust-src
