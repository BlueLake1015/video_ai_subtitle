PY ?= python3
PIP ?= $(PY) -m pip
ASR_MODEL ?= large-v3-turbo
MT_MODEL  ?= google/translategemma-12b-it
MODELS_DIR ?= models

.PHONY: help install install-dev install-translate install-all install-whispercpp \
        models-asr test lint fmt clean \
        whispercpp-submodule whispercpp-build \
        run run-live list-presets

help:
	@echo "Targets:"
	@echo "  install            - core runtime deps (ffmpeg + VAD + Whisper)"
	@echo "  install-translate  - + Gemma translation (transformers)"
	@echo "  install-all        - everything (translation, quant, whispercpp, gemini)"
	@echo "  install-dev        - dev extras (pytest, ruff)"
	@echo "  models-asr         - pre-warm faster-whisper model cache"
	@echo "  test               - run pytest"
	@echo "  lint               - ruff check"
	@echo "  fmt                - ruff format"
	@echo "  list-presets       - print preset names"
	@echo "  run IN=<file> OUT=<path> [TPRESET=...] [MPRESET=...] [TGT=ko]"
	@echo "  run-live IN=udp://... OUT=<path> [TPRESET=...] [MPRESET=...]"

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

install-translate:
	$(PIP) install -e ".[translate]"

install-all:
	$(PIP) install -e ".[all,dev]"

install-whispercpp:
	$(PIP) install -e ".[whispercpp]"

models-asr:
	@mkdir -p $(MODELS_DIR)
	$(PY) -c "from faster_whisper import WhisperModel; WhisperModel('$(ASR_MODEL)', device='cpu', compute_type='int8')"
	@echo "faster-whisper cached: $(ASR_MODEL)"

whispercpp-submodule:
	@test -d third_party/whisper.cpp || git submodule add https://github.com/ggerganov/whisper.cpp third_party/whisper.cpp
	git submodule update --init --recursive

whispercpp-build: whispercpp-submodule
	cd third_party/whisper.cpp && cmake -B build -DGGML_CUDA=1 -DCMAKE_BUILD_TYPE=Release && cmake --build build -j --config Release

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests

fmt:
	$(PY) -m ruff format src tests

clean:
	rm -rf build dist .pytest_cache .ruff_cache **/__pycache__ *.egg-info

list-presets:
	$(PY) -m vas list-presets

TPRESET ?= large-v3-turbo
MPRESET ?=
TGT ?=

run:
	$(PY) -m vas subtitle "$(IN)" -o "$(OUT)" -t $(TPRESET) \
		$(if $(MPRESET),-T $(MPRESET),) \
		$(if $(TGT),--tgt-lang $(TGT),)

run-live:
	$(PY) -m vas subtitle "$(IN)" -o "$(OUT)" -t $(TPRESET) \
		$(if $(MPRESET),-T $(MPRESET),) \
		$(if $(TGT),--tgt-lang $(TGT),)
