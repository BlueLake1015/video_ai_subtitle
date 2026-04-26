PYTHON_VERSION ?= 3.11
VENV ?= .venv
PY  ?= $(VENV)/bin/python
PIP ?= $(VENV)/bin/pip
ASR_MODEL ?= large-v3-turbo
MT_MODEL  ?= google/translategemma-12b-it
MODELS_DIR ?= models

.PHONY: help venv install install-dev install-translate install-all install-whispercpp \
        models-asr test lint fmt clean \
        whispercpp-submodule whispercpp-build \
        run run-live list-presets

help:
	@echo "Targets:"
	@echo "  venv               - create $(VENV) using python$(PYTHON_VERSION)"
	@echo "  install            - core runtime deps (ffmpeg + VAD + Whisper)"
	@echo "  install-translate  - + Gemma translation (transformers)"
	@echo "  install-all        - everything (translation, quant, whispercpp, gemini)"
	@echo "  install-dev        - dev extras (pytest, ruff)"
	@echo "  models-asr         - pre-warm faster-whisper model cache"
	@echo "  test               - run pytest in venv"
	@echo "  lint               - ruff check in venv"
	@echo "  fmt                - ruff format in venv"
	@echo "  list-presets       - print preset names"
	@echo "  run IN=<file> OUT=<path> [TPRESET=...] [MPRESET=...] [TGT=ko]"
	@echo "  run-live IN=udp://... OUT=<path> [TPRESET=...] [MPRESET=...]"

# Create the venv if missing. All other targets depend on it.
$(VENV)/bin/python:
	@command -v python$(PYTHON_VERSION) >/dev/null 2>&1 || { \
		echo "python$(PYTHON_VERSION) not found. Run: bash scripts/install_online.sh"; \
		exit 1; \
	}
	python$(PYTHON_VERSION) -m venv $(VENV)
	$(PIP) install --upgrade pip wheel setuptools

venv: $(VENV)/bin/python

install: venv
	$(PIP) install -e .

install-dev: venv
	$(PIP) install -e ".[dev]"

install-translate: venv
	$(PIP) install -e ".[translate]"

install-all: venv
	$(PIP) install -e ".[all,dev]"

install-whispercpp: venv
	$(PIP) install -e ".[whispercpp]"

models-asr: venv
	@mkdir -p $(MODELS_DIR)
	$(PY) -c "from faster_whisper import WhisperModel; WhisperModel('$(ASR_MODEL)', device='cpu', compute_type='int8')"
	@echo "faster-whisper cached: $(ASR_MODEL)"

whispercpp-submodule:
	@test -d third_party/whisper.cpp || git submodule add https://github.com/ggerganov/whisper.cpp third_party/whisper.cpp
	git submodule update --init --recursive

whispercpp-build: whispercpp-submodule
	cd third_party/whisper.cpp && cmake -B build -DGGML_CUDA=1 -DCMAKE_BUILD_TYPE=Release && cmake --build build -j --config Release

test: venv
	$(PY) -m pytest -q

lint: venv
	$(PY) -m ruff check src tests

fmt: venv
	$(PY) -m ruff format src tests

clean:
	rm -rf build dist .pytest_cache .ruff_cache **/__pycache__ *.egg-info

list-presets: venv
	$(PY) -m vas list-presets

TPRESET ?= large-v3-turbo
MPRESET ?=
TGT ?=

run: venv
	$(PY) -m vas subtitle "$(IN)" -o "$(OUT)" -t $(TPRESET) \
		$(if $(MPRESET),-T $(MPRESET),) \
		$(if $(TGT),--tgt-lang $(TGT),)

run-live: venv
	$(PY) -m vas subtitle "$(IN)" -o "$(OUT)" -t $(TPRESET) \
		$(if $(MPRESET),-T $(MPRESET),) \
		$(if $(TGT),--tgt-lang $(TGT),)
