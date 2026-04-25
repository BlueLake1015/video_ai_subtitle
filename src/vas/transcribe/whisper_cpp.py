from __future__ import annotations

import numpy as np

from ..config import TranscribeConfig
from ..types import Word
from ..utils.logging import get_logger
from .base import TranscribeOptions

log = get_logger(__name__)


class WhisperCppTranscriber:
    """whisper.cpp backend via pywhispercpp (in-process libwhisper bindings).

    Requires the 'whispercpp' extra:  pip install -e '.[whispercpp]'
    Model path should point to a ggml-*.bin file (or the Model enum short name
    like 'large-v3-turbo' which pywhispercpp will resolve and download).
    """

    def __init__(self, cfg: TranscribeConfig):
        self.cfg = cfg
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            try:
                from pywhispercpp.model import Model
            except ImportError as e:
                raise ImportError(
                    "pywhispercpp not installed. `pip install -e '.[whispercpp]'`"
                ) from e
            model_ref = self.cfg.model_path or self.cfg.model
            log.info("loading whisper.cpp model=%s", model_ref)
            self._model = Model(
                model=model_ref,
                n_threads=0,  # 0 = hardware default
                print_realtime=False,
                print_progress=False,
            )
        return self._model

    def transcribe(
        self, audio: np.ndarray, options: TranscribeOptions | None = None
    ) -> list[Word]:
        model = self._ensure_model()
        opts = options or TranscribeOptions()
        lang = opts.language or self.cfg.language or "auto"
        segments = model.transcribe(
            audio.astype(np.float32),
            language=lang,
            translate=False,
            token_timestamps=self.cfg.word_timestamps,
            split_on_word=True,
        )

        words: list[Word] = []
        off = opts.time_offset_s
        for seg in segments:
            # pywhispercpp Segment has t0, t1 in 10ms ticks, and .text.
            start = (seg.t0 / 100.0) + off
            end = (seg.t1 / 100.0) + off
            words.append(Word(
                text=seg.text.strip(),
                start_s=start,
                end_s=end,
                probability=1.0,
            ))
        return words

    def close(self) -> None:
        self._model = None
