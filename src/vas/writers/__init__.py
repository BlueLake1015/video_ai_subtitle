from .base import Writer  # noqa: F401
from .srt import SrtWriter  # noqa: F401
from .ttml import TtmlWriter  # noqa: F401
from .vtt import VttWriter  # noqa: F401


def writer_for(path_or_format: str) -> Writer:
    low = path_or_format.lower()
    if low.endswith(".srt") or low == "srt":
        return SrtWriter()
    if low.endswith(".ttml") or low.endswith(".xml") or low == "ttml":
        return TtmlWriter()
    if low.endswith(".vtt") or low == "vtt":
        return VttWriter()
    raise ValueError(f"Unknown output format: {path_or_format!r}")
