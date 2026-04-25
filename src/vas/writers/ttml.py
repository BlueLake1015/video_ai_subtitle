from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from ..types import Cue
from ..utils.timing import fmt_ttml_time


class TtmlWriter:
    """TTML 1.0 / IMSC1 text-profile writer.

    Produces a minimal, broadcast-safe document: clock-time timestamps,
    region anchored to bottom, lines rendered as <br/>-separated spans.
    """

    def __init__(self, language: str = "en"):
        self.language = language

    def render(self, cues: list[Cue]) -> str:
        header = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<tt xmlns="http://www.w3.org/ns/ttml"\n'
            '    xmlns:ttp="http://www.w3.org/ns/ttml#parameter"\n'
            '    xmlns:tts="http://www.w3.org/ns/ttml#styling"\n'
            '    xmlns:ittp="http://www.w3.org/ns/ttml/profile/imsc1#parameter"\n'
            f'    xml:lang="{escape(self.language)}"\n'
            '    ttp:timeBase="media"\n'
            '    ittp:aspectRatio="16 9">\n'
            '  <head>\n'
            '    <styling>\n'
            '      <style xml:id="default"\n'
            '             tts:fontFamily="proportionalSansSerif"\n'
            '             tts:fontSize="100%"\n'
            '             tts:textAlign="center"\n'
            '             tts:color="white"\n'
            '             tts:backgroundColor="rgba(0,0,0,192)"/>\n'
            '    </styling>\n'
            '    <layout>\n'
            '      <region xml:id="bottom"\n'
            '              tts:origin="10% 80%"\n'
            '              tts:extent="80% 20%"\n'
            '              tts:displayAlign="after"/>\n'
            '    </layout>\n'
            '  </head>\n'
            '  <body region="bottom" style="default">\n'
            '    <div>\n'
        )
        body: list[str] = []
        for i, c in enumerate(cues, start=1):
            escaped_lines = [escape(ln) for ln in c.lines]
            content = "<br/>".join(escaped_lines)
            body.append(
                f'      <p xml:id="s{i}" '
                f'begin="{fmt_ttml_time(c.start_s)}" '
                f'end="{fmt_ttml_time(c.end_s)}">{content}</p>'
            )
        footer = "\n    </div>\n  </body>\n</tt>\n"
        return header + "\n".join(body) + footer

    def write(self, cues: list[Cue], path: str | Path) -> None:
        Path(path).write_text(self.render(cues), encoding="utf-8")
