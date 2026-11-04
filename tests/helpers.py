"""Fakes and builders shared by the test files (plain functions and classes; fixtures live in conftest.py)."""
from __future__ import annotations

import re
from types import SimpleNamespace

from backend.models import Chunk, TranscriptLine, Video


# --- YouTube transcript fakes --------------------------------------------------------------------------
class FakeSnippet:
    def __init__(self, text, start, duration):
        self.text, self.start, self.duration = text, start, duration


class FakeTranscript:
    def __init__(self, lang, generated=False, count=20):
        self.language_code = lang
        self.language = {"en": "English", "hi": "Hindi", "es": "Spanish"}.get(lang, lang)
        self.is_generated = generated
        self.fetch_calls = 0
        self._lines = [FakeSnippet(f"line {i} about machine learning", i * 5.0, 5.0) for i in range(count)]

    def fetch(self):
        self.fetch_calls += 1
        return self._lines


class FakeApi:
    """Stands in for YouTubeTranscriptApi. `error` is raised once by list(), then cleared."""

    def __init__(self, transcripts, error=None):
        self.transcripts = transcripts
        self.error = error
        self.list_calls = 0

    def list(self, _video_id):
        self.list_calls += 1
        if self.error:
            err, self.error = self.error, None
            raise err
        return list(self.transcripts)


def yt_provider(api):
    """A YouTubeTranscriptProvider wired to a fake API, without touching the network."""
    from backend.ingest import transcript as tp

    provider = tp.YouTubeTranscriptProvider.__new__(tp.YouTubeTranscriptProvider)
    provider._api = api
    return provider


class Rate429(Exception):
    def __init__(self, retry_after=1):
        super().__init__("HTTP 429 Too Many Requests")
        self.response = SimpleNamespace(status_code=429, headers={"Retry-After": str(retry_after)})


# --- chunks / translation ------------------------------------------------------------------------------
def make_chunks(n=20, language="es", words_per_chunk=0):
    """n Spanish chunks. `words_per_chunk` pads the text so it is long enough for the truncation check."""
    pad = " palabra" * words_per_chunk
    return [
        Chunk(
            chunk_id=f"v:{i}", video_id="v", video_title="Demo", playlist_ids=["p"], start_sec=i * 90,
            end_sec=(i + 1) * 90, text=f"texto del fragmento {i} sobre aprendizaje automatico{pad}",
            source_language=language,
        )
        for i in range(n)
    ]


_MARKER = re.compile(r"<<<CHUNK (\d+)>>>\n")


def _passages(user: str) -> list[tuple[str, str]]:
    parts = _MARKER.split(user)  # [preamble, "1", text1, "2", text2, ...]
    return list(zip(parts[1::2], (p.strip() for p in parts[2::2])))


def echo_translator(calls: list | None = None, prefix: str = "EN: "):
    """A fake `_chat` that translates properly: batches come back under their markers, single passages as plain text."""

    def fake_chat(_system, user, role="translate", **_kwargs):
        if calls is not None:
            calls.append(user)
        passages = _passages(user)
        if passages:
            return "\n".join(f"<<<CHUNK {n}>>>\n{prefix}{text}" for n, text in passages)
        return prefix + user.split("\n\n", 1)[1]

    return fake_chat


def garbling_translator(calls: list | None = None, garble_when=lambda passages: True, prefix: str = "EN: "):
    """Like echo_translator, but a BATCH reply is unusable (markers missing) whenever garble_when(passages) is true.
    Single-passage requests always succeed, which is what the one-by-one fallback relies on."""

    def fake_chat(_system, user, role="translate", **_kwargs):
        if calls is not None:
            calls.append(user)
        passages = _passages(user)
        if not passages:
            return prefix + user.split("\n\n", 1)[1]
        if garble_when(passages):
            return "Sorry, here is a summary of those passages instead of a translation."
        return "\n".join(f"<<<CHUNK {n}>>>\n{prefix}{text}" for n, text in passages)

    return fake_chat


# --- transcript lines ------------------------------------------------------------------------------------
def make_lines(count=30, seconds=10, words=8):
    """`count` consecutive lines of `seconds` each, every line with its own distinct words."""
    return [
        TranscriptLine(i * seconds, (i + 1) * seconds, " ".join(f"w{i}x{j}" for j in range(words)))
        for i in range(count)
    ]


def demo_video(video_id="vid", title="Demo", duration=600, language=None):
    return Video(video_id, title, duration, language)
