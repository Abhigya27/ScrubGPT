"""Small JSON-on-disk caches used by local playlist ingestion.

No cache server or database is needed. Cache writes are atomic and failures are treated as
cache misses so the main indexing pipeline remains correct even when the cache is unavailable.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend import config
from backend.models import TranscriptLine, TranscriptResult

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _root(kind: str) -> Path:
    return Path(config.CACHE_DIR) / kind


def _safe_name(value: str) -> str:
    cleaned = _SAFE.sub("_", value.strip())
    return cleaned[:180] or "unknown"


def _read(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def _atomic_write(path: Path, data: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent), text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
    except OSError:
        # Caching must never make an otherwise valid ingest fail.
        return


def transcript_path(video_id: str) -> Path:
    return _root("transcripts") / f"{_safe_name(video_id)}.json"


def load_transcript(video_id: str) -> TranscriptResult | None:
    payload = _read(transcript_path(video_id))
    if not payload:
        return None
    try:
        lines = [TranscriptLine(float(x["start"]), float(x["end"]), str(x["text"])) for x in payload["lines"]]
        return TranscriptResult(
            lines=lines,
            language=str(payload["language"]),
            language_name=str(payload.get("language_name") or payload["language"]),
            is_generated=bool(payload.get("is_generated")),
            provider=str(payload.get("provider") or "youtube"),
            source_type=str(payload["source_type"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def save_transcript(video_id: str, result: TranscriptResult) -> None:
    _atomic_write(
        transcript_path(video_id),
        {
            "video_id": video_id,
            "lines": [{"start": x.start, "end": x.end, "text": x.text} for x in result.lines],
            "language": result.language,
            "language_name": result.language_name,
            "is_generated": result.is_generated,
            "provider": result.provider,
            "source_type": result.source_type,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def translation_cache_key(chunks, provider: str | None, model: str | None) -> str:
    h = hashlib.sha256()
    h.update((provider or "").encode())
    h.update(b"\0")
    h.update((model or "").encode())
    for chunk in chunks:
        h.update(b"\0")
        h.update(chunk.chunk_id.encode())
        h.update(b"\0")
        h.update(chunk.source_language.encode())
        h.update(b"\0")
        h.update(chunk.text.encode("utf-8"))
    return h.hexdigest()


def translation_path(cache_key: str) -> Path:
    return _root("translations") / f"{cache_key}.json"


def load_translation(cache_key: str, expected_count: int) -> list[str] | None:
    payload = _read(translation_path(cache_key))
    if not payload or payload.get("chunk_count") != expected_count:
        return None
    values = payload.get("translations")
    if not isinstance(values, list) or len(values) != expected_count or not all(isinstance(x, str) for x in values):
        return None
    return values


def save_translation(cache_key: str, provider: str | None, model: str | None, translations: list[str]) -> None:
    _atomic_write(
        translation_path(cache_key),
        {
            "cache_key": cache_key,
            "provider": provider,
            "model": model,
            "chunk_count": len(translations),
            "translations": translations,
            "cached_at": datetime.now(timezone.utc).isoformat(),
        },
    )
