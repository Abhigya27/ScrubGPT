"""Tiny persistent local library index backed by one JSON file.

Qdrant remains the source of truth for chunks. This file only remembers what playlists/videos
have been indexed locally so the UI can show a useful library after a process restart.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

from backend import config

_lock = threading.Lock()


def _path() -> Path:
    return Path(config.LIBRARY_METADATA_PATH)


def _load() -> dict:
    try:
        value = json.loads(_path().read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _save(data: dict) -> None:
    path = _path()
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


def upsert_source(source_id: str, kind: str, videos: list[dict]) -> None:
    with _lock:
        data = _load()
        sources = data.setdefault("sources", {})
        entry = sources.setdefault(source_id, {"source_id": source_id, "indexed_videos": {}})
        entry["kind"] = kind
        entry["videos"] = len(videos)
        entry["duration_seconds"] = sum(int(v.get("duration") or 0) for v in videos)
        entry["video_titles"] = {v["video_id"]: v.get("title") or v["video_id"] for v in videos}
        entry["updated_at"] = datetime.now(timezone.utc).isoformat()
        _save(data)


def record_video(source_id: str, video_id: str, title: str, duration: int, indexed: bool, reused: bool = False) -> None:
    with _lock:
        data = _load()
        sources = data.setdefault("sources", {})
        entry = sources.setdefault(source_id, {"source_id": source_id, "indexed_videos": {}})
        indexed_videos = entry.setdefault("indexed_videos", {})
        indexed_videos[video_id] = {
            "video_id": video_id,
            "title": title,
            "duration": duration,
            "indexed": indexed,
            "reused": reused,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _save(data)


def list_sources() -> list[dict]:
    with _lock:
        data = _load()
        sources = []
        for source_id, entry in data.get("sources", {}).items():
            videos = entry.get("indexed_videos", {})
            indexed_count = sum(1 for v in videos.values() if v.get("indexed"))
            sources.append({
                "source_id": source_id,
                "kind": entry.get("kind", "source"),
                "videos": int(entry.get("videos") or 0),
                "duration_seconds": int(entry.get("duration_seconds") or 0),
                "indexed_count": indexed_count,
                "updated_at": entry.get("updated_at"),
            })
        return sorted(sources, key=lambda x: x.get("updated_at") or "", reverse=True)
