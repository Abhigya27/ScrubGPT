"""Translation fallback statistics (spec 1.6).

    python -m evaluation.translation_stats                      # from data/library.json
    python -m evaluation.translation_stats --source PLAYLIST_ID
    python -m evaluation.translation_stats --job-id JOB --api-url http://127.0.0.1:8000    # from a live/finished job

Batched translation (backend/ingest/translate.py) validates the model's reply and, when it is unusable, retries as
two half-size batches and then chunk by chunk. Each translated video records how often that happened; this script
adds the videos up. Counters come from data/library.json (written by every ingest) or from a job's status.

  fallback_rate = fallback_batches / batches_attempted
  (fallback_batches counts each batch once however many fallbacks it needed; a translation-cache hit attempts no batch)

Only videos ingested AFTER the counters were added carry them, and a video translated from the on-disk cache
contributes no fresh batches. To measure, ingest non-English videos that are not in the translation cache.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from evaluation.common import fail, format_table, pct, write_results

COUNTERS = ("batches_attempted", "batches_fell_back_to_smaller", "batches_fell_back_to_individual",
            "fallback_batches", "chunks_translated", "chunks_from_cache")


def _primary(code: str | None) -> str:
    return (code or "").lower().replace("_", "-").split("-")[0] or "unknown"


def rows_from_library(library: dict, sources: list[str] | None = None) -> list[dict]:
    """One row per translated video (deduplicated across sources) that has recorded counters."""
    rows: dict[str, dict] = {}
    for source_id, entry in (library.get("sources") or {}).items():
        if sources and source_id not in sources:
            continue
        for video_id, video in (entry.get("indexed_videos") or {}).items():
            if video.get("translation") and video_id not in rows:
                rows[video_id] = {"source_id": source_id, "video_id": video_id, "title": video.get("title"),
                                  "language": video.get("language"), "translation": video["translation"]}
    return list(rows.values())


def rows_from_job(status: dict) -> list[dict]:
    return [
        {"source_id": status.get("source_id"), "video_id": v.get("video_id"), "title": v.get("title"),
         "language": v.get("language"), "translation": v["translation"]}
        for v in status.get("indexed") or [] if v.get("translation")
    ]


def aggregate(rows: list[dict]) -> dict:
    """Totals over the rows. fallback_rate is None when no batch was attempted (nothing to rate)."""
    total = {k: sum(int(r["translation"].get(k) or 0) for r in rows) for k in COUNTERS}
    attempted = total["batches_attempted"]
    return {
        "videos_translated": len(rows),
        "videos_fully_cached": sum(1 for r in rows if r["translation"].get("cache_hit") and not r["translation"].get("batches_attempted")),
        "languages": dict(Counter(_primary(r["language"]) for r in rows).most_common()),
        **total,
        "fallbacks_total": total["batches_fell_back_to_smaller"] + total["batches_fell_back_to_individual"],
        "fallback_rate": total["fallback_batches"] / attempted if attempted else None,
        "chunks_total": total["chunks_translated"] + total["chunks_from_cache"],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.translation_stats", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--library", help="library file (default LIBRARY_METADATA_PATH, normally data/library.json)")
    parser.add_argument("--source", action="append", help="only this source id (repeatable)")
    parser.add_argument("--job-id", help="read the counters from this ingest job instead of the library")
    parser.add_argument("--api-url", default=os.environ.get("API_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--out-dir")
    args = parser.parse_args(argv)

    if args.job_id:
        import requests

        try:
            response = requests.get(f"{args.api_url.rstrip('/')}/ingest/status/{args.job_id}", timeout=60)
            response.raise_for_status()
        except requests.RequestException as exc:
            return fail(f"could not read job {args.job_id} from {args.api_url}: {exc}")
        rows, origin = rows_from_job(response.json()), f"job {args.job_id}"
    else:
        from backend import config

        path = Path(args.library or config.LIBRARY_METADATA_PATH)
        try:
            library = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return fail(f"cannot read {path}: {exc}")
        rows, origin = rows_from_library(library, args.source), str(path)

    if not rows:
        return fail(
            f"no translation counters found in {origin}. They are recorded when a NON-English video is ingested "
            "(after the counters were added). Index a playlist with a non-English video, or delete that video's "
            "translation cache file and ingest it again.", 1,
        )

    stats = aggregate(rows)
    print(f"Translation fallback statistics ({origin})\n")
    print(format_table(
        ["video", "language", "batches", "smaller", "individual", "fallback rate", "chunks"],
        [[(r["title"] or r["video_id"])[:38], _primary(r["language"]), r["translation"].get("batches_attempted", 0),
          r["translation"].get("batches_fell_back_to_smaller", 0), r["translation"].get("batches_fell_back_to_individual", 0),
          pct(r["translation"].get("fallback_rate"), 0), (r["translation"].get("chunks_translated", 0) + r["translation"].get("chunks_from_cache", 0))]
         for r in rows],
    ))
    print()
    print(format_table(["total", "value"], [
        ["videos translated", stats["videos_translated"]],
        ["languages", ", ".join(f"{k} {v}" for k, v in stats["languages"].items())],
        ["batches attempted", stats["batches_attempted"]],
        ["fell back to smaller batches", stats["batches_fell_back_to_smaller"]],
        ["fell back to individual chunks", stats["batches_fell_back_to_individual"]],
        ["batches needing any fallback", stats["fallback_batches"]],
        ["fallback rate", pct(stats["fallback_rate"], 1)],
        ["chunks translated (LLM / cache)", f"{stats['chunks_translated']} / {stats['chunks_from_cache']}"],
    ]))
    if stats["batches_attempted"] == 0:
        print("\nnote: every translated video came from the translation cache, so no batch was attempted and there "
              "is no rate. Delete the translation cache file for a video (data/cache/translations/) and ingest it again.")
    path = write_results("translation", {"origin": origin, "summary": stats, "videos": rows}, args.out_dir)
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
