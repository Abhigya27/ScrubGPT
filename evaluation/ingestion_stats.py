"""Corpus size and the re-ingest benchmark (spec 1.5).

    python -m evaluation.ingestion_stats                                   # corpus statistics only (read-only)
    python -m evaluation.ingestion_stats --playlist-url URL                # + cold vs re-ingest timing
    python -m evaluation.ingestion_stats --playlist-url URL --phases cold,cache,reuse --reset --yes

1. CORPUS SIZE (read-only). From data/library.json: videos indexed, total hours, videos per language (language comes
   from the library entry, else the local transcript cache). From Qdrant: exact chunk counts. Duplicates (a video
   that sits in two sources) are counted once. --source limits all of it to one or more sources.

2. RE-INGEST BENCHMARK. POSTs the playlist to /ingest and polls /ingest/status/{job_id} until the job is done, three
   phases (wall-clock, submit to done):
     cold    nothing cached, nothing in Qdrant: YouTube transcripts, translation, embedding, upsert
     cache   (optional) Qdrant points removed but the local transcript/translation caches kept: isolates what the
             disk caches save
     reuse   the same playlist again with everything in place: every video is recognised in Qdrant and only re-tagged
   speedup = cold_seconds / reingest_seconds. The cold phase needs a cold state. If the playlist's videos already have
   points or cached transcripts, nothing is touched unless you pass --reset, which DELETES those videos' points from
   Qdrant (for every playlist they belong to) and their local cache files, after a confirmation (or --yes).

   --mode http (default) talks to a running backend (API_BASE_URL, default http://127.0.0.1:8000). The backend keeps
   its last 8 transcripts in memory, which no outside script can clear: restart it before the cold run, or use
   --mode inprocess, which runs the same job queue inside this process (embedding models are loaded before the clock
   starts, as the server does at boot).

   Run this locally: YouTube may block cloud IPs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from evaluation.common import fail, format_table, write_results

PHASES = ("cold", "cache", "reuse")
SECTION_TIMINGS = ("metadata_seconds", "transcript_seconds", "translation_seconds", "embedding_seconds", "indexing_seconds")


class BenchmarkError(RuntimeError):
    """The benchmark cannot produce a trustworthy number; the message says why and what to do."""


# --- 1. corpus size --------------------------------------------------------------------------------------
def _primary(code: str | None) -> str:
    return (code or "").lower().replace("_", "-").split("-")[0] or "unknown"


def _cached_language(video_id: str) -> str | None:
    """The language recorded in the local transcript cache, for libraries written before language was stored."""
    from backend.ingest import cache

    try:
        return json.loads(cache.transcript_path(video_id).read_text(encoding="utf-8")).get("language")
    except (OSError, ValueError, AttributeError):
        return None


def corpus_stats(library: dict, sources: list[str] | None = None, language_lookup=_cached_language) -> dict:
    """Videos, hours and language breakdown from the parsed library.json. Pure given `language_lookup`.

    A video counts once even if several sources contain it, and only if it is marked indexed.
    """
    unique: dict[str, dict] = {}
    scoped = []
    for source_id, entry in (library.get("sources") or {}).items():
        if sources and source_id not in sources:
            continue
        scoped.append(source_id)
        for video_id, video in (entry.get("indexed_videos") or {}).items():
            if not video.get("indexed"):
                continue
            if video_id not in unique:
                unique[video_id] = dict(video)  # a copy: never mutate the caller's data
            else:  # the same video under another source: keep a language/chunks value the first copy lacked
                for key in ("language", "chunks"):
                    if not unique[video_id].get(key) and video.get(key):
                        unique[video_id][key] = video[key]
    languages: Counter = Counter()
    for video_id, video in unique.items():
        languages[_primary(video.get("language") or language_lookup(video_id))] += 1
    seconds = sum(int(v.get("duration") or 0) for v in unique.values())
    return {
        "sources": scoped, "videos": len(unique), "total_seconds": seconds, "hours": round(seconds / 3600, 2),
        "languages": dict(languages.most_common()), "video_ids": sorted(unique),
        "non_english_videos": sum(n for lang, n in languages.items() if lang != "en"),
    }


def qdrant_chunk_counts(video_ids: list[str]) -> dict:
    """Exact chunk counts from Qdrant: for the given videos (deduplicated) and for the whole collection."""
    from backend.index import qdrant_store

    per_video = {vid: qdrant_store.count_video_points(vid) for vid in video_ids}
    return {
        "chunks_in_scope": sum(per_video.values()), "chunks_in_collection": qdrant_store.count_points(),
        "collection": qdrant_store.collection_name(), "per_video": per_video,
    }


# --- 2. the benchmark ------------------------------------------------------------------------------------
class HttpRunner:
    """Drives a running backend over HTTP."""

    mode = "http"

    def __init__(self, api_url: str):
        import requests

        self._requests, self.api_url = requests, api_url.rstrip("/")

    def submit(self, url: str) -> tuple[str, bool]:
        response = self._requests.post(f"{self.api_url}/ingest", json={"url": url}, timeout=60)
        if response.status_code == 409:
            raise BenchmarkError(f"the backend is busy with another source: {response.json().get('detail')}")
        response.raise_for_status()
        body = response.json()
        return body["job_id"], bool(body.get("already_running"))

    def status(self, job_id: str) -> dict:
        response = self._requests.get(f"{self.api_url}/ingest/status/{job_id}", timeout=60)
        response.raise_for_status()
        return response.json()

    def clear_memory_caches(self) -> bool:
        return False  # cannot reach into the server process


class InProcessRunner:
    """Runs the backend's own job queue inside this process (no server needed)."""

    mode = "inprocess"

    def __init__(self):
        from backend.index import qdrant_store
        from backend.index.embed import get_sparse

        qdrant_store.ensure_ready()  # loads the dense model and verifies the collection
        get_sparse()  # the server does both at boot; keep model loading out of the timed section

    def submit(self, url: str) -> tuple[str, bool]:
        from backend import jobs
        from backend.ingest.url_parser import parse_url

        kind, ident = parse_url(url)
        job, attached = jobs.submit_ingest(kind, ident)
        return job.job_id, attached

    def status(self, job_id: str) -> dict:
        from backend import jobs

        job = jobs.get_job(job_id)
        if job is None:
            raise BenchmarkError(f"job {job_id} disappeared")
        return asdict(job)

    def clear_memory_caches(self) -> bool:
        from backend.ingest import transcript

        transcript.clear_cache()
        return True


def wait_for_job(runner, job_id: str, poll_interval: float, timeout: float, clock=time.monotonic, sleep=time.sleep) -> dict:
    """Poll until the job is `done`. Anything else terminal (paused, error) means the timing is not valid."""
    deadline = clock() + timeout
    while True:
        status = runner.status(job_id)
        state = status.get("status")
        if state == "done":
            return status
        if state == "paused":
            raise BenchmarkError(
                f"the job paused ({status.get('pause_reason')}): {status.get('error')}. A paused run is not a timing; "
                "wait for the cooldown and run the benchmark again."
            )
        if state == "error":
            raise BenchmarkError(f"the job failed: {status.get('error') or status.get('detail')}")
        if clock() > deadline:
            raise BenchmarkError(f"the job did not finish within {timeout:.0f}s (last stage: {status.get('stage')})")
        sleep(poll_interval)


def aggregate_translation(indexed: list[dict]) -> dict | None:
    """Sum the per-video translation counters of a finished job. None when no video was translated."""
    parts = [v["translation"] for v in indexed if v.get("translation")]
    if not parts:
        return None
    keys = ("batches_attempted", "batches_fell_back_to_smaller", "batches_fell_back_to_individual",
            "fallback_batches", "chunks_translated", "chunks_from_cache")
    total = {k: sum(int(p.get(k) or 0) for p in parts) for k in keys}
    total["fallback_rate"] = total["fallback_batches"] / total["batches_attempted"] if total["batches_attempted"] else None
    return total


def time_ingest(runner, url: str, label: str, poll_interval: float = 1.0, timeout: float = 3600.0,
                clock=time.monotonic, sleep=time.sleep) -> dict:
    """Submit `url` and wait until the job is done. Elapsed is wall-clock from just before the POST to `done`."""
    started = clock()
    job_id, attached = runner.submit(url)
    if attached:
        raise BenchmarkError(
            "this playlist was already being indexed by a running job, so there is no clean start to time. "
            "Wait for it to finish and run again."
        )
    final = wait_for_job(runner, job_id, poll_interval, timeout, clock, sleep)
    elapsed = clock() - started
    indexed = final.get("indexed") or []
    return {
        "phase": label, "elapsed_seconds": round(elapsed, 2), "job_id": job_id,
        "videos_total": final.get("total_videos"), "videos_indexed": len(indexed),
        "videos_reused": sum(1 for v in indexed if v.get("reused")),
        "videos_skipped": len(final.get("skipped") or []), "skipped": final.get("skipped") or [],
        "server_timings": {k: round(float(v), 2) for k, v in (final.get("timings") or {}).items()},
        "translation": aggregate_translation(indexed),
    }


def speedup(slow_seconds: float, fast_seconds: float) -> float | None:
    """slow / fast, or None when the fast run took no measurable time."""
    return round(slow_seconds / fast_seconds, 2) if fast_seconds and fast_seconds > 0 else None


def build_caveats(phases: list[dict], mode: str) -> list[str]:
    caveats = []
    skipped = max((p["videos_skipped"] for p in phases), default=0)
    if skipped:
        caveats.append(
            f"{skipped} video(s) were skipped (no usable transcript). A skipped video has no points, so every run "
            "retries it on YouTube, which makes the re-ingest time look worse than it is. Use a playlist whose videos "
            "all have transcripts."
        )
    reuse = next((p for p in phases if p["phase"] == "reuse"), None)
    if reuse and reuse["videos_reused"] < reuse["videos_indexed"]:
        caveats.append("Not every video was recognised as already indexed in the reuse phase, so it is not a pure re-ingest.")
    if mode == "http":
        caveats.append(
            "HTTP mode: the backend caches its last 8 transcripts in memory and a script cannot clear that. If this "
            "playlist was processed by the same backend process before, the cold time is understated; restart the "
            "backend before the cold run, or use --mode inprocess."
        )
    caveats.append(
        "Every run includes the fixed pause between videos (TRANSCRIPT_BETWEEN_VIDEOS_SECONDS plus up to 50% jitter), "
        "so a re-ingest never drops to zero."
    )
    return caveats


# --- cold-state handling (destructive, therefore explicit) --------------------------------------------
def translation_cache_files(video, source_id: str, position: int) -> list[Path]:
    """The on-disk translation cache file for a video's chunks, if its transcript is cached (the cache key is a hash
    of the chunks, so it can only be recomputed from the cached transcript). Empty for English or uncached videos."""
    from backend import config
    from backend.ingest import cache, chunker

    cached = cache.load_transcript(video.video_id)
    if cached is None or _primary(cached.language) == "en":
        return []
    chunks = chunker.chunk_transcript(cached.lines, video, source_id, cached.language, cached.source_type, position)
    if not chunks:
        return []
    path = cache.translation_path(cache.translation_cache_key(chunks, config.TRANSLATE_PROVIDER, config.TRANSLATE_MODEL))
    return [path] if path.exists() else []


def inspect_state(videos: list, source_id: str) -> dict:
    """What already exists for these videos: Qdrant points, cached transcripts, cached translations."""
    from backend.index import qdrant_store
    from backend.ingest import cache

    points = {v.video_id: qdrant_store.count_video_points(v.video_id) for v in videos}
    transcripts = [cache.transcript_path(v.video_id) for v in videos if cache.transcript_path(v.video_id).exists()]
    translations = [p for pos, v in enumerate(videos) for p in translation_cache_files(v, source_id, pos)]
    return {"points": points, "transcript_files": transcripts, "translation_files": translations,
            "total_points": sum(points.values()), "is_cold": not (sum(points.values()) or transcripts or translations)}


def apply_reset(videos: list, state: dict, points: bool = True, caches: bool = True) -> None:
    from backend.index import qdrant_store

    if points:
        for video in videos:
            qdrant_store.delete_video_points(video.video_id)
    if caches:
        for path in [*state["transcript_files"], *state["translation_files"]]:
            path.unlink(missing_ok=True)


def confirm(message: str, assume_yes: bool) -> bool:
    print(message)
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        print("Not an interactive terminal and --yes was not given; refusing.")
        return False
    return input("Type 'yes' to continue: ").strip().lower() == "yes"


def run_benchmark(args, runner) -> dict:
    from backend.ingest import metadata
    from backend.ingest.url_parser import parse_url
    from backend.util import UserFacingError

    kind, ident = parse_url(args.playlist_url)
    try:
        videos = metadata.list_playlist(ident) if kind == "playlist" else [metadata.resolve_video(ident)]
    except UserFacingError as exc:
        raise BenchmarkError(str(exc)) from exc
    if not videos:
        raise BenchmarkError("that URL has no videos")
    print(f"{kind} {ident}: {len(videos)} video(s)")

    phases_wanted = [p.strip() for p in args.phases.split(",") if p.strip()]
    if not phases_wanted or phases_wanted[0] != "cold" or any(p not in PHASES for p in phases_wanted):
        raise BenchmarkError(f"--phases must start with cold and use only {', '.join(PHASES)} (e.g. cold,reuse)")

    state = inspect_state(videos, ident)
    if not state["is_cold"]:
        summary = (f"{state['total_points']} chunk(s) in Qdrant, {len(state['transcript_files'])} cached transcript(s), "
                   f"{len(state['translation_files'])} cached translation(s)")
        if not args.reset:
            raise BenchmarkError(
                f"this playlist is not in a cold state ({summary}), so a 'cold' time would be meaningless. Either use a "
                "playlist you have never indexed, or pass --reset to delete those points and cache files first."
            )
        if not confirm(
            f"--reset will DELETE {summary} for these {len(videos)} video(s). The points are removed from Qdrant for "
            "every playlist containing these videos.", args.yes,
        ):
            raise BenchmarkError("reset not confirmed; nothing was deleted")
        apply_reset(videos, state)
        print("Reset done.")

    if runner.clear_memory_caches():
        print("Cleared the in-process transcript cache.")
    results = []
    for phase in phases_wanted:
        if phase == "cache":  # keep the disk caches, drop the points: what do the caches save on their own?
            apply_reset(videos, state, points=True, caches=False)
        print(f"\n== {phase} ingest ==")
        outcome = time_ingest(runner, args.playlist_url, phase, args.poll_interval, args.timeout)
        results.append(outcome)
        print(f"{phase}: {outcome['elapsed_seconds']:.1f}s  ({outcome['videos_indexed']} indexed, "
              f"{outcome['videos_reused']} reused, {outcome['videos_skipped']} skipped)")

    by_phase = {r["phase"]: r["elapsed_seconds"] for r in results}
    out = {
        "playlist_url": args.playlist_url, "source_id": ident, "videos": len(videos), "mode": runner.mode,
        "phases": results, "cold_seconds": by_phase.get("cold"), "reingest_seconds": by_phase.get("reuse"),
        "cache_only_seconds": by_phase.get("cache"),
        "reingest_speedup": speedup(by_phase["cold"], by_phase["reuse"]) if "reuse" in by_phase else None,
        "cache_only_speedup": speedup(by_phase["cold"], by_phase["cache"]) if "cache" in by_phase else None,
    }
    out["caveats"] = build_caveats(results, runner.mode)
    return out


def print_benchmark(bench: dict) -> None:
    rows = [[r["phase"], f"{r['elapsed_seconds']:.1f}s", r["videos_indexed"], r["videos_reused"], r["videos_skipped"]]
            for r in bench["phases"]]
    print("\nRe-ingest benchmark")
    print(format_table(["phase", "elapsed", "indexed", "reused", "skipped"], rows))
    if bench["reingest_speedup"] is not None:
        print(f"\nre-ingest speedup (cold / reuse): {bench['reingest_speedup']}x   "
              f"[A] = {bench['cold_seconds']:.1f}s cold, [B] = {bench['reingest_seconds']:.1f}s re-ingest")
    if bench["cache_only_speedup"] is not None:
        print(f"disk caches alone (cold / cache): {bench['cache_only_speedup']}x")
    for caveat in bench["caveats"]:
        print(f"note: {caveat}")


def print_corpus(corpus: dict, counts: dict | None, counts_error: str | None) -> None:
    langs = ", ".join(f"{lang} {n}" for lang, n in corpus["languages"].items()) or "none"
    rows = [["videos indexed", corpus["videos"]], ["total hours", f"{corpus['hours']:.2f}"], ["languages (videos)", langs]]
    if counts:
        rows += [["chunks (these videos)", counts["chunks_in_scope"]], ["chunks (whole collection)", counts["chunks_in_collection"]]]
    print("Corpus size" + (f" (sources: {', '.join(corpus['sources'])})" if corpus["sources"] else ""))
    for label, value in rows:
        print(f"  {label:<28}{value}")
    if counts_error:
        print(f"note: chunk counts unavailable: {counts_error}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.ingestion_stats", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--playlist-url", help="a SMALL playlist (or video) to time; omit for corpus statistics only")
    parser.add_argument("--library", help="library file (default LIBRARY_METADATA_PATH, normally data/library.json)")
    parser.add_argument("--source", action="append", help="limit the corpus statistics to this source id (repeatable)")
    parser.add_argument("--skip-corpus", action="store_true")
    parser.add_argument("--mode", choices=("http", "inprocess"), default="http")
    parser.add_argument("--api-url", default=os.environ.get("API_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--phases", default="cold,reuse", help="cold,reuse (default) or cold,cache,reuse")
    parser.add_argument("--reset", action="store_true", help="delete the playlist's Qdrant points and cache files first (DESTRUCTIVE)")
    parser.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=3600.0, help="per-phase limit in seconds")
    parser.add_argument("--out-dir")
    args = parser.parse_args(argv)

    payload: dict = {"params": {k: v for k, v in vars(args).items() if k not in ("out_dir",)}}
    bench = None
    if args.playlist_url:
        try:
            runner = HttpRunner(args.api_url) if args.mode == "http" else InProcessRunner()
            bench = run_benchmark(args, runner)
        except BenchmarkError as exc:
            return fail(str(exc), 1)
        except Exception as exc:  # noqa: BLE001
            return fail(f"{type(exc).__name__}: {exc}", 1)
        payload["benchmark"] = bench
        print_benchmark(bench)

    if not args.skip_corpus:
        from backend import config

        library_path = Path(args.library or config.LIBRARY_METADATA_PATH)
        try:
            library = json.loads(library_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            if bench is None:
                return fail(f"cannot read {library_path}: {exc}")
            print(f"\nnote: corpus statistics skipped, cannot read {library_path}: {exc}")
        else:
            corpus = corpus_stats(library, args.source)
            counts, counts_error = None, None
            try:
                counts = qdrant_chunk_counts(corpus["video_ids"])
            except Exception as exc:  # noqa: BLE001 - statistics from the library are still useful without Qdrant
                counts_error = f"{type(exc).__name__}: {exc}"
            print()
            print_corpus(corpus, counts, counts_error)
            payload["corpus"] = {**{k: v for k, v in corpus.items() if k != "video_ids"}, "qdrant": counts,
                                 "qdrant_error": counts_error}

    path = write_results("ingestion", payload, args.out_dir)
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
