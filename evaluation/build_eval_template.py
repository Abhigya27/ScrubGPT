"""Write a skeleton eval set from the videos you have indexed (spec 1.1).

    python -m evaluation.build_eval_template                      # data/library.json -> evaluation/eval_set.json
    python -m evaluation.build_eval_template --source PLAYLIST_ID --cross-video 6 --unanswerable 12

For every indexed video it writes one stub per category (direct_fact, paraphrase, followup,
terminology_mismatch). For every source with two or more indexed videos it adds cross_video stubs, and for every
source it adds unanswerable stubs (expects_refusal: true). Every stub has `question: null`, `gold_answer_contains: null` and
`gold_answer: null`: YOU fill in the real questions by watching the videos, then delete the stubs you do
not use. The scripts ignore any entry whose question is still null.

It never overwrites an existing eval set unless you pass --force: that file holds hand-written work.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evaluation.common import DEFAULT_EVAL_SET, fail

PER_VIDEO_CATEGORIES = ("direct_fact", "paraphrase", "followup", "terminology_mismatch")


def _stub(category: str, source_id: str, video_id: str | None, video_title: str | None, expects_refusal: bool = False) -> dict:
    stub = {
        "id": "",
        "source_id": source_id,
        "video_id": video_id,
        "video_title": video_title,  # a reminder of which video this stub is for; ignored by the scripts
        "category": category,
        "question": None,
        "gold_answer_contains": None,
        "gold_answer": None,  # 1-3 sentence reference answer; enables correctness and context_recall
        "gold_timestamp_seconds": None,
        "expects_refusal": expects_refusal,
    }
    if category == "followup":
        stub["history"] = [{"question": None, "answer": None}]  # the earlier turn this follow-up depends on
    return stub


def build_template(
    library: dict, cross_video: int = 4, unanswerable: int = 12, source: str | None = None,
    categories: tuple[str, ...] = PER_VIDEO_CATEGORIES,
) -> list[dict]:
    """Stub entries for every indexed video in `library` (the parsed data/library.json). Pure."""
    stubs: list[dict] = []
    for source_id, entry in (library.get("sources") or {}).items():
        if source and source_id != source:
            continue
        videos = [v for v in (entry.get("indexed_videos") or {}).values() if v.get("indexed")]
        for video in videos:
            for category in categories:
                stubs.append(_stub(category, source_id, video["video_id"], video.get("title")))
        if len(videos) >= 2:
            stubs.extend(_stub("cross_video", source_id, None, None) for _ in range(cross_video))
        if videos:
            stubs.extend(_stub("unanswerable", source_id, None, None, expects_refusal=True) for _ in range(unanswerable))
    width = max(2, len(str(len(stubs))))
    for n, stub in enumerate(stubs, 1):
        stub["id"] = f"e{n:0{width}d}"
    return stubs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.build_eval_template", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--library", help="library file (default: LIBRARY_METADATA_PATH, normally data/library.json)")
    parser.add_argument("--out", default=str(DEFAULT_EVAL_SET), help="where to write the skeleton")
    parser.add_argument("--source", help="only this source id (a playlist or video id from the library)")
    parser.add_argument("--cross-video", type=int, default=4, help="cross_video stubs per source with 2+ videos (default 4)")
    parser.add_argument("--unanswerable", type=int, default=12, help="unanswerable stubs per source (default 12)")
    parser.add_argument("--force", action="store_true", help="overwrite an existing eval set")
    args = parser.parse_args(argv)

    if args.library:
        library_path = Path(args.library)
    else:
        from backend import config

        library_path = Path(config.LIBRARY_METADATA_PATH)
    try:
        library = json.loads(library_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return fail(f"{library_path} not found. Index a test playlist first (POST /ingest or the Streamlit app); "
                    "that writes the library file.")
    except ValueError as exc:
        return fail(f"{library_path} is not valid JSON: {exc}")

    stubs = build_template(library, args.cross_video, args.unanswerable, args.source)
    if not stubs:
        return fail("no indexed videos found in the library" + (f" for source {args.source!r}" if args.source else "") + ".")

    out = Path(args.out)
    if out.exists() and not args.force:
        return fail(f"{out} already exists and may hold your hand-written questions. Pass --force to overwrite it, "
                    "or --out to write somewhere else.")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stubs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    videos = len({(s["source_id"], s["video_id"]) for s in stubs if s["video_id"]})
    print(f"Wrote {len(stubs)} stubs for {videos} indexed video(s) to {out}")
    print("Next: fill in `question` and `gold_answer_contains` (and `gold_timestamp_seconds` for ~15-20 of them),")
    print("give each followup a `history` turn, delete the stubs you do not need, and aim for 40-50 questions in total")
    print("including 10-15 unanswerable ones. See evaluation/README.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
