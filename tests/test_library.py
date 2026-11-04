"""data/library.json: persistence and the merge rules the evaluation scripts rely on."""
import json

from backend import config, library


def record(**kwargs):
    base = dict(source_id="PL1", video_id="v1", title="One", duration=600, indexed=True)
    base.update(kwargs)
    library.record_video(**base)


def entry(video_id="v1", source_id="PL1"):
    return library._load()["sources"][source_id]["indexed_videos"][video_id]


def test_library_persists_across_loads():
    library.upsert_source("PL1", "playlist", [{"video_id": "v1", "title": "One", "duration": 600}])
    record()
    [source] = library.list_sources()
    assert (source["source_id"], source["kind"], source["videos"], source["indexed_count"]) == ("PL1", "playlist", 1, 1)
    assert source["duration_seconds"] == 600
    assert json.loads(open(config.LIBRARY_METADATA_PATH, encoding="utf-8").read())["sources"]["PL1"]["indexed_videos"]["v1"]["indexed"] is True


def test_a_missing_or_corrupt_library_reads_as_empty():
    assert library.list_sources() == []
    with open(config.LIBRARY_METADATA_PATH, "w", encoding="utf-8") as fh:
        fh.write("{ nope")
    assert library.list_sources() == []


def test_language_chunks_and_translation_are_stored():
    record(language="hi", chunks=12, translation={"batches_attempted": 2, "cache_hit": False})
    assert (entry()["language"], entry()["chunks"]) == ("hi", 12)
    assert entry()["translation"]["batches_attempted"] == 2


def test_a_reuse_does_not_erase_what_an_earlier_run_learned():
    record(language="hi", chunks=12, translation={"batches_attempted": 2, "cache_hit": False})
    record(reused=True)  # a re-ingest only reuses the video and knows nothing new
    assert entry()["reused"] is True
    assert (entry()["language"], entry()["chunks"]) == ("hi", 12)
    assert entry()["translation"]["batches_attempted"] == 2


def test_a_translation_cache_hit_does_not_overwrite_real_counters():
    record(language="hi", translation={"batches_attempted": 5, "fallback_batches": 1, "cache_hit": False})
    record(language="hi", translation={"batches_attempted": 0, "fallback_batches": 0, "cache_hit": True})
    assert entry()["translation"]["batches_attempted"] == 5


def test_fresh_counters_replace_older_ones():
    record(translation={"batches_attempted": 5, "cache_hit": False})
    record(translation={"batches_attempted": 9, "cache_hit": False})
    assert entry()["translation"]["batches_attempted"] == 9


def test_a_cache_hit_is_kept_when_there_is_nothing_better():
    record(translation={"batches_attempted": 0, "cache_hit": True})
    assert entry()["translation"]["cache_hit"] is True


def test_the_same_video_can_belong_to_several_sources():
    record(source_id="PL1")
    record(source_id="PL2")
    assert {s["source_id"] for s in library.list_sources()} == {"PL1", "PL2"}
