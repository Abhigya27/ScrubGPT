# Evaluation workflow

Everything here is meant to be run **locally**, against your real Qdrant collection, because YouTube may block
cloud IPs and the numbers should describe the system you actually run. Each script prints a table and writes
`evaluation/results/<name>_<UTC timestamp>.json` containing `generated_at` and the git commit, so results are
skimmable and diffable. `python -m evaluation.<script> --help` documents every option.

The goal: every bracketed number in the resume bullets comes from one of these scripts, never from memory.

## Order of work

1. **Run the offline tests** (`pytest -m "not integration" -q`). Cheap, and it makes later changes safer.
2. **Index a test playlist** of 10-20 videos, **including at least one non-English video** so the translation path is
   exercised. Use the Streamlit app or `POST /ingest`. Note the source id (the playlist id) in `data/library.json`.
3. **Get questions.** Either generate the skeleton, `python -m evaluation.build_eval_template`, and write the questions
   yourself, or let the LLM draft candidates from your real transcripts, `python -m evaluation.draft_eval_set`, and
   review them. Either way, **a person has to check the questions and reference answers** (below).
4. **Run everything**: `python -m evaluation.run_all` (add `--fast` to skip the two LLM-heavy steps). Or run the steps one by one:
   `ablation_eval`, `robustness_eval`, `timestamp_accuracy_eval`, `grounding_eval`, `answer_quality_eval`,
   `performance_eval`, `translation_stats`, `ingestion_stats`.
5. **Re-ingest benchmark** (not part of `run_all`, because it deletes data): `python -m evaluation.ingestion_stats --playlist-url <small playlist>`.
6. **Spot-check the judge**: `python -m evaluation.answer_quality_eval --audit 8`, then read `judge_audit_*.md`.
7. **Record the demo** (below).
8. **Fill the README and the resume**: `python -m evaluation.render_results --count-tests --write-readme` prints the
   bracketed numbers for the bullets and refreshes the README's Results section from the saved files.

## What is measured, and what is not

| Question | Metric | Script |
|---|---|---|
| Does retrieval put a relevant chunk in the top k? | **Recall@k (hit rate)**, **MRR** | `ablation_eval` |
| How much of the top k is relevant? | **Precision@k** | `ablation_eval` |
| Are relevant chunks ranked early? | **nDCG@k** | `ablation_eval` |
| How many of *all* the relevant chunks were found? | **ChunkRecall@k** (true recall) | `ablation_eval` |
| Does it work equally in Hindi and English? | Recall by language and by category | `ablation_eval` |
| Does retrieval survive typos, shouting, rambling? | Recall/MRR drop per perturbation | `robustness_eval` |
| Do citations point at the right moment? | **Timestamp accuracy** within +-15 s / +-30 s | `timestamp_accuracy_eval` |
| Does it refuse what it cannot answer, and only that? | **Refusal accuracy**, **false-refusal rate** | `grounding_eval` |
| Are the answer's claims supported by the retrieved text? | **Faithfulness** (hallucination) | `answer_quality_eval` |
| Does each citation support its sentence? | **Citation support**, citation coverage | `answer_quality_eval` |
| Does the answer address the question? | **Answer relevancy** | `answer_quality_eval` |
| Is the answer right? | **Correctness** vs your `gold_answer` | `answer_quality_eval` |
| Is the retrieved context useful / complete? | **Context precision**, **context recall** | `answer_quality_eval` |
| Is it fast and cheap enough? | Latency p50/p95 per stage, LLM calls and tokens per question | `performance_eval` |
| Does translation hold up? | Fallback rate | `translation_stats` |
| How big is the corpus, and does re-ingest use the caches? | Videos, hours, chunks, speedup | `ingestion_stats` |

**What none of these tell you:** whether real users are satisfied (they use a fixed question set you wrote), how the
system behaves on playlists unlike yours, or whether the judge is right (that needs your own spot-check). Retrieval
metrics are exact given your gold strings; the answer-quality metrics are a model's opinion.

## The eval set

`evaluation/eval_set.json` follows the `source_id` / `questions` / `q` / `expected` format shown in
`eval_set.example.json`. Each expected answer location has a `video_id` and accepted `start`/`end` timestamps.
For a representative, hand-reviewed set, aim for 40-50 questions: direct facts, paraphrases, follow-ups,
`terminology_mismatch` (worded differently from the transcript), `cross_video` (needs two videos), plus
**10-15 deliberately unanswerable ones** with `"expected": []`.

The current `eval_set.json` contains 200 answerable, fact-oriented timestamp lookups across the 41 indexed videos:
196 concise topic lookups with transcript evidence cues, plus four cross-video questions for RAG, LangGraph, MCP,
and vector databases. The set was generated locally from cached transcripts and matched English translations.
Question wording is intentionally straightforward and evidence-cued, so these results are an easy smoke test, not
an unbiased measure of keyword-only cross-video search. Wording has not been manually reviewed; do not treat the
file as a fully curated gold standard. The `id`, `category`, `gold_answer_contains`, `gold_answer`, and
`expects_refusal` fields are additional metadata consumed by evaluation scripts; the example-format fields remain
`q` and `expected`. This set has no unanswerable questions or follow-ups. Expected time ranges are transcript-chunk
bounds, not manually verified answer instants; use them for span-based evaluation, not the stricter point-timestamp
metric.

For a keyword-lookup calibration check, use `eval_set.keyword-smoke.json` (104 entries covering 52 distinct
transcript-backed topics). Each topic has two navigational phrasings with the same expected locations; both reduce
to the same retrieval query after navigation-word cleanup. Expected locations include matching cached
transcript/translation chunks, grouped by the configured period gap. Run it with
`python -m evaluation.run_eval run --set evaluation/eval_set.keyword-smoke.json`. On the current local index and
retrieval settings, it scores 79% precision, 71% recall, and 75% F1. These are 104 question-entry results, not
results from 104 independent topics: paired phrasings deliberately repeat each topic and label. Topics were
selected after retrieval testing, so this is a calibrated development smoke test—not a held-out, independent,
comprehensive, or representative golden benchmark. Its scores are not comparable to the 200-question set. It is
answerable-only and does not measure refusal accuracy.

```json
{
  "source_id": "PLAYLIST_ID",
  "questions": [{
    "q": "In the lesson on [topic], at what timestamp is [specific fact] discussed?",
    "expected": [{"video_id": "VIDEO_ID", "start": "5:42", "end": "6:30"}],
    "gold_answer_contains": "the transcript wording that identifies a relevant chunk",
    "gold_answer": "Optional reference answer used by answer-quality evaluation."
  }]
}
```

| Field | Meaning |
|---|---|
| `source_id` | The id you **ingested**, which is what search is scoped to: the playlist id for a playlist, or the video id for a video ingested on its own. |
| `q` | The question to search for. |
| `expected` | Accepted answer locations, each with a `video_id`, `start`, and `end` (`mm:ss` or `h:mm:ss`). Use an empty list when the videos do not answer the question. |
| `video_id` | Inside each expected location, the video that holds the answer. |
| `category` | `direct_fact`, `paraphrase`, `followup`, `terminology_mismatch`, `cross_video`, or `unanswerable`. |
| `gold_answer_contains` | A short phrase a **relevant transcript chunk** contains, matched case- and punctuation-insensitively. It is a string in the *transcript's* wording (use the English text for translated videos, since chunks are stored translated). A list means "any of these". Pick something specific: too generic and everything counts as relevant; too long and a correct chunk that splits the phrase counts as a miss. |
| `gold_answer` | Optional reference answer, 1-3 sentences, in your own words. Needed for **correctness** and **context recall**; without it those two are skipped. Write what a person who watched the video would say. |
| `expects_refusal` | `true` for questions the playlist cannot answer. When omitted, the loader infers it from an empty `expected` list. |
| `history` | Follow-ups only: the earlier turn(s), `[{"question": "...", "answer": "..."}]`, so the script can reproduce the conversation. |

Entries whose `q` is still `null` are ignored, so you can leave unused stubs in place while you work. So are
entries with `"needs_review": true`, which is how LLM drafts arrive (below).
`build_eval_template` refuses to overwrite an existing `eval_set.json` unless you pass `--force`.

## What each script reports

### `ablation_eval`: does 0.65 sparse / 0.35 dense hold up?

Fetches candidates once per question, then re-fuses them under dense-only, sparse-only, 50/50, 65/35 (default) and
80/20 (`--configs "0:1,1:0,0.65:0.35"` to change; pairs are `sparse:dense`).

- `Recall@5` = questions with a relevant chunk in the top 5 / questions. `MRR` = mean of 1/rank of the first relevant chunk (0 if none in the top 10).
- Prints a 95% Wilson interval beside each Recall. With ~40 questions the interval is wide; a difference of one or two questions is noise.
- **RANKING table** (quote this one): every candidate ranked by fused score, no threshold. It isolates fusion quality.
- **PRODUCTION table**: the app's real pipeline (absolute `RETRIEVAL_THRESHOLD` plus the precision guards). That threshold was tuned for the default weights, so other weights are disadvantaged there.
- It flags clearly if any non-default setting beats the default. If one does, consider changing `SPARSE_WEIGHT`/`DENSE_WEIGHT`, and re-run.
- `--enhance history` (default) resolves follow-ups only, so the comparison is about fusion; `--enhance full` adds the app's query expansion (one LLM call per question).

### `timestamp_accuracy_eval`

For each question with `gold_timestamp_seconds`, runs the app's search pipeline and looks at the periods returned **for the correct video**.

- **Gold inside `[start - T, end + T]`** of some period, for T = 15 s and 30 s. This is the definition in the build spec.
- **Period start within T** of the gold moment. Stricter, and closer to what a user experiences, because the start is the timestamp they click. A long merged period can contain the gold moment while starting minutes earlier. Report both, or say which one you quote.

### `grounding_eval`

Runs `/ask`'s pipeline for every question.

- `refusal_accuracy = correct_refusals / total_unanswerable` and `false_refusal_rate = wrongly_refused / total_answerable`. Always quote them together: a system that refuses everything scores 100% on the first.
- A response counts as a refusal when `grounded` is false or the answer is the refusal sentence. A stricter "refusal sentence only" figure is printed too, plus which layer refused (retrieval threshold, the answer model, or an uncited answer).
- Costs up to two LLM calls per question; use `--delay` to respect rate limits. If the quota runs out it saves partial results and exits with code 3.

### `ablation_eval`, the extra retrieval metrics

Besides Recall@5 and MRR it prints **Precision@5**, **nDCG@5** and **ChunkRecall@5**, and Recall by category and by the
language of the relevant text. These need to know *all* the chunks that contain each gold phrase, so the script first
reads the source's chunks from Qdrant (read-only; `--no-corpus-scan` skips it, and the extra metrics with it).

- That scan also **checks your gold phrases**: a `gold_answer_contains` that matches no chunk is an eval-set mistake (a typo, or Hindi wording where the stored text is the English translation). Those questions are listed and left out of the metrics (`--keep-unmatched` scores them anyway). Fix them rather than ignoring the warning.
- "Recall@5" in the build spec is a **hit rate** (any relevant chunk in the top 5). **ChunkRecall@5** is the share of all relevant chunks found. If a topic is covered in six places and retrieval finds one, the hit rate says 100% and ChunkRecall says about 17%.
- Precision@5 is low by design when only one or two chunks are relevant; compare configurations with it, do not read it in isolation.

### `robustness_eval`

Repeats retrieval on altered versions of each question (typos, UPPERCASE, filler words, a truncated question, an unrelated clause) and reports how much Recall@5 and MRR drop for dense-only, sparse-only and the default hybrid. It uses no LLM by default. The point is to see whether hybrid holds up better than either leg alone, and which kind of mess hurts most.

### `answer_quality_eval`: faithfulness, correctness, relevancy

Runs `/ask`'s pipeline for each answerable question, then a **judge model** scores the result. Metric definitions are in `evaluation/judge.py`.

- **Use a different judge from the answer model.** Set `EVAL_JUDGE_PROVIDER` and `EVAL_JUDGE_MODEL` in `.env` (or `--judge-provider/--judge-model`). A judge from the same family favours its own style and scores high. The script warns when they match, and the README Results section repeats the warning.
- **It costs LLM calls**: up to six judge calls per question, plus the app's own one or two. Trial it with `--limit 10` or `--metrics faithfulness,answer_relevancy` first. `--delay` helps with rate limits; on quota exhaustion it saves partial results and exits with code 3.
- **Check the judge yourself**: `--audit 8` writes `judge_audit_*.md` with sampled answers and the judge's reasons. If you disagree often, do not quote its scores.
- Scores come with bootstrap 95% intervals. With ~40 questions they are wide, so a difference of a few points is noise.
- Refused answers are not scored on the answer metrics (that is the grounding eval's job) but are counted in `answered_rate`, and their retrieved context is still scored.

### `performance_eval`

Times each stage per question (query enhancement, retrieval, answer), reports p50/p95/max, and counts LLM calls and tokens per question by role (answer, enhance, judge). The first questions are warm-up (`--warmup`) because the first search loads the embedding models. Add `--price-in` and `--price-out` (USD per million tokens, from your provider) for a cost per 100 questions; no price is built in. Numbers describe your machine and network.

### `run_all`

`python -m evaluation.run_all` runs every step above in order and continues past failures, then prints what ran and which result files were written. `--only`, `--skip`, `--fast`, `--delay`, and `--write-readme` are described in `--help`. It never runs the re-ingest benchmark.

### `ingestion_stats`

- **Corpus** (read-only): videos, total hours, videos per language, chunks. Add `--source PLAYLIST_ID` to describe only the test playlist.
- **Re-ingest benchmark**: `--playlist-url URL` times `POST /ingest` to `done`, cold and then again (`speedup = cold / re-ingest`).
  - The cold phase needs a cold state. If the playlist already has points or cached transcripts the script stops and tells you; `--reset` deletes that playlist's points from Qdrant (for every playlist containing those videos) and its cache files, after a confirmation (or `--yes`). **That is destructive**, so prefer a playlist you have never indexed.
  - Add `--phases cold,cache,reuse` to also time a run where the disk caches are kept but the Qdrant points are removed. That isolates what the transcript and translation caches save, which is the honest number for "caching cut re-ingestion time"; a plain re-ingest is mostly Qdrant recognising the videos as already indexed.
  - `--mode http` (default) uses the running backend (`--api-url`, default `http://127.0.0.1:8000`), which keeps its last 8 transcripts in memory: restart it before the cold run, or use `--mode inprocess`.
  - Videos with no usable transcript are retried on every run, which inflates the re-ingest time. Use a playlist where every video has a transcript. The report lists caveats like this.

### `translation_stats`

Reads the counters recorded for each translated video (in `data/library.json`, or from a job with `--job-id`). `fallback_rate = batches needing any fallback / batches attempted`. Counters exist only for videos ingested after they were added, and a video served from the translation cache attempts no batches; to measure, ingest non-English videos that are not cached.

## Drafting questions with the LLM

`python -m evaluation.draft_eval_set --source PLAYLIST_ID --per-video 1 --unanswerable 12` reads your indexed chunks, asks
the judge model for questions (one direct, one reworded) plus a reference answer and a verbatim evidence phrase for each,
and keeps only drafts that pass mechanical checks (the evidence must appear verbatim in the passage; no "in this excerpt").
Unanswerable drafts are generated from the video titles.

It writes `evaluation/eval_set.draft.json` with every entry marked `"needs_review": true`, and **every script ignores such
entries until you clear the flag**. For each draft: is it a question a person would ask, is the reference answer right, set
`gold_timestamp_seconds` to the real moment (the draft only knows where the passage *starts*), and check the unanswerable
ones really are not covered. Add the `cross_video` questions yourself.

Drafted questions are easier than real ones because the model wrote them with the passage in front of it, so treat scores on
them as an optimistic ceiling and keep some questions you wrote yourself. `lexical_overlap` in each draft shows how much of
the passage's wording the question reuses.

## Demo recording (checklist)

- [ ] 60-90 seconds: paste a playlist URL, show job progress, ask a question, show the answer with a timestamped citation, and click it so the video starts at the right moment.
- [ ] Record locally (OBS Studio, Xbox Game Bar on Windows, or QuickTime on macOS), since cloud-hosted ingestion may be blocked by YouTube.
- [ ] Export a GIF under ~10 MB, for example:
  `ffmpeg -i demo.mp4 -vf "fps=12,scale=900:-1:flags=lanczos" -loop 0 docs/demo.gif`
  (lower `fps` or `scale` if it is too big).
- [ ] Save it as `docs/demo.gif`, then uncomment the image line near the top of the README.
- [ ] Upload the full video as an unlisted YouTube video and link it from the resume.

## Resume bullet to script output

| Bracket | Comes from |
|---|---|
| `[N]` videos, `[H]` hours, `[C]` chunks | `ingestion_stats` corpus section (with `--source` for the test playlist) |
| `[X]%` dense-only, `[Y]%` hybrid Recall@5, eval-set size | `ablation_eval` RANKING table |
| `±[T]s` in `[Z]%` of cases | `timestamp_accuracy_eval` (say which definition) |
| refused `[X]%` of unanswerable questions | `grounding_eval` `refusal_accuracy` (quote `false_refusal_rate` with it) |
| faithfulness / correctness / relevancy | `answer_quality_eval` (say which judge model, and that it is an LLM judge) |
| latency, calls per question | `performance_eval` |
| re-ingestion `[A]` to `[B]` | `ingestion_stats` benchmark |

`python -m evaluation.render_results` prints all of these from the newest result files.
