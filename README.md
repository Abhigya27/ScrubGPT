# ScrubGPT

**ScrubGPT** turns YouTube playlist and into a searchable, timestamp-aware knowledge base. It combines YouTube metadata, transcript retrieval, multilingual translation, dense embeddings, BM25 keyword search, and grounded language-model answers in one small local-first application.

The project is designed for personal and local playlist-scale use. It favors resumability, caching, transparent progress, and a simple architecture over a distributed ingestion platform.

```text
YouTube video or playlist
        |
        v
   yt-dlp metadata
        |
        v
 YouTube transcript
        |
        v
 Timestamp-preserving chunks
        |
        +--> optional translation to English
        |
        v
 Dense + sparse embeddings
        |
        v
      Qdrant
        |
        v
 Query enhancement -> hybrid retrieval -> time periods
        |
        v
 Grounded answer with citations and timestamps
```

## What It Does

- Accepts a YouTube video URL or playlist URL.
- Resolves titles, durations, playlist order, and language hints with `yt-dlp`.
- Selects a manual transcript before an auto-generated transcript when available.
- Preserves timestamps by chunking the original transcript before translation.
- Translates non-English chunks to English through a configurable LLM provider.
- Stores dense and BM25 sparse vectors in Qdrant Cloud.
- Reuses already-indexed videos without downloading their transcripts again.
- Searches only within the submitted video or playlist.
- Merges nearby matching chunks into readable timestamp periods.
- Answers questions using retrieved transcript passages and grounding checks.
- Streams indexing and answer progress to the Streamlit interface.

## Design Principles

### Local-first

The application runs as two local processes: a FastAPI backend and a Streamlit frontend. Qdrant Cloud is the persistent source of truth for indexed chunks and vectors. Local disk is used for transcript/translation caches and lightweight library metadata.

### Sequential video processing

Only one ingestion job runs at a time per backend process, and videos inside a playlist are processed in playlist order. This avoids bursts of YouTube requests and makes progress, pause/resume, and failure reporting predictable.

Translation requests may run concurrently within one video, but the video pipeline itself remains sequential.

### Timestamps are fixed before translation

The pipeline chunks the original transcript first. Translation changes wording and length, so translating before chunking could move timestamps away from what was actually said.

### Failures keep their meaning

The job distinguishes between:

- **Expected:** no usable transcript, private video, unavailable video, or unusable text. The video is skipped and the playlist continues.
- **Temporary:** YouTube rate limiting, provider blocking, network trouble, or an LLM quota response. The job pauses and preserves completed work.
- **Unexpected:** a programming error, malformed provider response, invalid configuration, or Qdrant failure. The job enters an error state and records the traceback in backend logs.

## Architecture

```text
backend/
  main.py                 FastAPI app and HTTP routes
  jobs.py                 In-memory job store and one-worker ingest queue
  config.py               Environment-driven configuration
  guardrails.py           Input validation and rate limits
  library.py              Small local library metadata store
  models.py               Dataclasses shared across the pipeline
  util.py                 Errors, retries, quotas, and shared helpers
  index/
    embed.py              Dense and sparse embedding models
    qdrant_store.py       Collection setup, writes, deduplication, search
  ingest/
    url_parser.py         Video/playlist URL parsing
    metadata.py           YouTube metadata through yt-dlp
    transcript.py         Transcript selection, pacing, retries, caching
    chunker.py             Timestamp-preserving transcript chunking
    translate.py           Batched translation and translation cache
    cache.py               Persistent cache helpers
    pipeline.py            One-video end-to-end orchestration
  rag/
    enhance.py            Follow-up condensation and query expansion
    retrieve.py           Hybrid retrieval and period grouping
    answer.py              Grounded answer generation
    history.py             In-memory conversation history

frontend/
  streamlit_app.py         Complete Streamlit user interface

evaluation/
  run_eval.py              Retrieval evaluation runner
  eval_set.example.json    Example evaluation questions

tests/
  acceptance.py            Dependency-free acceptance checks

data/
  library.json             Local source/video metadata
  cache/                   Generated transcript and translation caches
```

## Ingestion Flow

### 1. Submit a source

The frontend sends a URL to `POST /ingest`. The backend parses it into a source kind and identifier, validates it, applies the ingest rate limit when enabled, and returns a job ID immediately.

The actual work runs on a single background worker so the API remains responsive.

### 2. Analyze metadata

For a playlist, `yt-dlp` obtains the complete flat entry list in playlist order. For a video, it obtains the title, duration, and language hint. Missing playlist durations are resolved individually only when necessary.

The backend checks configured playlist and duration limits, then records the source in `data/library.json`.

### 3. Reuse indexed videos

Before requesting a transcript, the pipeline asks Qdrant whether the video already has points. Existing videos are reused and tagged with the new source ID and playlist position. This makes re-submitting a playlist inexpensive and safe after a restart.

### 4. Retrieve a transcript

Transcript selection is performed by `backend/ingest/transcript.py`:

1. Manual transcript in the video's original language.
2. Manual English transcript.
3. Another manual transcript.
4. Auto-generated transcript when `ALLOW_GENERATED_TRANSCRIPT=true`.
5. Skip the video when no usable transcript exists.

Requests are paced, retried within configured bounds, and cached locally. A provider block or rate limit pauses the entire job instead of incorrectly marking the current video as unavailable.

### 5. Chunk the original transcript

`backend/ingest/chunker.py` creates overlapping time windows. Each chunk retains the video ID and title, source ID and playlist position, start and end timestamps, original language, transcript provenance, and a stable chunk ID.

### 6. Translate when needed

English transcripts bypass translation. Other languages are translated after chunking. Translation is batched with a character limit and concurrency cap, validates returned chunks, and falls back to smaller batches or individual requests when a model response is unusable.

Successful translations are cached by source content, provider, and model.

### 7. Embed and save

Each final chunk receives a dense vector from the configured FastEmbed model and a sparse BM25 vector from `Qdrant/bm25`. Qdrant writes are deterministic and idempotent. A video is counted as indexed only after Qdrant confirms the write.

## Retrieval and Answers

### Query enhancement

For follow-up questions, the backend uses recent conversation history to produce a standalone question. When enabled, it can also produce related rewrites and keywords. Enhancement is best-effort: if the model is unavailable, the typed question is still searched.

### Hybrid search

The retrieval layer searches dense and sparse vectors, then fuses the results:

```text
fused score = (sparse_weight * sparse_score + dense_weight * dense_score)
              / (sparse_weight + dense_weight)
```

The default weights are `0.65` sparse and `0.35` dense. Results must clear the configured absolute retrieval threshold. Additional relative-score and dense-similarity guards reduce unrelated keyword matches.

Every query is scoped to its `source_id`, so a question about one submitted video cannot retrieve chunks from another source.

### Period grouping

Matching chunks are grouped by video and time. Nearby chunks become one period; distant occurrences of the same topic remain separate. Final results are sorted by playlist position and then playback time, not by raw similarity score.

### Grounded answers

`/ask` uses the grouped transcript periods as context. The answer layer applies grounding checks and returns the answer text, cited timestamp periods, and whether the answer is grounded.

`/search` returns timestamped passages without generating a prose answer.

## Project Setup

### Requirements

- Python 3.12 or newer.
- A Qdrant Cloud collection or project with its URL and API key.
- At least one answer/translation provider key: Gemini or Groq.
- Network access to YouTube, Qdrant, and the selected LLM provider.

### Windows setup

From the repository root:

```powershell
py -3.12 -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r frontend\requirements.txt
```

Use the virtual-environment interpreter for both services. This avoids a common Windows failure where global Python can start Uvicorn but cannot import project dependencies.

### Linux/macOS setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r frontend/requirements.txt
```

### Environment configuration

Create a `.env` file in the repository root. At minimum:

```dotenv
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
GEMINI_API_KEY=your-gemini-api-key
```

The application loads `.env` from the current project directory. Never commit real API keys. If a key was exposed, revoke it and create a replacement.

## Running Locally

Start the backend in one terminal:

```powershell
& .\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Check readiness:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

A healthy response includes `"status": "ok"`, `"qdrant": true`, and `"llm": true`.

Start Streamlit in a second terminal:

```powershell
& .\.venv\Scripts\python.exe -m streamlit run frontend\streamlit_app.py
```

Open the URL printed by Streamlit, normally `http://localhost:8501`.

The frontend reads `API_BASE_URL`, which defaults to `http://localhost:8000`. If the backend uses another port, set the same value before starting Streamlit:

```powershell
$env:API_BASE_URL = "http://127.0.0.1:8001"
& .\.venv\Scripts\python.exe -m streamlit run frontend\streamlit_app.py
```

Stop local services with `Ctrl+C` in their terminal. To find leftover project processes on Windows:

```powershell
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match 'uvicorn|streamlit|backend.main|frontend\\streamlit_app.py' } |
  Select-Object ProcessId,CommandLine
```

## Configuration Reference

### Providers and storage

| Variable | Default | Purpose |
|---|---:|---|
| `QDRANT_URL` | empty | Qdrant Cloud endpoint |
| `QDRANT_API_KEY` | empty | Qdrant authentication |
| `GEMINI_API_KEY` | empty | Gemini answer, enhancement, and translation access |
| `GROQ_API_KEY` | empty | Groq answer access |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Dense embedding model |
| `SPARSE_MODEL` | `Qdrant/bm25` | Sparse embedding model |
| `CACHE_DIR` | `data/cache` | Transcript and translation cache directory |
| `LIBRARY_METADATA_PATH` | `data/library.json` | Local library metadata |

### Transcript acquisition

| Variable | Default | Purpose |
|---|---:|---|
| `ALLOW_GENERATED_TRANSCRIPT` | `true` | Allow YouTube auto-generated transcripts |
| `TRANSCRIPT_REQUEST_MIN_DELAY_SECONDS` | `1.0` | Minimum delay between transcript requests |
| `TRANSCRIPT_REQUEST_MAX_DELAY_SECONDS` | `2.5` | Maximum randomized transcript delay |
| `TRANSCRIPT_MAX_RETRIES` | `2` | Bounded retries for temporary failures |
| `TRANSCRIPT_RETRY_BASE_SECONDS` | `3.0` | Retry backoff base |
| `TRANSCRIPT_RETRY_MAX_SECONDS` | `30.0` | Maximum retry delay |
| `TRANSCRIPT_PAUSE_SECONDS` | `900` | Default pause after a temporary provider failure |
| `YOUTUBE_HTTP_PROXY` | empty | Optional HTTP proxy |
| `YOUTUBE_HTTPS_PROXY` | empty | Optional HTTPS proxy |

### Chunking, retrieval, and limits

| Variable | Default | Purpose |
|---|---:|---|
| `CHUNK_SECONDS` | `90` | Target chunk duration |
| `CHUNK_OVERLAP_SECONDS` | `20` | Time overlap between chunks |
| `MIN_CHUNK_WORDS` | `15` | Minimum useful chunk length |
| `RETRIEVAL_THRESHOLD` | `0.55` | Maximum accepted fused distance |
| `MAX_RESULTS` | `12` | Retrieval safety cap |
| `SPARSE_WEIGHT` | `0.65` | Sparse fusion weight |
| `DENSE_WEIGHT` | `0.35` | Dense fusion weight |
| `PERIOD_GAP_SECONDS` | `180` | Gap that starts a new timestamp period |
| `MAX_VIDEOS_PER_JOB` | `0` | Maximum videos; `0` means unlimited |
| `MAX_TOTAL_HOURS` | `0` | Maximum total duration; `0` means unlimited |
| `MAX_SINGLE_VIDEO_HOURS` | `0` | Maximum individual duration; `0` means unlimited |

### Translation

| Variable | Default | Purpose |
|---|---:|---|
| `TRANSLATE_PROVIDER` | `gemini` | Translation provider |
| `TRANSLATE_MODEL` | `gemini-3.5-flash-lite` | Translation model |
| `TRANSLATE_BATCH_SIZE` | `16` | Maximum chunks per translation request |
| `TRANSLATE_BATCH_MAX_CHARS` | `24000` | Maximum request text size |
| `TRANSLATE_MAX_CONCURRENCY` | `4` | Concurrent translation batches per video |
| `TRANSLATION_DAILY_CHUNK_BUDGET` | `0` | Daily limit; `0` means unlimited |
| `QUOTA_MAX_WAIT_SECONDS` | `90` | Maximum quota wait behavior |

## HTTP API

The FastAPI application is created in `backend/main.py`.

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/ingest` | Queue a video or playlist for indexing |
| `GET` | `/ingest/status/{job_id}` | Read live job progress and final results |
| `GET` | `/library` | List locally recorded sources |
| `POST` | `/search` | Retrieve timestamp periods without an answer |
| `POST` | `/search/stream` | Stream retrieval progress as NDJSON |
| `POST` | `/ask` | Generate a grounded answer |
| `POST` | `/ask/stream` | Stream answer progress as NDJSON |
| `GET` | `/healthz` | Liveness check; does not require Qdrant |
| `GET` | `/health` | Readiness and configuration detail |

Example ingestion request:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/ingest `
  -ContentType 'application/json' `
  -Body '{"url":"https://www.youtube.com/watch?v=VIDEO_ID"}'
```

Example search request:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/search `
  -ContentType 'application/json' `
  -Body '{"session_id":"demo","source_id":"VIDEO_ID","question":"What is the main idea?"}'
```

## Job States

Jobs are held in memory by the backend process. Their indexed data survives because the actual vectors live in Qdrant, but job status is lost if the backend restarts.

```text
queued -> processing -> done
                    \-> paused -> processing
                    \-> error
                    \-> rejected
```

Completed videos are detected through Qdrant when the same source is submitted again, so a restart does not require rebuilding them.

## Caches and Persistence

```text
data/
  library.json
  cache/
    transcripts/
    translations/
```

- Qdrant stores chunks, payloads, dense vectors, and sparse vectors.
- Transcript cache files avoid repeated YouTube transcript requests.
- Translation cache files avoid repeated LLM translation calls.
- `library.json` stores small source/video summaries for the frontend.
- The cache is an optimization, not the source of truth. It can be removed if necessary; indexed Qdrant data remains available.

## Testing and Evaluation

Run the dependency-light acceptance checks from the repository root:

```powershell
& .\.venv\Scripts\python.exe -m tests.acceptance
```

The checks cover URL parsing, transcript selection, generated-transcript behavior, pause semantics, local limits, translation batching, caching, and sequential job behavior without intentionally requiring live YouTube, Qdrant, or LLM calls.

Run retrieval evaluation with a configured Qdrant index:

```powershell
& .\.venv\Scripts\python.exe -m evaluation.run_eval run --set evaluation\eval_set.example.json
```

Useful evaluation questions include direct facts, paraphrases, follow-ups, terminology mismatches, and questions whose answer spans multiple videos.

## Troubleshooting

### `Can't reach the API`

The frontend and backend ports do not match, or the backend is stopped.

1. Check the backend directly: `Invoke-RestMethod http://127.0.0.1:8000/health`.
2. Start FastAPI with the `.venv` interpreter.
3. Ensure `API_BASE_URL` points to the same port used by Uvicorn.
4. Refresh the Streamlit page.

### `The indexing service encountered an unexpected error`

Read the backend terminal output and inspect the job status. The UI intentionally hides raw tracebacks, while the backend logs the underlying exception. Common causes are missing Qdrant settings, starting Uvicorn with global Python, Qdrant authentication/network failure, LLM quota/provider outage, or YouTube blocking the current network route.

### `/health` reports `unhealthy`

Check the `qdrant`, `llm`, and `config_warnings` fields. `healthz` only reports whether FastAPI is alive; `health` checks whether the required dependencies are ready.

### Port `8000` is already in use

Find the listener:

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen |
  Select-Object LocalPort,OwningProcess
```

Inspect the process:

```powershell
Get-Process -Id PROCESS_ID
```

Use another port and point Streamlit at it if the owner is unrelated:

```powershell
& .\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8001
$env:API_BASE_URL = "http://127.0.0.1:8001"
```

### YouTube transcript requests pause

This is intentional. Wait for the suggested cooldown and resume the job. Increasing concurrency or removing pacing usually makes provider blocking worse.

### Translation is slow or falls back to individual requests

The translation layer validates model output. Lower `TRANSLATE_BATCH_SIZE` or `TRANSLATE_BATCH_MAX_CHARS`, and keep `TRANSLATE_MAX_CONCURRENCY` modest. The fallback protects correctness when a provider returns malformed or incomplete batches.

## Deployment

`render.yaml` defines separate Render services for the FastAPI backend and Streamlit frontend. The backend uses `/healthz` as its platform liveness check so a temporary Qdrant outage does not restart the process and erase in-memory job state. Qdrant remains the persistent index because Render's free filesystem is ephemeral.

For deployment:

1. Create the backend service from the repository blueprint.
2. Add Qdrant and at least one LLM key as private environment variables.
3. Deploy the backend and copy its public URL.
4. Set the frontend's `API_BASE_URL` to that backend URL.
5. Do not expose provider keys to the frontend service.

YouTube may treat shared cloud IP ranges differently from a local connection. A deployment can therefore experience transcript blocks even when local development works.

## Intentional Boundaries

This project deliberately does not add Whisper or local speech recognition, video downloading or OCR, Redis/Celery/Kafka, LangGraph, a second vector database, a knowledge graph, local LLM serving, or a separate ingestion microservice.

The result is a compact system whose important behavior can be followed from URL parsing through Qdrant write and from question through grounded timestamped answer.

## Security Notes

- Keep `.env` out of version control.
- Never paste API keys into issues, screenshots, README files, or chat logs.
- Rotate any key that has been exposed.
- Use Qdrant API keys with the narrowest practical permissions.
- Keep the local API bound to `127.0.0.1` unless you intentionally need network access.
- The backend is designed not to log API keys or proxy credentials.

## Project Status

This repository is an actively developed local-first prototype. Review the repository's licensing and deployment requirements before distributing it or exposing it as a public service.