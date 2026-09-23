"""Retry helper, the one LLM entry point (`_chat`), and small shared functions."""
import contextvars
import functools
import logging
import re
import time

from backend import config

log = logging.getLogger(__name__)


class UserFacingError(Exception):
    """An error whose message is safe and useful to show the end user as-is."""


class QuotaExhausted(UserFacingError):
    """The LLM provider (or our own daily translation budget) says "not now, try later".

    Jobs treat this as a PAUSE, not a failure: the video isn't marked skipped, and
    re-submitting the link resumes because finished videos are skipped.
    """

    def __init__(self, message: str, retry_after: int | None = None):
        super().__init__(message)
        self.retry_after = retry_after  # seconds until trying again is sensible, if known


def retry(attempts: int = 3, base_delay: float = 1.0):
    """Retry a sync function with exponential backoff. UserFacingError is never retried."""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(1, attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except UserFacingError:
                    raise
                except Exception as exc:
                    if attempt == attempts:
                        raise
                    delay = base_delay * 2 ** (attempt - 1)
                    log.warning(
                        "%s failed (%s: %s); retrying in %.1fs (%d/%d)",
                        fn.__name__, type(exc).__name__, exc, delay, attempt, attempts - 1,
                    )
                    time.sleep(delay)

        return wrapper

    return decorator


def format_timestamp(seconds: int) -> str:
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def format_duration(seconds: float) -> str:
    """'2h 5m', '8m 38s', '45s': for messages, not timestamps."""
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m {secs}s" if minutes else f"{secs}s"


def body_text(text: str) -> str:
    """The displayable transcript text of a chunk.

    Chunk text used to be f"{title}\\n\\n{body}" and this stripped the prefix. The
    title is no longer embedded (it repeated in every chunk of a video and pulled
    every chunk toward the video's general topic, which cost precision), so the
    stored text IS the body. Kept as a function because the call sites read better
    with it, and because it keeps working if a prefix ever comes back.
    """
    return text


# --- turning a question into a search query --------------------------------------
# People ask this app *navigational* questions: "when was self-supervised learning
# taught", "where does he explain hashmaps". The navigational half ("when was",
# "taught", "explain") says nothing about the content being looked for, but it is
# embedded and BM25-matched all the same, which drags in transcript passages about
# teaching, classes and professors instead of the actual topic. Removing it makes
# both retrieval legs see just the topic: "self supervised learning".
_META_WORDS = frozenset(
    """
    a an the this that these those of in on at to for from by with about
    when where what which who whom whose why how
    is are was were be been being do does did done
    can could will would should shall may might must
    he she it they them his her their we you i me my our us your
    time timestamp timestamps moment moments point part parts section sections
    video videos clip lecture lectures course tutorial
    talk talks talked talking discuss discusses discussed discussing
    explain explains explained explaining explanation
    teach teaches taught teaching cover covers covered covering
    mention mentions mentioned mentioning introduce introduces introduced
    say says said tell tells told speak speaks spoke spoken
    show shows shown showed describe describes described
    find search look give get tell me us please exactly first start starts begin begins
    and or but if then than there here
    """.split()
)
_WORD_RE = re.compile(r"[^\w\s]", re.UNICODE)


def search_query(question: str) -> str:
    """The topic to retrieve on, with navigational phrasing removed.

    Falls back to the question as typed whenever stripping would leave nothing,
    so a question made entirely of these words still searches for something.
    """
    if not config.STRIP_NAVIGATION_WORDS:
        return question
    cleaned = _WORD_RE.sub(" ", question.lower())
    kept = [w for w in cleaned.split() if w not in _META_WORDS]
    return " ".join(kept) if kept else question


def _as_text(content) -> str:
    """Chat models sometimes return a list of content parts instead of a string."""
    if isinstance(content, str):
        return content
    return "".join(p if isinstance(p, str) else p.get("text", "") for p in content)


# --- live notices from inside LLM calls ------------------------------------------
# Whoever is waiting on an LLM call (a job, a streamed chat answer) can register a notifier so
# the UI can say "rate limited, waiting 23 s" instead of appearing to hang.
_notifier: contextvars.ContextVar = contextvars.ContextVar("llm_notifier", default=None)


def set_notifier(fn):
    """Register fn(message) for the current context (copied into worker threads by asyncio.to_thread)."""
    return _notifier.set(fn)


def current_notifier():
    """The notifier registered for this context, or None.

    A plain `concurrent.futures.ThreadPoolExecutor` (unlike `asyncio.to_thread`) does NOT copy
    context vars into the worker thread on its own, so code that spawns its own thread pool
    (translate.py, for concurrent translation batches) must read this in the calling thread and
    call `set_notifier` again inside each worker, or rate-limit "waiting Ns" messages silently
    stop reaching the UI once the job is running inside such a pool.
    """
    return _notifier.get()


def _notify(message: str) -> None:
    fn = _notifier.get()
    if fn is not None:
        try:
            fn(message)
        except Exception:
            log.exception("notifier failed")


# --- provider errors -----------------------------------------------------------------
def _status_of(exc: Exception) -> int | None:
    return getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)


def _is_rate_limit(exc: Exception) -> bool:
    if _status_of(exc) == 429:
        return True
    name, text = type(exc).__name__.lower(), str(exc).lower()
    return "ratelimit" in name or "resourceexhausted" in name or "429" in text or "rate limit" in text or "quota" in text


_WAIT_RE = re.compile(r"(?:try again|retry)\s+in\s+([0-9hms.\s]+)", re.I)
_UNIT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(ms|h|m|s)", re.I)
_UNIT_SECONDS = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}


def _parse_wait_text(text: str) -> float | None:
    """'Please try again in 8m38.4s' / 'retry in 20.3s' -> seconds."""
    match = _WAIT_RE.search(text)
    if not match:
        return None
    parts = _UNIT_RE.findall(match.group(1))
    if not parts:
        return None
    return sum(float(num) * _UNIT_SECONDS[unit.lower()] for num, unit in parts)


def _retry_after_seconds(exc: Exception) -> float | None:
    """How long the provider says to wait: the retry-after header if present, else the message text."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers:
        try:
            value = headers.get("retry-after") or headers.get("Retry-After")
            if value is not None:
                return float(value)
        except (TypeError, ValueError, AttributeError):
            pass
    return _parse_wait_text(str(exc))


# --- the LLM ---------------------------------------------------------------------------
def _settings(role: str) -> tuple[str | None, str | None]:
    if role == "translate":
        return config.TRANSLATE_PROVIDER, config.TRANSLATE_MODEL
    if role == "enhance":
        return config.ENHANCE_PROVIDER, config.ENHANCE_MODEL
    return config.LLM_PROVIDER, (config.GROQ_MODEL if config.LLM_PROVIDER == "groq" else config.GEMINI_MODEL)


@functools.lru_cache(maxsize=4)
def get_llm(role: str = "default"):
    """A LangChain chat model. role="translate" / "enhance" may point at their own provider/model (own quota)."""
    provider, model = _settings(role)
    if provider == "groq":
        if not config.GROQ_API_KEY:
            raise UserFacingError("GROQ_API_KEY isn't set, but Groq was selected. Set it in .env.")
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, api_key=config.GROQ_API_KEY, temperature=0)
    if provider == "gemini":
        if not config.GEMINI_API_KEY:
            raise UserFacingError("GEMINI_API_KEY isn't set, but Gemini was selected. Set it in .env.")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, google_api_key=config.GEMINI_API_KEY, temperature=0)
    raise UserFacingError("No LLM API key is configured. Set GROQ_API_KEY or GEMINI_API_KEY in .env.")


_ATTEMPTS = 4


def _chat(system: str, user: str, role: str = "default", *, attempts: int | None = None, max_wait: float | None = None) -> str:
    """The one place the LLM is called. Translation, query enhancement, condensation and answering all go through here.

    Quota-aware: on a rate limit it honors the provider's retry-after and waits it out if that's short
    (<= QUOTA_MAX_WAIT_SECONDS); a longer wait (a daily limit) raises QuotaExhausted so a job can pause
    instead of burning retries and marking every remaining video skipped.

    `attempts` and `max_wait` override those defaults for one call. Optional steps (query enhancement) pass small
    values so they give up quickly and let the caller carry on without them, instead of sleeping out a rate limit.
    """
    llm = get_llm(role)
    messages = [("system", system), ("human", user)]
    total = attempts or _ATTEMPTS
    limit = config.QUOTA_MAX_WAIT_SECONDS if max_wait is None else max_wait
    for attempt in range(1, total + 1):
        try:
            reply = llm.invoke(messages)
            return _as_text(reply.content).strip()
        except UserFacingError:
            raise
        except Exception as exc:
            if _status_of(exc) == 413:  # bigger than the model's per-request token limit: waiting won't help
                raise UserFacingError(
                    "That request is larger than the language model's token limit. Try a more specific question."
                ) from exc
            if _is_rate_limit(exc):
                wait = _retry_after_seconds(exc)
                if wait is not None and wait > limit:
                    raise QuotaExhausted(
                        f"The language model's quota is used up. It should reset in about {format_duration(wait)}.",
                        int(wait),
                    ) from exc
                if attempt == total:
                    raise QuotaExhausted(
                        "The language model keeps rate limiting requests. Please try again in a minute or two.",
                        int(wait or 60),
                    ) from exc
                delay = wait + 1.0 if wait is not None else min(5.0 * 2 ** (attempt - 1), 60.0)
                delay = min(delay, max(limit, 1.0) + 1.0)  # an optional step never sleeps longer than its own cap (+1s: the provider's 'wait N' slack)
                _notify(f"The language model is rate limiting us. Waiting {delay:.0f}s, then retrying...")
                log.warning("LLM rate limited (%s); waiting %.0fs (attempt %d/%d)", type(exc).__name__, delay, attempt, total)
                time.sleep(delay)
                continue
            if attempt == total:
                raise
            delay = 2.0 * 2 ** (attempt - 1)
            log.warning("LLM call failed (%s: %s); retrying in %.0fs (%d/%d)", type(exc).__name__, exc, delay, attempt, total - 1)
            time.sleep(delay)


def make_chain(system_template: str, human_template: str, role: str = "default", attempts: int | None = None, max_wait: float | None = None):
    """LangChain (LCEL) chain: ChatPromptTemplate -> `_chat`.

    Used by the condensation, query-enhancement and answer chains so every LLM use shares the single
    quota-aware `_chat(system, user)` helper. The result is a Runnable: it can be piped into an output parser
    (`make_chain(...) | JsonOutputParser()`), which is how rag/enhance.py uses it.

    Templates are f-string style: a literal brace in a prompt must be written {{ or }}.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda

    prompt = ChatPromptTemplate.from_messages([("system", system_template), ("human", human_template)])

    def _run(prompt_value) -> str:
        system_msg, human_msg = prompt_value.to_messages()
        return _chat(system_msg.content, human_msg.content, role=role, attempts=attempts, max_wait=max_wait)

    return prompt | RunnableLambda(_run)
