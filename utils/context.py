"""Conversation memory for the assistant: a transcript store with on-demand compaction.

The push-to-ask assistant (``utils/agent.py``) needs to answer questions that depend on what
was said earlier ("who did he say to email?"), but a long meeting transcript is too big and too
expensive to hand the model verbatim every turn. ``ConversationStore`` is a **TranscriptBus
subscriber** (no pipeline change — that is what the seam is for) that keeps the conversation in
two tiers:

* a **recent verbatim window** — the last ``recent_max`` finalized lines, kept word-for-word; and
* a **rolling summary** — older lines folded into a compact running summary by a cheap Gemini
  call, then evicted. This is the "compaction": detail collapses to gist as it ages.

**Compaction is on-demand, not background.** ``ingest`` only appends (cheap; it runs on the bus
publisher thread) and never calls Gemini. The fold-older-lines-into-summary step runs lazily and
synchronously the moment the assistant actually assembles context for an ask
(``compact_now``, called from utils/agent.py). That is the whole point: if the assistant is off —
or on but never invoked — the store makes **zero** Gemini calls, so it can't quietly drain the
free-tier quota in the background. Memory stays bounded without any API: past a hard cap
``ingest`` sheds the oldest verbatim lines (old context is lost, but it can't grow without
bound), and an ask folds whatever overflow remains into the summary in a single call (cached, so
back-to-back asks don't re-summarize).

The agent reaches in through three retrieval methods (``recent`` / ``summary`` / ``search``) plus
``compact_now`` — it pulls (and folds) context only when a question needs it. With no API key the
store degrades to a bounded raw window (no summary), so memory still can't grow without bound.

Auth: ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` (free tier); override the model with
``GEMINI_CONTEXT_MODEL``.
"""

import os
import threading
from collections import deque
from typing import List, Optional

from utils.gemini import make_client

DEFAULT_MODEL = "gemini-2.5-flash"

_COMPACT_SYSTEM = (
    "You maintain a running summary of a live, possibly bilingual (Korean/English) "
    "conversation as it streams in. You are given the summary so far plus the next batch of "
    "transcript lines that are about to scroll out of the verbatim window. Return an updated, "
    "concise summary (a few short bullets or paragraphs, under ~200 words) that preserves "
    "names, numbers, decisions, action items, questions asked, and the flow of topics. Do not "
    "invent anything — summarize only what is given. Output the summary text only."
)


def _fmt_line(line: dict) -> str:
    tag = f"[{line['source']}] " if line.get("source") else ""
    lang = f"({line['lang']}) " if line.get("lang") else ""
    return f"{tag}{lang}{line['text']}"


class ConversationStore:
    """Two-tier (verbatim window + compacted summary) transcript memory for the assistant."""

    def __init__(self, model: Optional[str] = None, recent_max: int = 30,
                 compact_threshold: int = 50):
        self.model = model or os.environ.get("GEMINI_CONTEXT_MODEL") or DEFAULT_MODEL
        self.recent_max = recent_max
        # An ask folds everything past recent_max into the summary; compact_threshold floors the
        # hard cap below (kept as a parameter for back-compat with the old background trigger).
        self.compact_threshold = max(compact_threshold, recent_max + 1)
        # Compaction is on-demand (compact_now), so when nothing consumes the store — assistant
        # off, or on but never asked — the window would otherwise grow unbounded. Cap it here and
        # let ingest shed the oldest with no API call; a later ask still folds the rest in.
        self._hard_cap = max(self.compact_threshold * 6, 200)

        self._recent: deque = deque()  # {ts, source, lang, text}; managed manually (no maxlen)
        self._summary: str = ""
        self._lock = threading.Lock()

        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        try:
            self._client = make_client()  # None if no key — store degrades to a bounded raw window
        except Exception:
            self._client = None

    # --- bus subscriber -----------------------------------------------------------------
    def ingest(self, event) -> None:
        """TranscriptBus callback. Runs on the publisher thread — keep it cheap and
        non-blocking (the bus contract): append-only, never calls Gemini. Only finalized
        conversation lines are stored; partials and push-to-ask commands are skipped."""
        if getattr(event, "partial", False) or getattr(event, "is_command", False):
            return
        text = (getattr(event, "display_text", None) or getattr(event, "text", "") or "").strip()
        if not text:
            return
        with self._lock:
            self._recent.append({
                "ts": getattr(event, "ts_start", 0.0) or 0.0,
                "source": getattr(event, "source", None),
                "lang": getattr(event, "source_lang", None),
                "text": text,
            })
            # Bound memory without an API call: if no ask has compacted the window for a long
            # time (assistant idle), shed the oldest verbatim lines past the hard cap. A normal
            # ask trims the window far below this by folding overflow into the summary.
            while len(self._recent) > self._hard_cap:
                self._recent.popleft()

    # --- lifecycle ----------------------------------------------------------------------
    def start(self) -> None:
        """No-op. Compaction is on-demand (see compact_now), so the store runs no background
        thread; kept for lifecycle symmetry with the other pipeline stages in main.py."""

    def stop(self) -> None:
        """No-op counterpart to start() — the store owns no thread to join."""

    def compact_now(self) -> None:
        """Fold any lines beyond the verbatim window into the rolling summary in ONE Gemini call,
        then evict them. Called synchronously by the assistant as it assembles context for an ask
        (utils/agent.py) — compaction happens on demand, only when the summary is about to be
        used, never on a background timer. So an assistant that's off (or on but never invoked)
        triggers no Gemini calls here at all.

        Best-effort: with no client (no key) it just sheds the overflow to bound memory; on a
        Gemini error it leaves the prior summary and the overflow in place for the next ask."""
        # Snapshot the overflow (oldest lines beyond the verbatim window) under the lock, but run
        # the Gemini call OUTSIDE it so ingest never blocks. Only this method pops from the left
        # and ingest only appends to the right, so the leftmost N stay stable across the call.
        # (Asks are serialized on the single agent worker thread, so two compactions never race.)
        with self._lock:
            overflow_n = len(self._recent) - self.recent_max
            if overflow_n <= 0:
                return
            overflow = [dict(self._recent[i]) for i in range(overflow_n)]
            prior_summary = self._summary

        if self._client is None:
            # No summarizer → just bound memory by dropping the oldest. Older context is lost,
            # but the recent window holds and memory can't grow without bound.
            with self._lock:
                for _ in range(overflow_n):
                    self._recent.popleft()
            return

        try:
            new_summary = self._summarize(prior_summary, overflow)
        except Exception as e:
            # Keep the overflow in the window (don't drop it un-summarized); the next ask retries.
            print(f"[context] compaction failed ({type(e).__name__}: {e}).")
            return

        with self._lock:
            for _ in range(overflow_n):
                self._recent.popleft()
            if new_summary:
                self._summary = new_summary

    def _summarize(self, prior_summary: str, lines: List[dict]) -> str:
        from google.genai import types
        body = ""
        if prior_summary:
            body += f"Summary so far:\n{prior_summary}\n\n"
        body += "New lines:\n" + "\n".join(_fmt_line(line) for line in lines)
        config = types.GenerateContentConfig(
            system_instruction=_COMPACT_SYSTEM,
            max_output_tokens=1024,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        resp = self._client.models.generate_content(
            model=self.model, contents=body, config=config
        )
        try:
            return (resp.text or "").strip()
        except Exception:
            return prior_summary  # keep what we had rather than wiping the summary

    # --- retrieval (exposed to the agent as tools) --------------------------------------
    def recent(self, n: int = 20) -> str:
        """The last ``n`` verbatim lines, formatted ``[Source] (lang) text``."""
        with self._lock:
            lines = list(self._recent)[-max(1, n):]
        return "\n".join(_fmt_line(line) for line in lines)

    def summary(self) -> str:
        """The rolling summary of older, evicted context ("" if nothing has been compacted)."""
        with self._lock:
            return self._summary

    def search(self, query: str, limit: int = 12) -> str:
        """Case-insensitive keyword match over the verbatim window (and the summary if it
        matches). Simple substring/keyword search — no vector store."""
        terms = [t for t in (query or "").strip().lower().split() if t]
        if not terms:
            return ""
        with self._lock:
            lines = list(self._recent)
            summ = self._summary
        hits = [line for line in lines if any(t in line["text"].lower() for t in terms)]
        out = [_fmt_line(line) for line in hits[-limit:]]
        if summ and any(t in summ.lower() for t in terms):
            out.insert(0, f"(from earlier summary) {summ}")
        return "\n".join(out)

    def is_empty(self) -> bool:
        with self._lock:
            return not self._recent and not self._summary
