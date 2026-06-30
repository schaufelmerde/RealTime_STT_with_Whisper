"""Cloud layer: the push-to-ask assistant (Google Gemini, free tier).

The push-to-ask key no longer forces a transcription language — holding it marks what you
say as a *command* for an AI assistant instead of conversation. Those command utterances
are transcribed by Whisper like everything else (the model can't hear audio), then routed
here — instead of to the Translator — as the prompt for the agent.

The agent runs on the **Gemini API** (free tier) with **Google Search grounding** so it can
look things up live, and — when a ``ConversationStore`` is supplied — with conversation context
folded in so it can answer questions that reference what was said earlier ("who did he say to
email?").

Each request is a **single grounded call**: any available context (the store's rolling summary
of older exchanges plus the most recent verbatim lines) is assembled locally — no extra Gemini
round-trip — and prepended to the one answer call. That keeps each ask at exactly one API
request, which matters on the free tier's per-minute/day limits. With no store, the request is
just the plain grounded call.

Like the Translator, it is an inline worker thread and degrades gracefully: no API key /
no SDK / API error → the command is surfaced with an error note rather than crashing the
pipeline.

Auth: set ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``) in your .env — get a free key at
https://aistudio.google.com/apikey. Override the model with ``GEMINI_MODEL`` if you want a
newer/larger flash model than the default.
"""

import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from utils.events import TranscriptEvent
from utils.gemini import get_api_key, make_client

# A fast flash model on the free tier that supports Google Search grounding. Overridable
# via GEMINI_MODEL (e.g. a newer flash release) without touching code.
DEFAULT_MODEL = "gemini-2.5-flash"

_SYSTEM = (
    "You are a real-time voice assistant embedded in a live-conversation transcriber. "
    "The user holds a key and speaks a request; you receive the transcribed text. "
    "Answer concisely and conversationally — your reply is read at a glance mid-conversation, "
    "so lead with the answer in a sentence or two, with no preamble. Use Google Search "
    "whenever the answer depends on current, live, or factual information you can't be sure "
    "of. If asked to perform an action you have no tool for, say briefly that you can't do "
    "that yet."
)

# How many recent verbatim lines to fold into the single grounded call as conversation context.
# Injected deterministically (no separate retrieval call), so a question referencing what was
# just said ("who did he say to email?") still resolves in ONE Gemini request.
_CONTEXT_RECENT_LINES = 20


@dataclass
class AssistantMessage:
    """One push-to-ask exchange, shown in the Assistant panel.

    Emitted twice under the same ``id``: first ``status="pending"`` (query known, answer in
    flight, so the panel shows a spinner), then ``status="done"`` / ``"error"`` with the
    answer. The UI upserts by ``id``.
    """

    query: str
    status: str = "pending"            # "pending" | "done" | "error"
    answer: Optional[str] = None
    citations: List[Tuple[str, str]] = field(default_factory=list)  # (title, url)
    # Channel tag of the command that triggered this exchange (the originating capture's
    # ``source``, e.g. "Phone"/"Mic"). Lets the UI echo a phone-initiated ask back to the
    # phone screen without the agent knowing anything about the phone server.
    source: Optional[str] = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    ts: float = field(default_factory=time.time)


class AgentService:
    """Consumes command TranscriptEvents, runs a Gemini agent, emits AssistantMessages."""

    def __init__(self, command_queue: queue.Queue, assistant_sink: queue.Queue,
                 enabled: bool = True, model: Optional[str] = None, context_store=None,
                 use_context: bool = True):
        self.command_queue = command_queue
        self.assistant_sink = assistant_sink
        self.model = model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL
        # Optional ConversationStore (utils/context.py): when set, the agent folds transcript
        # context into the single grounded call. None ⇒ plain grounded call.
        self.context_store = context_store
        # Whether to actually include that context per ask. Off ⇒ a leaner, self-contained call
        # (fewer tokens, no transcript bleed into general-knowledge answers). Toggled live from the
        # phone's Context button via set_use_context.
        self.use_context = bool(use_context)
        self._client = None
        self._types = None
        # Split "client usable" from "feature on" so the UI can toggle the assistant live
        # (set_enabled) mid-recording without re-initializing the Gemini client.
        self._client_ok = self._init_client()
        self.enabled = enabled and self._client_ok

        self._running = False
        self._thread = None

    def set_enabled(self, on: bool):
        """Turn the assistant on/off live (the _loop reads self.enabled per command). Never
        enables past a usable client, so toggling on without a key is a harmless no-op."""
        self.enabled = bool(on) and self._client_ok

    def set_use_context(self, on: bool):
        """Toggle live conversation context on/off live (read per ask in _run)."""
        self.use_context = bool(on)

    def _init_client(self) -> bool:
        if not get_api_key():
            return False
        try:
            from google.genai import types
            self._client = make_client()
            self._types = types
            return self._client is not None
        except Exception:
            return False

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            # Loop blocks up to 1.0s in command_queue.get(); join so a prior session's
            # worker doesn't linger past restart.
            self._thread.join(timeout=2.0)
            self._thread = None

    def _loop(self):
        while self._running:
            try:
                event: TranscriptEvent = self.command_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            query = (event.text or "").strip()
            if not query:
                continue
            source = getattr(event, "source", None)  # originating channel, e.g. "Phone"/"Mic"

            # Pending first so the panel shows the question + spinner immediately.
            self.assistant_sink.put(AssistantMessage(query=query, id=event.id, source=source))

            if not self.enabled:
                self.assistant_sink.put(AssistantMessage(
                    query=query, id=event.id, source=source, status="error",
                    answer="Assistant is off — set GEMINI_API_KEY to enable it.",
                ))
                continue

            try:
                answer, citations = self._run(query)
                self.assistant_sink.put(AssistantMessage(
                    query=query, id=event.id, source=source, status="done",
                    answer=answer or "(no answer)", citations=citations,
                ))
            except Exception as e:
                # A failed request must not take down the worker — surface it and move on.
                print(f"[agent] request failed ({type(e).__name__}: {e}).")
                self.assistant_sink.put(AssistantMessage(
                    query=query, id=event.id, source=source, status="error",
                    answer=self._error_text(e),
                ))

    def _run(self, query: str) -> Tuple[str, List[Tuple[str, str]]]:
        """Answer one request in a SINGLE grounded call. Any available conversation context
        (rolling summary + most recent verbatim lines) is folded in locally — no separate
        retrieval round-trip — so each ask costs exactly one Gemini request. Returns
        (text, citations)."""
        context_blob = ""
        if self.use_context and self.context_store is not None and not self.context_store.is_empty():
            try:
                context_blob = self._build_context()
            except Exception as e:
                # Context assembly is best-effort — a failure must not block the answer.
                print(f"[agent] context assembly failed ({type(e).__name__}: {e}); "
                      "answering without it.")
        return self._answer(query, context_blob)

    def _build_context(self) -> str:
        """Assemble conversation context for one ask: the rolling summary of older, evicted
        exchanges (if any) followed by the most recent verbatim lines. This is where the store's
        on-demand compaction is triggered — folding aged-out lines into the summary happens here,
        only because an ask is actually about to use it, so an idle assistant never compacts."""
        store = self.context_store
        # Fold any aged-out lines into the summary now (one Gemini call, only if there's overflow
        # and a key; cached so back-to-back asks don't re-summarize). compact_now swallows its own
        # errors, and _run wraps this whole method, so a failure just answers without context.
        store.compact_now()
        parts: List[str] = []
        summ = store.summary()
        if summ:
            parts.append("Earlier conversation (summary):\n" + summ)
        recent = store.recent(_CONTEXT_RECENT_LINES)
        if recent:
            parts.append("Most recent lines:\n" + recent)
        return "\n\n".join(parts)

    def _generate(self, **kwargs):
        """Call Gemini, retrying a couple of times on transient 5xx/overload errors — the
        free-tier flash models throw 503 "model is overloaded" / 500 fairly often, and a
        single blip shouldn't sink an otherwise-fine request. Re-raises after the last
        attempt so the caller's handler can surface it."""
        last = None
        for attempt in range(3):
            try:
                return self._client.models.generate_content(**kwargs)
            except Exception as e:
                if attempt == 2 or not self._is_transient(e):
                    raise
                last = e
                time.sleep(0.8 * (2 ** attempt))  # 0.8s, then 1.6s
        raise last  # unreachable (loop either returns or raises)

    @staticmethod
    def _is_transient(e) -> bool:
        """A retryable server-side hiccup (5xx / overload) vs. a real client error (bad key,
        rate limit, blocked content). Duck-typed so we don't depend on the SDK error classes."""
        code = getattr(e, "code", None)
        if code in (500, 502, 503, 504):
            return True
        name = type(e).__name__.lower()
        msg = str(getattr(e, "message", "") or e).lower()
        return ("servererror" in name or "unavailable" in name
                or "overloaded" in msg or "try again later" in msg)

    @classmethod
    def _error_text(cls, e) -> str:
        """Turn an exception into an actionable line for the Assistant panel."""
        code = getattr(e, "code", None)
        if cls._is_transient(e):
            return "Gemini is busy right now (overloaded). Hold the key and ask again in a moment."
        if code in (401, 403):
            return "Assistant request was rejected — check that GEMINI_API_KEY is valid."
        if code == 429:
            return "Hit the Gemini rate limit. Wait a moment, then ask again."
        return f"Couldn't complete that ({type(e).__name__})."

    def _answer(self, query: str, context_blob: str) -> Tuple[str, List[Tuple[str, str]]]:
        """Grounded answer turn: Gemini searches server-side and answers, with any retrieved
        conversation context prepended. The ``tools`` list is also the seam for future
        client-side task tools."""
        config = self._types.GenerateContentConfig(
            system_instruction=_SYSTEM,
            tools=[self._types.Tool(google_search=self._types.GoogleSearch())],
            max_output_tokens=1024,
        )
        if context_blob:
            contents = (
                "Relevant context from the live conversation (use it to resolve references "
                "like names or pronouns in the request; ignore if irrelevant):\n"
                f"{context_blob}\n\nUser request: {query}"
            )
        else:
            contents = query
        response = self._generate(model=self.model, contents=contents, config=config)
        return self._extract(response)

    @staticmethod
    def _extract(response) -> Tuple[str, List[Tuple[str, str]]]:
        # .text can raise if the response was blocked / has no text parts — guard it.
        try:
            text = (response.text or "").strip()
        except Exception:
            text = ""

        citations: List[Tuple[str, str]] = []
        seen = set()
        try:
            meta = response.candidates[0].grounding_metadata
            for chunk in (getattr(meta, "grounding_chunks", None) or []):
                web = getattr(chunk, "web", None)
                uri = getattr(web, "uri", None) if web else None
                if uri and uri not in seen:
                    seen.add(uri)
                    citations.append((getattr(web, "title", None) or uri, uri))
        except Exception:
            pass
        return text, citations
