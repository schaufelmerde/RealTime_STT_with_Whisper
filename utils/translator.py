"""Cloud layer: transcript cleanup + bidirectional KO↔EN translation via Gemini.

One Gemini call per utterance does two jobs at once: (1) clean up the raw, often
noisy/code-switched ASR text (spacing, punctuation, obvious errors) and (2) translate it
into the other language. This replaces both an NMT model (opus-mt is X→EN only and weak on
conversational Korean) and a separate Korean-spacing post-processor. Structured output (a
JSON schema) guarantees parseable output; thinking is disabled to keep this per-utterance
path low-latency.

The Translator is an inline transform stage: it consumes raw TranscriptEvents from
transcript_queue, enriches them, and publishes to the TranscriptBus. If the LLM is
unavailable (no API key, no SDK, network/API error) it degrades to passthrough — the
transcript is always published, translation just goes missing for that line.

Auth: set ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``); free key at
https://aistudio.google.com/apikey. Override the model with ``GEMINI_TRANSLATE_MODEL``.
"""

import difflib
import json
import os
import queue
import re
import threading
from collections import deque

from utils.events import TranscriptBus, TranscriptEvent
from utils.gemini import get_api_key, make_client

# Strip whitespace + common punctuation so two transcripts of the same sound compare equal
# regardless of ASR spacing/punctuation jitter (Korean spacing especially is unreliable).
_NORM_RE = re.compile(r"[\s.,!?;:'\"()\[\]{}…·~\-–—、。，！？　]+")

# Flash is fast and free-tier — fitting for the per-utterance hot path. Override via
# GEMINI_TRANSLATE_MODEL (e.g. a newer flash release) without touching code.
TRANSLATION_MODEL = "gemini-2.5-flash"

_LANG_NAMES = {"ko": "Korean", "en": "English"}

_SYSTEM = (
    "You are a real-time interpreter for Korean–English conversations. You receive a "
    "raw speech-to-text transcript of a single spoken utterance (which may be noisy, "
    "mis-spaced, or mix Korean and English), along with its detected source language "
    "and the requested target language.\n\n"
    "Return two things:\n"
    "1. clean_text: the original utterance with obvious ASR errors, spacing, and "
    "punctuation fixed. Do NOT add, remove, or paraphrase meaning. Keep the original "
    "language(s) — if the speaker code-switched, preserve that.\n"
    "2. translation: a natural, conversational translation of the utterance into the "
    "target language, faithful to spoken register.\n\n"
    "If the input is empty or is not real speech, return empty strings for both."
)

# Structured-output schema. A Pydantic model gives the SDK an exact schema and lets us read
# response.parsed. Defined defensively so a missing pydantic doesn't break import (in which
# case _init_client disables enrichment and the pipeline runs passthrough).
try:
    from pydantic import BaseModel

    class _Cleanup(BaseModel):
        clean_text: str
        translation: str
except Exception:
    _Cleanup = None


class Translator:
    """Enriches TranscriptEvents with cleanup + translation, then publishes to the bus.

    mode: "auto" (translate to the opposite language), "ko-en", or "en-ko".

    **Echo suppression** (``echo_suppress``): on shared-ground headset mics, the playback
    signal couples into the mic line (crosstalk), so system/call audio gets transcribed
    twice — once cleanly via loopback, once echoed through the mic. Crosstalk is strictly
    one-directional (loopback is a clean digital tap that can never contain the mic), so we
    drop a non-clean (mic) final when it overlaps a recent clean (loopback) final in time
    **and** matches its text. The user's own speech dominates the mic and won't match any
    loopback line, so it survives. ``clean_sources`` is the set of source tags treated as
    authoritative (the loopback channels).
    """

    def __init__(self, transcript_queue: queue.Queue, bus: TranscriptBus, mode: str = "auto",
                 enabled: bool = True, model: str = None,
                 echo_suppress: bool = False, clean_sources=None,
                 echo_window_s: float = 4.0, echo_ratio: float = 0.78,
                 get_glossary_block=None, cleanup: bool = False):
        self.transcript_queue = transcript_queue
        self.bus = bus
        self.mode = mode
        # Whether to show Gemini's *rewrite* of the original line (clean_text) instead of the raw
        # ASR. Off by default: instrumentation showed the raw Whisper text is usually MORE faithful
        # than the cleanup, which can drift. Translation is produced either way — this only governs
        # the displayed original. Toggleable live (set_cleanup); the _loop reads it per utterance.
        self.cleanup = bool(cleanup)
        self.model = model or os.environ.get("GEMINI_TRANSLATE_MODEL") or TRANSLATION_MODEL
        # Proper-noun glossary as a text backstop: a callable returning a "keep these exact
        # spellings" block (read live so UI edits apply next utterance). See utils/glossary.py.
        self._get_glossary_block = get_glossary_block
        self._client = None
        self._types = None
        # Split "client usable" from "feature on" so the UI can toggle translation live
        # (set_enabled) mid-recording without re-initializing the Gemini client.
        self._client_ok = self._init_client()
        self.enabled = enabled and self._client_ok

        self.echo_suppress = echo_suppress
        self._clean_sources = set(clean_sources or ())
        self.echo_window_s = echo_window_s
        self.echo_ratio = echo_ratio
        # Recent clean (loopback) finals to test mic lines against; bounded so it can't grow.
        self._recent_clean: deque = deque(maxlen=64)

        self._running = False
        self._thread = None

    def set_enabled(self, on: bool):
        """Turn translation on/off live (the _loop reads self.enabled per utterance). Never
        enables past a usable client, so toggling on without a key is a harmless no-op."""
        self.enabled = bool(on) and self._client_ok

    def set_cleanup(self, on: bool):
        """Turn the displayed-original cleanup (Gemini rewrite) on/off live. When off, the raw
        ASR text is shown; translation is unaffected. _enrich reads this per utterance."""
        self.cleanup = bool(on)

    def set_clean_sources(self, clean_sources, echo_suppress=None):
        """Live-update the authoritative (loopback/system) tag set when a source is added or
        removed mid-recording — otherwise this is baked at construction. Rebinding the set is
        atomic in CPython, so the _loop thread reading it concurrently in _is_echo/_remember_clean
        is safe. Pass echo_suppress to also flip the gate (a live-added system channel should turn
        dedup on; losing the last one should turn it off)."""
        self._clean_sources = set(clean_sources or ())
        if echo_suppress is not None:
            self.echo_suppress = bool(echo_suppress)

    def _init_client(self) -> bool:
        if not get_api_key():
            return False
        if _Cleanup is None:
            return False
        try:
            from google.genai import types
            self._client = make_client()
            self._types = types
            return self._client is not None
        except Exception:
            return False

    def _target_lang(self, source_lang: str) -> str:
        if self.mode == "ko-en":
            return "en"
        if self.mode == "en-ko":
            return "ko"
        # auto: the opposite of whatever was detected (default to EN for anything
        # that isn't clearly Korean).
        return "en" if source_lang == "ko" else "ko"

    @staticmethod
    def _norm(text: str) -> str:
        return _NORM_RE.sub("", (text or "").lower())

    @staticmethod
    def _similar(a: str, b: str, ratio: float) -> bool:
        """Same utterance? True if one is a (non-trivial) substring of the other — the mic
        bleed often yields a fragment of the clean line — or their char-sequence ratio
        clears ``ratio``."""
        if not a or not b:
            return False
        short, long = (a, b) if len(a) <= len(b) else (b, a)
        if len(short) >= 6 and short in long:
            return True
        return difflib.SequenceMatcher(None, a, b).ratio() >= ratio

    @staticmethod
    def _overlaps(event: TranscriptEvent, rec: dict, slack: float = 1.5) -> bool:
        """Do the two utterances overlap in wall-clock time (with slack for the two sources'
        independent VAD segmentation drifting apart)?"""
        return event.ts_start <= rec["ts_end"] + slack and rec["ts_start"] <= event.ts_end + slack

    def _is_echo(self, event: TranscriptEvent) -> bool:
        """True if this mic final merely echoes a recent clean (loopback) line. The clean
        side is authoritative and never suppressed."""
        if event.source in self._clean_sources:
            return False
        norm = self._norm(event.text)
        if len(norm) < 4:  # too short to match on without risking false positives
            return False
        return any(
            self._overlaps(event, rec) and self._similar(norm, rec["norm"], self.echo_ratio)
            for rec in self._recent_clean
        )

    def _remember_clean(self, event: TranscriptEvent) -> None:
        if event.source in self._clean_sources:
            self._recent_clean.append({
                "ts_start": event.ts_start,
                "ts_end": event.ts_end,
                "norm": self._norm(event.text),
            })

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            # Loop blocks up to 1.0s in transcript_queue.get(); join so a prior session's
            # worker doesn't linger past restart.
            self._thread.join(timeout=2.0)
            self._thread = None

    def _enrich(self, event: TranscriptEvent) -> TranscriptEvent:
        target = self._target_lang(event.source_lang)
        target_name = _LANG_NAMES.get(target, "English")
        try:
            config = self._types.GenerateContentConfig(
                system_instruction=_SYSTEM,
                response_mime_type="application/json",
                response_schema=_Cleanup,
                # No thinking on the per-utterance path — flash supports budget 0, which
                # cuts latency. (A "pro" override would reject 0; flash is the default.)
                thinking_config=self._types.ThinkingConfig(thinking_budget=0),
            )
            glossary = self._get_glossary_block() if self._get_glossary_block else ""
            response = self._client.models.generate_content(
                model=self.model,
                contents=(
                    (f"{glossary}\n" if glossary else "")
                    + f"Source language: {event.source_lang or 'unknown'}\n"
                    + f"Target language: {target_name}\n"
                    + f"Transcript: {event.text}"
                ),
                config=config,
            )
            data = response.parsed or _Cleanup(**json.loads(response.text))
            # Only let the rewrite become the displayed original when cleanup is on; otherwise
            # leave clean_text None so display_text falls back to the (more faithful) raw ASR.
            if self.cleanup:
                event.clean_text = data.clean_text or event.text
            event.translation = data.translation or None
            event.target_lang = target
        except Exception as e:
            # Translation failed — leave the transcript intact and publish anyway. Log once
            # so a misconfigured key / SDK / network is distinguishable from "model returned
            # nothing".
            print(f"[translator] enrichment failed ({type(e).__name__}: {e}); publishing raw transcript.")
            event.translation = None
        return event

    def _loop(self):
        while self._running:
            try:
                event = self.transcript_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if self.echo_suppress and self._is_echo(event):
                # Mic crosstalk of system audio already captured cleanly via loopback —
                # drop it entirely: no transcript line, no (wasted) translation call.
                continue
            # ORDERING ASSUMPTION: a mic echo is only suppressed if its matching clean line is
            # already in _recent_clean. Mic and loopback are independent captures feeding one
            # queue, processed here in arrival order — so this relies on the clean (loopback)
            # final reaching us before the mic echo. That holds in practice because loopback is a
            # zero-latency digital tap while the mic path adds acoustic + re-VAD delay; if the two
            # ever raced the other way the echo would slip through (a duplicate line), not corrupt
            # state. A hold-mic-finals-one-beat buffer would close that gap if it ever shows up.
            self._remember_clean(event)

            if self.enabled:
                event = self._enrich(event)

            self.bus.publish(event)
