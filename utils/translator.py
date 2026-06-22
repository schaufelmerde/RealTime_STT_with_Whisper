"""Cloud layer: transcript cleanup + bidirectional KO↔EN translation via Claude.

One Claude call per utterance does two jobs at once: (1) clean up the raw, often
noisy/code-switched ASR text (spacing, punctuation, obvious errors) and (2) translate
it into the other language. This replaces both an NMT model (opus-mt is X→EN only and
weak on conversational Korean) and a separate Korean-spacing post-processor, and it
reuses the same client the agent layer (PRD.md M1) will need.

The Translator is an inline transform stage: it consumes raw TranscriptEvents from
transcript_queue, enriches them, and publishes to the TranscriptBus. If the LLM is
unavailable (no API key, no SDK, network/API error) it degrades to passthrough — the
transcript is always published, translation just goes missing for that line.
"""

import json
import os
import queue
import threading

from utils.events import TranscriptBus, TranscriptEvent

# Latency-appropriate tier for the per-utterance real-time path. Raise to
# "claude-sonnet-4-6" or "claude-opus-4-8" when translation quality matters more
# than latency.
TRANSLATION_MODEL = "claude-haiku-4-5"

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

# Structured output guarantees parseable JSON (no prose, no markdown fences).
_SCHEMA = {
    "type": "object",
    "properties": {
        "clean_text": {"type": "string"},
        "translation": {"type": "string"},
    },
    "required": ["clean_text", "translation"],
    "additionalProperties": False,
}


class Translator:
    """Enriches TranscriptEvents with cleanup + translation, then publishes to the bus.

    mode: "auto" (translate to the opposite language), "ko-en", or "en-ko".
    """

    def __init__(self, transcript_queue: queue.Queue, bus: TranscriptBus, mode: str = "auto", enabled: bool = True):
        self.transcript_queue = transcript_queue
        self.bus = bus
        self.mode = mode
        self._client = None
        self.enabled = enabled and self._init_client()

        self._running = False
        self._thread = None

    def _init_client(self) -> bool:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return False
        try:
            import anthropic
            self._client = anthropic.Anthropic()
            return True
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

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _enrich(self, event: TranscriptEvent) -> TranscriptEvent:
        target = self._target_lang(event.source_lang)
        target_name = _LANG_NAMES.get(target, "English")
        try:
            response = self._client.messages.create(
                model=TRANSLATION_MODEL,
                max_tokens=1024,
                system=_SYSTEM,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Source language: {event.source_lang or 'unknown'}\n"
                        f"Target language: {target_name}\n"
                        f"Transcript: {event.text}"
                    ),
                }],
                output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
            )
            text = next(b.text for b in response.content if b.type == "text")
            data = json.loads(text)
            event.clean_text = data.get("clean_text") or event.text
            event.translation = data.get("translation") or None
            event.target_lang = target
        except Exception as e:
            # Translation failed — leave the transcript intact and publish anyway. Log once
            # so a misconfigured key / SDK / network is distinguishable from "model returned
            # nothing" (e.g. an SDK too old to accept output_config fails here every time).
            print(f"[translator] enrichment failed ({type(e).__name__}: {e}); publishing raw transcript.")
            event.translation = None
        return event

    def _loop(self):
        while self._running:
            try:
                event = self.transcript_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if self.enabled:
                event = self._enrich(event)

            self.bus.publish(event)
