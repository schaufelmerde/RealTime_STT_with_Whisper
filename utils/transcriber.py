import queue
import threading
from typing import Optional, Tuple

from utils.events import TranscriptEvent

MIN_AUDIO_SAMPLES = 8000  # 0.5s at 16kHz — skip segments shorter than this


class Transcriber:
    """
    Consumes speech segments ({"audio", "ts_start", "ts_end", "forced_lang"}) from
    segment_queue, runs faster-whisper in-memory, and puts a raw TranscriptEvent
    (text + source_lang + lang_source) on transcript_queue for the Translator to enrich.

    Language is decided per-segment, never from a single instance-wide setting (see
    PRD.md "Language selection (P1)"):

    * ``segment["forced_lang"]`` set  → force that decode language (hold key was down at
      onset); ``lang_source="forced"``.
    * ``forced_lang is None``         → auto-detect, then **constrain to the selected
      pair**: if Whisper's top guess lands outside ``lang_pair`` (common on short clips —
      KO mislabeled ja/zh), re-decode forcing the higher-scoring pair member.
      ``lang_source="detected"``.
    """

    def __init__(
        self,
        segment_queue: queue.Queue,
        transcript_queue: queue.Queue,
        model,
        lang_pair: Tuple[str, str] = ("en", "ko"),
    ):
        self.segment_queue = segment_queue
        self.transcript_queue = transcript_queue
        self.model = model
        self.lang_pair = tuple(lang.lower() for lang in lang_pair)

        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._transcribe_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _decode(self, audio, language: Optional[str]):
        segments, info = self.model.transcribe(
            audio,
            language=language,
            beam_size=3,
            vad_filter=False,
        )
        return segments, info

    def _best_in_pair(self, info) -> str:
        """Higher-scoring member of the selected pair from Whisper's language probs."""
        probs = dict(getattr(info, "all_language_probs", None) or [])
        a, b = self.lang_pair
        return a if probs.get(a, 0.0) >= probs.get(b, 0.0) else b

    def _transcribe_loop(self):
        while self._running:
            try:
                segment = self.segment_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            audio = segment["audio"]
            if len(audio) < MIN_AUDIO_SAMPLES:
                continue

            forced = segment.get("forced_lang")

            if forced:
                segments, info = self._decode(audio, forced)
                resolved_lang = forced.lower()
                lang_source = "forced"
            else:
                segments, info = self._decode(audio, None)
                detected = (info.language or "").lower()
                if detected in self.lang_pair:
                    resolved_lang = detected
                else:
                    # Top guess fell outside the pair — re-decode constrained to it.
                    resolved_lang = self._best_in_pair(info)
                    segments, info = self._decode(audio, resolved_lang)
                lang_source = "detected"

            text_parts = []
            for seg in segments:
                if seg.no_speech_prob > 0.6:
                    continue
                text_parts.append(seg.text)

            text = "".join(text_parts).strip()
            if text:
                self.transcript_queue.put(TranscriptEvent(
                    text=text,
                    source_lang=resolved_lang,
                    lang_source=lang_source,
                    ts_start=segment["ts_start"],
                    ts_end=segment["ts_end"],
                ))
