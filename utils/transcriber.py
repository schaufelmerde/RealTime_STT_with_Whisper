import os
import queue
import threading
from typing import Callable, Optional, Tuple

from utils.events import TranscriptEvent

SAMPLE_RATE = 16000

# Diagnostic: print the RAW Whisper output (before the translator's Gemini cleanup can touch it)
# plus the per-segment gate reasons to the console, so we can tell whether a wrong on-screen line
# came from Whisper or from the cleanup stage. (With "Polish transcript" off — the default — the UI
# now shows this raw text directly; with it on, the UI shows Gemini's rewrite instead, so this is
# still the only window onto what the ASR actually produced.) Off by default — it walks the
# segments again and writes to stdout on every final decode; set STT_DEBUG_ASR=1 to enable.
_DEBUG_ASR = os.environ.get("STT_DEBUG_ASR", "0") not in ("0", "", "false", "False")

MIN_AUDIO_SAMPLES = 4000   # 0.25s at 16kHz — a real monosyllable clears this; anything shorter is
                           # almost always a VAD blip whose decode tends to hallucinate, so skip it.
# Interim (partial) decodes are throwaway live-preview snapshots that re-decode the whole
# utterance-so-far every partial interval. On a long monologue (up to max_segment_s ≈ 25s)
# that re-runs Whisper over ~25s of audio every second — wasted GPU that can starve nothing
# critical (finals are prioritized) but burns cycles for no benefit. Cap the partial decode to
# the most recent slice: the live line only needs the tail, and the FINAL flush still decodes
# the full audio, so the committed transcript is always complete.
_PARTIAL_MAX_SAMPLES = 10 * SAMPLE_RATE  # 10s tail
# Per-segment quality gates, read from faster-whisper's own decoder signals. These drop
# hallucinations (silence/repetition) without a blunt length floor that would erase real short words.
_NO_SPEECH_MAX = 0.8       # drop near-certain non-speech. High enough that genuine quiet/short words
                           # — which can carry an elevated no-speech score — still survive.
_COMPRESSION_MAX = 2.4     # drop repetitive hallucination ("word word word…" / a regurgitated run);
                           # real speech doesn't compress this well. Whisper's own default threshold.


class Transcriber:
    """
    Consumes speech segments ({"audio", "ts_start", "ts_end", "forced_lang", "command"})
    from segment_queue, runs faster-whisper in-memory, and emits a raw TranscriptEvent
    (text + source_lang + lang_source). Segments flagged ``command`` (push-to-ask) go to
    ``command_queue`` for the agent; all others go to ``transcript_queue`` for the
    Translator to enrich.

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
        command_queue: Optional[queue.Queue] = None,
        partial_queue: Optional[queue.Queue] = None,
        get_hotwords: Optional[Callable[[], str]] = None,
        beam_size: int = 5,
    ):
        self.segment_queue = segment_queue
        self.transcript_queue = transcript_queue
        self.command_queue = command_queue
        self.partial_queue = partial_queue
        self.model = model
        self.lang_pair = tuple(lang.lower() for lang in lang_pair)
        # Beam width for FINAL (committed) decodes. Higher = more accurate but slower; 1 = greedy.
        # Set once per session from the UI (fixed for the run, like the model/language). Interim
        # partials always use beam_size=1 (cheap throwaway), regardless of this.
        self.beam_size = max(1, int(beam_size))
        # Proper-noun biasing: read live (per decode) so vocabulary edits in the UI apply to the
        # next utterance with no restart. See utils/glossary.py — this is decode-time prompt
        # conditioning, not training.
        self._get_hotwords = get_hotwords

        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        if _DEBUG_ASR:
            # Printed the moment a fresh worker spins up, so you can confirm the new code is live
            # (the worker is a daemon thread started once — editing this file does nothing until
            # recording is restarted) and know this window is where RAW lines will appear.
            print("[asr] diagnostic active — RAW Whisper output prints here. Set STT_DEBUG_ASR=0 to silence.",
                  flush=True)
        self._thread = threading.Thread(target=self._transcribe_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            # The loop blocks up to 1.0s in segment_queue.get(); give it room to exit so a
            # prior session's worker doesn't linger past restart.
            self._thread.join(timeout=2.0)
            self._thread = None

    def _decode(self, audio, language: Optional[str], beam_size: Optional[int] = None,
                hotwords: Optional[str] = None):
        # None ⇒ use the session's configured beam (final decodes). Partials pass beam_size=1.
        if beam_size is None:
            beam_size = self.beam_size
        segments, info = self.model.transcribe(
            audio,
            language=language,
            beam_size=beam_size,
            vad_filter=False,
            # Hint phrases (proper nouns) biasing the spelling prior; None/"" → no effect.
            hotwords=hotwords or None,
            # Decode each utterance independently. Conditioning on prior text is the main driver
            # of Whisper's runaway repetition: one hallucinated word becomes the seed for the next.
            # Our segments are already single utterances, so there's nothing to gain and a lot to
            # lose by carrying text across them.
            condition_on_previous_text=False,
        )
        return segments, info

    @staticmethod
    def _debug_log(segments, source, lang, lang_source, n_samples):
        """Print raw Whisper output + per-segment gate signals for one final decode. ``segments``
        must already be materialized (a list), since the caller also gates over it. Shows what the
        ASR actually produced — the on-screen line is the post-Gemini rewrite, not this."""
        dur = n_samples / SAMPLE_RATE
        kept, dropped = [], []
        for s in segments:
            ns = getattr(s, "no_speech_prob", 0.0)
            cr = getattr(s, "compression_ratio", 0.0)
            if ns > _NO_SPEECH_MAX:
                dropped.append((s.text.strip(), f"no_speech={ns:.2f}"))
            elif cr > _COMPRESSION_MAX:
                dropped.append((s.text.strip(), f"compression={cr:.2f}"))
            else:
                kept.append(s.text.strip())
        raw = "  ".join(t for t in kept if t) or "(nothing kept)"
        print(f"[asr] {source or '?':<8} {lang}/{lang_source} {dur:4.2f}s  RAW: {raw}", flush=True)
        for text, why in dropped:
            print(f"[asr]          dropped ({why}): {text!r}", flush=True)

    @staticmethod
    def _text_of(segments) -> str:
        """Join the kept segments, dropping per-segment hallucinations by faster-whisper's own
        signals: near-certain non-speech (silence) and repetitive/regurgitated runs. This is a
        quality gate, not the old blunt no_speech_prob<=0.6 cut that erased real quiet words."""
        kept = []
        for s in segments:
            if s.no_speech_prob > _NO_SPEECH_MAX:
                continue
            if getattr(s, "compression_ratio", 0.0) > _COMPRESSION_MAX:
                continue
            kept.append(s.text)
        return "".join(kept).strip()

    def _best_in_pair(self, info) -> str:
        """Higher-scoring member of the selected pair from Whisper's language probs."""
        probs = dict(getattr(info, "all_language_probs", None) or [])
        a, b = self.lang_pair
        return a if probs.get(a, 0.0) >= probs.get(b, 0.0) else b

    def _next_batch(self) -> list:
        """Block for one segment, then drain everything else immediately available and
        collapse it: keep all finals (and the latest partial per segment), so a backlog of
        stale interim decodes can't make the live line lag behind the speaker."""
        try:
            first = self.segment_queue.get(timeout=1.0)
        except queue.Empty:
            return []
        items = [first]
        while True:
            try:
                items.append(self.segment_queue.get_nowait())
            except queue.Empty:
                break
        return self._collapse(items)

    @staticmethod
    def _collapse(items: list) -> list:
        finals = [it for it in items if not it.get("partial")]
        final_ids = {it.get("segment_id") for it in finals}
        latest_partial = {}
        for it in items:
            # A partial is moot once its segment has a final in the same batch; drop it.
            if it.get("partial") and it.get("segment_id") not in final_ids:
                latest_partial[it.get("segment_id")] = it  # later items win → newest kept
        # Finals first (commit ASAP), then the freshest interim decode per open segment.
        return finals + list(latest_partial.values())

    def _transcribe_loop(self):
        while self._running:
            for segment in self._next_batch():
                if not self._running:
                    break
                if segment.get("partial"):
                    self._handle_partial(segment)
                else:
                    self._handle_final(segment)

    def _handle_partial(self, segment):
        """Tentative, throwaway decode of an in-progress utterance for live display. One
        greedy pass (beam_size=1) keeps it cheap; the final commit re-decodes with full
        context. Never reaches the translator/agent — display only."""
        if self.partial_queue is None:
            return
        audio = segment["audio"]
        if len(audio) < MIN_AUDIO_SAMPLES:
            return
        # Only decode the recent tail (see _PARTIAL_MAX_SAMPLES) — bounds the cost of partials on
        # long utterances. The final commit re-decodes the full audio, so nothing is lost.
        if len(audio) > _PARTIAL_MAX_SAMPLES:
            audio = audio[-_PARTIAL_MAX_SAMPLES:]
        forced = segment.get("forced_lang")
        segments, info = self._decode(audio, forced, beam_size=1)
        text = self._text_of(segments)
        if not text:
            return
        resolved_lang = (forced or info.language or "").lower()
        self.partial_queue.put(TranscriptEvent(
            text=text,
            source_lang=resolved_lang,
            lang_source="forced" if forced else "detected",
            source=segment.get("source"),
            ts_start=segment["ts_start"],
            ts_end=segment["ts_end"],
            segment_id=segment.get("segment_id"),
            partial=True,
        ))

    def _handle_final(self, segment):
        audio = segment["audio"]
        if len(audio) < MIN_AUDIO_SAMPLES:
            return

        forced = segment.get("forced_lang")
        command = bool(segment.get("command"))
        # Proper-noun hint for this decode (read live so UI edits apply next utterance). Only on
        # finals — partials stay cheap. Empty/None ⇒ no biasing.
        hotwords = self._get_hotwords() if self._get_hotwords else None

        if forced:
            segments, info = self._decode(audio, forced, hotwords=hotwords)
            resolved_lang = forced.lower()
            lang_source = "forced"
        else:
            segments, info = self._decode(audio, None, hotwords=hotwords)
            detected = (info.language or "").lower()
            if detected in self.lang_pair:
                resolved_lang = detected
            else:
                # Top guess fell outside the pair — re-decode constrained to it.
                resolved_lang = self._best_in_pair(info)
                segments, info = self._decode(audio, resolved_lang, hotwords=hotwords)
            lang_source = "detected"

        # Materialize once: the segments generator is single-pass, and both the debug log and the
        # quality gate need to walk it. (Decoding already happened lazily on first iteration.)
        segments = list(segments)
        if _DEBUG_ASR:
            self._debug_log(segments, segment.get("source"), resolved_lang, lang_source, len(audio))
        text = self._text_of(segments)
        if text:
            event = TranscriptEvent(
                text=text,
                source_lang=resolved_lang,
                lang_source=lang_source,
                source=segment.get("source"),
                ts_start=segment["ts_start"],
                ts_end=segment["ts_end"],
                is_command=command,
                segment_id=segment.get("segment_id"),
            )
            # Commands (push-to-ask) split off to the agent and never reach the
            # transcript/translator; conversation flows on as before.
            if command and self.command_queue is not None:
                self.command_queue.put(event)
                # The key may have been pressed mid-utterance, after interim partials for this
                # segment already streamed into the live preview. Re-send the (command) final on
                # the partial channel so the UI can drop that lingering line — its text belongs
                # in the Assistant panel, not the transcript.
                if self.partial_queue is not None:
                    self.partial_queue.put(event)
            else:
                self.transcript_queue.put(event)
        elif not command and self.partial_queue is not None and segment.get("segment_id"):
            # The committed decode produced nothing (the quality gate dropped it as silence or
            # repetitive hallucination), yet interim partials may already be showing a live
            # preview line for this segment. With no final event to supersede it, that line would
            # hang in the UI forever — so send a no-text final on the preview channel purely to
            # retire it. (Command segments resend their own final above.)
            self.partial_queue.put(TranscriptEvent(
                text="",
                source_lang=resolved_lang,
                lang_source=lang_source,
                source=segment.get("source"),
                ts_start=segment["ts_start"],
                ts_end=segment["ts_end"],
                segment_id=segment.get("segment_id"),
                partial=False,
            ))
