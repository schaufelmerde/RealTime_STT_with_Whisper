import math
import time
import threading
import queue
from collections import deque
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
import torch

SAMPLE_RATE = 16000
CHUNK_SIZE = 512  # Silero VAD requires exactly 512 samples at 16kHz


class AudioCapture:
    """
    Captures microphone audio via sounddevice, runs Silero VAD per-chunk, and emits
    complete speech segments to segment_queue as dicts:

        {"audio": np.float32[], "ts_start": float, "ts_end": float, "forced_lang": str|None}

    Audio stays in memory the entire time — nothing is written to disk (see PRD.md:
    the v1 disk round-trip was the original race-condition bug). Timestamps are
    wall-clock so downstream consumers (and the future agent layer) can order and
    correlate utterances.

    Two refinements over the naive VAD loop (see PRD.md "Behavior", "Language selection"):

    * **Pre-roll** — a small rolling buffer of pre-onset chunks is prepended to each
      segment so VAD's detection latency doesn't clip the first phoneme.
    * **forced_lang latch** — the decode language is read from ``get_forced_lang`` once,
      at speech onset, and frozen onto the segment. It is never re-read at transcribe
      time, so an async ASR backlog can't race a hold key the user has since released.
    * **max-segment flush** — a speaker who never pauses is force-flushed at
      ``max_segment_s`` (kept under Whisper's 30s window) so latency and memory stay bounded.
    """

    def __init__(
        self,
        segment_queue: queue.Queue,
        vad_model,
        silence_ms: int = 800,
        threshold: float = 0.5,
        get_forced_lang: Optional[Callable[[], Optional[str]]] = None,
        preroll_ms: int = 200,
        max_segment_s: float = 25.0,
    ):
        self.segment_queue = segment_queue
        self.vad_model = vad_model
        self.silence_ms = silence_ms
        self.threshold = threshold
        self._get_forced_lang = get_forced_lang

        self._preroll_chunks = max(1, math.ceil(preroll_ms / 1000 * SAMPLE_RATE / CHUNK_SIZE))
        self._max_samples = int(max_segment_s * SAMPLE_RATE)

        self._frame_deque: deque = deque()
        self._running = False
        self._stream = None
        self._vad_thread = None

    def start(self):
        self._running = True
        self._vad_thread = threading.Thread(target=self._vad_loop, daemon=True)
        self._vad_thread.start()
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SIZE,
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self):
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _audio_callback(self, indata, frames, time_info, status):
        # Runs in a C thread — only append, no heavy work
        self._frame_deque.append(indata[:, 0].copy())

    def _latch_forced_lang(self) -> Optional[str]:
        """Read the hold-key state once, at onset. Never raises into the loop."""
        if self._get_forced_lang is None:
            return None
        try:
            return self._get_forced_lang()
        except Exception:
            return None

    def _flush(self, buffer: list, ts_start: float, forced_lang: Optional[str]):
        if not buffer:
            return
        self.segment_queue.put({
            "audio": np.concatenate(buffer),
            "ts_start": ts_start,
            "ts_end": time.time(),
            "forced_lang": forced_lang,
        })

    def _vad_loop(self):
        from silero_vad import VADIterator

        vad_iter = VADIterator(
            self.vad_model,
            sampling_rate=SAMPLE_RATE,
            threshold=self.threshold,
            min_silence_duration_ms=self.silence_ms,
        )

        preroll: deque = deque(maxlen=self._preroll_chunks)
        speech_buffer: list = []
        seg_samples = 0
        in_speech = False
        seg_start = 0.0
        forced_lang: Optional[str] = None

        while self._running:
            if not self._frame_deque:
                time.sleep(0.01)
                continue

            chunk = self._frame_deque.popleft()
            if len(chunk) != CHUNK_SIZE:
                continue

            chunk_tensor = torch.from_numpy(chunk)
            result = vad_iter(chunk_tensor, return_seconds=False)

            if result:
                if "start" in result:
                    in_speech = True
                    speech_buffer = list(preroll)  # pre-roll: chunks captured just before onset
                    seg_samples = sum(len(c) for c in speech_buffer)
                    seg_start = time.time() - seg_samples / SAMPLE_RATE
                    forced_lang = self._latch_forced_lang()
                elif "end" in result:
                    in_speech = False
                    self._flush(speech_buffer, seg_start, forced_lang)
                    speech_buffer = []
                    seg_samples = 0

            if in_speech:
                speech_buffer.append(chunk)
                seg_samples += len(chunk)
                if seg_samples >= self._max_samples:
                    # Forced mid-utterance flush — speaker never paused. Keep capturing:
                    # the next chunk starts a fresh segment with no pre-roll (no silence gap).
                    self._flush(speech_buffer, seg_start, forced_lang)
                    speech_buffer = []
                    seg_samples = 0
                    seg_start = time.time()
                    forced_lang = self._latch_forced_lang()

            preroll.append(chunk)
