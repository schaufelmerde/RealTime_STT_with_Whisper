import math
import time
import threading
import queue
import uuid
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import sounddevice as sd
import torch

# WASAPI loopback flags a "data discontinuity" on its first buffer and every time the
# render side goes digitally silent (loopback delivers no frames during silence, so when
# audio resumes it looks like a gap). It's benign for us — VAD only fires on real speech
# and segment timestamps are wall-clock, not sample-counted — but soundcard warns on every
# occurrence, flooding the console. Silence just that one message; other warnings still show.
#
# NOTE: this module-level filter is NOT enough on its own. soundcard does
# `warnings.simplefilter('always', SoundcardRuntimeWarning)` at import (mediafoundation.py),
# which prepends to the filter list and overrides this ignore — and soundcard is imported
# lazily (in LoopbackReader._run), AFTER this runs. So LoopbackReader re-asserts this same
# filter right after importing soundcard, putting our ignore back in front. See _run below.
warnings.filterwarnings("ignore", message="data discontinuity in recording")

SAMPLE_RATE = 16000
CHUNK_SIZE = 512  # Silero VAD requires exactly 512 samples at 16kHz
LOOPBACK_RATE = 48000  # rate we pull loopback at, then resample to SAMPLE_RATE
PARTIAL_MIN_SAMPLES = 8000  # 0.5s — don't emit an interim decode of less audio than this
CMD_RELEASE_TAIL_S = 0.3  # keep capturing this long after the push-to-ask key is released, so
                          # the final word of a request isn't clipped before the hold is committed


# --------------------------------------------------------------------------------------
# Sources & readers
#
# An AudioCapture no longer hard-wires the default microphone. Capture is split into a
# pluggable *reader* (where samples come from) and the VAD/segmentation loop (unchanged).
# A reader delivers float32 **mono @ 16kHz** sample blocks of arbitrary length to a
# callback; AudioCapture re-frames them into the exact 512-sample chunks Silero needs.
#
# Two readers ship:
#   * MicReader      — a sounddevice input stream (any input device).
#   * LoopbackReader — WASAPI loopback of a render device ("system audio" / what you hear),
#                      via the `soundcard` package, downmixed + resampled to 16kHz.
#   * WebMicReader   — a phone's mic, streamed over Wi-Fi to a local HTTPS+WSS server
#                      (utils/phone_server.py) that the phone reaches by scanning a QR code.
#                      The server is a process-wide singleton; this reader is just its consumer
#                      side, resampling the browser's float32 PCM to 16kHz.
#   * ProcessReader  — per-application capture of one program (and its child processes) via
#                      the Windows Process Loopback API — the same technique OBS uses for
#                      "Application Audio Capture". Non-destructive: the target app keeps
#                      playing to your speakers (no lag, you still hear it) while we get a
#                      private, isolated copy of just that app — no virtual cable needed.
#                      The COM/ctypes plumbing lives in utils/process_loopback.py. The API is
#                      documented for build 20348+, but verified working on Win10 22H2 (19045).
# --------------------------------------------------------------------------------------


@dataclass
class AudioSource:
    """A selectable capture source surfaced to the UI's source picker."""
    key: str          # stable id, e.g. "mic:default", "mic:7", "loopback:default"
    label: str        # full human label, e.g. "Mic: Webcam", "App: Brave"
    kind: str         # "mic" | "loopback" | "process"
    tag: str          # short channel tag stamped onto each event, e.g. "Mic", "System"
    ref: object = None  # device index (mic) or speaker name (loopback); None = default
    exe: str = ""        # process image name (process sources only) — splits App vs Browser
    target_label: str = ""  # concise name shown once a category is chosen (picker's 2nd step)


# UI categories for the "Add source" picker. "App" and "Browser" both map to the same
# underlying "process" reader kind — they're split only so a web browser (which a user
# thinks of separately from a regular program) gets its own bucket in the picker.
SOURCE_CATEGORIES = ["Microphone", "Phone", "System Audio", "App", "Browser"]

_BROWSER_EXES = {
    "chrome.exe", "msedge.exe", "edge.exe", "firefox.exe", "brave.exe", "opera.exe",
    "opera_gx.exe", "vivaldi.exe", "arc.exe", "chromium.exe", "librewolf.exe",
    "waterfox.exe", "thorium.exe", "browser.exe",  # browser.exe = Yandex
}


def source_category(src: "AudioSource") -> str:
    """Bucket a source into one of SOURCE_CATEGORIES for the picker."""
    if src.kind == "mic":
        return "Microphone"
    if src.kind == "phone":
        return "Phone"
    if src.kind == "loopback":
        return "System Audio"
    if src.kind == "process":
        return "Browser" if (src.exe or "").lower() in _BROWSER_EXES else "App"
    return "App"


# Set by list_sources() to the reason loopback enumeration produced nothing, so the UI
# can show *why* (missing package vs COM error) instead of a silent "none found".
_LAST_LOOPBACK_ERROR: Optional[str] = None


def loopback_error() -> Optional[str]:
    """The last system-audio enumeration error, if any (for surfacing in the UI)."""
    return _LAST_LOOPBACK_ERROR


def _ensure_com():
    """Give the current thread a COM apartment before calling `soundcard`.

    soundcard's WASAPI/MediaFoundation calls raise CO_E_NOTINITIALIZED (0x800401f0) on
    any thread without a COM apartment. soundcard only initializes COM on the thread that
    *imports* it, so once PortAudio (sounddevice) owns COM on the main thread, Streamlit's
    script-runner thread and our loopback thread have none. Any apartment works for what we
    do (enumeration + a polling recorder), so we tolerate "already initialized"
    (S_FALSE / RPC_E_CHANGED_MODE) and just make sure one exists. No-op off Windows.
    """
    try:
        import ctypes
        ctypes.windll.ole32.CoInitializeEx(None, 0x0)  # COINIT_MULTITHREADED
    except (AttributeError, OSError):
        pass  # not Windows / no ole32


def list_sources() -> List[AudioSource]:
    """Enumerate available capture sources: input devices + system-audio loopbacks.

    Best-effort and defensive — a backend that isn't present (e.g. `soundcard` not
    installed, no WASAPI) just contributes nothing rather than raising, but records the
    reason in ``loopback_error()`` so the UI can explain a missing system-audio source.
    """
    global _LAST_LOOPBACK_ERROR
    sources: List[AudioSource] = [
        AudioSource("mic:default", "Microphone (default)", "mic", "Mic", None,
                    target_label="Default"),
    ]

    # Named input devices. Filter to the WASAPI host API when present so we don't list
    # the same physical mic 3–4× (MME / DirectSound / WDM-KS duplicates).
    try:
        hostapis = sd.query_hostapis()
        wasapi_idx = next((i for i, h in enumerate(hostapis) if "WASAPI" in h["name"]), None)
        seen = set()
        for idx, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] < 1:
                continue
            if wasapi_idx is not None and dev["hostapi"] != wasapi_idx:
                continue
            name = dev["name"]
            if name in seen:
                continue
            seen.add(name)
            sources.append(AudioSource(f"mic:{idx}", f"Mic: {name}", "mic", _short_tag(name), idx,
                                       target_label=name))
    except Exception:
        pass

    # Phone mic over Wi-Fi. A single, always-offered virtual source: adding it shows a QR the
    # phone scans to start streaming (see utils/phone_server.py). The server is started lazily
    # by the UI / make_reader, not here, so merely enumerating sources never binds a port.
    sources.append(AudioSource("phone:default", "Phone mic (scan QR)", "phone", "Phone", None,
                               target_label="Phone (scan QR)"))

    # System-audio loopback (what you hear) via soundcard. One per render device.
    _LAST_LOOPBACK_ERROR = None
    try:
        import soundcard as sc

        _ensure_com()  # the calling thread (Streamlit's) needs a COM apartment
        default_spk = sc.default_speaker()
        default_name = getattr(default_spk, "name", None)
        sources.append(
            AudioSource("loopback:default", f"System audio ({default_name})", "loopback", "System",
                        None, target_label=default_name or "System audio")
        )
        for spk in sc.all_speakers():
            if spk.name == default_name:
                continue
            sources.append(
                AudioSource(f"loopback:{spk.name}", f"System audio: {spk.name}", "loopback",
                            _short_tag(spk.name), spk.name, target_label=spk.name)
            )
    except ImportError:
        _LAST_LOOPBACK_ERROR = "soundcard not installed (pip install soundcard)"
    except Exception as e:
        _LAST_LOOPBACK_ERROR = f"{type(e).__name__}: {e}"

    # Per-application capture (Windows Process Loopback API — OBS-style). One entry per app
    # currently holding an audio session, rolled up to its root process. Best-effort and
    # Windows-only: any failure (non-Windows, unsupported build) contributes nothing.
    try:
        from utils.process_loopback import list_audio_sessions
        for sess in list_audio_sessions():
            sources.append(AudioSource(
                f"process:{sess['pid']}", f"App: {sess['name']}", "process",
                _short_tag(sess["name"]), sess["pid"],
                exe=sess["exe"], target_label=sess["name"]))
    except Exception:
        pass

    return sources


def _short_tag(name: str) -> str:
    """A compact channel tag from a device name (first word, capped)."""
    word = (name or "").strip().split()
    return (word[0][:12] if word else "Src")


def make_reader(source: AudioSource):
    """Build the reader for a selected source."""
    if source.kind == "mic":
        return MicReader(device=source.ref)
    if source.kind == "phone":
        return WebMicReader()
    if source.kind == "loopback":
        return LoopbackReader(speaker_name=source.ref)
    if source.kind == "process":
        return ProcessReader(pid=source.ref)
    raise ValueError(f"unknown source kind: {source.kind!r}")


class MicReader:
    """sounddevice input stream → 16kHz mono float32 blocks (512 samples each)."""

    def __init__(self, device: Optional[int] = None):
        self.device = device
        self._stream = None

    def start(self, on_samples: Callable[[np.ndarray], None]):
        def callback(indata, frames, time_info, status):
            # Runs in a C thread — only copy + hand off, no heavy work.
            on_samples(indata[:, 0].copy())

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SIZE,
            device=self.device,
            callback=callback,
        )
        self._stream.start()

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class WebMicReader:
    """Phone-mic-over-Wi-Fi → 16kHz mono float32 blocks.

    The consumer side of the PhoneAudioServer singleton (utils/phone_server.py): the phone
    streams float32 PCM over a secure WebSocket, the server queues it, and this reader drains
    that queue on its own thread, resampling each block from the browser's audio rate (usually
    44.1/48kHz) to SAMPLE_RATE — same one-shot-per-block approach as LoopbackReader.

    Starting/stopping this reader does NOT start/stop the server: it's a process-wide singleton
    so the phone stays "plugged in" across Start/Stop just like a real mic. On start we flush
    any audio buffered while nothing was consuming, so capture begins on live samples.
    """

    def __init__(self):
        self._server = None
        self._running = False
        self._thread = None

    def start(self, on_samples: Callable[[np.ndarray], None]):
        from utils.phone_server import get_server
        self._server = get_server()  # raises with a user-facing message if deps/cert/bind fail
        self._server.flush()
        self._running = True
        self._thread = threading.Thread(target=self._run, args=(on_samples,), daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _run(self, on_samples: Callable[[np.ndarray], None]):
        try:
            import soxr

            # Stateful streaming resampler, built lazily once we see the phone's rate and rebuilt
            # if that rate ever changes (a reconnect can negotiate a different one). Keeps filter
            # continuity across blocks instead of resetting per block.
            resampler = None
            resampler_rate = None
            while self._running:
                item = self._server.read_block(timeout=0.1)
                if item is None:
                    continue  # no audio yet (phone idle / between blocks) — poll again
                if not self._running:
                    break
                samples, rate = item
                if samples is None or samples.size == 0:
                    continue
                if rate != SAMPLE_RATE:
                    if resampler is None or rate != resampler_rate:
                        resampler = soxr.ResampleStream(rate, SAMPLE_RATE, 1, dtype=np.float32)
                        resampler_rate = rate
                    samples = resampler.resample_chunk(
                        np.ascontiguousarray(samples, dtype=np.float32))
                    if samples.size == 0:
                        continue  # resampler still priming
                on_samples(np.ascontiguousarray(samples, dtype=np.float32))
        except Exception as e:
            print(f"[phone] capture stopped ({type(e).__name__}: {e}).")


class LoopbackReader:
    """WASAPI loopback of a render device → 16kHz mono float32 blocks.

    soundcard's recorder is blocking/pull-based, so it runs on its own thread. Audio is
    captured at LOOPBACK_RATE (the typical Windows mix rate; WASAPI shared mode converts
    the device's native rate for us), downmixed to mono, then resampled to SAMPLE_RATE.
    A capture failure is logged and ends this reader's thread without taking down the app.
    """

    def __init__(self, speaker_name: Optional[str] = None, block_frames: int = 2048):
        self.speaker_name = speaker_name
        self.block_frames = block_frames
        self._running = False
        self._thread = None

    def start(self, on_samples: Callable[[np.ndarray], None]):
        self._running = True
        self._thread = threading.Thread(target=self._run, args=(on_samples,), daemon=True)
        self._thread.start()

    def stop(self):
        # NOTE: soundcard's rec.record() blocks until block_frames arrive, and WASAPI
        # loopback yields *no* frames during digital silence (see module header). So if the
        # render side is silent at stop time, the worker is parked inside record() and won't
        # observe _running until the next audio frame. We bound the join so the UI never
        # hangs; the thread is a daemon and self-terminates on its next wakeup (it re-checks
        # _running before emitting, so it can't leak a stale block into a new session).
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self, on_samples: Callable[[np.ndarray], None]):
        try:
            import soundcard as sc
            import soxr

            # soundcard's import ran `simplefilter('always', SoundcardRuntimeWarning)`, which
            # jumped ahead of the module-level ignore and re-enabled the "data discontinuity"
            # flood. Re-assert the ignore now (after the import) so ours sits back in front —
            # otherwise the spam buries everything else on the console.
            warnings.filterwarnings("ignore", message="data discontinuity in recording")

            _ensure_com()  # this capture thread needs its own COM apartment
            if self.speaker_name:
                mic = sc.get_microphone(self.speaker_name, include_loopback=True)
            else:
                mic = sc.get_microphone(sc.default_speaker().name, include_loopback=True)

            # One stateful resampler for the whole capture: a streaming resampler carries the
            # anti-alias filter state across blocks, so there's no filter-reset transient at every
            # block seam (which a fresh soxr.resample() per block would introduce). May return an
            # empty array while it primes — guard on size before emitting.
            resampler = soxr.ResampleStream(LOOPBACK_RATE, SAMPLE_RATE, 1, dtype=np.float32)
            with mic.recorder(samplerate=LOOPBACK_RATE) as rec:
                while self._running:
                    data = rec.record(numframes=self.block_frames)  # (frames, channels) float32
                    # record() may have unblocked because we're stopping (silence then
                    # shutdown) — don't push a final stale block past stop().
                    if not self._running:
                        break
                    if data is None or len(data) == 0:
                        continue
                    mono = data.mean(axis=1) if data.ndim > 1 else data
                    resampled = resampler.resample_chunk(
                        np.ascontiguousarray(mono, dtype=np.float32))
                    if resampled.size:
                        on_samples(np.ascontiguousarray(resampled, dtype=np.float32))
        except Exception as e:
            print(f"[loopback] capture stopped ({type(e).__name__}: {e}).")


class ProcessReader:
    """Per-application WASAPI process loopback → 16kHz mono float32 blocks.

    Captures only the target process and its child processes (Chromium et al. play audio
    from a child utility process, so we target the app's root PID and include its tree). The
    app keeps playing to your speakers — this is a non-destructive copy, no virtual cable, no
    added latency. Like LoopbackReader the capture is blocking/pull-based, so it runs on its
    own thread: pulled at CAPTURE_RATE stereo, downmixed to mono, resampled to SAMPLE_RATE.
    A capture failure is logged and ends this reader's thread without taking down the app.
    """

    def __init__(self, pid: int):
        self.pid = int(pid)
        self._running = False
        self._thread = None

    def start(self, on_samples: Callable[[np.ndarray], None]):
        self._running = True
        self._thread = threading.Thread(target=self._run, args=(on_samples,), daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.5)
            self._thread = None

    def _run(self, on_samples: Callable[[np.ndarray], None]):
        try:
            import soxr
            from utils.process_loopback import ProcessLoopbackCapture, CAPTURE_RATE

            # Stateful streaming resampler: keeps filter continuity across blocks (no per-block
            # reset transient). May return empty while priming — guard on size before emitting.
            resampler = soxr.ResampleStream(CAPTURE_RATE, SAMPLE_RATE, 1, dtype=np.float32)
            with ProcessLoopbackCapture(self.pid) as cap:
                while self._running:
                    data = cap.read()  # (frames, channels) float32; empty when no audio yet
                    if data is None or data.shape[0] == 0:
                        time.sleep(0.005)
                        continue
                    mono = data.mean(axis=1) if data.ndim > 1 else data
                    resampled = resampler.resample_chunk(
                        np.ascontiguousarray(mono, dtype=np.float32))
                    if resampled.size:
                        on_samples(np.ascontiguousarray(resampled, dtype=np.float32))
        except Exception as e:
            print(f"[process] capture stopped ({type(e).__name__}: {e}).")


class AudioCapture:
    """
    Drives one capture source: pulls 16kHz mono float32 from a reader, runs Silero VAD,
    and emits complete speech segments to segment_queue as dicts:

        {"audio": np.float32[], "ts_start": float, "ts_end": float,
         "forced_lang": str|None, "command": bool, "source": str|None}

    ``source`` is a short channel tag (e.g. "Mic"/"System") so several captures can share
    one segment_queue/Transcriber and the resulting events stay attributable to a channel.

    Audio stays in memory the entire time — nothing is written to disk (see PRD.md: the v1
    disk round-trip was the original race-condition bug). Timestamps are wall-clock so
    downstream consumers (and the future agent layer) can order and correlate utterances.

    Two refinements over the naive VAD loop (see PRD.md "Behavior", "Language selection"):

    * **Pre-roll** — a small rolling buffer of pre-onset chunks is prepended to each
      segment so VAD's detection latency doesn't clip the first phoneme.
    * **onset latch** — the decode language (``get_forced_lang``) and the push-to-ask flag
      (``get_command``) are read once, at speech onset, and frozen onto the segment. They
      are never re-read at transcribe time, so an async ASR backlog can't race a key the
      user has since released.
    * **max-segment flush** — a speaker who never pauses is force-flushed at
      ``max_segment_s`` (kept under Whisper's 30s window) so latency and memory stay bounded.
    * **interim partials** — while a segment is still open, a snapshot of the audio-so-far is
      flushed every ``partial_interval_s`` marked ``partial=True``, sharing the segment's
      ``segment_id`` with the eventual final flush. These drive a live, tentative line in the
      UI; the final (non-partial) flush is the committed, full-context decode. Set
      ``partial_interval_s=0`` to disable. Command (push-to-ask) segments emit no partials —
      they go to the agent, not the transcript, so a live preview would only leak the question.

    Each AudioCapture owns its **own** VAD model instance: Silero's iterator is stateful
    (an RNN), so sources must not share one model or their interleaved chunks corrupt it.
    """

    def __init__(
        self,
        segment_queue: queue.Queue,
        vad_model,
        reader,
        source: Optional[str] = None,
        silence_ms: int = 800,
        threshold: float = 0.5,
        get_forced_lang: Optional[Callable[[], Optional[str]]] = None,
        get_command: Optional[Callable[[], bool]] = None,
        preroll_ms: int = 200,
        max_segment_s: float = 25.0,
        partial_interval_s: float = 1.0,
        muted: bool = False,
    ):
        self.segment_queue = segment_queue
        self.vad_model = vad_model
        self.reader = reader
        self.source = source
        self.silence_ms = silence_ms
        self.threshold = threshold
        self.partial_interval_s = partial_interval_s
        self._get_forced_lang = get_forced_lang
        self._get_command = get_command
        self._muted = muted

        self._preroll_chunks = max(1, math.ceil(preroll_ms / 1000 * SAMPLE_RATE / CHUNK_SIZE))
        self._max_samples = int(max_segment_s * SAMPLE_RATE)

        self._raw: deque = deque()  # 16kHz mono float32 blocks straight from the reader
        self._level = 0.0  # smoothed input RMS (fast attack / slow release) for the UI meter
        self._running = False
        self._vad_thread = None

    def start(self):
        self._running = True
        self._vad_thread = threading.Thread(target=self._vad_loop, daemon=True)
        self._vad_thread.start()
        self.reader.start(self._ingest)

    def stop(self):
        self._running = False
        try:
            self.reader.stop()
        except Exception:
            pass
        # Join our OWN VAD loop too. The readers own (and join) their threads, but this thread is
        # ours and was never joined — so on a fast Stop→Start the dying loop could briefly overlap
        # a fresh one, and because VAD models are reused per cache slot across restarts, two
        # iterators could then drive the same Silero RNN and corrupt its hidden state (exactly what
        # the per-slot design exists to prevent). The loop polls _raw every ~10ms and the reader is
        # already stopped, so it exits almost immediately; bound the join so the UI never hangs, and
        # it's a daemon thread regardless.
        if self._vad_thread:
            self._vad_thread.join(timeout=1.0)
            self._vad_thread = None

    def set_muted(self, muted: bool):
        """Silence (or unsilence) this capture in place — lets a source be muted/unmuted
        mid-recording without tearing down its reader/VAD thread. While muted the VAD loop
        discards incoming audio and emits no segments; unmuting resumes from the next
        utterance (the in-flight one is dropped, not stitched across the gap)."""
        self._muted = bool(muted)

    def level(self) -> float:
        """Current smoothed input level as a linear RMS amplitude (~0..1), for a UI meter.
        Reading the single float is atomic in CPython, so no lock is needed across the
        reader/UI threads — a torn read at worst shows a slightly stale bar."""
        return self._level

    def _ingest(self, samples: np.ndarray):
        # Called from the reader's thread — keep it light. Update the input-level meter (a
        # cheap RMS with fast attack / slow release so even a coarse UI poll catches recent
        # peaks) and hand the block to the VAD loop. While muted, ease the meter toward zero
        # so the row's bar reads as silenced even though the reader stays open.
        if samples.size:
            rms = float(np.sqrt(np.mean(samples * samples)))
            if self._muted:
                self._level *= 0.6
            elif rms > self._level:
                self._level = rms                               # instant attack on louder audio
            else:
                self._level = self._level * 0.85 + rms * 0.15   # gentle release
        self._raw.append(samples)

    def _latch_forced_lang(self) -> Optional[str]:
        """Read the explicit-language state once, at onset. Never raises into the loop."""
        if self._get_forced_lang is None:
            return None
        try:
            return self._get_forced_lang()
        except Exception:
            return None

    def _latch_command(self) -> bool:
        """Read the push-to-ask key state once, at onset. Never raises into the loop."""
        if self._get_command is None:
            return False
        try:
            return bool(self._get_command())
        except Exception:
            return False

    def _command_key_down(self) -> bool:
        """Live read of whether the push-to-ask key is *currently* held (vs. the onset latch).
        Keeps a held command open as ONE segment — internal pauses included — so the whole
        hold is a single prompt for the agent, instead of VAD splitting it into many segments
        (and thus many separate, rate-limited agent calls)."""
        return self._latch_command()

    def _flush(self, buffer: list, ts_start: float, forced_lang: Optional[str], command: bool,
               segment_id: Optional[str], partial: bool = False):
        if not buffer:
            return
        self.segment_queue.put({
            "audio": np.concatenate(buffer),  # snapshot — concatenate copies, so the open
            "ts_start": ts_start,             # buffer can keep growing after a partial flush
            "ts_end": time.time(),
            "forced_lang": forced_lang,
            "command": command,
            "source": self.source,
            "segment_id": segment_id,
            "partial": partial,
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
        command = False
        segment_id: Optional[str] = None
        last_partial = 0.0  # wall-clock of the last interim flush for the current segment
        cmd_release_at: Optional[float] = None  # deadline to commit a command hold after release
        leftover = np.empty(0, dtype=np.float32)  # carries <512 samples between reader blocks
        muted_latched = False  # True once we've reset segmentation for the current muted spell

        while self._running:
            if self._muted:
                # Muted mid-stream: drop incoming audio and emit nothing. On the muting edge,
                # abandon any in-flight utterance and reset the VAD's RNN state so unmuting
                # starts a clean segment instead of resuming a half-captured one. The reader
                # stays open (cheap) so unmute is instant; we just discard what it delivers.
                self._raw.clear()
                if not muted_latched:
                    in_speech = False
                    speech_buffer = []
                    seg_samples = 0
                    leftover = np.empty(0, dtype=np.float32)
                    try:
                        vad_iter.reset_states()
                    except Exception:
                        pass
                    muted_latched = True
                time.sleep(0.05)
                continue
            muted_latched = False

            if not self._raw:
                time.sleep(0.01)
                continue

            block = self._raw.popleft()
            if leftover.size:
                block = np.concatenate([leftover, block])

            n_frames = len(block) // CHUNK_SIZE
            leftover = block[n_frames * CHUNK_SIZE:].copy()

            for i in range(n_frames):
                chunk = block[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE]
                result = vad_iter(torch.from_numpy(chunk), return_seconds=False)

                key_down = self._command_key_down()  # live push-to-ask key state, this chunk

                if result:
                    if "start" in result:
                        # While a command hold is still open, VAD re-fires "start" after each
                        # pause we absorbed — ignore it, or we'd discard the audio captured so
                        # far and split the hold. Otherwise begin a fresh segment as usual.
                        if not (in_speech and command):
                            in_speech = True
                            speech_buffer = list(preroll)  # pre-roll: chunks just before onset
                            seg_samples = sum(len(c) for c in speech_buffer)
                            seg_start = time.time() - seg_samples / SAMPLE_RATE
                            forced_lang = self._latch_forced_lang()
                            command = self._latch_command()
                            segment_id = uuid.uuid4().hex[:8]
                            last_partial = time.time()
                            cmd_release_at = None
                    elif "end" in result:
                        # A pause. For a command whose key is still held, don't cut here — keep
                        # the segment open and absorb the pause so the whole hold stays ONE
                        # prompt (release is what ends it, handled below).
                        if not (command and key_down):
                            in_speech = False
                            self._flush(speech_buffer, seg_start, forced_lang, command, segment_id)
                            speech_buffer = []
                            seg_samples = 0
                            command = False
                            cmd_release_at = None

                if in_speech:
                    # Push-to-ask is forgiving about timing: the hold key only has to be down at
                    # SOME point during the utterance, not precisely at VAD onset. Latch it sticky
                    # (onset value OR'd with every chunk after) so "start talking, then press" still
                    # routes the whole utterance to the agent — matching the live "Listening" pill.
                    if not command:
                        command = self._latch_command()
                    speech_buffer.append(chunk)
                    seg_samples += len(chunk)

                    # A command hold is one segment, bounded by the key — not by VAD pauses.
                    # While held, keep capturing (cancel any pending release); once released,
                    # run a short tail (so the last word isn't clipped) then commit the whole
                    # hold as a single command segment → a single agent call.
                    if command:
                        if key_down:
                            cmd_release_at = None
                        elif cmd_release_at is None:
                            cmd_release_at = time.time() + CMD_RELEASE_TAIL_S

                    if command and cmd_release_at is not None and time.time() >= cmd_release_at:
                        in_speech = False
                        self._flush(speech_buffer, seg_start, forced_lang, command, segment_id)
                        speech_buffer = []
                        seg_samples = 0
                        command = False
                        cmd_release_at = None
                        # We may have cut while VAD still considered itself mid-speech (key let
                        # go on the last word). Reset it so speech that continues straight into
                        # conversation re-onsets as a fresh segment instead of being swallowed.
                        try:
                            vad_iter.reset_states()
                        except Exception:
                            pass
                    elif seg_samples >= self._max_samples:
                        # Forced mid-utterance flush — speaker never paused. Keep capturing:
                        # the next chunk starts a fresh segment (new id) with no pre-roll.
                        self._flush(speech_buffer, seg_start, forced_lang, command, segment_id)
                        speech_buffer = []
                        seg_samples = 0
                        seg_start = time.time()
                        forced_lang = self._latch_forced_lang()
                        command = self._latch_command()
                        segment_id = uuid.uuid4().hex[:8]
                        last_partial = time.time()
                        cmd_release_at = None
                    elif (self.partial_interval_s > 0 and not command
                          and seg_samples >= PARTIAL_MIN_SAMPLES
                          and time.time() - last_partial >= self.partial_interval_s):
                        # Interim, tentative decode of the utterance-so-far for live display.
                        # Shares segment_id with the eventual final flush so the UI replaces
                        # the live line in place once the committed decode lands.
                        self._flush(speech_buffer, seg_start, forced_lang, command,
                                    segment_id, partial=True)
                        last_partial = time.time()

                preroll.append(chunk)


class LevelMonitor:
    """A lightweight input-level meter for a source that ISN'T being transcribed.

    Owns just a reader and computes the same fast-attack / slow-release RMS that AudioCapture
    does, so the UI's per-source meter can show a live level (and confirm audio is flowing)
    while idle — before Start, between sessions — without spinning up VAD/ASR. It's a drop-in
    for the UI: exposes the same ``level()`` as AudioCapture. Cheap by design — one RMS per
    block on the reader's own thread, and the samples are dropped (nothing consumes them).

    When the user hits Start, monitors are torn down and the real AudioCaptures take over the
    devices; on Stop they're rebuilt. So a given device is only ever opened by one of the two.
    """

    def __init__(self, reader):
        self.reader = reader
        self._level = 0.0
        self._running = False

    def start(self):
        self._running = True
        self.reader.start(self._ingest)

    def stop(self):
        self._running = False
        try:
            self.reader.stop()
        except Exception:
            pass

    def level(self) -> float:
        """Current smoothed input level (~0..1 linear RMS) for the UI meter. A single-float
        read is atomic in CPython, so no lock is needed across the reader/UI threads."""
        return self._level

    def _ingest(self, samples: np.ndarray):
        # Runs on the reader's thread — keep it light: a cheap RMS with fast attack / slow
        # release so even a coarse UI poll catches recent peaks. Guard on _running so a final
        # block delivered after stop() (loopback can lag) can't twitch a torn-down meter.
        if not self._running or not samples.size:
            return
        rms = float(np.sqrt(np.mean(samples * samples)))
        if rms > self._level:
            self._level = rms                               # instant attack on louder audio
        else:
            self._level = self._level * 0.85 + rms * 0.15   # gentle release
