# RealTime_STT_with_Whisper

Real-time **Korean–English** assistant transcriber. Speech-to-text runs **on-device**
(faster-whisper + Silero VAD, fully offline); a hybrid **cloud** layer adds translation
and, later, agent delegation. The full design lives in [PRD.md](PRD.md) — this README is
the short version.

> This repo pivoted from an earlier on-device Korean **dictation** tool (v1–v5) to a
> bilingual **conversation** transcriber. See [Project history](#project-history) for why.

---

## Status — Phase 1: local STT

Delivery is staged so the first build is small and fully local. **Phase 1 = English +
Korean speech-to-text only** — no LLM, no network, no API key.

| Phase | Scope | Network | LLM |
|-------|-------|---------|-----|
| **P1 — Local STT (current)** | KO + EN mic transcription, utterance-final, on the bus | none (offline) | none |
| P2 — Translation + cleanup | bidirectional KO↔EN translation + ASR cleanup | cloud | Claude Haiku |
| P3 — Agents | task delegation over the transcript stream | cloud | Sonnet/Opus |

It is **utterance-final, not streaming**: a line appears after the speaker pauses (VAD
detects end-of-speech), not word-by-word.

---

## How it works

```
Microphone
  ▼  sounddevice (callback stream, float32 @ 16kHz, in-memory only)
[1] Silero VAD ─ detects speech start/end ─► segment {audio, ts_start, ts_end, forced_lang}
  ▼
[2] faster-whisper (int8, CPU) ─► TranscriptEvent {text, source_lang, lang_source}
  ▼  (Phase 2: Translator enriches the event here — bypassed in P1)
TranscriptBus  ─ thread-safe pub/sub ─►  [UI sink]   [Agent orchestrator (P3)]
```

Audio never touches disk — everything is passed as in-memory numpy. The **TranscriptBus**
is the seam: the UI is one subscriber today; the future agent layer subscribes instead of
forcing a rewrite. See [PRD.md](PRD.md) for the full architecture and rationale.

---

## Language selection — hold to force

ASR is bound to an **explicit language pair** (default English + Korean) rather than
open-ended detection, because Whisper's auto-detect is fragile on short clips. Each
utterance's language is latched at speech onset:

- **Key not held → constrained auto-detect** over the pair. The hands-free default — this
  is what transcribes the *other party*, who never touches a key.
- **Hold key → force the secondary language.** Deterministic; for when *you* code-switch
  and want certainty. Forced lines are marked 🔒 in the transcript.

The hold key is a global OS hotkey (**Right Ctrl** by default, configurable in the
sidebar) because the browser/Streamlit UI can't reliably see key-up. If `pynput` isn't
available, the app runs auto-detect-only and tells you so.

---

## Quickstart

**Prerequisites (Windows 10/11):**
- Python 3.10+ (3.12 used in development), CPU-only is fine.
- [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
  ("Desktop development with C++") for building some native wheels.

**Install & run:**

```bash
python -m venv venv
venv\Scripts\activate          # Windows (PowerShell: venv\Scripts\Activate.ps1)
pip install -r requirements.txt

streamlit run main.py          # NOT `python main.py` — it's a Streamlit app
```

In the UI: pick the model size and language pair in the sidebar, press **Start**, and
speak. Hold **Right Ctrl** to force the secondary language. Phase 1 needs no API key.

---

## Configuration

Set in the sidebar (locked while recording):

| Setting | Default | Notes |
|---------|---------|-------|
| Whisper model | `small` | `small` = best speed/accuracy on CPU; drop to `tiny`/`base` if latency is high |
| Primary language | English | auto-detected (constrained to the pair) |
| Secondary language | Korean | what the hold key forces |
| Hold key | Right Ctrl | global OS hotkey |
| Silence cutoff | 800 ms | silence that ends a speech segment |
| VAD sensitivity | 0.5 | lower = picks up quieter speech |

**Phase 2 (translation):** copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`. The
"Translate (Claude)" toggle then enables KO↔EN translation + cleanup. Without a key,
transcription still works — translation just stays off.

---

## Project layout

| Path | Role |
|------|------|
| [main.py](main.py) | Streamlit UI + pipeline wiring (throwaway MVP host) |
| [utils/audio.py](utils/audio.py) | mic capture, Silero VAD, segmentation (pre-roll, `forced_lang` latch, max-length flush) |
| [utils/transcriber.py](utils/transcriber.py) | faster-whisper decode; per-segment forced lang / constrained auto-detect |
| [utils/hotkey.py](utils/hotkey.py) | `LanguageController` — language pair + OS hold-key state |
| [utils/events.py](utils/events.py) | `TranscriptEvent` schema + `TranscriptBus` (the consumer seam) |
| [utils/translator.py](utils/translator.py) | Phase 2 — Claude cleanup + KO↔EN translation |
| [PRD.md](PRD.md) | full product/design doc |

---

## Roadmap

- **P1 (now)** — local KO/EN STT, offline, on the bus.
- **P2** — bidirectional KO↔EN translation + ASR cleanup via Claude Haiku, enriching each
  event without changing the schema or local pipeline.
- **P3** — an agent orchestrator subscribes to the bus for task delegation.
- **Later** — migrate the UI off Streamlit to FastAPI + WebSocket before the agent layer;
  lightweight speaker attribution.

---

## Project history

Versions **v1–v5** (Jun–Jul 2024) were a single-speaker, offline Korean **dictation** GUI
with on-device demos (PC, Raspberry Pi, Jetson Orin Nano). Key lessons carried into v3 and
baked into the current architecture:

- v1 wrote `tmp.wav` and read it back across threads → constant file-locking races on
  Windows. **Fix: never touch disk — pass numpy in-memory.**
- "Real-time" had degraded into record → stop → batch-transcribe (push-to-talk with lag),
  and a 10s sliding re-transcription window that lost context and couldn't keep up on CPU.
- `webrtcvad` was frame-size/encoding sensitive and failed silently → replaced with Silero.

v3 is a deliberate re-scope to a **bilingual conversation** transcriber with a cloud/agent
seam — a different product, not an increment. See [PRD.md](PRD.md) for the full "why".

---

## Credits & license

Built on: [faster-whisper](https://github.com/SYSTRAN/faster-whisper),
[silero-vad](https://github.com/snakers4/silero-vad),
[sounddevice](https://python-sounddevice.readthedocs.io/),
[PyTorch](https://pytorch.org/), [Streamlit](https://streamlit.io/),
[pynput](https://github.com/moses-palmer/pynput), and (Phase 2) the
[Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python). Their respective
licenses apply.

Author's blog (Korean): https://blog.naver.com/112fkdldjs/223513947371
