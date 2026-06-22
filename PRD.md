# PRD: Korean–English Real-Time Assistant Transcriber v3 (Hybrid, phased)

## What changed from v2 → v3

v2 scoped a **single-speaker, offline, no-LLM Korean dictation tool**. The actual
goal is a **real-time assistant transcriber + translator for Korean–English
*conversations*, with agent delegation later**. Those two are not the same product,
and v2 ruled out (offline-only, no cloud, no LLM) exactly what the assistant/agent
goal requires.

v3 commits to a **hybrid** split:

- **Local** (offline, low-latency, private): microphone capture, VAD, ASR. This is
  the part that genuinely benefits from running on-device.
- **Cloud** (quality, reasoning): an LLM (Claude) for translation + transcript
  cleanup now, and the agent layer later. Bidirectional KO↔EN, code-switch-tolerant
  cleanup, and agent delegation all fall out of one decision.

The seam between the two halves is an explicit **transcript event stream**
(`TranscriptBus`), so the future agent layer *subscribes* instead of forcing a
rewrite.

> **Privacy note (the cost of hybrid):** transcript text leaves the device for the
> LLM in layers 3–4. Layers 1–2 (audio + ASR) stay local. If some conversations are
> sensitive, gate which utterances are sent to the cloud (see Risks).
> **In Phase 1 (below) nothing leaves the device** — there is no LLM call at all; this
> cost only applies from Phase 2 on.

---

## Delivery phases

The hybrid vision above is unchanged — only the **delivery order** is staged so the
first shippable build is small and fully local. **Initial features = English + Korean
speech-to-text only.** Translation and cleanup come afterward.

| Phase | Scope | Network | LLM |
|-------|-------|---------|-----|
| **P1 — Local STT (initial features)** | KO + EN microphone transcription, utterance-final, on the bus | none (fully offline) | none |
| P2 — Translation + cleanup | bidirectional KO↔EN translation + ASR cleanup enriching each event | cloud | Claude Haiku |
| P3 — Agents (M1) | task delegation over the transcript stream | cloud | Sonnet/Opus |

**Phase 1 is the current target.** It is exactly the local half of the architecture
(capture → VAD → ASR → event), with raw events going straight to the `TranscriptBus`
and the UI. No `anthropic` dependency, no API key, nothing leaves the device. Phase 2
adds the Translator stage between the Transcriber and the bus **without changing the
event schema or the local pipeline** — `clean_text` / `translation` / `target_lang` go
from always-`None` to populated.

---

## Why the original v1 failed (still relevant — keep avoiding these)

1. **File I/O race conditions** — v1 wrote `tmp.wav` and read it back from another
   thread; on Windows this caused constant file-locking errors and a `while True:
   os.remove(...)` retry loop. **Fix: never touch disk — pass numpy float32 in-memory.**
2. **Real-time was silently abandoned** — by v2–v5 it became record → silence →
   batch transcribe → display (push-to-talk with lag), not streaming.
3. **CPU inference too slow for full re-transcription** — re-transcribing a 10s
   sliding buffer every loop never kept up.
4. **The 10s sliding window was architecturally wrong** — lost context, did
   redundant compute, and forced the `last_del=True` "always drop the last word" hack.
5. **webrtcvad was brittle** — frame-size/encoding sensitive, failed silently.
6. **UX was never real-time** — three manual clicks per utterance.

---

## Problem statement

Bilingual Korean–English conversations have no good lightweight real-time
transcription + translation tool. Cloud captioners are English-first and don't
handle KO↔EN code-switching well; offline tools can't translate bidirectionally or
reason. v3 keeps ASR on-device for latency/privacy and uses an LLM for the parts
that need quality and (later) reasoning.

---

## Goals

| #  | Goal | Phase |
|----|------|-------|
| 1  | Continuously transcribe Korean **and** English speech from the microphone in near real-time | **P1** |
| 2  | Show each utterance within ~1–2s of the speaker pausing | **P1** |
| 3  | Run ASR locally/offline on a modern consumer CPU | **P1** |
| 6  | Emit a structured transcript event stream that any consumer (UI now, agents later) can subscribe to | **P1** |
| 7  | Bound ASR to a user-selected language pair, with a hold-to-override secondary trigger | **P1** |
| 4  | Translate each utterance bidirectionally (KO↔EN) via the LLM | P2 |
| 5  | Clean up noisy ASR output (spacing, punctuation, obvious errors, code-switch) via the LLM | P2 |
| M1 | Agent delegation / task hand-offs over the transcript stream | P3 (stretch) |
| M2 | Lightweight speaker attribution (turn/channel-based, not full diarization) | Stretch |

## Non-goals (v3)

- Full speaker diarization (who-said-what via embeddings). The event schema carries
  an optional `speaker` field so this can be added later without migration.
- Custom vocabulary fine-tuning.
- Mobile / embedded deployment.
- True word-by-word live captioning (we are **utterance-final**, see below).
- EXE packaging (after the core is stable).

**Deferred, not cut (Phase 2):** bidirectional translation and LLM cleanup are *not* in
Phase 1, but they remain planned — see Delivery phases above, not the permanent
exclusions in this list.

---

## "Real-time" — be precise

This system is **utterance-final**, not streaming: results appear after the speaker
pauses (VAD detects end-of-speech), not word-by-word as they talk. That is the right
tradeoff for *conversation* transcription and translation. True live captioning needs
a LocalAgreement streaming policy (e.g. `whisper_streaming`) — a different, more
complex design we explicitly do not want here.

---

## Architecture

```
LOCAL — Phase 1 (offline, low-latency)                 CLOUD — Phase 2+ (quality, reasoning)
┌───────────────────────────────┐
│ Microphone                    │
│   ▼                           │
│ sounddevice (callback stream) │  raw float32 @ 16kHz, in-memory only
│   ▼                           │
│ [1] Silero VAD (per 512-frame)│  detects speech start/end
│   ▼  segment_queue            │  {audio, ts_start, ts_end, forced_lang}
│ [2] faster-whisper (int8 CPU) │  → TranscriptEvent{source_lang, text}
└──────────────┬────────────────┘
               │ transcript_queue
               ▼
        ┌─────────────────────────────────┐
        │ [3] Translator (Claude Haiku 4.5)│  PHASE 2: cleanup + bidirectional translation,
        │     enriches the event           │  structured JSON output
        └──────────────┬──────────────────┘
                       ▼  bus.publish(event)   P1: raw event published here directly;
                                              P2: routed through [3] first, enriched
              ╔═══════════════════════╗
              ║  TranscriptBus (seam) ║  thread-safe pub/sub
              ╚═══════╦═══════╦═══════╝
                      ▼       ▼
                 [UI sink]  [Agent orchestrator]  ← Phase 3 (M1), subscribes later
```

In **Phase 1** the Translator stage ([3]) is bypassed: the Transcriber publishes the
raw event straight to the bus. **Phase 2** inserts the Translator without touching the
local pipeline or the event schema.

**Key decisions vs v2:**

| Issue | v2 | v3 |
|-------|----|----|
| Audio I/O | in-memory numpy ✅ (keep) | in-memory numpy |
| VAD | Silero ✅ (keep) | Silero |
| ASR timing | segment-on-VAD ✅ (keep) | segment-on-VAD |
| Translation | Whisper `task=translate` (X→EN only — can't do EN→KO) | LLM, bidirectional KO↔EN |
| Cleanup/spacing | `kss` post-processor | folded into the LLM cleanup pass |
| Language detection | per-segment auto-detect, fragile on code-switch | auto-detect **constrained to the selected language pair** + hold-to-override the secondary (P1); LLM cleans residual mislabels (P2) |
| Consumer coupling | hardwired into Streamlit `session_state` | `TranscriptBus` seam; UI is one subscriber, agents another |
| Assistant/agents | out of scope ("v3 feature") | the reason the seam exists |

---

## Segment schema (VAD → Transcriber)

What the VAD stage puts on `segment_queue`. Short-lived and internal — it exists only
between speech onset and the ASR call, and never carries to the bus:

```python
segment = {
    "audio":       np.ndarray,  # float32 @ 16kHz mono, in-memory only (never disk)
    "ts_start":    float,       # wall-clock speech onset (includes pre-roll, see Behavior)
    "ts_end":      float,       # wall-clock end-of-speech (silence threshold crossed)
    "forced_lang": str | None,  # "ko"/"en" forces the decode; None → constrained auto-detect
}
```

`forced_lang` is the single language primitive (see "Language selection (P1)"). It is
latched **at speech onset** from the live hold-key state and never re-read at transcribe
time, so an async ASR backlog can't race a key the user has since released.

---

## Transcript event schema

The single representation that flows through the pipeline and over the bus
(`utils/events.py`):

```python
TranscriptEvent(
    id:          str,            # short unique id
    ts_start:    float,          # wall-clock speech start
    ts_end:      float,          # wall-clock speech end
    source_lang: str,            # resolved decode language: "ko" | "en" | ...
    lang_source: str | None,     # provenance: "forced" (hold key) | "detected" (constrained auto-detect)
    text:        str,            # raw ASR output
    speaker:     str | None,     # unused in MVP — reserved for M2
    clean_text:  str | None,     # LLM-cleaned original
    translation: str | None,     # LLM translation into target_lang
    target_lang: str | None,     # "ko" | "en"
)
```

Raw events (`text` + `source_lang` + `lang_source`) are produced by the Transcriber.
`lang_source` records *how* the language was decided — `"forced"` when the hold key
pinned it, `"detected"` when constrained auto-detect chose it — so the UI/QA can tell a
missed key-press from a detection miss. In **Phase 1** events go straight to the bus with
`clean_text` / `translation` / `target_lang` left `None`.
In **Phase 2** the Translator fills those fields before publishing. Either way consumers
see one shape — the optional fields just populate later.

---

## Component stack

**Phase 1 needs only** Capture, VAD, ASR, the event seam, and the UI — no `anthropic`,
no API key. The Translation/cleanup row below is Phase 2.

| Layer | Tech | Notes |
|-------|------|-------|
| Capture | `sounddevice` | more reliable than PyAudio on Windows |
| VAD | `silero-vad` (+ `torch`) | neural, noise-robust; 512-sample frames @ 16kHz |
| ASR | `faster-whisper`, `compute_type="int8"`, CPU | model size selectable; `small` default |
| Translation/cleanup | `anthropic` SDK, `claude-haiku-4-5` | **Phase 2** — latency-appropriate tier; structured JSON output |
| Event seam | `utils/events.py` (`TranscriptBus`) | thread-safe pub/sub; the agent on-ramp |
| UI | `Streamlit` | MVP validation only — see "UI" below |

**Model choice for translation (Phase 2).** `claude-haiku-4-5` is the latency-appropriate tier
for the per-utterance real-time path. It's a single constant
(`TRANSLATION_MODEL` in `utils/translator.py`) — raise it to `claude-sonnet-4-6` or
`claude-opus-4-8` when translation quality matters more than latency. Korean
translation quality from an LLM is materially better than `opus-mt` NMT on
conversational/idiomatic speech, and it reuses the same client the agent layer needs.

---

## Behavior

**Phase 1 (local STT):**
1. App starts → load Silero VAD + faster-whisper (cached).
2. Silero VAD listens continuously on the callback stream.
3. On voice onset → start buffering, record `ts_start`, **prepend ~200ms of pre-roll**
   (a rolling pre-onset ring buffer) so the first phoneme isn't clipped, and **latch
   `forced_lang`** from the current hold-key state.
4. Flush the segment `{audio, ts_start, ts_end, forced_lang}` when **either** silence
   exceeds `silence_ms` (default 800ms) **or** the buffer reaches `max_segment_s`
   (default ~25s, under Whisper's 30s window). The length cap is a forced mid-utterance
   flush that bounds latency and memory for a speaker who never pauses; on a length flush
   the next segment starts immediately (no pre-roll, since there was no silence) and
   re-latches `forced_lang`.
5. Transcriber runs faster-whisper in-memory → raw `TranscriptEvent`.
6. The raw event is published to the bus; the UI sink appends it; the transcript shows
   the original text (with its detected language) per line.

**Phase 2 inserts one stage between 5 and 6:**
- Translator (if enabled) calls Claude once: returns `clean_text` + `translation`; on any
  error it leaves the raw transcript intact (translation degrades, transcript never
  breaks). The enriched event — original + translation per line — is what reaches the bus.

**Acceptance criteria — Phase 1:**
- KO and EN utterances both transcribed and labeled correctly within ~2s of pause.
- No filesystem writes during operation.
- Runs on a CPU-only Windows 10/11 machine.
- Fully offline — no network calls, no API key required to run.
- Raw events reach the bus and render in the UI.
- Stable for 30+ minutes without memory growth or crashes.

**Acceptance criteria — Phase 2 (adds):**
- Bidirectional translation: KO→EN and EN→KO both produced.
- ASR cleanup (spacing / punctuation / code-switch) applied to `clean_text`.
- Translation failure (no key / network / API error) never blocks the transcript.

---

## Language selection (P1)

ASR runs against an **explicitly chosen language pair**, not open-ended detection. The
user selects the two languages in play (default: English + Korean) and designates which
one the **hold key forces** (the "secondary"). Bounding ASR to two known languages is
the biggest hands-free reliability win: Whisper's auto-detect is most fragile on short
clips (mislabeling KO as ja/zh, or flip-flopping), and restricting the decision to the
chosen pair removes that for the **partner's** utterances, which must work with zero
input.

A segment's language is decided one of two ways, latched onto `segment["forced_lang"]`
at VAD speech onset (so async transcription can't race the key state):

- **Key NOT held → constrained auto-detect** over the selected pair. The default path;
  covers the partner hands-free.
- **Key held → force the secondary language.** Deterministic; what you reach for when
  you code-switch and want certainty.

Forcing the *wrong* language produces garbage text (not just a wrong label), so a missed
key-press degrades to a constrained guess in the right language — never to a forced
language on the partner. That asymmetry is why the default is detect, not force.

**One primitive.** Selection defines the candidate set; the hold key pins one member of
it; both just set `forced_lang` (or leave it `None` for auto). A future automatic
language-ID step writes the same field against the same selected pair — selection is
also what would bound *its* candidates.

**UI surface:** a language-pair selector, an active/forced-language indicator, and the
hold-key state. Capture the hold key via an **OS-level hotkey** (`keyboard` / `pynput`)
on the pipeline thread, not Streamlit (which can't reliably see keyup); the browser UI
later gets keydown/keyup natively.

**faster-whisper 1.0.2 note.** No detect-only API in this version, so the pair
constraint is applied post-hoc: transcribe with `language=None`; if `info.language`
lands outside the pair, re-decode forcing the higher-scoring member of
`info.all_language_probs`. The extra pass only fires on the rare out-of-set guess.

---

## UI

Streamlit is the **MVP validation UI only.** The `time.sleep + st.rerun` polling
loop plus background threads parked in `session_state` is fragile for a long-running
real-time service and is the wrong host for the future agent layer. The pipeline is
deliberately decoupled from Streamlit (it lives in `utils/`, talks over queues + the
bus), so the planned migration is: **FastAPI + WebSocket backend with a thin
frontend, before the agent layer lands.** Until then, Streamlit is acceptable as a
throwaway.

---

## Stretch M1 — Agent delegation (later iteration)

Not built in the MVP. The design enables it cheaply: an orchestrator subscribes to
the `TranscriptBus`, watches the enriched event stream for actionable intent, and
delegates task hand-offs to sub-agents (Claude Agent SDK / multi-agent coordinator).
Because consumers subscribe to the bus, adding the orchestrator is a new subscriber,
not a pipeline rewrite. Orchestration runs at a higher tier (Sonnet/Opus) than the
translation path.

## Stretch M2 — Lightweight speaker attribution

Turn-based or dual-input (two mics / stereo channels) speaker tagging populating the
existing `speaker` field. Not full embedding-based diarization.

---

## Risks & mitigations

| Risk | Mitigation |
|------|------------|
| KO↔EN intra-sentence code-switching mislabeled by Whisper | P1: detection constrained to the selected pair + hold-to-override the secondary; P2: LLM cleanup corrects residual mixed/mislabeled text. Set expectations — no VAD/param tuning fixes Whisper's one-language-per-segment limit |
| CPU latency too high for `small` model | model-size selector in UI; `tiny` fallback |
| Translation adds latency to each line **(P2)** | utterance-final budget tolerates ~0.3–1s; Haiku tier keeps it low; translation is async (own thread) and never blocks the transcript |
| Transcript text leaves device (privacy) **(P2)** | no network in P1 at all; from P2 hybrid is layers 3–4 only; add per-utterance gating / "local-only" toggle if needed |
| No `ANTHROPIC_API_KEY` set **(P2)** | not needed in P1; from P2 the Translator degrades to transcription-only; UI warns; transcript still works |
| Silero VAD cuts speech mid-sentence | tune `silence_ms` (slider, default 800ms) |
| Windows audio device conflicts | `sounddevice.query_devices()` for device selection |
| Streamlit unsuitable for low-latency UI | known; planned move to FastAPI+WebSocket before M1 |

---

## Out of scope for v3

- Full diarization, EXE packaging, mobile.
- True streaming/live-caption ASR (LocalAgreement).
- Local-LLM-only mode (we chose hybrid; a fully-offline LLM path is a possible future
  fork but is heavier on CPU and lower quality for KO and for agent delegation).
