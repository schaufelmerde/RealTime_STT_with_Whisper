# PRD: Korean–English Real-Time Assistant Transcriber v4 (as-built + planned)

## What changed from v3 → v4 (drift reconciliation)

v3 described the *intended* hybrid architecture. The code then diverged from it on several
load-bearing points. v4 reconciles the document to what actually shipped, keeps the parts of the
vision still being pursued, and folds in several new features (conversation context for the
assistant, a proper-noun vocabulary, and a phone companion). The substantive changes:

| v3 said | As built (v4) |
|---------|---------------|
| Cloud LLM = **Anthropic / Claude** (Haiku for translation, Sonnet/Opus for agents); `ANTHROPIC_API_KEY` | **Google Gemini** for *everything* cloud-side — translation, the assistant, and the report — via `google-genai`, model `gemini-2.5-flash`; `GEMINI_API_KEY` (free tier). No `anthropic` dependency. |
| Single **microphone** source | **Multi-source capture**: microphone + system loopback + per-app/per-browser process loopback + a **phone-as-mic over Wi-Fi**, each its own VAD+ASR channel, tagged on the transcript. Per-source level meters, mute/unmute, and retarget — all live. |
| The hold key **forces the secondary language** (Goal 7) | The hold key is **push-to-ask**: holding it marks your speech as a *command to the assistant* instead of conversation. The `forced_lang` machinery still exists in the schema but is **dormant** (not wired to any UI control). |
| "**Utterance-final, not streaming**, no word-by-word" | Still utterance-final for the committed line, **plus an optional throwaway live preview** (a cheap greedy decode shown while someone is still speaking). Not LocalAgreement streaming. |
| Agent layer = **P3 stretch**, lands after a FastAPI/WebSocket rewrite | A **push-to-ask assistant shipped now, on Streamlit** (Gemini + Google Search grounding). The FastAPI migration has **not** happened. |
| (no report) | A **post-meeting bilingual report** generator (EN + KO Markdown) shipped. |
| (no phone bridge) | A **phone companion**: a phone joins over Wi-Fi as a wireless mic *and* a remote control surface (translation / mute / push-to-ask / context), via a local HTTPS+WSS server. |
| "Custom vocabulary fine-tuning" = non-goal | Clarified: we add **prompt-biasing** vocabulary (Whisper `hotwords` + an LLM glossary). That is *not* fine-tuning/retraining — see "Proper-noun vocabulary". |

The **seam** (`TranscriptBus`) held up exactly as designed: the assistant, the conversation
store, and the report are all bus consumers / post-bus actions, added without a pipeline rewrite.

> **Privacy note (the cost of hybrid):** with translation or the assistant enabled, transcript
> text leaves the device for **Google's Gemini API**. Audio + ASR stay local. STT-only operation
> (no key) makes **no network calls at all**. If some conversations are sensitive, keep the
> assistant/translation off or gate which utterances are sent.

---

## Delivery phases (status)

| Phase | Scope | Network | LLM | Status |
|-------|-------|---------|-----|--------|
| **P1 — Local STT** | KO + EN transcription, utterance-final, on the bus | none (offline) | none | **Shipped** + extended (multi-source, live preview) |
| **P2 — Translation + cleanup** | bidirectional KO↔EN translation + ASR cleanup per event | cloud | Gemini Flash | **Shipped** |
| **P3 — Assistant** | push-to-ask grounded Q&A over the transcript stream + conversation memory | cloud | Gemini Flash + Google Search | **Shipped** (Q&A + conversation context) |
| P4 — Backend migration | FastAPI + WebSocket host replacing Streamlit | — | — | Planned, not started |

P1 remains the fully-local core (capture → VAD → ASR → event). P2 inserts the Translator between
the Transcriber and the bus without changing the event schema. P3 subscribes to the bus.

---

## Why v1 failed (still relevant — keep avoiding these)

1. **File I/O race conditions** — v1 wrote `tmp.wav` and read it back from another thread; on
   Windows this caused constant file-locking errors and a `while True: os.remove(...)` retry
   loop. **Fix: never touch disk — pass numpy float32 in-memory.**
2. **Real-time was silently abandoned** — by v2–v5 it became record → silence → batch
   transcribe → display (push-to-talk with lag), not streaming.
3. **CPU inference too slow for full re-transcription** — re-transcribing a 10s sliding buffer
   every loop never kept up.
4. **The 10s sliding window was architecturally wrong** — lost context, did redundant compute,
   forced the `last_del=True` "always drop the last word" hack.
5. **webrtcvad was brittle** — frame-size/encoding sensitive, failed silently.
6. **UX was never real-time** — three manual clicks per utterance.

---

## Problem statement

Bilingual Korean–English conversations have no good lightweight real-time transcription +
translation tool. Cloud captioners are English-first and don't handle KO↔EN code-switching well;
offline tools can't translate bidirectionally or reason. v4 keeps ASR on-device for
latency/privacy and uses an LLM for the parts that need quality and reasoning.

---

## Goals

| #  | Goal | Phase | Status |
|----|------|-------|--------|
| 1  | Continuously transcribe Korean **and** English speech in near real-time | P1 | ✅ |
| 2  | Show each utterance within ~1–2s of the speaker pausing | P1 | ✅ |
| 3  | Run ASR locally/offline on a modern consumer machine (GPU or CPU) | P1 | ✅ |
| 6  | Emit a structured transcript event stream any consumer can subscribe to | P1 | ✅ |
| 7  | Constrain ASR to a user-selected language pair | P1 | ✅ (hold-to-force-language **dropped** — see Language selection) |
| 8  | Capture **multiple audio sources** (mic + system/app loopback) as tagged channels | P1+ | ✅ |
| 4  | Translate each utterance bidirectionally (KO↔EN) via the LLM | P2 | ✅ |
| 5  | Clean up noisy ASR output via the LLM | P2 | ✅ |
| 9  | Push-to-ask assistant: hold a key, ask a question, get a grounded answer | P3 | ✅ |
| 10 | Assistant can pull **conversation context** on demand | P3 | ✅ |
| 11 | **Proper-noun vocabulary** to bias ASR + LLM toward names | P1+ | ✅ |
| 12 | Post-meeting **bilingual report** | P3 | ✅ |
| 13 | **Phone companion**: a phone as a wireless mic + remote control surface | P1+/P3 | ✅ |
| M2 | Lightweight speaker attribution | Stretch | Partial (channel `source` tag) |

## Non-goals (v4)

- Full speaker diarization (who-said-what via embeddings). The `speaker` field is reserved.
- **Custom-vocabulary _fine-tuning / retraining_.** We bias decoding with `hotwords` and an LLM
  glossary (prompt conditioning, no weight changes) — see "Proper-noun vocabulary". Genuine
  fine-tuning is still out of scope.
- Mobile / embedded deployment.
- True word-by-word LocalAgreement streaming (we are utterance-final + a tentative preview).
- EXE packaging (after the core is stable).

---

## "Real-time" — be precise

The committed transcript is **utterance-final**: the authoritative line appears after the speaker
pauses (VAD end-of-speech), not word-by-word. On top of that we render an **optional live
preview** — a cheap greedy (`beam_size=1`) decode of the in-progress utterance, shown as a
tentative italic line and replaced by the finalized (and translated) line at segment end. The
preview is display-only: it is never translated, stored, or sent to the assistant. This is *not*
a LocalAgreement streaming policy (e.g. `whisper_streaming`) — that remains a deliberately
unwanted, more complex design.

---

## Architecture

```
LOCAL (offline, low-latency)                          CLOUD (Gemini, quality + reasoning)
┌─────────────────────────────────────┐
│ N audio sources (mic / system / app  │  sounddevice + soundcard (WASAPI loopback) +
│   / browser loopback / phone-as-mic) │  phone over Wi-Fi; 16kHz float32, in-memory only
│   ▼  (one pipeline per source)       │
│ [1] Silero VAD (per 512-frame)       │  speech start/end; ~200ms pre-roll
│   ▼  segment_queue                   │  {audio, ts_start, ts_end, forced_lang, command, source}
│ [2] faster-whisper (GPU fp16 / CPU   │  → TranscriptEvent{source, source_lang, text}
│     int8); hotwords=vocab (opt-in)   │  (+ a cheap partial decode → live preview)
└──────────────┬───────────────────────┘
   command?     │ transcript_queue
   ├── yes ─────┼──────────────► command_queue ──► [Agent] (Gemini + Google Search;
   │            ▼                                    conversation context folded in locally)
   │     ┌──────────────────────────────┐
   │     │ [3] Translator (Gemini Flash) │  cleanup + KO↔EN translation; glossary-aware;
   │     │     enriches the event        │  echo-suppresses mic crosstalk of loopback
   │     └──────────────┬────────────────┘
   │                    ▼  bus.publish(event)
   │           ╔════════════════════════╗
   └──────────►║  TranscriptBus (seam)  ║  thread-safe pub/sub
              ╚═══╦═════════╦═════════╦══╝
                  ▼         ▼         ▼
             [UI sink] [ConversationStore] [future consumers]
                            │ (on-demand compacting transcript memory)
                            └──► read by the Agent locally when assembling an ask

Post-meeting (one-shot): transcript ──► [Reporter] (Gemini Flash) ──► EN + KO Markdown report
```

**Key decisions vs v3:**

| Issue | v3 plan | v4 as-built |
|-------|---------|-------------|
| Cloud provider | Anthropic / Claude | **Google Gemini** (`gemini-2.5-flash`) across translate / assistant / report |
| Sources | one mic | **many** tagged channels (mic + loopback + process + **phone over Wi-Fi**) |
| Hold key | force secondary language | **push-to-ask** assistant command |
| Live feedback | none (utterance-final only) | + tentative **live preview** |
| Assistant memory | (later) | **ConversationStore** behind the bus; context folded into the ask locally (no model tools) |
| Names/jargon | (non-goal) | **vocabulary** via `hotwords` + LLM glossary |
| Phone | (none) | **phone-as-mic + remote control** over a local HTTPS/WSS bridge |

---

## Segment schema (VAD → Transcriber)

What the VAD stage puts on `segment_queue`. Short-lived and internal — never carries to the bus:

```python
segment = {
    "audio":       np.ndarray,  # float32 @ 16kHz mono, in-memory only (never disk)
    "ts_start":    float,       # wall-clock speech onset (includes pre-roll)
    "ts_end":      float,       # wall-clock end-of-speech (silence threshold crossed)
    "forced_lang": str | None,  # decode-language override; dormant (no UI wires it now)
    "command":     bool,        # latched from the push-to-ask key at onset → route to the agent
    "source":      str | None,  # audio-channel tag (e.g. "Mic", "System")
    "segment_id":  str | None,  # ties interim partials to their final commit
    "partial":     bool,        # tentative in-progress decode for live preview
}
```

`command` and `forced_lang` are latched **at speech onset** from live key state and never re-read
at transcribe time, so an async ASR backlog can't race a key the user has since released.

---

## Transcript event schema (`utils/events.py`)

```python
TranscriptEvent(
    text:        str,            # raw ASR output
    source_lang: str,            # resolved decode language: "ko" | "en" | ...
    ts_start:    float,
    ts_end:      float,
    id:          str,
    lang_source: str | None,     # "forced" | "detected"
    source:      str | None,     # audio-channel tag (mic/system/app) — technical origin
    speaker:     str | None,     # reserved (M2 diarization), unused
    is_command:  bool,           # push-to-ask command → agent, not the transcript
    segment_id:  str | None,     # stable per-utterance id (partial ↔ final)
    partial:     bool,           # tentative live-preview decode (never translated/stored)
    clean_text:  str | None,     # LLM-cleaned original (P2)
    translation: str | None,     # LLM translation into target_lang (P2)
    target_lang: str | None,     # "ko" | "en"
)
```

In P1 events go to the bus with `clean_text` / `translation` / `target_lang` = `None`. In P2 the
Translator fills them before publishing. Consumers see one shape either way.

---

## Component stack

| Layer | Tech | Notes |
|-------|------|-------|
| Capture | `sounddevice` + `soundcard` | mic + WASAPI **loopback**; `soxr` resamples 48k→16k |
| Phone bridge | `utils/phone_server.py` + `aiohttp` / `segno` / `cryptography` | phone-as-mic over a local HTTPS/WSS server (QR pairing, self-signed cert, per-launch token); doubles as a remote control surface |
| VAD | `silero-vad` (+ `torch`) | one stateful instance **per source** |
| ASR | `faster-whisper` 1.0.2 | GPU `float16` (CTranslate2 CUDA), CPU `int8` fallback; **`large-v3` default** (most accurate, esp. Korean); `hotwords` = vocabulary (**opt-in, off by default**) |
| Translation / cleanup | `google-genai`, `gemini-2.5-flash` | structured JSON; `thinking_budget=0` for latency; glossary-aware; `GEMINI_TRANSLATE_MODEL` overrides |
| Assistant | `google-genai`, `gemini-2.5-flash` | Google Search grounding; conversation context folded into the single call locally; `GEMINI_MODEL` overrides |
| Conversation memory | `utils/context.py` (`ConversationStore`) | bus subscriber; recent verbatim window + rolling summary, compacted **on demand** at ask time |
| Vocabulary | `utils/glossary.py` (`Glossary`) | `vocabulary.json`; feeds `hotwords` + LLM glossary block |
| Report | `google-genai`, `gemini-2.5-flash` | one-shot bilingual Markdown; `GEMINI_REPORT_MODEL` overrides |
| Gemini client | `utils/gemini.py` | shared `get_api_key` / `make_client` — the single `GEMINI_API_KEY → GOOGLE_API_KEY` fallback every cloud module (translate / assistant / context / report) uses |
| Hotkey | `pynput` | OS-level push-to-ask (Streamlit can't see keyup) |
| Event seam | `utils/events.py` (`TranscriptBus`) | thread-safe pub/sub |
| UI | `Streamlit` | MVP host — see "UI" |
| Settings | `utils/settings.py` | cross-session sidebar state (model, languages, sources, VAD/beam, toggles) persisted to gitignored `user_settings.json`; seeds the widgets at launch |

**Model choice.** `gemini-2.5-flash` is the latency-appropriate free tier for the per-utterance
hot path; each module exposes an env override. Raise the model where quality matters more than
latency (e.g. report). Auth: `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) in `.env` / `.env.local`;
free key at https://aistudio.google.com/apikey.

---

## Multi-source capture & echo suppression

The app captures **any number of audio sources** in parallel, each as an independent
VAD+ASR pipeline writing to the shared segment queue tagged with its channel (`source`):

- **Microphone** — an input device.
- **System Audio** — everything you hear, via WASAPI loopback (`soundcard`).
- **App / Browser** — one program's audio via Windows **process loopback** (the OBS technique;
  no virtual cable). Only programs currently playing audio are listed.
- **Phone** — a phone used as a **wireless microphone**: scan a QR code, open the served HTTPS
  page, grant mic access, and the phone streams audio over a secure WebSocket to a small local
  server (`utils/phone_server.py`) that feeds the same VAD/ASR path. The phone doubles as a
  **remote control surface** — see "Phone companion".

Per-source UI: a live input-level meter (driven by the capture while recording, or a lightweight
`LevelMonitor` while idle), **mute/unmute** (silences a source live without losing its row), and
**retarget** (re-point a source at a different device/app in place). Unresolved sources (an app
that closed) are flagged, not silently dropped.

**Echo suppression.** On shared-ground (TRRS) headsets the mic electrically picks up playback
(crosstalk), so call/system audio gets transcribed twice — once cleanly via loopback, once echoed
through the mic. Crosstalk is strictly one-directional (loopback is a clean digital tap that can
never contain the mic), so the Translator drops a **mic** final that overlaps a recent **clean
(loopback)** final in time *and* matches its text. Your own speech dominates the mic and won't
match any loopback line, so it survives. Toggleable per mic; only active when both a mic and a
loopback source are live.

---

## Phone companion

A phone joins the session over Wi-Fi as **both** a capture source and a remote control, via a
small local server (`utils/phone_server.py`) that is deliberately **separate from Streamlit**.

- **Phone-as-mic.** The phone scans a QR code, opens a served page, grants mic access, and streams
  32-bit float PCM over a secure WebSocket. A `WebMicReader` (`utils/audio.py`) drains those
  samples into the same VAD → ASR path every other source uses, so the phone is just another
  tagged channel (`source="Phone"`).
- **Why a separate server.** `getUserMedia` requires a **secure context** (HTTPS or `localhost`),
  but the phone reaches the host by LAN IP — never localhost — so the capture page must be served
  over HTTPS. A standalone `aiohttp` app with a self-signed cert runs on its own port; Streamlit
  keeps serving plain HTTP to the desktop. Audio crosses the LAN **TLS-encrypted**, and stays on
  the local machine from there (no cloud) — the privacy note's "audio stays local" still holds.
- **Security.** The QR/URL carries a random **per-launch token**; the WebSocket handshake rejects
  any connection without it, so another device on the same Wi-Fi can't push audio. The self-signed
  cert is persisted (`~/.realtime_stt`) so the phone's one-time trust prompt sticks across restarts.
- **Lifecycle.** A process-wide singleton, started lazily the first time a Phone source is used and
  left running for the app's life — it must outlive Streamlit's ~3/s reruns and stay "plugged in"
  across Start/Stop, like a real mic.
- **Remote control surface.** The phone screen also exposes buttons that drive the desktop session
  as a **parallel control channel**: toggle translation, mute, **push-to-ask** (a two-tap listening
  window marks phone speech as a command), and the assistant's context toggle. Assistant answers to
  phone-initiated asks are echoed back to the phone screen. All optional: with the phone extras not
  installed, a Phone source simply reports itself unavailable instead of crashing.

---

## Language selection (P1)

ASR runs against an **explicitly chosen language pair** (default English + Korean), not open-ended
detection. Bounding to two known languages is the biggest hands-free reliability win: Whisper's
auto-detect is most fragile on short clips (mislabeling KO as ja/zh), and restricting the decision
to the pair removes that for the partner's utterances, which must work with zero input.

The language is decided by **constrained auto-detect**: transcribe with `language=None`; if
`info.language` lands outside the pair (`faster-whisper` 1.0.2 has no detect-only API), re-decode
forcing the higher-scoring pair member. `lang_source="detected"`.

> **Drift note.** v3's *hold-to-force-the-secondary-language* (Goal 7) was **not built**. The hold
> key was repurposed for **push-to-ask** (see below). The `forced_lang` segment field and the
> Transcriber's forced path remain in place (so a future language-ID step or a re-introduced
> force control can set it with `lang_source="forced"`), but **nothing currently writes it**.

---

## Push-to-ask assistant (P3)

Hold the configured OS hotkey (default Left Ctrl) and speak: the utterance is transcribed like any
other, but flagged `command` at onset and routed to the **AgentService** instead of the transcript
/ translator. The agent runs on Gemini with **Google Search grounding** and answers concisely in
the Assistant panel (with citations). Push-to-ask is wired to **mic** sources only, and only when
the assistant is enabled; loopback (the other party) is never a command channel. Best-effort: no
`pynput` / no key → transcription-only, no crash.

### Conversation context

The assistant previously saw only the single utterance you spoke. It now has **on-demand access to
the conversation** via `ConversationStore` (`utils/context.py`), a bus subscriber that keeps the
transcript in two tiers:

- a **recent verbatim window** (last N finalized lines, word-for-word); and
- a **rolling summary** — older lines folded into a compact running summary by a cheap Gemini
  call, then evicted (the **compaction**), so token cost and memory stay bounded over a long
  session.

Compaction is **on-demand, not background**: `ingest` only appends (never calls Gemini), and the
fold-into-summary step runs synchronously inside an ask (`compact_now`), exactly when the summary
is about to be used. So when the assistant is off — or on but never invoked — the store issues
**zero** Gemini calls and can't drain the free-tier quota in the background; memory is still
bounded by a hard cap that sheds the oldest verbatim lines with no API call.

The agent assembles context **locally**, with no extra model round-trip: when context is enabled
and the conversation isn't empty, it reads the store's rolling summary plus the most recent
verbatim lines and **prepends them to the single grounded answer call** (Gemini + Google Search).
So every ask is exactly **one** grounded request — which matters on the free tier's per-minute/day
limits. (The store's `recent` / `summary` / `search` methods are a plain Python API the agent calls
directly; they are *not* exposed to the model as function tools — and Gemini won't reliably mix
Google Search grounding with custom tools in one call anyway.) The only extra call an ask can
trigger is the **on-demand compaction** above, and only when there's aged-out backlog to fold
(then cached). With no key or an empty conversation, context is skipped and the ask is just the
plain grounded call.

---

## Proper-noun vocabulary

Whisper mis-hears uncommon proper nouns (company names, people, jargon). A user-editable list
(`utils/glossary.py`, persisted to the gitignored `vocabulary.json`) biases recognition toward
those terms. **This is not training.**

- **Primary layer — LLM glossary (always on, no retraining).** The terms are handed to the **Gemini
  cleanup/translation pass** (and the report) as a "keep these exact spellings" glossary. Gemini
  sees the ASR text and only fixes a spelling that's *already there*, so it can't invent a term
  from nothing — the safe path whenever a Gemini stage runs. No gradients, no weight changes.
- **Secondary layer — Whisper `hotwords` (opt-in, OFF by default).** The same terms can also be fed
  to `faster-whisper`'s `hotwords` parameter, which tokenizes them into the decoder's *previous-text
  context* to re-weight its spelling prior. In principle this rescues a name that fails on the
  spelling prior rather than acoustics; **in practice it conditions the decode too strongly and
  over-forces** the terms — emitting them on short/quiet clips even when unspoken, the same
  conditioning-hallucination we removed from `initial_prompt`. So it sits behind a **default-off UI
  toggle** (`hotwords_gate`), for the occasional name Whisper keeps mangling. faster-whisper also
  truncates the hint to half the context, so keep the list focused.
- **Live.** The Transcriber/Translator read the glossary (and the hotwords gate) per utterance via
  callables, so edits and the toggle take effect on the next utterance with no restart.

---

## Post-meeting report (P3)

Once recording stops and a transcript exists, "Generate report" makes one Gemini call returning a
structured result: a title plus an **English** and a **Korean** report (Summary / Key Points /
Decisions / Action Items), both Markdown and downloadable. Synchronous (run behind a spinner),
glossary-aware, and grounded strictly in the transcript. Truncation (output-budget overflow) is
detected and surfaced as an actionable message.

---

## Behavior

**P1 (local STT):** load Silero VAD + faster-whisper (cached) → per-source VAD listens
continuously → on voice onset, buffer with ~200ms pre-roll and latch `command`/`forced_lang` →
flush the segment on `silence_ms` of silence or a `max_segment_s` length cap → Transcriber decodes
in-memory (with optional `hotwords` biasing, off by default) → raw `TranscriptEvent` published to
the bus → UI appends it. An
optional partial decode streams a tentative preview line meanwhile.

**P2 inserts the Translator** between decode and publish: one Gemini call returns
`clean_text` + `translation`; on any error it leaves the raw transcript intact (translation
degrades, the transcript never breaks). Echo-suppressed mic duplicates are dropped before the call.

**P3:** `command` utterances route to the agent (a single grounded call with any conversation
context folded in locally); all finalized lines also feed the ConversationStore.

**Acceptance criteria — P1:** KO + EN both transcribed and labeled within ~2s of pause; no
filesystem writes during operation; runs on Windows 10/11 (GPU or CPU-only); fully offline with no
key; raw events reach the bus and render; stable 30+ min without memory growth or crashes
(including the ConversationStore's bounded window/compaction).

**Acceptance criteria — P2:** bidirectional KO→EN and EN→KO; ASR cleanup applied to `clean_text`;
translation failure (no key / network / API error) never blocks the transcript.

**Acceptance criteria — P3:** push-to-ask produces a grounded answer with citations; a follow-up
that references earlier conversation is answered using locally-folded context; every ask is a
single grounded pass; assistant off / no key degrades gracefully.

---

## UI

Streamlit is the **MVP validation host.** The `time.sleep + st.rerun` polling loop plus
background threads parked in `session_state` is fragile for a long-running real-time service and is
the wrong host for the agent layer. The pipeline is deliberately decoupled from Streamlit (it lives
in `utils/`, talks over queues + the bus), so the planned migration remains **FastAPI + WebSocket
backend with a thin frontend**. That migration has not happened; the assistant shipped on Streamlit
in the interim. Until then, Streamlit is acceptable as a throwaway.

---

## Stretch — remaining

- **M1 — Agent delegation / task tools.** The assistant answers and searches today; the next step
  is client-side *task* tools (function calling) for actions, using the same `tools` seam in
  `AgentService` (currently just Google Search; conversation context is folded in locally rather
  than via model tools). Auto-suggesting glossary terms from the conversation is a candidate here.
- **M2 — Lightweight speaker attribution.** Turn/channel-based tagging populating `speaker` (the
  `source` channel tag is a partial down-payment). Not full embedding-based diarization.

---

## Risks & mitigations

| Risk | Mitigation |
|------|------------|
| KO↔EN intra-sentence code-switching mislabeled by Whisper | detection constrained to the pair; LLM cleanup corrects residual mixed/mislabeled text. No VAD tuning fixes Whisper's one-language-per-segment limit |
| ASR latency too high | model-size selector; GPU `float16` path with CPU `int8` fallback; `tiny`/`base` for CPU-only |
| Translation/assistant add latency | utterance-final budget tolerates it; Flash + `thinking_budget=0`; both async (own threads), never block the transcript |
| Many Gemini consumers share one free-tier quota | translate / assistant / context-compaction / report all draw the same ~20/day key; compaction is **on-demand** (no background drain) and each ask is a single grounded call; use a separate key (or paid tier) if you hit the cap |
| Transcript text leaves device (privacy) | no network in STT-only mode; with cloud features, text goes to **Gemini**; keep features off / gate utterances if sensitive |
| Proper-noun list poisons decoding (false insertions) | Whisper `hotwords` biasing is **off by default** (it over-forces terms); the always-on LLM glossary is the higher-precision backstop; if hotwords is enabled, keep the list focused and it's reversible per utterance |
| Vocabulary / settings hold personal or machine-specific data | `vocabulary.json` and `user_settings.json` are gitignored |
| No `GEMINI_API_KEY` | STT-only still runs fully; translation/assistant/report degrade with a UI warning; transcript always works |
| Mic crosstalk double-transcribes call audio | echo suppression (loopback authoritative, mic dupes dropped) |
| Silero VAD cuts speech mid-sentence | tune `silence_ms` slider |
| Windows audio device conflicts | `sounddevice` device selection; loopback via `soundcard`; idle monitors released before captures open |
| Streamlit unsuitable for low-latency UI | known; planned move to FastAPI+WebSocket |

---

## Out of scope for v4

- Full diarization, EXE packaging, mobile.
- True streaming/live-caption ASR (LocalAgreement).
- Custom-vocabulary **fine-tuning / model retraining** (we prompt-bias instead).
- Local-LLM-only mode (we chose hybrid; a fully-offline LLM path is a possible future fork but is
  heavier and lower quality for KO and for the assistant).
