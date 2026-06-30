"""On-demand meeting report: turn a finished transcript into a formatted write-up.

Unlike the Translator/AgentService (always-on worker threads), report generation is a
one-shot, user-initiated action after recording stops — so it is a plain synchronous call
the Streamlit UI runs behind a spinner. One Gemini call returns a structured result with a
title plus an English and a Korean report, both Markdown.

Auth: ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` (free tier). Override the model with
``GEMINI_REPORT_MODEL`` — thinking is left on (default) for report quality, since this is
not a latency-sensitive path.
"""

import json
import os
from typing import List, Optional

from utils.events import TranscriptEvent
from utils.gemini import get_api_key

REPORT_MODEL = "gemini-2.5-flash"

_SYSTEM = (
    "You are a meeting/discussion summarizer. You receive a raw, multi-channel "
    "speech-to-text transcript of a conversation (Korean and/or English, possibly noisy "
    "and code-switched). Each line is prefixed with its audio channel (e.g. [Mic], "
    "[System]) and detected language. Produce a clear, well-structured written report of "
    "what was discussed.\n\n"
    "Return three fields:\n"
    "- title: a short, descriptive title for the meeting/discussion.\n"
    "- report_en: a complete report in English, in Markdown, with these sections:\n"
    "  ## Summary (2–4 sentences), ## Key Points (bullets), ## Decisions (bullets; "
    "\"None\" if there were none), ## Action Items (bullets, with owner if identifiable; "
    "\"None\" if there were none).\n"
    "- report_ko: the same report written in natural, fluent Korean (a native write-up, "
    "not a word-for-word translation), with the matching sections: ## 요약, ## 주요 논의 "
    "내용, ## 결정 사항, ## 액션 아이템.\n\n"
    "Base the report ONLY on the transcript — do not invent facts, names, or decisions. If "
    "the transcript is empty or too short to summarize, say so briefly in both languages."
)

# Structured-output schema. A Pydantic model gives the SDK an exact schema and lets us read
# response.parsed. Defined defensively so a missing pydantic doesn't break import.
try:
    from pydantic import BaseModel

    class Report(BaseModel):
        title: str
        report_en: str
        report_ko: str
except Exception:
    Report = None


def report_available() -> bool:
    """True if a report can be generated (key + SDK schema present)."""
    has_key = bool(get_api_key())
    return has_key and Report is not None


def _normalize_report_text(s: Optional[str]) -> Optional[str]:
    """Undo double-escaped whitespace in a model's structured output.

    Some structured-output (JSON-schema) responses return Markdown where newlines arrive as the
    literal two-character sequence ``\\n`` (a backslash followed by an ``n``) instead of a real
    newline — so ``st.markdown`` renders the whole report as one unbroken, unformatted line. A
    correctly-formed report has real newlines (which contain no backslash), so unescaping these
    literal sequences is a no-op on the good case and the fix on the bad one.
    """
    if not s:
        return s
    if "\\n" in s or "\\t" in s or "\\r" in s:
        s = (s.replace("\\r\\n", "\n").replace("\\n", "\n")
              .replace("\\r", "\n").replace("\\t", "\t"))
    return s


def _format_transcript(events: List[TranscriptEvent]) -> str:
    lines = []
    for e in events:
        tag = f"[{e.source}] " if getattr(e, "source", None) else ""
        lang = f"({e.source_lang}) " if e.source_lang else ""
        lines.append(f"{tag}{lang}{e.display_text}")
    return "\n".join(lines)


def generate_report(events: List[TranscriptEvent], model: Optional[str] = None,
                    glossary_block: str = "",
                    transcript_text: Optional[str] = None) -> "Report":
    """Generate a bilingual (EN + KO) Markdown report from transcript events.

    Synchronous and blocking — call it behind a UI spinner. Raises on failure so the caller
    can surface an error; it never partially mutates app state. ``glossary_block`` is an
    optional "keep these exact spellings" line (see utils/glossary.py) so proper nouns stay
    consistent in the write-up.

    ``transcript_text``: when the user has freeform-edited the transcript (utils/sessions.py),
    pass the edited string here and it is summarized verbatim instead of ``events``. ``None``
    (the default) falls back to formatting ``events`` — so the common path is unchanged.
    """
    if Report is None:
        raise RuntimeError("google-genai / pydantic not available")
    key = get_api_key()
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key)
    config = types.GenerateContentConfig(
        system_instruction=_SYSTEM,
        response_mime_type="application/json",
        response_schema=Report,
        # Thinking is on for quality, and the output is a bilingual (EN+KO) Markdown
        # document — both eat into the output budget. A long meeting can blow past a small
        # cap and truncate the JSON mid-object, so give it generous headroom (flash allows
        # up to 65536) and detect truncation explicitly below.
        max_output_tokens=32768,
    )
    body = transcript_text if transcript_text is not None else _format_transcript(events)
    response = client.models.generate_content(
        model=model or os.environ.get("GEMINI_REPORT_MODEL") or REPORT_MODEL,
        contents=(
            (f"{glossary_block}\n\n" if glossary_block else "")
            + f"Transcript:\n{body}"
        ),
        config=config,
    )

    # If the model ran out of output budget, response.text is truncated JSON — turn the
    # cryptic JSONDecodeError into a clear, actionable message for the user.
    try:
        finish = str(getattr(response.candidates[0].finish_reason, "name", "") or "")
    except (AttributeError, IndexError, TypeError):
        finish = ""
    if finish.upper() == "MAX_TOKENS":
        raise RuntimeError(
            "Transcript too long — the report exceeded the model's output limit. "
            "Try generating from a shorter recording."
        )

    if response.parsed is not None:
        result = response.parsed
    elif response.text:
        result = Report(**json.loads(response.text))
    else:
        raise RuntimeError("Report generation returned no content.")

    # Repair double-escaped Markdown (literal "\n" instead of real newlines) so the report renders
    # with its headings and bullets intact rather than as one flat line. See _normalize_report_text.
    result.title = _normalize_report_text(result.title)
    result.report_en = _normalize_report_text(result.report_en)
    result.report_ko = _normalize_report_text(result.report_ko)
    return result
