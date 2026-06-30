"""Tiny shared helpers for the Gemini (google-genai) cloud layer.

Every cloud stage — translator, agent, context store, reporter — needs the same two things:
the API key (with the GEMINI_API_KEY → GOOGLE_API_KEY fallback) and a configured client. They
all share ONE free-tier key and its quota, so centralizing the lookup keeps that single source
of truth in one place instead of re-implementing the fallback in four modules.

No client is cached here: each stage owns its own ``genai.Client`` for its lifetime (clients are
cheap — they just hold the key + transport), and this module stays import-light so it can be
pulled in anywhere without dragging in google-genai unless a client is actually built.
"""

import os
from typing import Optional


def get_api_key() -> Optional[str]:
    """The Gemini API key, preferring GEMINI_API_KEY and falling back to GOOGLE_API_KEY.
    Returns None if neither is set (callers degrade to passthrough/no-op)."""
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def make_client():
    """A configured ``genai.Client``, or None if no key is set. Imports google-genai lazily so
    a missing SDK surfaces as an ImportError for the caller to catch (same as before), and only
    when a client is actually requested."""
    key = get_api_key()
    if not key:
        return None
    from google import genai
    return genai.Client(api_key=key)
