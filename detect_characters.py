"""
DubStudio — Character Detection (Stage 1d)
LLM-based text analysis + optional Pyannote audio diarization + reconciliation.
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Callable

import anthropic
from dotenv import load_dotenv

load_dotenv()

ARGUS_API_KEY = os.getenv("ARGUS_API_KEY")
ARGUS_BASE_URL = os.getenv("ARGUS_BASE_URL", "https://api.anthropic.com")
ARGUS_MODEL = os.getenv("ARGUS_MODEL", "claude-opus-4-5")

LLM_BATCH_SIZE = 80
TIEBREAK_BATCH = 20


# ── LLM detection ─────────────────────────────────────────────────────────────

def detect_llm(segments: list, log_fn: Callable) -> dict:
    """
    Ask the LLM to label each segment with a character name.
    Returns dict: { "0": "NARRATOR", "1": "ALICE", ... }
    """
    client = anthropic.Anthropic(api_key=ARGUS_API_KEY, base_url=ARGUS_BASE_URL)
    characters: dict[str, str] = {}
    n = len(segments)
    n_batches = (n + LLM_BATCH_SIZE - 1) // LLM_BATCH_SIZE

    for b in range(n_batches):
        start = b * LLM_BATCH_SIZE
        end = min(start + LLM_BATCH_SIZE, n)
        batch = segments[start:end]
        log_fn(f"    LLM character batch {b + 1}/{n_batches} (segs {start}–{end - 1})...")

        lines = [f"{start + i}: {seg.get('text', '')}" for i, seg in enumerate(batch)]

        prompt = (
            "Label each dialogue segment with its speaker.\n\n"
            "Rules:\n"
            "- Third-person narration or descriptive prose → NARRATOR\n"
            "- A character speaking aloud → THEIR_NAME (uppercase, consistent)\n"
            "- Internal monologue / voice-over → THEIR_NAME_VO\n"
            "- Unknown / ambiguous → UNKNOWN\n\n"
            f'Return ONLY a JSON object: {{"0": "NARRATOR", "1": "ALICE", ...}}\n\n'
            "Segments:\n" + "\n".join(lines)
        )

        try:
            resp = client.messages.create(
                model=ARGUS_MODEL,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                result = json.loads(m.group())
                for k, v in result.items():
                    characters[str(k)] = str(v).upper()
            else:
                log_fn(f"    Warning: no JSON in response for batch {b + 1}")
                for i in range(start, end):
                    characters[str(i)] = "UNKNOWN"
        except Exception as exc:
            log_fn(f"    LLM character batch {b + 1} failed: {exc}")
            for i in range(start, end):
                characters[str(i)] = "UNKNOWN"

        time.sleep(0.3)

    return characters


# ── Pyannote diarization ──────────────────────────────────────────────────────

def detect_pyannote(audio_path: Path, segments: list, log_fn: Callable) -> dict:
    """
    Run Pyannote speaker diarization on the full audio and map each
    segment to its dominant speaker by time overlap.
    Requires: pip install pyannote.audio torch torchaudio
    """
    try:
        from pyannote.audio import Pipeline as PyannotePipeline
    except ImportError:
        raise RuntimeError(
            "pyannote.audio not installed.\n"
            "Run: pip install pyannote.audio torch torchaudio"
        )

    hf_token = Path("hf_token.txt").read_text().strip()

    log_fn("    Loading Pyannote model (may take a moment on first run)...")
    pipeline = PyannotePipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )

    log_fn("    Running speaker diarization on full audio...")
    diarization = pipeline(str(audio_path))

    # Build timeline
    timeline: list[tuple[float, float, str]] = [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]

    characters: dict[str, str] = {}
    for i, seg in enumerate(segments):
        s_start, s_end = seg["start"], seg["end"]
        speaker_overlap: dict[str, float] = {}
        for t_start, t_end, speaker in timeline:
            overlap = min(s_end, t_end) - max(s_start, t_start)
            if overlap > 0:
                speaker_overlap[speaker] = speaker_overlap.get(speaker, 0.0) + overlap
        if speaker_overlap:
            dominant = max(speaker_overlap, key=speaker_overlap.get)
            label = dominant.upper().replace("SPEAKER_", "SPK_")
        else:
            label = "UNKNOWN"
        characters[str(i)] = label

    return characters


# ── Reconciliation ────────────────────────────────────────────────────────────

def reconcile(
    llm_chars: dict,
    hf_chars: dict,
    segments: list,
    log_fn: Callable,
) -> dict:
    """
    Merge LLM and Pyannote labels:
    - Agreement → keep
    - One side is UNKNOWN → use the other
    - Both confident but different → ask LLM with context to tiebreak
    """
    client = anthropic.Anthropic(api_key=ARGUS_API_KEY, base_url=ARGUS_BASE_URL)
    final: dict[str, str] = {}
    disagreements: list[int] = []

    for i in range(len(segments)):
        sid = str(i)
        llm = llm_chars.get(sid, "UNKNOWN")
        hf = hf_chars.get(sid, "UNKNOWN")

        if llm == hf:
            final[sid] = llm
        elif hf == "UNKNOWN":
            final[sid] = llm
        elif llm == "UNKNOWN":
            final[sid] = hf
        else:
            disagreements.append(i)

    if not disagreements:
        log_fn(f"    No disagreements — reconciliation done.")
        return final

    log_fn(f"    Tiebreaking {len(disagreements)} disagreements via LLM...")
    n_batches = (len(disagreements) + TIEBREAK_BATCH - 1) // TIEBREAK_BATCH

    for b in range(n_batches):
        batch = disagreements[b * TIEBREAK_BATCH:(b + 1) * TIEBREAK_BATCH]
        lines = []
        for i in batch:
            seg = segments[i]
            prev = segments[i - 1]["text"][:50] if i > 0 else ""
            nxt = segments[i + 1]["text"][:50] if i < len(segments) - 1 else ""
            lines.append(
                f"SEG_{i}|LLM:{llm_chars.get(str(i),'?')}|"
                f"AUDIO:{hf_chars.get(str(i),'?')}|"
                f"PREV:{prev}|TEXT:{seg.get('text','')[:80]}|NEXT:{nxt}"
            )

        prompt = (
            "These segments have conflicting speaker labels from text analysis (LLM) "
            "and audio analysis (AUDIO). Choose the correct speaker for each.\n"
            f"Return ONLY JSON: {{\"SEG_N\": \"LABEL\", ...}}\n\n"
            + "\n".join(lines)
        )

        try:
            resp = client.messages.create(
                model=ARGUS_MODEL,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            m = re.search(r"\{.*\}", resp.content[0].text, re.DOTALL)
            if m:
                result = json.loads(m.group())
                for key, val in result.items():
                    idx = int(key.replace("SEG_", ""))
                    final[str(idx)] = str(val).upper()
        except Exception as exc:
            log_fn(f"    Tiebreak batch {b + 1} failed ({exc}) — using LLM labels")
            for i in batch:
                final[str(i)] = llm_chars.get(str(i), "UNKNOWN")

        time.sleep(0.3)

    return final
