"""
DubStudio — Stage 1: SCRIPT
Audio extraction → Whisper transcription → sentence merging →
character detection → translation → Excel cue sheet
"""

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

load_dotenv()

WHISPER_MODEL_SIZE = "base"  # tiny/base/small/medium — larger = slower but more accurate
MERGE_GAP_THRESHOLD = 0.3   # seconds — don't merge across gaps larger than this
MAX_SEGMENT_DURATION = 15.0  # seconds — split merged segments longer than this
SENTENCE_ENDINGS = (".", "!", "?", "…", "...", ".'", "!'", "?'")


def run_stage1(video_path: str, project_dir: Path, target_lang: str, log_fn: Callable):
    video_path = Path(video_path)
    project_dir = Path(project_dir)

    log_fn(f"=== Stage 1: SCRIPT ===")
    log_fn(f"Project: {project_dir.name}  |  Target: {target_lang}  |  Video: {video_path.name}")

    # ── 1a. Extract audio ──────────────────────────────────────────────────────
    audio_path = project_dir / "audio.mp3"
    if audio_path.exists():
        log_fn("1a. Audio already extracted — skipping.")
    else:
        log_fn("1a. Extracting audio track...")
        _extract_audio(video_path, audio_path)
        log_fn(f"    Done → audio.mp3 ({audio_path.stat().st_size // 1024} KB)")

    # ── 1b. Transcribe ────────────────────────────────────────────────────────
    segments_path = project_dir / "segments.json"
    if segments_path.exists():
        stored = json.loads(segments_path.read_text(encoding="utf-8-sig"))
        raw_segments = stored.get("raw", stored) if isinstance(stored, dict) else stored
        log_fn(f"1b. Transcription already exists ({len(raw_segments)} raw segments) — skipping.")
    else:
        log_fn("1b. Transcribing with OpenAI Whisper...")
        raw_segments = _transcribe_chunked(audio_path, log_fn)
        segments_path.write_text(json.dumps({"raw": raw_segments}, ensure_ascii=False, indent=2), encoding="utf-8")
        log_fn(f"    Done → {len(raw_segments)} raw segments")

    # ── 1c. Merge sentences ───────────────────────────────────────────────────
    stored = json.loads(segments_path.read_text(encoding="utf-8-sig"))
    if isinstance(stored, dict) and "merged" in stored:
        merged = stored["merged"]
        log_fn(f"1c. Sentence merging already done ({len(merged)} merged segments) — skipping.")
    else:
        log_fn("1c. Merging segments at sentence boundaries...")
        merged = _merge_sentences(raw_segments)
        if isinstance(stored, dict):
            stored["merged"] = merged
        else:
            stored = {"raw": raw_segments, "merged": merged}
        segments_path.write_text(json.dumps(stored, ensure_ascii=False, indent=2), encoding="utf-8")
        log_fn(f"    Done → {len(merged)} segments after merging")

    # ── 1d. Detect characters ─────────────────────────────────────────────────
    chars_path = project_dir / "characters_final.json"
    if chars_path.exists():
        final_chars = json.loads(chars_path.read_text(encoding="utf-8-sig"))
        log_fn(f"1d. Characters already detected ({len(set(final_chars.values()))} unique) — skipping.")
    else:
        log_fn("1d. Detecting characters (LLM)...")
        from detect_characters import detect_llm, detect_pyannote, reconcile

        llm_chars = detect_llm(merged, log_fn)
        (project_dir / "characters_llm.json").write_text(
            json.dumps(llm_chars, ensure_ascii=False, indent=2)
        )

        hf_token_path = Path("hf_token.txt")
        if hf_token_path.exists():
            log_fn("    HF token found — running Pyannote diarization...")
            try:
                hf_chars = detect_pyannote(audio_path, merged, log_fn)
                (project_dir / "characters_hf.json").write_text(
                    json.dumps(hf_chars, ensure_ascii=False, indent=2)
                )
                final_chars = reconcile(llm_chars, hf_chars, merged, log_fn)
            except Exception as exc:
                log_fn(f"    Pyannote failed ({exc}) — using LLM-only labels.")
                final_chars = llm_chars
        else:
            log_fn("    No hf_token.txt — using LLM-only character detection.")
            final_chars = llm_chars

        chars_path.write_text(json.dumps(final_chars, ensure_ascii=False, indent=2), encoding="utf-8")
        unique = set(final_chars.values())
        log_fn(f"    Done → {len(unique)} characters: {', '.join(sorted(unique))}")

    # ── 1e. Translate ─────────────────────────────────────────────────────────
    trans_path = project_dir / "translations.json"
    existing_trans = json.loads(trans_path.read_text(encoding="utf-8-sig")) if trans_path.exists() else {}

    log_fn(f"1e. Translating to {target_lang}...")
    from translate import translate_all
    translations = translate_all(merged, final_chars, target_lang, existing_trans, log_fn, project_name=project_dir.name)
    trans_path.write_text(json.dumps(translations, ensure_ascii=False, indent=2), encoding="utf-8")
    log_fn(f"    Done → {len(translations)} segments translated")

    # ── 1f. Generate cue sheet ────────────────────────────────────────────────
    log_fn("1f. Generating Excel cue sheet...")
    from translate import write_excel, write_review_doc
    write_excel(project_dir, merged, translations, final_chars, target_lang)
    log_fn("    Done → cue_sheet_final.xlsx")

    # ── 1g. Generate Word review document ─────────────────────────────────────
    log_fn("1g. Generating review document...")
    try:
        write_review_doc(project_dir, merged, translations, final_chars, target_lang, project_dir.name)
        log_fn(f"    Done → {project_dir.name}_stage1_review.docx")
    except Exception as exc:
        log_fn(f"    Warning: review doc skipped ({exc})")

    log_fn("=== Stage 1 complete ===")


# ── Audio extraction ──────────────────────────────────────────────────────────

def _extract_audio(video_path: Path, audio_path: Path):
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "libmp3lame",
        "-q:a", "2",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed:\n{result.stderr[-500:]}")


def _get_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
        capture_output=True, text=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


# ── Whisper transcription (local, free via faster-whisper) ────────────────────

def _transcribe_chunked(audio_path: Path, log_fn: Callable) -> list:
    from faster_whisper import WhisperModel

    log_fn(f"    Loading faster-whisper model ({WHISPER_MODEL_SIZE})...")
    model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")

    log_fn("    Transcribing (this may take a few minutes)...")
    segments_gen, info = model.transcribe(str(audio_path), beam_size=5)

    log_fn(f"    Detected language: {info.language} ({info.language_probability:.0%} confidence)")

    all_segments = []
    for seg in segments_gen:
        all_segments.append({
            "id": len(all_segments),
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
        })
        if len(all_segments) % 50 == 0:
            log_fn(f"    ...{len(all_segments)} segments transcribed")

    return all_segments


# ── Sentence merging ──────────────────────────────────────────────────────────

def _merge_sentences(segments: list) -> list:
    if not segments:
        return []

    merged = []
    current: dict | None = None

    for seg in segments:
        if current is None:
            current = _new_merged(seg)
            continue

        gap = seg["start"] - current["end"]
        text_ends_sentence = current["text"].rstrip().endswith(SENTENCE_ENDINGS)

        if text_ends_sentence or gap > MERGE_GAP_THRESHOLD:
            merged.extend(_split_long(current))
            current = _new_merged(seg)
        else:
            current["text"] = current["text"].rstrip() + " " + seg["text"].lstrip()
            current["end"] = seg["end"]
            if "id" in seg:
                current["source_ids"].append(seg["id"])

    if current:
        merged.extend(_split_long(current))

    return merged


def _new_merged(seg: dict) -> dict:
    return {
        "start": seg["start"],
        "end": seg["end"],
        "text": seg["text"],
        "source_ids": [seg["id"]] if "id" in seg else [],
    }


def _split_long(seg: dict) -> list:
    duration = seg["end"] - seg["start"]
    if duration <= MAX_SEGMENT_DURATION:
        return [seg]

    sentences = re.split(r'(?<=[.!?…])\s+', seg["text"].strip())
    if len(sentences) <= 1:
        return [seg]

    total_words = max(1, len(seg["text"].split()))
    result = []
    t = seg["start"]

    for sentence in sentences:
        frac = len(sentence.split()) / total_words
        dur = (seg["end"] - seg["start"]) * frac
        result.append({
            "start": round(t, 3),
            "end": round(t + dur, 3),
            "text": sentence,
            "source_ids": seg.get("source_ids", []),
        })
        t += dur

    return result
