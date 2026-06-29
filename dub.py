"""
DubStudio — Stage 3: TTS + Stage 4: Stitch & Mux
Stage 3: ElevenLabs voice synthesis with iterative LLM rewrite loop
Stage 4: NumPy audio timeline stitching + FFmpeg video assembly
"""

import json
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from argus import make_client
import numpy as np
from dotenv import load_dotenv
from pydub import AudioSegment

load_dotenv()

ARGUS_MODEL = os.getenv("ARGUS_MODEL", "claude-sonnet-4-6")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

TTS_WORKERS = 10
TTS_TOLERANCE = 0.3      # seconds — acceptable duration mismatch
MAX_REWRITE_ITERS = 3    # max LLM rewrite attempts per segment
EXTEND_DROP_THRESHOLD = 0.15  # seconds — brief overruns are acceptable

SAMPLE_RATE = 44100
AUDIO_BITRATE = "192k"

LANG_NAMES = {
    "de": "German", "fr": "French", "it": "Italian",
    "es": "Spanish", "pt": "Portuguese",
}


# ── Stage 3: TTS ───────────────────────────────────────────────────────────────

def run_stage3_tts(project_dir: Path, target_lang: str, voice_config: dict, log_fn: Callable):
    log_fn("=== Stage 3: TTS ===")

    # Save voice config for use by regenerate-tts endpoint
    (project_dir / "voice_config.json").write_text(
        json.dumps(voice_config, indent=2)
    )

    raw = json.loads((project_dir / "segments.json").read_text(encoding="utf-8-sig"))
    segments = raw.get("merged", raw) if isinstance(raw, dict) else raw
    translations = json.loads((project_dir / "translations.json").read_text(encoding="utf-8-sig"))

    audio_dir = project_dir / "segments_audio"
    audio_dir.mkdir(exist_ok=True)

    dub_path = project_dir / "dub_state.json"
    dub_state: dict = json.loads(dub_path.read_text(encoding="utf-8-sig")) if dub_path.exists() else {}

    # Pre-populate character info from characters_final
    chars: dict = {}
    chars_path = project_dir / "characters_final.json"
    if chars_path.exists():
        chars = json.loads(chars_path.read_text(encoding="utf-8-sig"))

    # Segments still needing TTS
    pending = [i for i in range(len(segments))
               if dub_state.get(str(i), {}).get("status") != "done"]

    if not pending:
        log_fn("All segments already done — skipping TTS generation.")
        return

    log_fn(f"Generating TTS for {len(pending)} segments ({TTS_WORKERS} parallel workers)...")

    argus_client = make_client()
    lang_name = LANG_NAMES.get(target_lang, target_lang)

    def process(i: int) -> tuple[str, dict]:
        seg = segments[i]
        sid = str(i)
        trans = translations.get(sid, {})

        # Pick best available translation
        text = None
        for key in (
            f"llm_{target_lang}_local", f"llm_{target_lang}",
            f"google_{target_lang}_local", f"google_{target_lang}",
        ):
            if trans.get(key):
                text = trans[key]
                break

        if not text:
            return sid, {"status": "done", "english": seg.get("text", ""), "translated_text": "",
                         "start": seg["start"], "end": seg["end"], "skipped": True}

        # Pick voice — characters_final.json is keyed by merged segment index
        char = chars.get(str(i), "UNKNOWN")
        vc_lower = {k.lower(): v for k, v in voice_config.items()}
        voice_id = vc_lower.get(char.lower()) or vc_lower.get("default") or list(voice_config.values())[0]

        target_dur = seg["end"] - seg["start"]
        english = seg.get("text", "")
        audio_file = str(audio_dir / f"{i:04d}.mp3")
        current_text = text
        actual_dur = None
        iterations = 0

        for attempt in range(MAX_REWRITE_ITERS + 1):
            ok, tts_err = _generate_tts(current_text, voice_id, audio_file)
            if not ok:
                return sid, {"status": "error", "error": f"TTS failed: {tts_err}"}

            actual_dur = _mp3_duration(audio_file)
            diff = actual_dur - target_dur
            iterations = attempt + 1

            if abs(diff) <= TTS_TOLERANCE or attempt >= MAX_REWRITE_ITERS:
                break

            # Ask LLM to rewrite shorter/longer
            prev = segments[i - 1].get("text", "")[:60] if i > 0 else ""
            nxt = segments[i + 1].get("text", "")[:60] if i < len(segments) - 1 else ""
            rewritten = _rewrite_for_timing(
                argus_client, english, current_text,
                target_dur, actual_dur, lang_name, prev, nxt
            )
            if rewritten:
                current_text = rewritten

        # Tempo-fit: stretch/compress audio to reduce gaps and freeze frames
        if actual_dur:
            actual_dur = _adjust_audio_tempo(audio_file, target_dur, actual_dur)

        extend_by = max(0.0, actual_dur - target_dur - EXTEND_DROP_THRESHOLD) if actual_dur else 0.0

        return sid, {
            "status": "done",
            "english": english,
            "source_text": text,
            "translated_text": current_text,
            "character": char,
            "voice": voice_id,
            "start": seg["start"],
            "end": seg["end"],
            "target_duration": round(target_dur, 3),
            "actual_duration": round(actual_dur, 3) if actual_dur else None,
            "iterations": iterations,
            "audio_file": audio_file,
            "extend_by": round(extend_by, 3),
        }

    import cancel_flag
    project_name = project_dir.name
    completed = 0
    with ThreadPoolExecutor(max_workers=TTS_WORKERS) as pool:
        futures = {pool.submit(process, i): i for i in pending}
        for future in as_completed(futures):
            sid, result = future.result()
            dub_state[sid] = result
            completed += 1
            if completed % 10 == 0 or completed == len(pending):
                log_fn(f"    TTS progress: {completed}/{len(pending)}")
                dub_path.write_text(json.dumps(dub_state, ensure_ascii=False, indent=2), encoding="utf-8")
            if cancel_flag.is_cancelled(project_name):
                log_fn(f"    TTS cancelled after {completed}/{len(pending)} segments.")
                pool.shutdown(wait=False, cancel_futures=True)
                break

    dub_path.write_text(json.dumps(dub_state, ensure_ascii=False, indent=2), encoding="utf-8")

    log_fn("Optimizing freeze-frame extensions...")
    dub_state = _optimize_extends(segments, dub_state)
    dub_path.write_text(json.dumps(dub_state, ensure_ascii=False, indent=2), encoding="utf-8")

    extends = sum(1 for s in dub_state.values() if s.get("extend_by", 0) > 0)
    errors = sum(1 for s in dub_state.values() if s.get("status") == "error")

    log_fn("Generating dubbed script review document...")
    try:
        from translate import write_stage3_review_doc
        write_stage3_review_doc(project_dir, segments, dub_state, target_lang, project_dir.name)
        log_fn(f"    Done → {project_dir.name}_stage3_dubbed.docx")
    except Exception as exc:
        log_fn(f"    Warning: stage3 doc skipped ({exc})")

    log_fn(f"=== Stage 3 complete — {extends} freeze-frame segments, {errors} errors ===")


def _adjust_audio_tempo(audio_file: str, target_dur: float, actual_dur: float) -> float:
    """Stretch or compress audio to fit target_dur using FFmpeg atempo (±30% max)."""
    if actual_dur <= 0 or target_dur <= 0:
        return actual_dur
    ratio = actual_dur / target_dur  # <1 = too short (slow down), >1 = too long (speed up)
    if abs(ratio - 1.0) < 0.05:     # within 5% — not worth adjusting
        return actual_dur
    if not (0.7 <= ratio <= 1.3):   # beyond ±30% — sounds unnatural, skip
        return actual_dur
    tmp = audio_file + ".tempo.mp3"
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", audio_file,
         "-filter:a", f"atempo={ratio:.4f}",
         "-q:a", "2", tmp],
        capture_output=True,
    )
    if result.returncode == 0:
        os.replace(tmp, audio_file)
        return _mp3_duration(audio_file)
    return actual_dur


def _generate_tts(text: str, voice: str, audio_file: str) -> tuple[bool, str]:
    """Generate TTS via ElevenLabs and save to mp3 file. Returns (ok, error_message)."""
    import httpx
    from dotenv import load_dotenv
    load_dotenv(override=True)
    api_key = os.getenv("ELEVENLABS_API_KEY", "")
    if not api_key:
        return False, "ELEVENLABS_API_KEY not set"
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}"
    headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }
    try:
        r = httpx.post(url, headers=headers, json=payload, timeout=60.0)
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}: {r.text[:200]}"
        if len(r.content) < 100:
            return False, f"Response too short ({len(r.content)} bytes) — possible empty audio"
        with open(audio_file, "wb") as f:
            f.write(r.content)
        return True, ""
    except Exception as e:
        return False, str(e)


def _mp3_duration(path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", path],
        capture_output=True, text=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def _rewrite_for_timing(
    client,
    english: str,
    current: str,
    target: float,
    actual: float,
    lang_name: str,
    prev: str,
    nxt: str,
) -> str | None:
    import re as _re
    direction = "shorter" if actual > target else "longer"
    max_words = len(current.split())
    prompt = (
        f"Rewrite this {lang_name} subtitle to be {direction} "
        f"to fit in {target:.1f}s (current TTS is {actual:.1f}s).\n"
        f"Preserve meaning. Do NOT exceed {max_words} words.\n\n"
        f"English: {english}\n"
        f"Current {lang_name}: {current}\n"
        f"Context before: {prev}\n"
        f"Context after: {nxt}"
    )
    try:
        resp = client.messages.create(
            model=ARGUS_MODEL,
            max_tokens=120,
            system="You rewrite subtitles. Output ONLY the rewritten subtitle — one line, no explanation, no alternatives, no markdown.",
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": ""},
            ],
        )
        raw = resp.content[0].text.strip()
        # Extract first clean line (no markdown, no reasoning)
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        result = lines[0] if lines else raw
        # Strip markdown bold/italic
        result = _re.sub(r'\*+', '', result).strip().strip('"').strip()
        return result or None
    except Exception:
        return None


def _optimize_extends(segments: list, dub_state: dict) -> dict:
    for i in range(len(segments)):
        sid = str(i)
        if dub_state.get(sid, {}).get("extend_by", 0) <= 0:
            continue

        # Check if next segment has slack (TTS finished early)
        next_sid = str(i + 1)
        if next_sid in dub_state:
            nxt = dub_state[next_sid]
            if nxt.get("target_duration") and nxt.get("actual_duration"):
                slack = nxt["target_duration"] - nxt["actual_duration"]
                if slack >= dub_state[sid]["extend_by"]:
                    dub_state[sid]["extend_by"] = 0.0
                    continue

        if dub_state[sid]["extend_by"] <= EXTEND_DROP_THRESHOLD:
            dub_state[sid]["extend_by"] = 0.0

    return dub_state


# ── Stage 4: Stitch & Mux ─────────────────────────────────────────────────────

def run_stage4_stitch(project_dir: Path, target_lang: str, log_fn: Callable):
    log_fn("=== Stage 4: STITCH & MUX ===")

    meta = json.loads((project_dir / "metadata.json").read_text(encoding="utf-8-sig"))
    video_path = Path(meta["video_path"])
    dub_state = json.loads((project_dir / "dub_state.json").read_text(encoding="utf-8-sig"))
    lang_name = LANG_NAMES.get(target_lang, target_lang.lower())

    # 4a. Build extended video if any segment needs freeze frames
    extends = {k: v for k, v in dub_state.items() if v.get("extend_by", 0) > 0}
    if extends:
        log_fn(f"4a. Building extended video ({len(extends)} freeze-frame segments)...")
        extended_path = project_dir / "video_extended.mp4"
        _build_extended_video(video_path, extended_path, dub_state, log_fn)
        working_video = extended_path
    else:
        log_fn("4a. No freeze-frame extensions needed.")
        working_video = video_path

    video_duration = _video_duration(working_video)
    log_fn(f"    Video duration: {video_duration:.2f}s")

    # 4b. Stitch dubbed audio track
    log_fn("4b. Stitching audio track (sample-level NumPy placement)...")
    audio_out = project_dir / f"{target_lang}_audio.mp3"
    _stitch_audio(dub_state, video_duration, audio_out, log_fn)

    # 4c. Mux
    log_fn("4c. Muxing video + dubbed audio...")
    final_name = video_path.stem + f"_{target_lang}.mp4"
    final_path = project_dir / final_name
    _mux(working_video, audio_out, final_path, log_fn)

    log_fn(f"=== Stage 4 complete — {final_name} ===")


def _build_extended_video(
    video_path: Path,
    output_path: Path,
    dub_state: dict,
    log_fn: Callable,
):
    sorted_extends = sorted(
        [(int(k), v) for k, v in dub_state.items() if v.get("extend_by", 0) > 0],
        key=lambda x: x[0],
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        chunks: list[str] = []
        prev_end = 0.0

        for idx, (seg_id, seg) in enumerate(sorted_extends):
            seg_end = seg["end"]
            extend = seg["extend_by"]

            # Normal playback chunk — re-encode for frame-accurate cuts, strip audio
            normal = tmp / f"n{idx}.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-ss", str(prev_end),
                 "-i", str(video_path),
                 "-to", str(seg_end - prev_end),
                 "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-an",
                 str(normal)],
                capture_output=True, check=True,
            )
            chunks.append(str(normal))

            # Freeze frame: extract last frame, loop it
            freeze = tmp / f"f{idx}.mp4"
            fps = 25
            n_frames = max(1, int(extend * fps))
            subprocess.run(
                ["ffmpeg", "-y",
                 "-ss", str(max(0, seg_end - 0.04)),
                 "-i", str(video_path),
                 "-vframes", "1",
                 "-vf", f"loop={n_frames}:1,settb=AVTB,fps={fps}",
                 "-t", str(extend),
                 "-c:v", "libx264", "-preset", "fast", "-an",
                 str(freeze)],
                capture_output=True, check=True,
            )
            chunks.append(str(freeze))
            prev_end = seg_end

        # Final chunk to end of video
        final_chunk = tmp / "final.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(prev_end),
             "-i", str(video_path),
             "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-an",
             str(final_chunk)],
            capture_output=True, check=True,
        )
        chunks.append(str(final_chunk))

        # Concatenate — reset timestamps so each chunk starts where the previous ended
        concat_txt = tmp / "concat.txt"
        concat_txt.write_text("\n".join(f"file '{c}'" for c in chunks))
        result = subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", str(concat_txt), "-c", "copy", str(output_path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Extended video build failed:\n{result.stderr[-400:]}")

    log_fn(f"    Extended video → {output_path.name}")


def _stitch_audio(
    dub_state: dict,
    video_duration: float,
    output_path: Path,
    log_fn: Callable,
):
    total_samples = int((video_duration + 5.0) * SAMPLE_RATE)
    timeline = np.zeros(total_samples, dtype=np.float64)

    done = sorted(
        [(int(k), v) for k, v in dub_state.items()
         if v.get("status") == "done" and v.get("audio_file")],
        key=lambda x: x[0],
    )

    # Build cumulative extension offset
    cumul_ext = 0.0
    ext_by_id: dict[int, float] = {int(k): v.get("extend_by", 0.0) for k, v in dub_state.items()}

    for seg_id, seg in done:
        audio_file = seg["audio_file"]
        if not Path(audio_file).exists():
            log_fn(f"    Warning: audio missing for seg {seg_id} — skipping")
            continue

        placement = seg["start"] + cumul_ext
        offset = int(placement * SAMPLE_RATE)

        try:
            audio = AudioSegment.from_mp3(audio_file)
            audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1)
            samples = np.array(audio.get_array_of_samples(), dtype=np.float64)

            end_idx = offset + len(samples)
            if end_idx > len(timeline):
                timeline = np.pad(timeline, (0, end_idx - len(timeline)))

            timeline[offset:end_idx] += samples
        except Exception as exc:
            log_fn(f"    Warning: failed to load seg {seg_id} audio: {exc}")

        cumul_ext += ext_by_id.get(seg_id, 0.0)

    # Prevent clipping
    peak = np.max(np.abs(timeline))
    if peak > 32767:
        timeline = timeline * (32767.0 / peak)

    pcm = timeline.astype(np.int16)
    out = AudioSegment(pcm.tobytes(), frame_rate=SAMPLE_RATE, sample_width=2, channels=1)
    out.export(str(output_path), format="mp3", bitrate=AUDIO_BITRATE)
    log_fn(f"    Audio stitched → {output_path.name}")


def _mux(video_path: Path, audio_path: Path, output_path: Path, log_fn: Callable):
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-map", "0:v:0",
            "-map", "1:a:0",
            str(output_path),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg mux failed:\n{result.stderr[-400:]}")
    log_fn(f"    Final video → {output_path.name}")


def _video_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
        capture_output=True, text=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])
