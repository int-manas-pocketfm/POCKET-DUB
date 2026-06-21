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

import anthropic
import numpy as np
from dotenv import load_dotenv
from pydub import AudioSegment

load_dotenv()

ARGUS_API_KEY = os.getenv("ARGUS_API_KEY")
ARGUS_BASE_URL = os.getenv("ARGUS_BASE_URL", "https://api.anthropic.com")
ARGUS_MODEL = os.getenv("ARGUS_MODEL", "claude-opus-4-5")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

TTS_WORKERS = 10
TTS_TOLERANCE = 0.3      # seconds — acceptable duration mismatch
MAX_REWRITE_ITERS = 3    # max LLM rewrite attempts per segment
EXTEND_DROP_THRESHOLD = 0.5  # seconds — brief overruns are acceptable

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

    raw = json.loads((project_dir / "segments.json").read_text())
    segments = raw.get("merged", raw) if isinstance(raw, dict) else raw
    translations = json.loads((project_dir / "translations.json").read_text())

    audio_dir = project_dir / "segments_audio"
    audio_dir.mkdir(exist_ok=True)

    dub_path = project_dir / "dub_state.json"
    dub_state: dict = json.loads(dub_path.read_text()) if dub_path.exists() else {}

    # Pre-populate character info from characters_final
    chars: dict = {}
    chars_path = project_dir / "characters_final.json"
    if chars_path.exists():
        chars = json.loads(chars_path.read_text())

    # Segments still needing TTS
    pending = [i for i in range(len(segments))
               if dub_state.get(str(i), {}).get("status") != "done"]

    if not pending:
        log_fn("All segments already done — skipping TTS generation.")
        return

    log_fn(f"Generating TTS for {len(pending)} segments ({TTS_WORKERS} parallel workers)...")

    argus_client = anthropic.Anthropic(api_key=ARGUS_API_KEY, base_url=ARGUS_BASE_URL)
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
            return sid, {"status": "error", "error": "No translation text available"}

        # Pick voice
        char = "UNKNOWN"
        for src_id in seg.get("source_ids", [i]):
            if str(src_id) in chars:
                char = chars[str(src_id)]
                break
        voice_id = voice_config.get(char) or voice_config.get("default") or list(voice_config.values())[0]

        target_dur = seg["end"] - seg["start"]
        english = seg.get("text", "")
        audio_file = str(audio_dir / f"{i:04d}.mp3")
        current_text = text
        actual_dur = None
        iterations = 0

        for attempt in range(MAX_REWRITE_ITERS + 1):
            ok = _generate_tts(current_text, voice_id, audio_file)
            if not ok:
                return sid, {"status": "error", "error": "edge-tts generation failed"}

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

        extend_by = max(0.0, actual_dur - target_dur - EXTEND_DROP_THRESHOLD) if actual_dur else 0.0

        return sid, {
            "status": "done",
            "english": english,
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


def _generate_tts(text: str, voice: str, audio_file: str) -> bool:
    """Generate TTS via edge-tts (free Microsoft neural voices) and save to file."""
    import asyncio
    import edge_tts

    async def _run():
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(audio_file)

    try:
        asyncio.run(_run())
        return True
    except Exception:
        return False


def _mp3_duration(path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", path],
        capture_output=True, text=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def _rewrite_for_timing(
    client: anthropic.Anthropic,
    english: str,
    current: str,
    target: float,
    actual: float,
    lang_name: str,
    prev: str,
    nxt: str,
) -> str | None:
    direction = "shorter" if actual > target else "longer"
    max_words = len(current.split())
    prompt = (
        f"Rewrite this {lang_name} subtitle to be {direction} "
        f"to fit in {target:.1f}s (current TTS is {actual:.1f}s).\n"
        f"Preserve meaning. Do NOT exceed {max_words} words. "
        "Return ONLY the rewritten text.\n\n"
        f"English: {english}\n"
        f"Current {lang_name}: {current}\n"
        f"Context before: {prev}\n"
        f"Context after: {nxt}"
    )
    try:
        resp = client.messages.create(
            model=ARGUS_MODEL,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip().strip('"')
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

    meta = json.loads((project_dir / "metadata.json").read_text())
    video_path = Path(meta["video_path"])
    dub_state = json.loads((project_dir / "dub_state.json").read_text())
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

            # Normal playback chunk
            normal = tmp / f"n{idx}.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(video_path),
                 "-ss", str(prev_end), "-to", str(seg_end),
                 "-c", "copy", str(normal)],
                capture_output=True, check=True,
            )
            chunks.append(str(normal))

            # Freeze frame: extract last frame, loop it
            freeze = tmp / f"f{idx}.mp4"
            fps = 25
            n_frames = max(1, int(extend * fps))
            subprocess.run(
                ["ffmpeg", "-y",
                 "-i", str(video_path),
                 "-ss", str(max(0, seg_end - 0.04)),
                 "-vframes", "1",
                 "-vf", f"loop={n_frames}:1,settb=AVTB,fps={fps}",
                 "-t", str(extend),
                 "-c:v", "libx264", "-preset", "fast",
                 str(freeze)],
                capture_output=True, check=True,
            )
            chunks.append(str(freeze))
            prev_end = seg_end

        # Final chunk to end of video
        final_chunk = tmp / "final.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path),
             "-ss", str(prev_end), "-c", "copy", str(final_chunk)],
            capture_output=True, check=True,
        )
        chunks.append(str(final_chunk))

        # Concatenate
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
