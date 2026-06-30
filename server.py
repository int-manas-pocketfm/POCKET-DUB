"""
DubStudio — FastAPI backend
Manages projects, runs pipeline stages, streams logs via SSE.
"""

import asyncio
import json
import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

import os

# Allow large multipart uploads (python-multipart default is 1 MB per field)
os.environ.setdefault("MULTIPART_MAX_SIZE", str(10 * 1024 * 1024 * 1024))

BASE_DIR = Path(__file__).parent
# Vercel has a read-only filesystem except /tmp
OUTPUT_DIR = Path("/tmp/dubstudio") if os.getenv("VERCEL") else BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

app = FastAPI(title="DubStudio", version="1.0.0")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# In-memory job state per project
# { project_name: { "status": "idle"|"running"|"done"|"error", "stage": int|None, "queue": Queue } }
_jobs: dict[str, dict] = {}

LANG_NAMES = {
    "de": "German",
    "fr": "French",
    "it": "Italian",
    "es": "Spanish",
    "pt": "Portuguese",
    "hi": "Hindi",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _project_dir(name: str) -> Path:
    d = OUTPUT_DIR / name
    if not d.exists():
        raise HTTPException(status_code=404, detail=f"Project '{name}' not found")
    return d


def _load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (UnicodeDecodeError, ValueError):
        try:
            return json.loads(path.read_text(encoding="latin-1"))
        except Exception:
            return None


def _load_metadata(name: str) -> dict:
    d = _project_dir(name)
    meta = _load_json(d / "metadata.json")
    if meta is None:
        raise HTTPException(status_code=404, detail="Project metadata not found")
    return meta


def _ensure_job(name: str) -> dict:
    if name not in _jobs:
        _jobs[name] = {"status": "idle", "stage": None, "queue": queue.Queue(), "cancelled": False}
    return _jobs[name]


def _run_in_thread(fn, name: str, stage: int, *args):
    job = _ensure_job(name)
    job["status"] = "running"
    job["stage"] = stage
    job["cancelled"] = False
    import cancel_flag
    cancel_flag.clear(name)

    def log(msg: str):
        job["queue"].put({"type": "log", "message": msg})

    def worker():
        try:
            fn(*args, log_fn=log)
            if not job["cancelled"]:
                job["status"] = "done"
                job["queue"].put({"type": "done", "message": f"Stage {stage} complete"})
        except Exception as exc:
            if not job["cancelled"]:
                job["status"] = "error"
                job["queue"].put({"type": "error", "message": str(exc)})

    threading.Thread(target=worker, daemon=True).start()


def _stage_flags(d: Path, meta: dict) -> dict:
    lang = meta.get("target_lang", "de")
    trans = _load_json(d / "translations.json") or {}

    has_local = any(
        f"llm_{lang}_local" in v or f"google_{lang}_local" in v
        for v in trans.values()
    )
    dub = _load_json(d / "dub_state.json") or {}
    tts_done = bool(dub) and all(
        s.get("status") == "done" or not s.get("translated_text", "").strip()
        for s in dub.values()
    )
    final_videos = list(d.glob(f"*_{lang}.mp4"))

    return {
        "stage1_done": (d / "translations.json").exists(),
        "stage2_done": has_local,
        "stage3_done": tts_done,
        "stage4_done": bool(final_videos),
        "final_video": final_videos[0].name if final_videos else None,
    }


# ── Routes: serve frontend ─────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


# ── Routes: projects ───────────────────────────────────────────────────────────

@app.get("/api/projects")
async def list_projects():
    result = []
    for d in sorted(OUTPUT_DIR.iterdir()):
        if not d.is_dir():
            continue
        meta = _load_json(d / "metadata.json")
        if meta is None:
            continue
        flags = _stage_flags(d, meta)
        result.append({**meta, **flags})
    return result


@app.post("/api/projects/link")
async def create_project_from_path(request: Request):
    """Create a project by pointing to an existing local file — no browser upload."""
    data = await request.json()
    name = (data.get("name") or "").strip()
    target_lang = (data.get("target_lang") or "de").strip()
    video_path = (data.get("video_path") or "").strip()

    if not name:
        raise HTTPException(status_code=400, detail="Project name is required")
    if not video_path:
        raise HTTPException(status_code=400, detail="Video path is required")

    path = Path(video_path)
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"File not found: {video_path}")
    if not path.is_file():
        raise HTTPException(status_code=400, detail=f"Not a file: {video_path}")

    project_dir = OUTPUT_DIR / name
    if project_dir.exists():
        raise HTTPException(status_code=400, detail=f"Project '{name}' already exists")
    project_dir.mkdir(parents=True)

    meta = {
        "name": name,
        "video_path": str(path.resolve()),
        "video_filename": path.name,
        "target_lang": target_lang,
        "target_lang_name": LANG_NAMES.get(target_lang, target_lang),
        "source_lang": "en",
        "fps": 25,
        "created_at": datetime.now().isoformat(),
    }
    (project_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


@app.post("/api/projects")
async def create_project(
    name: str = Form(...),
    target_lang: str = Form(...),
    video: UploadFile = File(...),
):
    project_dir = OUTPUT_DIR / name
    if project_dir.exists():
        raise HTTPException(status_code=400, detail=f"Project '{name}' already exists")

    project_dir.mkdir(parents=True)

    video_path = project_dir / video.filename
    with open(video_path, "wb") as f:
        while chunk := await video.read(1024 * 1024):  # 1 MB chunks
            f.write(chunk)

    meta = {
        "name": name,
        "video_path": str(video_path),
        "video_filename": video.filename,
        "target_lang": target_lang,
        "target_lang_name": LANG_NAMES.get(target_lang, target_lang),
        "source_lang": "en",
        "fps": 25,
        "created_at": datetime.now().isoformat(),
    }
    (project_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


@app.get("/api/projects/{name}")
async def get_project(name: str):
    d = _project_dir(name)
    meta = _load_metadata(name)
    flags = _stage_flags(d, meta)

    job = _jobs.get(name, {})
    return {
        **meta,
        **flags,
        "job_status": job.get("status", "idle"),
        "job_stage": job.get("stage"),
    }


@app.delete("/api/projects/{name}")
async def delete_project(name: str):
    import shutil
    d = _project_dir(name)
    shutil.rmtree(d)
    if name in _jobs:
        del _jobs[name]
    return {"status": "deleted"}


@app.post("/api/projects/{name}/cancel")
async def cancel_job(name: str):
    job = _jobs.get(name, {})
    if job.get("status") != "running":
        raise HTTPException(status_code=400, detail="No running job to cancel")
    import cancel_flag
    cancel_flag.set_cancel(name)
    job["cancelled"] = True
    job["status"] = "cancelled"
    job["queue"].put({"type": "error", "message": "Cancelled by user"})
    return {"status": "cancelled"}


# ── Routes: stage execution ────────────────────────────────────────────────────

def _assert_idle(name: str):
    job = _jobs.get(name, {})
    if job.get("status") == "running":
        raise HTTPException(status_code=400, detail="A stage is already running for this project")


@app.post("/api/projects/{name}/stage1")
async def run_stage1(name: str):
    meta = _load_metadata(name)
    _assert_idle(name)
    _ensure_job(name)

    from pipeline import run_stage1 as _run_stage1
    _run_in_thread(
        _run_stage1, name, 1,
        meta["video_path"],
        OUTPUT_DIR / name,
        meta["target_lang"],
    )
    return {"status": "started", "stage": 1}


@app.post("/api/projects/{name}/stage2")
async def run_stage2(name: str, localization_csv: UploadFile = File(...)):
    meta = _load_metadata(name)
    _assert_idle(name)

    d = OUTPUT_DIR / name
    csv_path = d / "localization.csv"
    with open(csv_path, "wb") as f:
        f.write(await localization_csv.read())

    _ensure_job(name)
    from translate import run_stage2_localize
    _run_in_thread(run_stage2_localize, name, 2, d, csv_path, meta["target_lang"])
    return {"status": "started", "stage": 2}


@app.post("/api/projects/{name}/stage2/inline")
async def run_stage2_inline(name: str, request: Request):
    meta = _load_metadata(name)
    _assert_idle(name)

    body = await request.json()
    replacements = body.get("replacements", [])
    if not replacements:
        raise HTTPException(status_code=400, detail="No replacements provided")

    import csv as _csv
    d = OUTPUT_DIR / name
    csv_path = d / "localization.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["Original", "Localized"])
        for r in replacements:
            if r.get("original") and r.get("localized"):
                w.writerow([r["original"], r["localized"]])

    _ensure_job(name)
    from translate import run_stage2_localize
    _run_in_thread(run_stage2_localize, name, 2, d, csv_path, meta["target_lang"])
    return {"status": "started", "stage": 2}


@app.post("/api/projects/{name}/stage3")
async def run_stage3(name: str, request: Request):
    meta = _load_metadata(name)
    _assert_idle(name)

    body = await request.json()
    voice_config: dict = body.get("voice_config", {})
    if not voice_config:
        raise HTTPException(status_code=400, detail="voice_config is required (character → ElevenLabs voice ID)")

    d = OUTPUT_DIR / name
    _ensure_job(name)
    from dub import run_stage3_tts
    _run_in_thread(run_stage3_tts, name, 3, d, meta["target_lang"], voice_config)
    return {"status": "started", "stage": 3}


@app.post("/api/projects/{name}/stage4")
async def run_stage4(name: str):
    meta = _load_metadata(name)
    _assert_idle(name)

    d = OUTPUT_DIR / name
    _ensure_job(name)
    from dub import run_stage4_stitch
    _run_in_thread(run_stage4_stitch, name, 4, d, meta["target_lang"])
    return {"status": "started", "stage": 4}


# ── Routes: log streaming (SSE) ────────────────────────────────────────────────

@app.get("/api/projects/{name}/logs")
async def stream_logs(name: str, request: Request):
    job = _ensure_job(name)
    q = job["queue"]

    async def generator():
        while True:
            if await request.is_disconnected():
                break
            try:
                msg = q.get_nowait()
                yield {"data": json.dumps(msg)}
                if msg.get("type") in ("done", "error"):
                    break
            except queue.Empty:
                await asyncio.sleep(0.03)

    return EventSourceResponse(generator())


# ── Routes: segments & translations ───────────────────────────────────────────

@app.get("/api/projects/{name}/segments")
async def get_segments(name: str):
    d = _project_dir(name)

    raw = _load_json(d / "segments.json")
    if raw is None:
        return []
    segments = raw.get("merged", raw.get("segments", raw)) if isinstance(raw, dict) else raw

    translations = _load_json(d / "translations.json") or {}
    characters = _load_json(d / "characters_final.json") or {}
    dub_state = _load_json(d / "dub_state.json") or {}

    result = []
    for i, seg in enumerate(segments):
        sid = str(i)
        char = characters.get(str(i), "?")
        result.append({
            "id": i,
            "start": seg["start"],
            "end": seg["end"],
            "text": seg.get("text", ""),
            "character": char,
            "translations": translations.get(sid, {}),
            "dub": dub_state.get(sid, {}),
        })
    return result


@app.put("/api/projects/{name}/segments/{seg_id}")
async def update_segment(name: str, seg_id: int, request: Request):
    d = _project_dir(name)
    trans_path = d / "translations.json"
    if not trans_path.exists():
        raise HTTPException(status_code=404, detail="No translations found — run Stage 1 first")

    body = await request.json()
    translations = json.loads(trans_path.read_text(encoding="utf-8-sig"))
    sid = str(seg_id)
    if sid not in translations:
        translations[sid] = {}
    translations[sid].update(body.get("translations", {}))
    trans_path.write_text(json.dumps(translations, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "updated"}


@app.post("/api/projects/{name}/segments/{seg_id}/regenerate-tts")
async def regenerate_tts(name: str, seg_id: int):
    meta = _load_metadata(name)
    _assert_idle(name)

    d = OUTPUT_DIR / name
    dub_path = d / "dub_state.json"
    if not dub_path.exists():
        raise HTTPException(status_code=404, detail="No dub state — run Stage 3 first")

    # Mark segment as pending so Stage 3 will re-process it
    dub_state = json.loads(dub_path.read_text(encoding="utf-8-sig"))
    dub_state[str(seg_id)] = {"status": "pending"}
    dub_path.write_text(json.dumps(dub_state, ensure_ascii=False, indent=2), encoding="utf-8")

    # Load voice config from last run or raise
    vc_path = d / "voice_config.json"
    if not vc_path.exists():
        raise HTTPException(status_code=400, detail="voice_config.json not found — re-run Stage 3 first")
    voice_config = json.loads(vc_path.read_text(encoding="utf-8-sig"))

    _ensure_job(name)
    from dub import run_stage3_tts
    _run_in_thread(run_stage3_tts, name, 3, d, meta["target_lang"], voice_config)
    return {"status": "started", "segment": seg_id}


@app.post("/api/projects/{name}/regen-edited")
async def regen_edited(name: str):
    meta = _load_metadata(name)
    _assert_idle(name)
    d = OUTPUT_DIR / name

    dub_path = d / "dub_state.json"
    if not dub_path.exists():
        raise HTTPException(status_code=400, detail="No dub state — run Stage 3 first")

    vc_path = d / "voice_config.json"
    if not vc_path.exists():
        raise HTTPException(status_code=400, detail="voice_config.json not found — re-run Stage 3 first")

    dub_state = json.loads(dub_path.read_text(encoding="utf-8-sig"))
    translations = _load_json(d / "translations.json") or {}
    lang = meta["target_lang"]

    reset_count = 0
    for sid, dub in dub_state.items():
        if dub.get("status") != "done":
            continue
        trans = translations.get(sid, {})
        # Only regen if we have a source_text baseline (set since the fix was applied)
        # Without it we can't tell what was manually edited vs timing-rewritten
        if not dub.get("source_text"):
            continue
        current = (trans.get(f"llm_{lang}_local") or trans.get(f"llm_{lang}") or
                   trans.get(f"google_{lang}_local") or trans.get(f"google_{lang}") or "").strip()
        tts_text = dub.get("source_text", "").strip()
        if current and current != tts_text:
            dub_state[sid]["status"] = "pending"
            reset_count += 1

    if reset_count == 0:
        return {"status": "nothing_to_regen", "reset": 0}

    dub_path.write_text(json.dumps(dub_state, ensure_ascii=False, indent=2), encoding="utf-8")
    voice_config = json.loads(vc_path.read_text(encoding="utf-8-sig"))
    _ensure_job(name)
    from dub import run_stage3_tts
    _run_in_thread(run_stage3_tts, name, 3, d, lang, voice_config)
    return {"status": "started", "reset": reset_count}


# ── Routes: writer script import ──────────────────────────────────────────────

@app.post("/api/projects/{name}/import-writer-script")
async def import_writer_script(name: str, file: UploadFile = File(...)):
    d = _project_dir(name)
    meta = _load_metadata(name)
    target_lang = meta["target_lang"]

    trans_path = d / "translations.json"
    if not trans_path.exists():
        raise HTTPException(status_code=400, detail="No translations — run Stage 1 first")

    # Save uploaded file to a temp path
    import tempfile, openpyxl as _xl
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp_path = tmp.name
        while chunk := await file.read(1024 * 1024):
            tmp.write(chunk)

    try:
        # Load without read_only so numeric IDs aren't lost
        wb = _xl.load_workbook(tmp_path, data_only=True)
        ws = wb.active

        # Detect header row and column positions
        writer_col = feedback_col = id_col = None
        header_row_idx = None
        for row_idx, row in enumerate(ws.iter_rows(values_only=True), 1):
            row_lower = [str(c).lower().strip() if c else "" for c in row]
            joined = " ".join(row_lower)
            if "writer" in joined:
                for ci, val in enumerate(row_lower):
                    if val in ("id", "#"):
                        id_col = ci
                    if "writer" in val and ("script" in val or "approved" in val):
                        writer_col = ci
                    if "feedback" in val:
                        feedback_col = ci
                if writer_col is not None:
                    header_row_idx = row_idx
                    break

        if writer_col is None:
            raise HTTPException(status_code=400, detail="Could not find a 'Writer Script' or 'Writer approved' column")

        translations = json.loads(trans_path.read_text(encoding="utf-8-sig"))
        writer_key = f"writer_{target_lang}"
        feedback_key = f"writer_feedback_{target_lang}"

        updated = skipped = 0
        data_start = (header_row_idx or 3) + 1
        for row_offset, row in enumerate(ws.iter_rows(min_row=data_start, values_only=True)):
            writer_text = str(row[writer_col]).strip() if row[writer_col] else ""
            feedback = str(row[feedback_col]).strip() if (feedback_col is not None and row[feedback_col]) else ""

            # Try ID column first, fall back to sequential row offset
            raw_id = row[id_col] if id_col is not None else None
            try:
                sid = str(int(float(raw_id)))
            except (TypeError, ValueError):
                sid = str(row_offset)  # fall back to position-based index

            if sid not in translations:
                translations[sid] = {}

            if writer_text:
                translations[sid][writer_key] = writer_text
                if feedback:
                    translations[sid][feedback_key] = feedback
                updated += 1
            else:
                skipped += 1

        trans_path.write_text(json.dumps(translations, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        import os as _os
        try: _os.unlink(tmp_path)
        except Exception: pass

    return {"status": "ok", "updated": updated, "skipped": skipped}


# ── Routes: cue sheet re-export ───────────────────────────────────────────────

@app.post("/api/projects/{name}/export-cue-sheet")
async def export_cue_sheet(name: str):
    d = _project_dir(name)
    meta = _load_metadata(name)

    segs_path = d / "segments.json"
    trans_path = d / "translations.json"
    chars_path = d / "characters_final.json"

    if not segs_path.exists():
        raise HTTPException(status_code=400, detail="No segments — run Stage 1 first")
    if not trans_path.exists():
        raise HTTPException(status_code=400, detail="No translations — run Stage 1 first")

    segs_raw = json.loads(segs_path.read_text(encoding="utf-8-sig"))
    segments = segs_raw.get("merged", segs_raw) if isinstance(segs_raw, dict) else segs_raw
    translations = json.loads(trans_path.read_text(encoding="utf-8-sig"))
    characters = json.loads(chars_path.read_text(encoding="utf-8-sig")) if chars_path.exists() else {}

    from translate import write_excel
    write_excel(d, segments, translations, characters, meta["target_lang"])
    return {"status": "ok"}


# ── Routes: file access ────────────────────────────────────────────────────────

@app.get("/api/projects/{name}/files/{filename:path}")
async def get_file(name: str, filename: str, download: bool = False):
    d = _project_dir(name)
    path = d / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found")
    suffix = path.suffix.lower()
    # Video files served inline so the browser player works; ?download=1 forces named download
    if suffix in (".mp4", ".webm", ".ogg", ".mov") and not download:
        return FileResponse(path, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    stem = path.stem
    safe_project = name.replace(" ", "_")
    download_name = f"{safe_project}__{stem}{suffix}"
    return FileResponse(path, filename=download_name)


@app.get("/api/projects/{name}/characters")
async def get_characters(name: str):
    d = _project_dir(name)
    for fname in ("characters_final.json", "characters_llm.json"):
        data = _load_json(d / fname)
        if data:
            from collections import Counter
            counts = Counter(data.values())
            return {"characters": dict(counts), "map": data}
    return {"characters": {}, "map": {}}


@app.get("/api/projects/{name}/name-candidates")
async def get_name_candidates(name: str):
    """Extract likely character names from English source text (capitalized mid-sentence words)."""
    import re as _re
    from collections import Counter as _Counter
    d = _project_dir(name)
    raw = _load_json(d / "segments.json")
    if not raw:
        return {"names": []}
    segs = raw.get("merged", raw) if isinstance(raw, dict) else raw

    word_counts: _Counter = _Counter()
    for seg in segs:
        text = seg.get("text", "")
        # Find capitalized words that are NOT at the start of the sentence
        words = _re.findall(r"(?<=[.!?]\s{0,5}|\s)[A-Z][a-z]{1,}", text)
        # Also find consecutive capitalized words (full names like "Damon Blake")
        full_names = _re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b", text)
        for w in words:
            word_counts[w] += 1
        for fn in full_names:
            word_counts[fn] += 2  # weight full names higher

    # Return names appearing more than once, sorted by frequency
    candidates = [w for w, c in word_counts.most_common(100) if c > 1 and len(w) > 2]
    return {"names": candidates}


@app.get("/api/projects/{name}/dub-state")
async def get_dub_state(name: str):
    d = _project_dir(name)
    dub = _load_json(d / "dub_state.json")
    if dub is None:
        return {}
    return dub


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8501, reload=True,
                h11_max_incomplete_event_size=10 * 1024 * 1024 * 1024)
