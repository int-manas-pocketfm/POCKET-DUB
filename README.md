# DubStudio — PocketFM

A 4-stage video dubbing pipeline that converts English MP4s into dubbed videos in German, French, Italian, Spanish, or Portuguese.

## How it works

| Stage | What it does |
|---|---|
| **1. SCRIPT** | Extracts audio → transcribes with Whisper → detects characters → translates with Claude |
| **2. LOCALIZE** | Replaces character names and cultural references for the target language |
| **3. TTS** | Generates dubbed audio per segment using Microsoft neural voices (edge-tts) |
| **4. STITCH** | Assembles the dubbed audio + original video into a final MP4 |

## Stack

- **Backend** — Python 3.12, FastAPI, SSE streaming
- **Frontend** — Alpine.js single-page app
- **Transcription** — faster-whisper (local, free)
- **Translation** — Google Translate + Claude (Argus) rewrite
- **TTS** — edge-tts (Microsoft neural voices, free)
- **Video** — FFmpeg + NumPy audio stitching

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add your Argus API key
python server.py       # runs on localhost:8501
```

## Environment variables

```
ARGUS_API_KEY=...
ARGUS_BASE_URL=...
ARGUS_MODEL=claude-sonnet-4-6
```

## Requirements

- Python 3.12+
- FFmpeg on PATH
- Whisper model (~150 MB, auto-downloads on first run)
