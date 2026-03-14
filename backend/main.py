"""
backend/main.py
----------------
FastAPI backend for the Smart Campus Navigation System.
Supports BOTH text and audio queries.

Endpoints:
  POST /navigate          — Text query → intent → recommend facility
  POST /navigate/audio    — Audio file upload → transcribe → intent → recommend
  GET  /facilities        — List all facilities (optional ?category= filter)
  GET  /health            — Health check + STT capability info

Audio transcription priority:
  1. OpenAI Whisper (best accuracy, fully offline)
  2. SpeechRecognition + Google STT (requires internet)
  3. Graceful 501 error if neither available (browser Web Speech API handles it)
"""

import os
import sys
import math
import sqlite3
import pickle
import tempfile
import shutil
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Project root on path ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, '..')
sys.path.append(ROOT_DIR)

from utils.preprocessing import preprocess_text

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(ROOT_DIR, 'model', 'intent_model.pkl')
DB_PATH    = os.path.join(ROOT_DIR, 'database', 'campus.db')

# ── Load ML model ─────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        f"Model not found at {MODEL_PATH}. Run: python model/train_model.py"
    )
with open(MODEL_PATH, 'rb') as f:
    MODEL = pickle.load(f)

# ── Optional STT libraries ────────────────────────────────────────────────────
WHISPER_AVAILABLE = False
SR_AVAILABLE      = False
_whisper_model    = None
_recognizer       = None

try:
    import whisper as _whisper_lib
    _whisper_model    = _whisper_lib.load_model("base")
    WHISPER_AVAILABLE = True
    print("[STT] ✓ OpenAI Whisper loaded (base model)")
except ImportError:
    pass

if not WHISPER_AVAILABLE:
    try:
        import speech_recognition as _sr_lib
        _recognizer  = _sr_lib.Recognizer()
        SR_AVAILABLE = True
        print("[STT] ✓ SpeechRecognition loaded (Google STT backend)")
    except ImportError:
        print("[STT] ✗ No server STT — browser Web Speech API will handle transcription")

STT_BACKEND = (
    "whisper"           if WHISPER_AVAILABLE else
    "google-speech-api" if SR_AVAILABLE      else
    "browser-only"
)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Smart Campus Navigation API — Multimodal",
    description="Text + Audio intent-based campus facility recommender",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic schemas ──────────────────────────────────────────────────────────
class NavigationRequest(BaseModel):
    query:    str             = Field(..., min_length=2, example="I want a quiet place to study")
    user_lat: Optional[float] = Field(None, example=3.1380)
    user_lon: Optional[float] = Field(None, example=101.6860)

class FacilityOut(BaseModel):
    id:           int
    name:         str
    category:     str
    description:  str
    latitude:     float
    longitude:    float
    opening_time: str
    closing_time: str
    building:     str
    floor:        str
    is_open:      bool
    distance_m:   Optional[float]
    maps_link:    Optional[str]

class NavigationResponse(BaseModel):
    query:         str
    input_mode:    str                   # "text" | "audio"
    transcription: Optional[str]         # filled only for audio input
    intent:        str
    confidence:    float
    recommended:   Optional[FacilityOut]
    alternatives:  list[FacilityOut]
    message:       str
    stt_backend:   Optional[str]         # which STT engine was used

# ── DB helpers ────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    with open(os.path.join(ROOT_DIR, 'database', 'db_setup.sql')) as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()

if not os.path.exists(DB_PATH):
    init_db()

# ── Geo & time helpers ────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin(math.radians(lat2 - lat1) / 2) ** 2
         + math.cos(p1) * math.cos(p2)
         * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return round(R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)), 1)

def is_open(open_t: str, close_t: str) -> bool:
    if open_t == '00:00' and close_t == '23:59':
        return True
    return open_t <= datetime.now().strftime('%H:%M') <= close_t

def enrich(row, ulat=None, ulon=None) -> dict:
    d = dict(row)
    d['is_open'] = is_open(d['opening_time'], d['closing_time'])
    if ulat is not None:
        d['distance_m'] = haversine(ulat, ulon, d['latitude'], d['longitude'])
        d['maps_link']  = f"https://www.google.com/maps/dir/{ulat},{ulon}/{d['latitude']},{d['longitude']}"
    else:
        d['distance_m'] = None
        d['maps_link']  = f"https://www.google.com/maps/search/?api=1&query={d['latitude']},{d['longitude']}"
    return d

# ── Intent classification ─────────────────────────────────────────────────────
def classify(query: str) -> dict:
    clean  = preprocess_text(query)
    intent = MODEL.predict([clean])[0]
    probas = MODEL.predict_proba([clean])[0]
    return {"intent": intent, "confidence": round(float(max(probas)) * 100, 1)}

# ── Recommendation engine ─────────────────────────────────────────────────────
def recommend(intent: str, ulat=None, ulon=None):
    conn  = get_db()
    rows  = conn.execute("SELECT * FROM facilities WHERE category=?", (intent,)).fetchall()
    conn.close()
    if not rows:
        return None, []
    items = [enrich(r, ulat, ulon) for r in rows]
    # Sort: open first → then by distance
    items.sort(key=lambda f: (0 if f['is_open'] else 1, f['distance_m'] or 999_999))
    return items[0], items[1:]

# ── Audio transcription ───────────────────────────────────────────────────────
def transcribe_audio(file_path: str) -> str:
    """
    Transcribe audio → text.
    Priority: Whisper > SpeechRecognition/Google > HTTP 501
    """
    if WHISPER_AVAILABLE:
        result = _whisper_model.transcribe(file_path)
        text   = result.get("text", "").strip()
        if not text:
            raise HTTPException(422, "Audio transcription returned empty text.")
        return text

    if SR_AVAILABLE:
        import speech_recognition as _sr_lib
        with _sr_lib.AudioFile(file_path) as src:
            audio = _recognizer.record(src)
        try:
            return _recognizer.recognize_google(audio)
        except _sr_lib.UnknownValueError:
            raise HTTPException(422, "Could not understand audio. Please speak clearly.")
        except _sr_lib.RequestError as e:
            raise HTTPException(503, f"Google STT service error: {e}")

    raise HTTPException(
        501,
        "No server-side STT installed. "
        "pip install openai-whisper  OR  pip install SpeechRecognition. "
        "Alternatively use the browser microphone (Web Speech API)."
    )

# ── Shared core logic ─────────────────────────────────────────────────────────
def _navigate(query, input_mode, transcription, ulat, ulon) -> NavigationResponse:
    cls    = classify(query)
    intent = cls['intent']
    conf   = cls['confidence']
    best, alts = recommend(intent, ulat, ulon)

    if best is None:
        return NavigationResponse(
            query=query, input_mode=input_mode, transcription=transcription,
            intent=intent, confidence=conf, recommended=None, alternatives=[],
            message=f"No facilities found for intent: {intent}",
            stt_backend=STT_BACKEND if input_mode == "audio" else None
        )

    dist_str = f", {best['distance_m']}m away" if best['distance_m'] else ""
    status   = "open ✓" if best['is_open'] else "closed ✗"
    msg      = (f"Found {len(alts)+1} {intent} facilities. "
                f"Best: '{best['name']}' — {status}{dist_str}.")

    return NavigationResponse(
        query=query, input_mode=input_mode, transcription=transcription,
        intent=intent, confidence=conf,
        recommended=FacilityOut(**best),
        alternatives=[FacilityOut(**f) for f in alts],
        message=msg,
        stt_backend=STT_BACKEND if input_mode == "audio" else None
    )

# ── API routes ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":             "ok",
        "model_loaded":       True,
        "stt_backend":        STT_BACKEND,
        "whisper":            WHISPER_AVAILABLE,
        "speech_recognition": SR_AVAILABLE,
    }

@app.get("/facilities")
def list_facilities(category: Optional[str] = None):
    conn = get_db()
    rows = (
        conn.execute("SELECT * FROM facilities WHERE category=?",
                     (category.upper(),)).fetchall()
        if category else
        conn.execute("SELECT * FROM facilities").fetchall()
    )
    conn.close()
    return [enrich(r) for r in rows]

@app.post("/navigate", response_model=NavigationResponse)
def navigate_text(req: NavigationRequest):
    """
    TEXT query → intent classification → facility recommendation.

    Request body:
      { "query": "I need a quiet place to study",
        "user_lat": 3.138, "user_lon": 101.686 }
    """
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty.")
    return _navigate(req.query, "text", None, req.user_lat, req.user_lon)


@app.post("/navigate/audio", response_model=NavigationResponse)
async def navigate_audio(
    audio:    UploadFile       = File(..., description="WAV / MP3 / WEBM / OGG / M4A"),
    user_lat: Optional[float]  = Form(None),
    user_lon: Optional[float]  = Form(None),
):
    """
    AUDIO query → transcribe → intent → facility recommendation.

    Accepts: WAV, MP3, WEBM, OGG, FLAC, M4A  (max ~10 MB recommended)

    curl example:
      curl -X POST http://localhost:8000/navigate/audio \\
        -F "audio=@query.wav" \\
        -F "user_lat=3.138" \\
        -F "user_lon=101.686"
    """
    allowed = {'.wav', '.mp3', '.webm', '.ogg', '.flac', '.m4a'}
    ext     = os.path.splitext(audio.filename or '')[1].lower()
    if ext not in allowed:
        raise HTTPException(415, f"Unsupported format '{ext}'. Use: {', '.join(allowed)}")

    # Save upload to temp file → transcribe → delete
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext or '.wav') as tmp:
        shutil.copyfileobj(audio.file, tmp)
        tmp_path = tmp.name
    try:
        transcription = transcribe_audio(tmp_path)
    finally:
        os.unlink(tmp_path)

    return _navigate(transcription, "audio", transcription, user_lat, user_lon)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
