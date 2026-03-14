# Smart_navugation_system🧭
### *From Intention to Destination.*

> A Semantic Intent-Based, Context-Aware Smart Campus Navigation System that understands natural language — by text or voice — and guides you to the right place on campus.

---

## 📌 Overview

**IntentRoute** is a full-stack AI-powered campus navigation application. Users can type or speak a natural language query like *"I need a quiet place to study"* or *"Where can I get food?"*, and the system intelligently classifies the intent, filters relevant campus facilities, and returns the best recommendation with a Google Maps navigation link.

---

## ✨ Features

- 🔤 **Natural Language Input** — Type queries in plain English
- 🎙 **Voice Input** — Speak via browser microphone (Web Speech API)
- 📁 **Audio File Upload** — Upload WAV/MP3/WEBM recordings for server-side transcription
- 🤖 **Intent Classification** — TF-IDF + Logistic Regression ML pipeline
- 🏫 **Campus Facility Database** — 30 seeded facilities across 6 categories
- ⏰ **Context-Aware Filtering** — Filters by open/closed status and distance from user
- 🗺 **Google Maps Navigation** — Direct route link to recommended facility
- ⚡ **REST API** — FastAPI backend with interactive Swagger docs

---

## 🗂 Project Structure

```
smart-campus-navigation/
│
├── dataset/
│   └── intents.csv           # 180 labelled training queries (30 per intent)
│
├── model/
│   ├── train_model.py        # TF-IDF + Logistic Regression training script
│   └── intent_model.pkl      # Saved trained model (generated after training)
│
├── backend/
│   └── main.py               # FastAPI backend — text + audio endpoints
│
├── database/
│   ├── db_setup.sql          # Schema + 30 seeded campus facilities
│   └── campus.db             # SQLite database (auto-created on first run)
│
├── frontend/
│   └── index.html            # Single-page UI (Text / Mic / Upload modes)
│
├── utils/
│   └── preprocessing.py      # NLP pipeline (NLTK + built-in fallback)
│
└── requirements.txt
```

---

## 🧠 Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python, FastAPI, Uvicorn |
| **ML Model** | Scikit-learn — TF-IDF Vectorizer + Logistic Regression |
| **NLP** | NLTK (tokenization, stopwords, lemmatization) |
| **Database** | SQLite (via Python `sqlite3`) |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript |
| **Speech (Browser)** | Web Speech API |
| **Speech (Server)** | OpenAI Whisper / SpeechRecognition + Google STT |
| **Maps** | Google Maps Directions URL |

---

## 🎯 Supported Intents

| Intent | Example Queries |
|---|---|
| `STUDY` | *"I want a quiet place to study"*, *"Where is the library?"* |
| `FOOD` | *"Where can I eat?"*, *"I'm hungry"*, *"Find me a canteen"* |
| `MEDICAL` | *"I need a doctor"*, *"Where is the clinic?"*, *"I feel sick"* |
| `ADMIN` | *"Where is the registrar?"*, *"I need my transcript"* |
| `LAB` | *"Where is the computer lab?"*, *"I need to use MATLAB"* |
| `HOSTEL` | *"Where is my hostel?"*, *"I locked myself out of my room"* |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip
- A modern browser (Chrome / Edge recommended for microphone support)

### 1. Clone / Extract the Project

```bash
cd smart-campus-navigation
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> For server-side audio transcription, install one of:
> ```bash
> pip install openai-whisper       # Recommended — offline, highly accurate
> pip install SpeechRecognition    # Fallback — uses Google STT (needs internet)
> ```

### 3. Train the ML Model

```bash
python model/train_model.py
```

This will:
- Load `dataset/intents.csv`
- Preprocess text using NLTK
- Train a TF-IDF + Logistic Regression pipeline
- Save the model to `model/intent_model.pkl`
- Print accuracy and classification report

### 4. Start the Backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

API will be live at: `http://localhost:8000`  
Interactive docs at: `http://localhost:8000/docs`

### 5. Open the Frontend

Simply open `frontend/index.html` in your browser.

> No build step required — it's a single HTML file.

---

## 🔌 API Reference

### `GET /health`
Returns server status and available STT backend.

```json
{
  "status": "ok",
  "model_loaded": true,
  "stt_backend": "whisper",
  "whisper": true,
  "speech_recognition": false
}
```

---

### `POST /navigate` — Text Query

**Request:**
```json
{
  "query": "I need a quiet place to study",
  "user_lat": 3.1380,
  "user_lon": 101.6860
}
```

**Response:**
```json
{
  "query": "I need a quiet place to study",
  "input_mode": "text",
  "transcription": null,
  "intent": "STUDY",
  "confidence": 89.9,
  "recommended": {
    "name": "24-Hour Study Hall",
    "building": "Student Hub",
    "floor": "Ground Floor",
    "is_open": true,
    "opening_time": "00:00",
    "closing_time": "23:59",
    "distance_m": 22.2,
    "maps_link": "https://www.google.com/maps/dir/3.138,101.686/3.1382,101.686"
  },
  "alternatives": [ ... ],
  "message": "Found 5 STUDY facilities. Best: '24-Hour Study Hall' — open ✓, 22.2m away."
}
```

---

### `POST /navigate/audio` — Audio File Upload

**curl example:**
```bash
curl -X POST http://localhost:8000/navigate/audio \
  -F "audio=@my_query.wav" \
  -F "user_lat=3.138" \
  -F "user_lon=101.686"
```

**Supported formats:** WAV · MP3 · WEBM · OGG · FLAC · M4A

**Response:** Same as `/navigate` with additional fields:
```json
{
  "input_mode": "audio",
  "transcription": "where can I get food",
  "stt_backend": "whisper",
  ...
}
```

---

### `GET /facilities?category=FOOD`
Returns all campus facilities, optionally filtered by category.

---

## 🎙 Audio Input Modes

| Mode | Where | How |
|---|---|---|
| **Microphone** | Browser | Web Speech API — real-time transcription, no install |
| **File Upload** | Browser → Server | Sent to `/navigate/audio`, transcribed by Whisper or Google STT |

### STT Priority (Server-side)
```
1. OpenAI Whisper     → fully offline, most accurate
2. SpeechRecognition  → Google STT, requires internet
3. Browser Web Speech API → fallback, no server install needed
```

---

## 🗃 Database Schema

```sql
CREATE TABLE facilities (
  id            INTEGER PRIMARY KEY,
  name          TEXT,
  category      TEXT,      -- STUDY | FOOD | MEDICAL | ADMIN | LAB | HOSTEL
  description   TEXT,
  latitude      REAL,
  longitude     REAL,
  opening_time  TEXT,      -- HH:MM (24h)
  closing_time  TEXT,
  building      TEXT,
  floor         TEXT
);
```

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Training samples | 180 (30 × 6 intents) |
| Test accuracy | ~69% (small dataset) |
| Feature extraction | TF-IDF (unigrams + bigrams, 5000 features) |
| Classifier | Logistic Regression (C=5.0, lbfgs solver) |

> Accuracy improves significantly with more training data. Add more examples to `dataset/intents.csv` and retrain.

---

## 📈 Extending the Project

### Add more training data
Edit `dataset/intents.csv` and add more rows, then retrain:
```bash
python model/train_model.py
```

### Add a new intent (e.g. SPORTS)
1. Add labelled rows to `intents.csv` with `intent=SPORTS`
2. Add facility rows to `database/db_setup.sql` with `category='SPORTS'`
3. Retrain the model
4. Add the intent chip in `frontend/index.html`

### Switch to MySQL
Replace the `sqlite3` calls in `backend/main.py` with `mysql-connector-python` or `SQLAlchemy`.

---

## 📄 License

This project is open for academic and educational use.

---

*IntentRoute — From Intention to Destination.*
