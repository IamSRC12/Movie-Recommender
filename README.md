# NEXUS — Screen Media Recommendation Engine

A cyberpunk-themed, locally hosted recommendation engine for **Movies, Series, Anime, and Donghua**, powered by FastAPI + Scikit-learn content-based filtering with automated LLM-driven data updates.

---

## Quick Start

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Set Your OpenRouter API Key
Open `main.py` and replace the placeholder on line 18:
```python
OPENROUTER_API_KEY = "sk-or-xxxxxxxxxxxxxxxxxxxx"   # ← Your key here
```
Get a free key at [openrouter.ai](https://openrouter.ai) — the `meta-llama/llama-3-8b-instruct:free` model has no cost.

### 3. Start the Server
```powershell
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Open in Browser
```
http://localhost:8000
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        NEXUS SYSTEM                         │
│                                                             │
│  ┌──────────────┐    ┌──────────────────────────────────┐   │
│  │  index.html  │◄──►│         FastAPI (main.py)        │   │
│  │  styles.css  │    │                                  │   │
│  │  (Frontend)  │    │  ┌──────────┐  ┌──────────────┐  │   │
│  └──────────────┘    │  │  Pandas  │  │  APScheduler │  │   │
│                      │  │ + TF-IDF │  │  (6h update) │  │   │
│                      │  │ Cosine   │  │              │  │   │
│                      │  │ Sim.     │  │  OpenRouter  │  │   │
│                      │  └────┬─────┘  │  LLM API     │  │   │
│                      │       │        └──────┬───────┘  │   │
│                      └───────┼───────────────┼──────────┘   │
│                              │               │              │
│                         media_db.csv ◄───────┘              │
└─────────────────────────────────────────────────────────────┘
```

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve the frontend |
| `/api/media` | GET | All media entries |
| `/api/media/{category}` | GET | Filter by category (Anime/Movie/Series/Donghua) |
| `/api/hero` | GET | Highest-rated title for hero banner |
| `/recommend/{title}` | GET | Top-5 TF-IDF content-based recommendations |
| `/api/trigger-update` | POST | Manually trigger the LLM updater |

---

## ML Engine

- **Algorithm**: Content-Based Filtering
- **Vectorizer**: `TfidfVectorizer` on `Genre + Description` columns
  - Genre is weighted 2× by doubling in the corpus
- **Similarity**: Cosine Similarity matrix
- **Returns**: Top 5 matches by similarity score (excludes self)

---

## Background Updater

- Runs every **6 hours** via APScheduler
- Calls OpenRouter's `meta-llama/llama-3-8b-instruct:free` model
- Prompts the LLM for 3 trending titles in JSON format
- Validates and deduplicates before appending to `media_db.csv`
- Trigger manually from the footer button in the UI

---

## Project Structure

```
📁 Antigravity projects/
├── main.py          ← FastAPI backend + ML engine + updater
├── index.html       ← Frontend UI (single page)
├── styles.css       ← Cyberpunk design system
├── media_db.csv     ← Media database (auto-growing)
├── requirements.txt ← Python dependencies
└── README.md        ← This file
```

---

## Design Features

- **Dark Cyberpunk Theme** — Deep black + neon green/blue accents
- **CRT Scanline Effect** — Applied to card posters on hover
- **Glassmorphism Navbar** — Frosted glass top bar
- **Hero Banner** — Full-bleed poster with blurred background
- **Category Carousels** — Horizontal scrolling rows per category
- **Real-time Search** — Instant client-side filtering
- **Detail Modal** — Slides in with poster, description + live ML recommendations
- **Toast Notifications** — Contextual feedback messages
- **Responsive** — Adapts to mobile screens
