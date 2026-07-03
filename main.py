import os
import json
import logging
import hashlib
import pandas as pd
import requests
import urllib.parse
from fastapi import FastAPI, HTTPException, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from contextlib import asynccontextmanager
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
TMDB_API_KEY       = os.environ.get("TMDB_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL   = "google/gemma-2-9b-it:free"
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
DB_PATH            = "media_db.csv"
USERS_PATH         = "users.json"
SECRET_KEY         = "cyberpunk_secret_key_recommend_engine"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# DATA & ML ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def load_df() -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        logger.info(f"Database file {DB_PATH} not found. Running fallback initialization...")
        df_init = pd.DataFrame([])
        df_init.to_csv(DB_PATH, index=False, encoding="utf-8")

    df = pd.read_csv(DB_PATH, encoding="utf-8")
    df.columns = df.columns.str.strip()
    df["Title"]        = df["Title"].str.strip()
    df["Category"]     = df["Category"].str.strip()
    df["Genre"]        = df["Genre"].fillna("").str.strip()
    df["Description"]  = df["Description"].fillna("").str.strip()
    df["Rating"]       = pd.to_numeric(df["Rating"], errors="coerce").fillna(0.0)
    df["Cover_URL"]    = df["Cover_URL"].fillna("").str.strip()
    df["Backdrop_URL"] = df["Backdrop_URL"].fillna("").str.strip()
    df["TMDB_ID"]      = pd.to_numeric(df["TMDB_ID"], errors="coerce").fillna(0).astype(int)
    
    df = df.drop_duplicates(subset=["Title"]).reset_index(drop=True)
    return df


def build_model(df: pd.DataFrame):
    if df.empty or len(df) < 2:
        return None
    corpus = (df["Genre"] + " " + df["Genre"] + " " + df["Description"]).tolist()
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    sim_matrix   = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return sim_matrix


def get_recommendations(title: str, df: pd.DataFrame, sim_matrix: np.ndarray, 
                        likes: list[str] = None, dislikes: list[str] = None, 
                        watched: list[str] = None, top_n: int = 5):
    if df.empty or sim_matrix is None:
        return []
        
    if likes is None: likes = []
    if dislikes is None: dislikes = []
    if watched is None: watched = []
    
    title_lower  = title.lower().strip()
    titles_lower = df["Title"].str.lower().str.strip()
    matches      = titles_lower[titles_lower == title_lower]
    if matches.empty:
        matches = titles_lower[titles_lower.str.contains(title_lower, na=False, regex=False)]
    if matches.empty:
        return []

    idx = matches.index[0]
    
    likes_indices = []
    for l_title in likes:
        l_matches = titles_lower[titles_lower == l_title.lower().strip()]
        if not l_matches.empty:
            likes_indices.append(l_matches.index[0])
            
    dislikes_indices = []
    for d_title in dislikes:
        d_matches = titles_lower[titles_lower == d_title.lower().strip()]
        if not d_matches.empty:
            dislikes_indices.append(d_matches.index[0])

    watched_set = {t.lower().strip() for t in watched}
    
    candidate_scores = []
    for i in range(len(df)):
        if i == idx:
            continue
            
        candidate_title = df.iloc[i]["Title"].lower().strip()
        if candidate_title in watched_set:
            continue
            
        base_score = sim_matrix[idx][i]
        
        like_bonus = 0.0
        for l_idx in likes_indices:
            like_bonus += sim_matrix[l_idx][i]
        if likes_indices:
            like_bonus /= len(likes_indices)
            
        dislike_penalty = 0.0
        for d_idx in dislikes_indices:
            dislike_penalty += sim_matrix[d_idx][i]
        if dislikes_indices:
            dislike_penalty /= len(dislikes_indices)
            
        adjusted_score = base_score + 0.3 * like_bonus - 0.3 * dislike_penalty
        candidate_scores.append((i, adjusted_score))
        
    candidate_scores = sorted(candidate_scores, key=lambda x: x[1], reverse=True)
    scores = candidate_scores[:top_n]

    results = []
    for i, score in scores:
        row = df.iloc[i]
        results.append({
            "title":       row["Title"],
            "category":    row["Category"],
            "genre":       row["Genre"],
            "description": row["Description"],
            "rating":      float(row["Rating"]),
            "cover_url":   row["Cover_URL"],
            "score":       round(float(score), 4),
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# ML MODEL CACHE
# ─────────────────────────────────────────────────────────────────────────────

class MLModelCache:
    def __init__(self):
        self.df = None
        self.sim_matrix = None

    def reload(self):
        try:
            logger.info("Reloading and rebuilding ML model cache...")
            df = load_df()
            sim_matrix = build_model(df)
            self.df = df
            self.sim_matrix = sim_matrix
            logger.info(f"ML Model Cache successfully updated. Loaded {len(df)} records.")
        except Exception as e:
            logger.error(f"Error rebuilding ML model cache: {e}")

ml_cache = MLModelCache()


# ─────────────────────────────────────────────────────────────────────────────
# USER DATABASE & AUTHENTICATION
# ─────────────────────────────────────────────────────────────────────────────

class UserAuth(BaseModel):
    username: str
    password: str

class InteractionRequest(BaseModel):
    title: str
    type: str  # "like", "dislike", "watched", "unwatched", "none"


def load_users() -> dict:
    if not os.path.exists(USERS_PATH):
        return {}
    try:
        with open(USERS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            for u in data:
                if "likes" not in data[u]: data[u]["likes"] = []
                if "dislikes" not in data[u]: data[u]["dislikes"] = []
                if "watched" not in data[u]: data[u]["watched"] = []
                if "watchlist_folders" not in data[u]: data[u]["watchlist_folders"] = {}
                if "ratings_reviews" not in data[u]: data[u]["ratings_reviews"] = {}
                if "custom_collections" not in data[u]: data[u]["custom_collections"] = {}
            return data
    except Exception as e:
        logger.error(f"Error loading users: {e}")
        return {}


def save_users(users: dict):
    try:
        with open(USERS_PATH, "w", encoding="utf-8") as f:
            json.dump(users, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving users: {e}")


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def generate_token(username: str) -> str:
    signature = hashlib.sha256(f"{username}:{SECRET_KEY}".encode("utf-8")).hexdigest()
    return f"{username}:{signature}"


def verify_token(token: str) -> str:
    if not token or ":" not in token:
        return None
    try:
        username, signature = token.split(":", 1)
        expected_signature = hashlib.sha256(f"{username}:{SECRET_KEY}".encode("utf-8")).hexdigest()
        if signature == expected_signature:
            return username
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# TMDB API INTEGRATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def fetch_tmdb_details(tmdb_id: int, category: str) -> dict:
    search_type = "movie" if category.lower() == "movie" else "tv"
    url = f"https://api.tmdb.org/3/{search_type}/{tmdb_id}?api_key={TMDB_API_KEY}&append_to_response=credits,videos,reviews"
    try:
        resp = requests.get(url, timeout=4.0)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"Error fetching details from TMDB: {e}")
        return {}


def fetch_tmdb_recommendations(tmdb_id: int, category: str) -> list:
    search_type = "movie" if category.lower() == "movie" else "tv"
    url = f"https://api.tmdb.org/3/{search_type}/{tmdb_id}/recommendations?api_key={TMDB_API_KEY}"
    try:
        resp = requests.get(url, timeout=4.0)
        resp.raise_for_status()
        results = resp.json().get("results", [])[:5]
        recommendations = []
        for r in results:
            title = r.get("name") if search_type == "tv" else r.get("title")
            recommendations.append({
                "title": title,
                "category": "Series" if search_type == "tv" else "Movie",
                "genre": "",
                "description": r.get("overview", ""),
                "rating": float(r.get("vote_average", 0.0)),
                "cover_url": f"https://image.tmdb.org/t/p/w500{r.get('poster_path')}" if r.get('poster_path') else "",
                "score": 0.8
            })
        return recommendations
    except Exception as e:
        logger.error(f"Error fetching recommendations from TMDB: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# AUTOMATED UPDATER (APScheduler + OpenRouter)
# ─────────────────────────────────────────────────────────────────────────────

UPDATER_PROMPT = """You are a media database curator. Your ONLY task is to return a JSON array
of exactly 3 trending or newly popular media titles mixing Anime, Movies, Series, and Donghua.
Each object MUST have these exact keys: Title, Category, Genre, Description, Rating, Cover_URL.

Rules:
- Category must be one of: Movie, Series
- Genre: 2-4 genre tags separated by spaces (including "Anime" or "Donghua" if applicable, e.g. "Action Thriller Sci-Fi Anime")
- Description: 1-2 sentences, engaging and informative
- Rating: a float between 1.0 and 10.0
- Cover_URL: use a real TMDB poster URL if known, otherwise use an empty string ""
- Return ONLY the raw JSON array — no markdown, no explanation, no code fences."""


def fetch_new_titles() -> list[dict]:
    if not OPENROUTER_API_KEY:
        logger.warning("OpenRouter API key not set — skipping updater fetch.")
        return []

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "http://localhost:8000",
        "X-Title":       "MediaRecommendEngine",
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": UPDATER_PROMPT}],
        "temperature": 0.8,
        "max_tokens":  800,
    }
    try:
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        
        start_idx = content.find("[")
        end_idx = content.rfind("]")
        if start_idx != -1 and end_idx != -1:
            content = content[start_idx:end_idx+1]
            
        data = json.loads(content)
        if isinstance(data, list):
            return data
    except Exception as e:
        logger.error(f"OpenRouter fetch error: {e}")
    return []


def append_to_db(new_entries: list[dict]):
    if not new_entries:
        return
    df_existing = load_df()
    existing_titles = set(df_existing["Title"].str.lower().str.strip())
    rows = []
    for entry in new_entries:
        title = str(entry.get("Title", "")).strip()
        if not title or title.lower() in existing_titles:
            logger.info(f"Skipping duplicate or empty title: '{title}'")
            continue
            
        category = str(entry.get("Category", "Movie")).strip()
        if category.lower() not in ["movie", "series"]:
            category = "Series" if category.lower() in ["anime", "donghua"] else "Movie"
            
        genre = str(entry.get("Genre", "")).strip()
        
        search_type = "tv" if category.lower() == "series" else "movie"
        search_url = f"https://api.tmdb.org/3/search/{search_type}?api_key={TMDB_API_KEY}&query={urllib.parse.quote(title)}"
        tmdb_id = 0
        cover_url = str(entry.get("Cover_URL", "")).strip()
        bd_url = ""
        
        try:
            r = requests.get(search_url, timeout=10)
            results = r.json().get("results", [])
            if results:
                best = results[0]
                tmdb_id = best["id"]
                cover_url = f"https://image.tmdb.org/t/p/w500{best.get('poster_path')}" if best.get("poster_path") else cover_url
                bd_url = f"https://image.tmdb.org/t/p/original{best.get('backdrop_path')}" if best.get("backdrop_path") else ""
                title = best.get("name") if search_type == "tv" else best.get("title")
        except Exception as e:
            logger.error(f"Error enriching appended title '{title}': {e}")
            
        rows.append({
            "Title":       title,
            "Category":    category,
            "Genre":       genre,
            "Description": str(entry.get("Description", "")).strip(),
            "Rating":      float(entry.get("Rating", 7.0)),
            "Cover_URL":   cover_url,
            "Backdrop_URL": bd_url,
            "TMDB_ID":     tmdb_id
        })
    if rows:
        df_new = pd.DataFrame(rows)
        df_new.to_csv(DB_PATH, mode="a", header=False, index=False, encoding="utf-8")
        logger.info(f"Appended {len(rows)} new entries to {DB_PATH}")
        ml_cache.reload()
    else:
        logger.info("No new unique entries to append.")


def updater_job():
    logger.info("Running scheduled media updater...")
    new_entries = fetch_new_titles()
    append_to_db(new_entries)
    logger.info("Updater job complete.")


def enrich_db_with_tmdb_popular():
    try:
        logger.info("Enriching local database with TMDB popular movies and series...")
        df_existing = load_df()
        existing_ids = set(df_existing["TMDB_ID"].tolist())
        existing_titles = set(df_existing["Title"].str.lower().str.strip())
        
        new_rows = []
        
        # 1. Popular Movies (first 10 pages = 200 items)
        for page in range(1, 11):
            url = f"https://api.tmdb.org/3/movie/popular?api_key={TMDB_API_KEY}&page={page}"
            try:
                r = requests.get(url, timeout=10)
                if r.status_code == 200:
                    results = r.json().get("results", [])
                    for m in results:
                        t_id = m.get("id")
                        title = m.get("title", "").strip()
                        if not title or t_id in existing_ids or title.lower() in existing_titles:
                            continue
                            
                        # Get details to have genres
                        details_url = f"https://api.tmdb.org/3/movie/{t_id}?api_key={TMDB_API_KEY}"
                        details = requests.get(details_url, timeout=5).json()
                        genres = " ".join([g["name"] for g in details.get("genres", [])])
                        
                        # Add Anime/Donghua genre tags for movies
                        is_anime = False
                        is_donghua = False
                        if "Animation" in genres:
                            origin_country_codes = [c.get("iso_3166_1", "") for c in details.get("production_countries", [])]
                            original_lang = details.get("original_language", "")
                            if "JP" in origin_country_codes or original_lang == "ja":
                                is_anime = True
                            elif "CN" in origin_country_codes or original_lang == "zh":
                                is_donghua = True
                        if is_anime and "Anime" not in genres:
                            genres += " Anime"
                        elif is_donghua and "Donghua" not in genres:
                            genres += " Donghua"
                        
                        cover_url = f"https://image.tmdb.org/t/p/w500{m.get('poster_path')}" if m.get('poster_path') else ""
                        bd_url = f"https://image.tmdb.org/t/p/original{m.get('backdrop_path')}" if m.get('backdrop_path') else ""
                        
                        new_rows.append({
                            "Title": title,
                            "Category": "Movie",
                            "Genre": genres,
                            "Description": m.get("overview", ""),
                            "Rating": float(m.get("vote_average", 0.0)),
                            "Cover_URL": cover_url,
                            "Backdrop_URL": bd_url,
                            "TMDB_ID": t_id
                        })
                        existing_ids.add(t_id)
                        existing_titles.add(title.lower())
            except Exception as e:
                logger.error(f"Error enriching popular movies page {page}: {e}")
                
        # 2. Popular TV Shows (first 10 pages = 200 items)
        for page in range(1, 11):
            url = f"https://api.tmdb.org/3/tv/popular?api_key={TMDB_API_KEY}&page={page}"
            try:
                r = requests.get(url, timeout=10)
                if r.status_code == 200:
                    results = r.json().get("results", [])
                    for m in results:
                        t_id = m.get("id")
                        title = m.get("name", "").strip()
                        if not title or t_id in existing_ids or title.lower() in existing_titles:
                            continue
                            
                        # Get details to have genres
                        details_url = f"https://api.tmdb.org/3/tv/{t_id}?api_key={TMDB_API_KEY}"
                        details = requests.get(details_url, timeout=5).json()
                        genres = " ".join([g["name"] for g in details.get("genres", [])])
                        
                        # Add Anime and Donghua genre tag if it's animation and has Japanese/Chinese origin country or language
                        is_anime = False
                        is_donghua = False
                        if "Animation" in genres:
                            origin_country = m.get("origin_country", [])
                            original_lang = m.get("original_language", "")
                            if "JP" in origin_country or original_lang == "ja":
                                is_anime = True
                            elif "CN" in origin_country or original_lang == "zh":
                                is_donghua = True
                        if is_anime and "Anime" not in genres:
                            genres += " Anime"
                        elif is_donghua and "Donghua" not in genres:
                            genres += " Donghua"
                            
                        cover_url = f"https://image.tmdb.org/t/p/w500{m.get('poster_path')}" if m.get('poster_path') else ""
                        bd_url = f"https://image.tmdb.org/t/p/original{m.get('backdrop_path')}" if m.get('backdrop_path') else ""
                        
                        new_rows.append({
                            "Title": title,
                            "Category": "Series",
                            "Genre": genres,
                            "Description": m.get("overview", ""),
                            "Rating": float(m.get("vote_average", 0.0)),
                            "Cover_URL": cover_url,
                            "Backdrop_URL": bd_url,
                            "TMDB_ID": t_id
                        })
                        existing_ids.add(t_id)
                        existing_titles.add(title.lower())
            except Exception as e:
                logger.error(f"Error enriching popular tv page {page}: {e}")
                
        if new_rows:
            df_new = pd.DataFrame(new_rows)
            df_new.to_csv(DB_PATH, mode="a", header=False, index=False, encoding="utf-8")
            logger.info(f"Successfully enriched database with {len(new_rows)} new items from TMDB.")
        else:
            logger.info("No new items to add to database from TMDB popular endpoints.")
            
    except Exception as e:
        logger.error(f"Enrichment task failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

scheduler = BackgroundScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Reload immediately to cache existing local DB so endpoints work instantly
    ml_cache.reload()
    
    # Run the heavy network DB popular enrichment in a background daemon thread
    import threading
    threading.Thread(target=enrich_db_with_tmdb_popular, daemon=True).start()
    
    scheduler.add_job(
        updater_job,
        trigger=IntervalTrigger(hours=6),
        id="media_updater",
        replace_existing=True,
        misfire_grace_time=300,
    )
    scheduler.start()
    logger.info("APScheduler started — updater runs every 6 hours.")
    yield
    scheduler.shutdown()
    logger.info("APScheduler shut down.")


app = FastAPI(title="Media Recommendation Engine", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="."), name="static")


@app.get("/")
async def serve_index():
    return FileResponse("index.html")


@app.get("/api/media")
async def get_all_media():
    try:
        if ml_cache.df is None:
            ml_cache.reload()
        records = ml_cache.df.to_dict(orient="records")
        return JSONResponse(content=records)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/media/{category}")
async def get_media_by_category(category: str):
    try:
        if ml_cache.df is None:
            ml_cache.reload()
        df = ml_cache.df
        cat_lower = category.lower().strip()
        
        if cat_lower == "anime":
            cat = df[df["Genre"].str.lower().str.contains("anime", na=False)]
        elif cat_lower == "donghua":
            cat = df[df["Genre"].str.lower().str.contains("donghua", na=False)]
        elif cat_lower == "animated":
            cat = df[df["Genre"].str.lower().str.contains("animation", na=False) & 
                     ~df["Genre"].str.lower().str.contains("anime|donghua", na=False, regex=True)]
        elif cat_lower == "movie":
            cat = df[(df["Category"].str.lower() == "movie") & 
                     ~df["Genre"].str.lower().str.contains("anime|donghua|animation", na=False, regex=True)]
        elif cat_lower == "series":
            cat = df[(df["Category"].str.lower() == "series") & 
                     ~df["Genre"].str.lower().str.contains("anime|donghua|animation", na=False, regex=True)]
        else:
            cat = df[df["Category"].str.lower() == cat_lower]
            
        if cat.empty:
            raise HTTPException(status_code=404, detail=f"No media found for category: {category}")
        return JSONResponse(content=cat.to_dict(orient="records"))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_local_discover(category: str, genre: str = None, platform: str = None, page: int = 1, page_size: int = 20) -> tuple[list[dict], int]:
    if ml_cache.df is None:
        ml_cache.reload()
    df = ml_cache.df
    if df.empty:
        return [], 1
        
    cat_lower = category.lower().strip() if category else ""
    page = max(1, page)
    
    if cat_lower == "anime":
        filtered_df = df[df["Genre"].str.lower().str.contains("anime", na=False)]
    elif cat_lower == "donghua":
        filtered_df = df[df["Genre"].str.lower().str.contains("donghua", na=False)]
    elif cat_lower == "animated":
        filtered_df = df[df["Genre"].str.lower().str.contains("animation", na=False) & 
                         ~df["Genre"].str.lower().str.contains("anime|donghua", na=False, regex=True)]
    elif cat_lower == "movie":
        filtered_df = df[(df["Category"].str.lower() == "movie") & 
                         ~df["Genre"].str.lower().str.contains("anime|donghua|animation", na=False, regex=True)]
    elif cat_lower == "series":
        filtered_df = df[(df["Category"].str.lower() == "series") & 
                         ~df["Genre"].str.lower().str.contains("anime|donghua|animation", na=False, regex=True)]
    else:
        filtered_df = df
        
    if genre:
        g_lower = genre.lower().strip()
        filtered_df = filtered_df[filtered_df["Genre"].str.lower().str.contains(g_lower, na=False)]
        
    if platform:
        p_lower = platform.lower().strip()
        filtered_df = filtered_df[filtered_df["Description"].str.lower().str.contains(p_lower, na=False) |
                                   filtered_df["Genre"].str.lower().str.contains(p_lower, na=False)]
        
    # Sort by Rating descending
    filtered_df = filtered_df.sort_values("Rating", ascending=False)
    
    total_items = len(filtered_df)
    total_pages = max(1, (total_items + page_size - 1) // page_size)
    
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    sliced_df = filtered_df.iloc[start_idx:end_idx]
    
    results = []
    for _, row in sliced_df.iterrows():
        results.append({
            "Title": row["Title"],
            "Category": row["Category"],
            "Genre": row["Genre"],
            "Description": row["Description"],
            "Rating": float(row["Rating"]),
            "Cover_URL": row["Cover_URL"],
            "Backdrop_URL": row.get("Backdrop_URL", ""),
            "TMDB_ID": int(row.get("TMDB_ID", 0)),
            "Source": "local"
        })
    return results, total_pages


# Caching systems and Fallbacks
import time
trending_cache = None
trending_cache_time = 0
schedule_cache = {}
discover_cache = {}


@app.get("/api/discover")
def get_discover_media(
    category: str = None, 
    media_type: str = None,
    page: int = 1, 
    genre: str = None, 
    platform: str = None,
    watch_region: str = "IN"
):
    try:
        if not category and media_type:
            category = media_type
            
        cat_lower = category.lower().strip() if category else ""
        plat_lower = platform.lower().strip() if platform else None
        
        # 1. Check Memory Cache First
        cache_key = f"{cat_lower}_{genre}_{platform}_{watch_region}_{page}"
        if cache_key in discover_cache:
            cache_time, cache_data = discover_cache[cache_key]
            if time.time() - cache_time < 3600:
                logger.info(f"Serving discover results from cache for key: {cache_key}")
                return JSONResponse(content=cache_data)
        
        # If TMDB_API_KEY is not configured, fallback to local DB instantly
        if not TMDB_API_KEY:
            logger.info("TMDB_API_KEY is empty. Using local fallback for discover.")
            local_res, total_pages = get_local_discover(category, genre, platform, page)
            response_data = {
                "page": page,
                "total_pages": total_pages,
                "results": local_res
            }
            discover_cache[cache_key] = (time.time(), response_data)
            return JSONResponse(content=response_data)

        
        # 2. Watch Provider mapping
        provider_id = None
        if plat_lower:
            providers_map = {
                "netflix": 8,
                "hotstar": 122,
                "disney+ hotstar": 122,
                "disney": 122 if watch_region == "IN" else 337,
                "prime": 9,
                "prime video": 9,
                "jiocinema": 220,
                "zee5": 232,
                "apple": 350,
                "apple tv": 350,
                "apple tv+": 350,
                "crunchyroll": 283,
                "hbo": 1899, # Max
                "hbo max": 1899,
                "max": 1899
            }
            # Normalize platform name
            for k, v in providers_map.items():
                if k in plat_lower:
                    provider_id = v
                    break
            if not provider_id:
                # Default fallback
                provider_id = 8
            
            # For HBO Max, since it is not in India, default region to US if IN was passed
            if provider_id == 1899 and watch_region == "IN":
                watch_region = "US"

        # 3. Genre mapping for TMDB
        genre_map_movie = {
            "action": 28, "adventure": 12, "animation": 16, "comedy": 35, 
            "crime": 80, "documentary": 99, "drama": 18, "family": 10751, 
            "fantasy": 14, "history": 36, "horror": 27, "music": 10402, 
            "mystery": 9648, "romance": 10749, "sci-fi": 878, "science fiction": 878, 
            "thriller": 53, "war": 10752, "western": 37
        }
        
        genre_map_tv = {
            "action": 10759, "adventure": 10759, "animation": 16, "comedy": 35, 
            "crime": 80, "documentary": 99, "drama": 18, "family": 10751, 
            "fantasy": 10765, "mystery": 9648, "reality": 10764, "sci-fi": 10765, 
            "science fiction": 10765, "soap": 10766, "talk": 10767, "war": 10768, "western": 37
        }
        
        # Determine search types to query
        search_types = []
        if plat_lower:
            if cat_lower in ["movie"]:
                search_types = ["movie"]
            elif cat_lower in ["series", "tv"]:
                search_types = ["tv"]
            else:
                search_types = ["movie", "tv"]
        else:
            if cat_lower == "anime" or cat_lower == "donghua":
                # User wants both movies and TV shows for Anime & Donghua
                search_types = ["movie", "tv"]
            elif cat_lower == "animated":
                search_types = ["movie", "tv"]
            elif cat_lower == "series" or cat_lower == "tv":
                search_types = ["tv"]
            else:
                search_types = ["movie"]

        raw_results = []
        total_pages = 1
        
        # 4. Fetch from TMDB
        for s_type in search_types:
            extra_params = "&primary_release_date.gte=1990-01-01&primary_release_date.lte=2026-12-31" if s_type == "movie" else "&first_air_date.gte=1990-01-01&first_air_date.lte=2026-12-31"
            
            # Apply platform watch provider filters
            if provider_id:
                extra_params += f"&with_watch_providers={provider_id}&watch_region={watch_region}"
            
            # Apply category-specific filters (like language or genre animation)
            if not plat_lower:
                if cat_lower == "anime":
                    extra_params += "&with_genres=16&with_original_language=ja"
                elif cat_lower == "donghua":
                    extra_params += "&with_genres=16&with_original_language=zh"
                elif cat_lower == "animated":
                    extra_params += "&with_genres=16"
                elif cat_lower == "series":
                    extra_params += "&without_genres=16"
                elif cat_lower == "movie":
                    extra_params += "&without_genres=16"
            
            # Apply genre filters
            if genre:
                g_lower = genre.lower().strip()
                g_map = genre_map_movie if s_type == "movie" else genre_map_tv
                if g_lower in g_map:
                    extra_params += f"&with_genres={g_map[g_lower]}"
                    
            url = f"https://api.tmdb.org/3/discover/{s_type}?api_key={TMDB_API_KEY}&page={page}{extra_params}"
            
            try:
                r = requests.get(url, timeout=5)
                if r.status_code == 200:
                    data = r.json()
                    results = data.get("results", [])
                    total_pages = max(total_pages, data.get("total_pages", 1))
                    for m in results:
                        title = m.get("name") if s_type == "tv" else m.get("title")
                        if not title:
                            continue
                            
                        original_lang = m.get("original_language", "")
                        
                        # Filter out Anime and Donghua from Global Animated tab (original lang ja/zh)
                        if cat_lower == "animated" and original_lang in ["ja", "zh"]:
                            continue
                            
                        cover_url = f"https://image.tmdb.org/t/p/w500{m.get('poster_path')}" if m.get('poster_path') else ""
                        bd_url = f"https://image.tmdb.org/t/p/original{m.get('backdrop_path')}" if m.get('backdrop_path') else ""
                        
                        # Map category display
                        display_category = "Series" if s_type == "tv" else "Movie"
                        
                        # Set genre display
                        if cat_lower in ["anime", "donghua", "animated"]:
                            display_genre = category.capitalize()
                        else:
                            display_genre = display_category
                            
                        raw_results.append({
                            "Title": title,
                            "Category": display_category,
                            "Genre": display_genre,
                            "Description": m.get("overview", ""),
                            "Rating": float(m.get("vote_average", 0.0)),
                            "Cover_URL": cover_url,
                            "Backdrop_URL": bd_url,
                            "TMDB_ID": m.get("id"),
                            "Popularity": float(m.get("popularity", 0.0)),
                            "Source": "tmdb"
                        })
            except Exception as e:
                logger.error(f"Error fetching discover page {page} for {s_type}: {e}")
                
        # 5. Merge and sort by popularity
        if len(search_types) > 1:
            raw_results = sorted(raw_results, key=lambda x: x.get("Popularity", 0.0), reverse=True)
            # Remove popularity key to clean response
            for item in raw_results:
                item.pop("Popularity", None)
        
        if not raw_results:
            logger.info("TMDB discover returned no results. Trying local fallback.")
            raw_results, total_pages = get_local_discover(category, genre, platform, page)
        
        response_data = {
            "page": page,
            "total_pages": total_pages,
            "results": raw_results
        }
        
        discover_cache[cache_key] = (time.time(), response_data)
        return JSONResponse(content=response_data)
    except Exception as e:
        logger.error(f"Error in /api/discover: {e}")
        return JSONResponse(content={"page": page, "results": [], "total_pages": 1})

OPENROUTER_FALLBACK_MODEL = "openrouter/free"


@app.get("/api/search")
def search_all(query: str, page: int = 1):
    query_clean = query.strip()
    if not query_clean:
        return JSONResponse(content={"results": [], "page": page, "total_pages": 1})
        
    # Helper for local search paging
    def run_local_search():
        try:
            if ml_cache.df is None:
                ml_cache.reload()
            df = ml_cache.df
            if df.empty:
                return [], 1
            matched_df = df[df["Title"].str.lower().str.contains(query_clean.lower(), na=False, regex=False)]
            total_items = len(matched_df)
            page_size = 20
            total_pages = max(1, (total_items + page_size - 1) // page_size)
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            sliced_df = matched_df.iloc[start_idx:end_idx]
            
            results = []
            for _, row in sliced_df.iterrows():
                results.append({
                    "Title": row["Title"],
                    "Category": row["Category"],
                    "Genre": row["Genre"],
                    "Description": row["Description"],
                    "Rating": float(row["Rating"]),
                    "Cover_URL": row["Cover_URL"],
                    "Backdrop_URL": row.get("Backdrop_URL", ""),
                    "TMDB_ID": int(row.get("TMDB_ID", 0)),
                    "Source": "local"
                })
            return results, total_pages
        except Exception as e:
            logger.error(f"Error in local search helper: {e}")
            return [], 1

    if not TMDB_API_KEY:
        local_results, total_pages = run_local_search()
        return JSONResponse(content={"results": local_results, "page": page, "total_pages": total_pages})

    local_results = []
    try:
        if page == 1:
            if ml_cache.df is None:
                ml_cache.reload()
            df = ml_cache.df
            if not df.empty:
                matched_df = df[df["Title"].str.lower().str.contains(query_clean.lower(), na=False, regex=False)]
                for _, row in matched_df.iterrows():
                    local_results.append({
                        "Title": row["Title"],
                        "Category": row["Category"],
                        "Genre": row["Genre"],
                        "Description": row["Description"],
                        "Rating": float(row["Rating"]),
                        "Cover_URL": row["Cover_URL"],
                        "Backdrop_URL": row.get("Backdrop_URL", ""),
                        "TMDB_ID": int(row.get("TMDB_ID", 0)),
                        "Source": "local"
                    })
    except Exception as e:
        logger.error(f"Error in local search: {e}")
        
    tmdb_results = []
    total_pages = 1
    url = f"https://api.tmdb.org/3/search/multi?api_key={TMDB_API_KEY}&query={urllib.parse.quote(query_clean)}&page={page}"
    try:
        resp = requests.get(url, timeout=3.5)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        total_pages = data.get("total_pages", 1)
        for r in results:
            media_type = r.get("media_type")
            if media_type not in ["movie", "tv"]:
                continue
            title = r.get("name") if media_type == "tv" else r.get("title")
            
            if page == 1:
                local_exists = any(lr["Title"].lower().strip() == title.lower().strip() for lr in local_results)
                if local_exists:
                    continue
                
            tmdb_results.append({
                "Title": title,
                "Category": "Series" if media_type == "tv" else "Movie",
                "Genre": "",
                "Description": r.get("overview", ""),
                "Rating": float(r.get("vote_average", 0.0)),
                "Cover_URL": f"https://image.tmdb.org/t/p/w500{r.get('poster_path')}" if r.get('poster_path') else "",
                "Backdrop_URL": f"https://image.tmdb.org/t/p/original{r.get('backdrop_path')}" if r.get('backdrop_path') else "",
                "TMDB_ID": int(r.get("id", 0)),
                "Source": "tmdb"
            })
    except Exception as e:
        logger.error(f"Error in TMDB search: {e}")
        
    combined = local_results + tmdb_results if page == 1 else tmdb_results
    if not combined:
        logger.info("TMDB and page 1 local search returned no results. Running local paginated fallback.")
        combined, total_pages = run_local_search()
        
    return JSONResponse(content={"results": combined, "page": page, "total_pages": total_pages})



@app.get("/api/media-details/{title}")
def get_media_details(title: str, tmdb_id: int = None):
    try:
        category = "Movie"
        genre = ""
        description = ""
        rating = 0.0
        cover_url = ""
        bd_url = ""
        
        if not tmdb_id:
            if ml_cache.df is None:
                ml_cache.reload()
            df = ml_cache.df
            row = df[df["Title"].str.lower() == title.lower().strip()]
            if row.empty:
                row = df[df["Title"].str.lower().str.contains(title.lower().strip(), na=False, regex=False)]
            if not row.empty:
                tmdb_id = int(row.iloc[0].get("TMDB_ID", 0))
                category = row.iloc[0].get("Category", "Movie")
                genre = row.iloc[0]["Genre"]
                description = row.iloc[0]["Description"]
                rating = float(row.iloc[0]["Rating"])
                cover_url = row.iloc[0]["Cover_URL"]
                bd_url = row.iloc[0].get("Backdrop_URL", "")
                
        if not tmdb_id:
            for s_type in ["tv", "movie"]:
                search_url = f"https://api.tmdb.org/3/search/{s_type}?api_key={TMDB_API_KEY}&query={urllib.parse.quote(title)}"
                try:
                    r = requests.get(search_url, timeout=3.5)
                    res = r.json().get("results", [])
                    if res:
                        best = res[0]
                        tmdb_id = best["id"]
                        category = "Series" if s_type == "tv" else "Movie"
                        description = best.get("overview", "")
                        rating = float(best.get("vote_average", 0.0))
                        cover_url = f"https://image.tmdb.org/t/p/w500{best.get('poster_path')}" if best.get('poster_path') else ""
                        bd_url = f"https://image.tmdb.org/t/p/original{best.get('backdrop_path')}" if best.get('backdrop_path') else ""
                        title = best.get("name") if s_type == "tv" else best.get("title")
                        break
                except Exception as e:
                    logger.error(f"Error in TMDB search details fallback: {e}")
                    
        if not tmdb_id:
            raise HTTPException(status_code=404, detail=f"Title '{title}' not found on TMDB or locally.")
            
        tmdb_data = fetch_tmdb_details(tmdb_id, category)
        
        result = {
            "title": tmdb_data.get("name") if category == "Series" else tmdb_data.get("title", title),
            "category": category,
            "genre": " ".join([g["name"] for g in tmdb_data.get("genres", [])]) if tmdb_data else genre,
            "description": tmdb_data.get("overview") if tmdb_data else description,
            "rating": float(tmdb_data.get("vote_average", rating)) if tmdb_data else rating,
            "cover_url": f"https://image.tmdb.org/t/p/w500{tmdb_data.get('poster_path')}" if tmdb_data and tmdb_data.get('poster_path') else cover_url,
            "backdrop_url": f"https://image.tmdb.org/t/p/original{tmdb_data.get('backdrop_path')}" if tmdb_data and tmdb_data.get('backdrop_path') else bd_url,
            "tmdb_id": tmdb_id,
            "runtime": 0,
            "budget": 0,
            "revenue": 0,
            "cast": [],
            "trailer_key": None,
            "reviews": []
        }
        
        if tmdb_data:
            if category == "Movie":
                result["runtime"] = tmdb_data.get("runtime", 0)
                result["budget"] = tmdb_data.get("budget", 0)
                result["revenue"] = tmdb_data.get("revenue", 0)
            else:
                runtimes = tmdb_data.get("episode_run_time", [])
                result["runtime"] = runtimes[0] if runtimes else 0
                
            cast_list = tmdb_data.get("credits", {}).get("cast", [])[:8]
            formatted_cast = []
            for c in cast_list:
                formatted_cast.append({
                    "name": c.get("name"),
                    "character": c.get("character"),
                    "profile_url": f"https://image.tmdb.org/t/p/w185{c.get('profile_path')}" if c.get("profile_path") else ""
                })
            result["cast"] = formatted_cast
            
            videos = tmdb_data.get("videos", {}).get("results", [])
            for v in videos:
                if v.get("site", "").lower() == "youtube" and v.get("type", "").lower() == "trailer":
                    result["trailer_key"] = v.get("key")
                    break
                    
            reviews = tmdb_data.get("reviews", {}).get("results", [])[:5]
            formatted_reviews = []
            for r in reviews:
                formatted_reviews.append({
                    "author": r.get("author"),
                    "content": r.get("content"),
                    "rating": r.get("author_details", {}).get("rating")
                })
            result["reviews"] = formatted_reviews
            
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in media-details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommend/{title}")
def recommend(title: str, tmdb_id: int = None, authorization: str = Header(None)):
    try:
        if ml_cache.df is None or ml_cache.sim_matrix is None:
            ml_cache.reload()
            
        username = None
        if authorization and authorization.startswith("Bearer "):
            token = authorization.split(" ")[1]
            username = verify_token(token)
            
        likes, dislikes, watched = [], [], []
        if username:
            users = load_users()
            if username in users:
                likes = users[username].get("likes", [])
                dislikes = users[username].get("dislikes", [])
                watched = users[username].get("watched", [])
                
        local_recs = get_recommendations(title, ml_cache.df, ml_cache.sim_matrix, likes, dislikes, watched, top_n=6)
        
        tmdb_recs = []
        if not tmdb_id:
            df = ml_cache.df
            row = df[df["Title"].str.lower() == title.lower().strip()]
            if row.empty:
                row = df[df["Title"].str.lower().str.contains(title.lower().strip(), na=False, regex=False)]
            if not row.empty:
                tmdb_id = int(row.iloc[0].get("TMDB_ID", 0))
                category = row.iloc[0].get("Category", "Movie")
            else:
                for s_type in ["tv", "movie"]:
                    search_url = f"https://api.tmdb.org/3/search/{s_type}?api_key={TMDB_API_KEY}&query={urllib.parse.quote(title)}"
                    try:
                        r = requests.get(search_url, timeout=3.5)
                        res = r.json().get("results", [])
                        if res:
                            tmdb_id = res[0]["id"]
                            category = "Series" if s_type == "tv" else "Movie"
                            break
                    except Exception as e:
                        logger.error(f"Error in TMDB recommend search fallback: {e}")
        else:
            df = ml_cache.df
            row = df[df["TMDB_ID"] == tmdb_id]
            if not row.empty:
                category = row.iloc[0]["Category"]
            else:
                category = "Movie"
                
        if tmdb_id:
            tmdb_recs = fetch_tmdb_recommendations(tmdb_id, category)
            
        seen = {title.lower().strip()}
        watched_set = {t.lower().strip() for t in watched}
        
        blended = []
        for rec in local_recs:
            r_title_lower = rec["title"].lower().strip()
            if r_title_lower not in seen:
                seen.add(r_title_lower)
                blended.append(rec)
                
        for rec in tmdb_recs:
            r_title_lower = rec["title"].lower().strip()
            if r_title_lower not in seen and r_title_lower not in watched_set:
                seen.add(r_title_lower)
                blended.append(rec)
                
        if not blended:
            logger.info(f"No recommendations found for '{title}'. Using top rated fallbacks.")
            df = ml_cache.df
            if not df.empty:
                cat_filter = "Movie"
                if category and category.lower() == "series":
                    cat_filter = "Series"
                fallback_df = df[df["Category"].str.lower() == cat_filter.lower()]
                if fallback_df.empty:
                    fallback_df = df
                top_rated = fallback_df.sort_values("Rating", ascending=False).head(8)
                for _, row in top_rated.iterrows():
                    r_title_lower = row["Title"].lower().strip()
                    if r_title_lower not in seen and r_title_lower not in watched_set:
                        seen.add(r_title_lower)
                        blended.append({
                            "title":       row["Title"],
                            "category":    row["Category"],
                            "genre":       row["Genre"],
                            "description": row["Description"],
                            "rating":      float(row["Rating"]),
                            "cover_url":   row["Cover_URL"],
                            "score":       0.5
                        })
                        
        blended = blended[:8]
        return JSONResponse(content={"query": title, "recommendations": blended})
    except Exception as e:
        logger.error(f"Error in recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/recommendations/personalized")
def get_personalized_recommendations(authorization: str = Header(None), top_n: int = 10):
    try:
        username = None
        if authorization and authorization.startswith("Bearer "):
            token = authorization.split(" ")[1]
            username = verify_token(token)
            
        if ml_cache.df is None or ml_cache.sim_matrix is None:
            ml_cache.reload()
            
        df = ml_cache.df
        sim_matrix = ml_cache.sim_matrix
        
        likes, dislikes, watched = [], [], []
        if username:
            users = load_users()
            if username in users:
                likes = users[username].get("likes", [])
                dislikes = users[username].get("dislikes", [])
                watched = users[username].get("watched", [])
                
        # If the user has no history, return top rated titles
        if not likes and not watched:
            top_rated = df.sort_values("Rating", ascending=False).head(top_n)
            results = []
            for _, row in top_rated.iterrows():
                results.append({
                    "title":       row["Title"],
                    "category":    row["Category"],
                    "genre":       row["Genre"],
                    "description": row["Description"],
                    "rating":      float(row["Rating"]),
                    "cover_url":   row["Cover_URL"],
                    "score":       1.0
                })
            return JSONResponse(content={"recommendations": results, "source": "popular_fallback"})
            
        # Get lower case sets
        likes_set = {t.lower().strip() for t in likes}
        dislikes_set = {t.lower().strip() for t in dislikes}
        watched_set = {t.lower().strip() for t in watched}
        
        # Get indices
        titles_lower = df["Title"].str.lower().str.strip()
        likes_indices = [titles_lower[titles_lower == t].index[0] for t in likes_set if not titles_lower[titles_lower == t].empty]
        dislikes_indices = [titles_lower[titles_lower == t].index[0] for t in dislikes_set if not titles_lower[titles_lower == t].empty]
        watched_indices = [titles_lower[titles_lower == t].index[0] for t in watched_set if not titles_lower[titles_lower == t].empty]
        
        candidate_scores = []
        for i in range(len(df)):
            candidate_title = df.iloc[i]["Title"].lower().strip()
            # Don't recommend things they already watched or disliked
            if candidate_title in watched_set or candidate_title in dislikes_set:
                continue
                
            # Score is based on similarity to liked items + similarity to watched items (slightly lower weight)
            like_score = 0.0
            for l_idx in likes_indices:
                like_score += sim_matrix[l_idx][i]
            if likes_indices:
                like_score /= len(likes_indices)
                
            watch_score = 0.0
            for w_idx in watched_indices:
                watch_score += sim_matrix[w_idx][i]
            if watched_indices:
                watch_score /= len(watched_indices)
                
            dislike_penalty = 0.0
            for d_idx in dislikes_indices:
                dislike_penalty += sim_matrix[d_idx][i]
            if dislikes_indices:
                dislike_penalty /= len(dislikes_indices)
                
            # Combine scores: likes are weighted highest (0.6), watched is middle (0.4)
            final_score = 0.6 * like_score + 0.4 * watch_score - 0.5 * dislike_penalty
            
            # Boost slightly based on rating
            rating_boost = float(df.iloc[i]["Rating"]) / 10.0 * 0.1
            final_score += rating_boost
            
            candidate_scores.append((i, final_score))
            
        candidate_scores = sorted(candidate_scores, key=lambda x: x[1], reverse=True)
        scores = candidate_scores[:top_n]
        
        results = []
        for i, score in scores:
            row = df.iloc[i]
            results.append({
                "title":       row["Title"],
                "category":    row["Category"],
                "genre":       row["Genre"],
                "description": row["Description"],
                "rating":      float(row["Rating"]),
                "cover_url":   row["Cover_URL"],
                "score":       round(float(score), 4)
            })
        return JSONResponse(content={"recommendations": results, "source": "personalized_matrix"})
    except Exception as e:
        logger.error(f"Error in personalized recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/trending")
def get_trending():
    global trending_cache, trending_cache_time
    import time
    if trending_cache and (time.time() - trending_cache_time < 3600):
        logger.info("Serving trending from memory cache.")
        return JSONResponse(content=trending_cache)

    url = f"https://api.tmdb.org/3/trending/all/day?api_key={TMDB_API_KEY}"
    try:
        resp = requests.get(url, timeout=3.5)
        resp.raise_for_status()
        results = resp.json().get("results", [])[:10]
        trending_list = []
        for r in results:
            media_type = r.get("media_type", "movie")
            if media_type not in ["movie", "tv"]:
                continue
            title = r.get("name") if media_type == "tv" else r.get("title")
            trending_list.append({
                "Title": title,
                "Category": "Series" if media_type == "tv" else "Movie",
                "Genre": "",
                "Description": r.get("overview", ""),
                "Rating": float(r.get("vote_average", 0.0)),
                "Cover_URL": f"https://image.tmdb.org/t/p/w500{r.get('poster_path')}" if r.get('poster_path') else "",
                "Backdrop_URL": f"https://image.tmdb.org/t/p/original{r.get('backdrop_path')}" if r.get('backdrop_path') else "",
                "TMDB_ID": r.get("id")
            })
        trending_cache = trending_list
        trending_cache_time = time.time()
        return JSONResponse(content=trending_list)
    except Exception as e:
        logger.error(f"Error fetching trending: {e}")
        try:
            if ml_cache.df is None:
                ml_cache.reload()
            df = ml_cache.df
            if not df.empty:
                trending_list = []
                top_rated = df.sort_values("Rating", ascending=False).head(10)
                for _, row in top_rated.iterrows():
                    trending_list.append({
                        "Title": row["Title"],
                        "Category": row["Category"],
                        "Genre": row["Genre"],
                        "Description": row["Description"],
                        "Rating": float(row["Rating"]),
                        "Cover_URL": row["Cover_URL"],
                        "Backdrop_URL": row.get("Backdrop_URL", ""),
                        "TMDB_ID": int(row.get("TMDB_ID", 0))
                    })
                logger.info(f"Served {len(trending_list)} trending fallback items from local database.")
                trending_cache = trending_list
                trending_cache_time = time.time()
                return JSONResponse(content=trending_list)
        except Exception as ex:
            logger.error(f"Error in trending fallback: {ex}")
        return JSONResponse(content=[])



@app.get("/api/schedule")
def get_release_schedule(year: int = 2026, month: int = None):
    import datetime
    
    cache_key = f"{year}_{month}"
    if cache_key in schedule_cache:
        logger.info(f"Serving schedule for {cache_key} from memory cache.")
        return JSONResponse(content=schedule_cache[cache_key])
        
    releases = []
    
    # Calculate start and end date
    if month:
        start_date = f"{year}-{month:02d}-01"
        if month == 12:
            end_date = f"{year}-12-31"
        else:
            next_month_first = datetime.date(year, month + 1, 1)
            end_date = (next_month_first - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        
    movie_url = (
        f"https://api.tmdb.org/3/discover/movie?api_key={TMDB_API_KEY}"
        f"&primary_release_date.gte={start_date}&primary_release_date.lte={end_date}"
        f"&sort_by=primary_release_date.asc"
    )
    
    tv_url = (
        f"https://api.tmdb.org/3/discover/tv?api_key={TMDB_API_KEY}"
        f"&first_air_date.gte={start_date}&first_air_date.lte={end_date}"
        f"&sort_by=first_air_date.asc"
    )
    
    try:
        # Movies
        try:
            r_movies = requests.get(movie_url, timeout=3.5)
            if r_movies.status_code == 200:
                for m in r_movies.json().get("results", []):
                    rel_date = m.get("release_date")
                    if rel_date:
                        releases.append({
                            "Title": m.get("title"),
                            "Category": "Movie",
                            "Release_Date": rel_date,
                            "Description": m.get("overview", ""),
                            "Cover_URL": f"https://image.tmdb.org/t/p/w500{m.get('poster_path')}" if m.get('poster_path') else "",
                            "Backdrop_URL": f"https://image.tmdb.org/t/p/original{m.get('backdrop_path')}" if m.get('backdrop_path') else "",
                            "Rating": float(m.get("vote_average", 0.0)),
                            "TMDB_ID": m.get("id")
                        })
        except Exception as e:
            logger.error(f"Error fetching movies for schedule: {e}")
                    
        # TV/Series
        try:
            r_tv = requests.get(tv_url, timeout=3.5)
            if r_tv.status_code == 200:
                for m in r_tv.json().get("results", []):
                    air_date = m.get("first_air_date")
                    if air_date:
                        releases.append({
                            "Title": m.get("name"),
                            "Category": "Series",
                            "Release_Date": air_date,
                            "Description": m.get("overview", ""),
                            "Cover_URL": f"https://image.tmdb.org/t/p/w500{m.get('poster_path')}" if m.get('poster_path') else "",
                            "Backdrop_URL": f"https://image.tmdb.org/t/p/original{m.get('backdrop_path')}" if m.get('backdrop_path') else "",
                            "Rating": float(m.get("vote_average", 0.0)),
                            "TMDB_ID": m.get("id")
                        })
        except Exception as e:
            logger.error(f"Error fetching TV for schedule: {e}")
                    
        # Sort all by release date ascending
        releases = sorted(releases, key=lambda x: x["Release_Date"])
        
        if releases:
            schedule_cache[cache_key] = releases
            return JSONResponse(content=releases)
        else:
            raise Exception("No release data retrieved from TMDB")
    except Exception as e:
        logger.error(f"Error fetching schedule: {e}")
        # Fallback: Generate mock calendar entries using local database
        logger.info("Using local database fallback for schedule.")
        try:
            if ml_cache.df is None:
                ml_cache.reload()
            df = ml_cache.df
            
            # Select some items
            fallback_items = df.sample(min(len(df), 15), random_state=42).to_dict(orient="records") if not df.empty else []
            
            import random
            rng = random.Random(year + (month if month else 1))
            
            releases = []
            for idx, item in enumerate(fallback_items):
                # Distribute across days of the month
                day = rng.randint(1, 28)
                m = month if month else rng.randint(1, 12)
                date_str = f"{year}-{m:02d}-{day:02d}"
                
                releases.append({
                    "Title": item["Title"],
                    "Category": item["Category"],
                    "Release_Date": date_str,
                    "Description": item["Description"],
                    "Cover_URL": item["Cover_URL"],
                    "Backdrop_URL": item.get("Backdrop_URL", ""),
                    "Rating": float(item["Rating"]),
                    "TMDB_ID": int(item.get("TMDB_ID", 0))
                })
            releases = sorted(releases, key=lambda x: x["Release_Date"])
            schedule_cache[cache_key] = releases
            return JSONResponse(content=releases)
        except Exception as ex:
            logger.error(f"Error in schedule fallback: {ex}")
            return JSONResponse(content=[])


@app.get("/api/hero")
def get_hero():
    try:
        if ml_cache.df is None:
            ml_cache.reload()
        df = ml_cache.df
        if df.empty:
            raise HTTPException(status_code=404, detail="No media available for hero.")
        hero = df.sort_values("Rating", ascending=False).iloc[0]
        return JSONResponse(content={
            "title":       hero["Title"],
            "category":    hero["Category"],
            "genre":       hero["Genre"],
            "description": hero["Description"],
            "rating":      float(hero["Rating"]),
            "cover_url":   hero["Cover_URL"],
            "backdrop_url": hero.get("Backdrop_URL", ""),
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/trigger-update")
def trigger_update():
    try:
        updater_job()
        return JSONResponse(content={"status": "Update job triggered successfully."})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# USER API ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/register")
def register(auth: UserAuth):
    try:
        users = load_users()
        username = auth.username.strip()
        if not username:
            raise HTTPException(status_code=400, detail="Username cannot be empty.")
        if len(username) < 3:
            raise HTTPException(status_code=400, detail="Username must be at least 3 characters.")
        if len(auth.password) < 4:
            raise HTTPException(status_code=400, detail="Password must be at least 4 characters.")
        
        if username.lower() in [u.lower() for u in users]:
            raise HTTPException(status_code=400, detail="Access code conflict. Username already exists.")
        
        users[username] = {
            "password_hash": hash_password(auth.password),
            "likes": [],
            "dislikes": [],
            "watched": []
        }
        save_users(users)
        logger.info(f"User registered successfully: {username}")
        return JSONResponse(content={"status": "success", "message": "Access decrypted. User registered."})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/login")
def login(auth: UserAuth):
    try:
        users = load_users()
        username = auth.username.strip()
        matched_user = None
        for u in users:
            if u.lower() == username.lower():
                matched_user = u
                break
                
        if not matched_user:
            raise HTTPException(status_code=400, detail="Access credentials invalid.")
        
        if users[matched_user]["password_hash"] != hash_password(auth.password):
            raise HTTPException(status_code=400, detail="Access credentials invalid.")
        
        token = generate_token(matched_user)
        logger.info(f"User logged in successfully: {matched_user}")
        return JSONResponse(content={"status": "success", "token": token, "username": matched_user})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/interact")
def interact(req: InteractionRequest, authorization: str = Header(None)):
    try:
        username = None
        if authorization and authorization.startswith("Bearer "):
            token = authorization.split(" ")[1]
            username = verify_token(token)
            
        if not username:
            raise HTTPException(status_code=401, detail="Authentication required.")
            
        users = load_users()
        if username not in users:
            raise HTTPException(status_code=401, detail="User session invalid.")
            
        title = req.title.strip()
        interaction_type = req.type.strip().lower()
        
        if not title:
            raise HTTPException(status_code=400, detail="Title is required.")
            
        if "likes" not in users[username]:
            users[username]["likes"] = []
        if "dislikes" not in users[username]:
            users[username]["dislikes"] = []
        if "watched" not in users[username]:
            users[username]["watched"] = []
            
        # Dynamically append TMDB item to local database if not present
        df_local = load_df()
        exists = not df_local[df_local["Title"].str.lower() == title.lower().strip()].empty
        if not exists:
            try:
                search_url = f"https://api.tmdb.org/3/search/multi?api_key={TMDB_API_KEY}&query={urllib.parse.quote(title)}"
                r = requests.get(search_url, timeout=3.5)
                results = r.json().get("results", [])
                if results:
                    best = results[0]
                    media_type = best.get("media_type", "movie")
                    search_type = "tv" if media_type == "tv" else "movie"
                    t_id = best["id"]
                    
                    details_url = f"https://api.tmdb.org/3/{search_type}/{t_id}?api_key={TMDB_API_KEY}"
                    details_data = requests.get(details_url, timeout=3.5).json()
                    genre_list = [g["name"] for g in details_data.get("genres", [])]
                    
                    is_anime = any(kw in title.lower() or kw in details_data.get("overview", "").lower() for kw in ["anime", "manga", "japanese animation"])
                    if is_anime and "Anime" not in genre_list:
                        genre_list.append("Anime")
                        
                    genre_str = " ".join(genre_list)
                    category = "Series" if search_type == "tv" else "Movie"
                    cover_url = f"https://image.tmdb.org/t/p/w500{details_data.get('poster_path')}" if details_data.get('poster_path') else ""
                    bd_url = f"https://image.tmdb.org/t/p/original{details_data.get('backdrop_path')}" if details_data.get('backdrop_path') else ""
                    
                    row_data = {
                        "Title":       details_data.get("name") if search_type == "tv" else details_data.get("title", title),
                        "Category":    category,
                        "Genre":       genre_str,
                        "Description": details_data.get("overview", ""),
                        "Rating":      float(details_data.get("vote_average", 7.0)),
                        "Cover_URL":   cover_url,
                        "Backdrop_URL": bd_url,
                        "TMDB_ID":     t_id
                    }
                    
                    df_new = pd.DataFrame([row_data])
                    df_new.to_csv(DB_PATH, mode="a", header=False, index=False, encoding="utf-8")
                    logger.info(f"Dynamically appended '{title}' to local database.")
                    ml_cache.reload()
            except Exception as e:
                logger.error(f"Failed to dynamically append TMDB title '{title}' to local database: {e}")

        if interaction_type in ["like", "dislike", "none"]:
            if title in users[username]["likes"]:
                users[username]["likes"].remove(title)
            if title in users[username]["dislikes"]:
                users[username]["dislikes"].remove(title)
                
            if interaction_type == "like":
                users[username]["likes"].append(title)
                logger.info(f"User '{username}' liked: {title}")
            elif interaction_type == "dislike":
                users[username]["dislikes"].append(title)
                logger.info(f"User '{username}' disliked: {title}")
            else:
                logger.info(f"User '{username}' cleared rating for: {title}")
                
        elif interaction_type == "watched":
            if title not in users[username]["watched"]:
                users[username]["watched"].append(title)
                logger.info(f"User '{username}' marked watched: {title}")
        elif interaction_type == "unwatched":
            if title in users[username]["watched"]:
                users[username]["watched"].remove(title)
                logger.info(f"User '{username}' marked unwatched: {title}")
                
        save_users(users)
        return JSONResponse(content={
            "status": "success", 
            "likes": users[username].get("likes", []), 
            "dislikes": users[username].get("dislikes", []),
            "watched": users[username].get("watched", [])
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/user/profile")
def get_profile(authorization: str = Header(None)):
    try:
        username = None
        if authorization and authorization.startswith("Bearer "):
            token = authorization.split(" ")[1]
            username = verify_token(token)
            
        if not username:
            raise HTTPException(status_code=401, detail="Authentication required.")
            
        users = load_users()
        if username not in users:
            raise HTTPException(status_code=401, detail="User session invalid.")
            
        return JSONResponse(content={
            "username": username,
            "likes": users[username].get("likes", []),
            "dislikes": users[username].get("dislikes", []),
            "watched": users[username].get("watched", []),
            "watchlist_folders": users[username].get("watchlist_folders", {}),
            "ratings_reviews": users[username].get("ratings_reviews", {}),
            "custom_collections": users[username].get("custom_collections", {})
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class UserDataUpdateRequest(BaseModel):
    watchlist_folders: dict = None
    ratings_reviews: dict = None
    custom_collections: dict = None


@app.post("/api/user/update-data")
def update_user_data(req: UserDataUpdateRequest, authorization: str = Header(None)):
    try:
        username = None
        if authorization and authorization.startswith("Bearer "):
            token = authorization.split(" ")[1]
            username = verify_token(token)
            
        if not username:
            raise HTTPException(status_code=401, detail="Authentication required.")
            
        users = load_users()
        if username not in users:
            raise HTTPException(status_code=401, detail="User session invalid.")
            
        if req.watchlist_folders is not None:
            users[username]["watchlist_folders"] = req.watchlist_folders
        if req.ratings_reviews is not None:
            users[username]["ratings_reviews"] = req.ratings_reviews
        if req.custom_collections is not None:
            users[username]["custom_collections"] = req.custom_collections
            
        save_users(users)
        return JSONResponse(content={"status": "success", "message": "User data updated."})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


@app.post("/api/ai-chat")
def ai_chat(req: ChatRequest, authorization: str = Header(None)):
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=400, detail="OpenRouter API key is not configured.")
        
    username = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
        username = verify_token(token)
        
    likes, watched = [], []
    if username:
        users = load_users()
        if username in users:
            likes = users[username].get("likes", [])
            watched = users[username].get("watched", [])
            
    taste_context = ""
    if likes or watched:
        taste_context = (
            f" The user is logged in as {username}."
            f" Their liked titles: {', '.join(likes)}."
            f" Their watched history: {', '.join(watched)}."
            " Use this information to tailor your suggestions to their tastes if relevant."
        )
        
    system_prompt = {
        "role": "system",
        "content": (
            "You are NEXUS AI, a sophisticated cyberpunk-themed movie, series, anime, and donghua recommender. "
            "Help the user find the perfect media to watch based on their request. Be concise, engaging, and "
            "format media titles clearly like **Movie Title (Year)**. Mention genres and streaming platforms if known."
            f"{taste_context}"
        )
    }
    
    formatted_messages = [system_prompt]
    for msg in req.messages:
        formatted_messages.append({"role": msg.role, "content": msg.content})
        
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "http://localhost:8000",
        "X-Title":       "MediaRecommendEngine",
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": formatted_messages,
        "temperature": 0.7,
        "max_tokens":  800,
    }
    
    try:
        # Try primary model first with 12s timeout
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=12)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        return JSONResponse(content={"content": content})
    except Exception as e:
        logger.warning(f"Primary model {OPENROUTER_MODEL} failed: {e}. Trying fallback model {OPENROUTER_FALLBACK_MODEL}...")
        payload["model"] = OPENROUTER_FALLBACK_MODEL
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=12)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            return JSONResponse(content={"content": content})
        except Exception as ex:
            logger.error(f"Fallback model failed: {ex}")
            raise HTTPException(status_code=500, detail=f"AI chat error: {str(ex)}")


@app.get("/api/ai-insights/{title}")
def get_ai_insights(title: str):
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=400, detail="OpenRouter API key is not configured.")
        
    prompt = f"""You are a movie and TV recommendation expert. Find details for the title: "{title}".
Please search/retrieve:
1. Streaming platforms (e.g. Netflix, Crunchyroll, Prime Video, Disney+) where it is available.
2. Estimated ratings from IMDb and Moctale (or your best estimation).
3. The release schedule or current status (e.g. "Completed", "Ongoing - Airing Sundays").
4. A brief, engaging AI suggestion/tip (1-2 sentences) about why someone should watch it.

Return ONLY a valid JSON object matching the following structure (no markdown code blocks, no other text):
{{
  "imdb_rating": "8.5/10",
  "moctale_rating": "8.8/10",
  "platforms": ["Netflix", "Prime Video"],
  "schedule": "Completed",
  "ai_suggestion": "A mind-bending thriller with incredible visuals and performance. Perfect for fans of complex mystery sci-fi."
}}
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "http://localhost:8000",
        "X-Title":       "MediaRecommendEngine",
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens":  600,
    }
    try:
        # Try primary model first with 12s timeout
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=12)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        
        start_idx = content.find("{")
        end_idx = content.rfind("}")
        if start_idx != -1 and end_idx != -1:
            content = content[start_idx:end_idx+1]
        
        data = json.loads(content)
        return JSONResponse(content=data)
    except Exception as e:
        logger.warning(f"Primary model insight failed: {e}. Trying fallback model {OPENROUTER_FALLBACK_MODEL}...")
        payload["model"] = OPENROUTER_FALLBACK_MODEL
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=12)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            
            start_idx = content.find("{")
            end_idx = content.rfind("}")
            if start_idx != -1 and end_idx != -1:
                content = content[start_idx:end_idx+1]
            
            data = json.loads(content)
            return JSONResponse(content=data)
        except Exception as ex:
            logger.error(f"Fallback model insight failed: {ex}")
            return JSONResponse(content={
                "imdb_rating": "N/A",
                "moctale_rating": "N/A",
                "platforms": ["Check local listings"],
                "schedule": "N/A",
                "ai_suggestion": f"Unable to fetch real-time AI details at this moment. Please check TMDB directly."
            })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
