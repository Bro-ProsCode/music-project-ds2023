import os
import time
import json
import base64
import requests
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()  # reads SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET from a .env file, if present

# ---------- CONFIG ----------
INPUT_CSV = "largeSpotify.csv"           # the file downloaded from Kaggle
OUTPUT_CSV = "spotify_tracks_with_release.csv"
CACHE_FILE = "release_cache.json"         # lets you resume if interrupted
ID_COL = "track_id"                       # <-- change if your column is named differently
CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET")
# -----------------------------


def get_token():
    auth = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
    r = requests.post(
        "https://accounts.spotify.com/api/token",
        headers={"Authorization": f"Basic {auth}"},
        data={"grant_type": "client_credentials"},
        timeout=30,
    )
    r.raise_for_status()
    tok = r.json()
    return tok["access_token"], time.time() + tok["expires_in"] - 60


def fetch_track(track_id, token):
    """Fetch one track, handling rate limits. Returns None if the ID no longer exists.

    Spotify's batch endpoint (/v1/tracks?ids=...) returns 403 for Development Mode
    apps, so we use the single-track endpoint instead.
    """
    retries = 0
    while True:
        try:
            r = requests.get(
                f"https://api.spotify.com/v1/tracks/{track_id}",
                headers={"Authorization": f"Bearer {token}"},
                timeout=30,
            )
        except (requests.ConnectionError, requests.Timeout):
            # Spotify sometimes drops connections; back off and retry instead of crashing
            retries += 1
            if retries > 8:
                raise
            time.sleep(min(2 ** retries, 120))
            continue
        if r.status_code == 429:  # rate limited
            wait = int(r.headers.get("Retry-After", 5)) + 1
            tqdm.write(f"Rate limited, waiting {wait}s")
            time.sleep(wait)
            continue
        if r.status_code >= 500:  # temporary server error
            retries += 1
            if retries > 8:
                r.raise_for_status()
            time.sleep(min(2 ** retries, 120))
            continue
        if r.status_code in (400, 404):  # bad or removed ID
            return None
        r.raise_for_status()
        return r.json()


def main():
    if not CLIENT_ID or not CLIENT_SECRET:
        raise SystemExit("Missing Spotify credentials. Copy .env.example to .env and fill in your keys.")

    df = pd.read_csv(INPUT_CSV)
    if ID_COL not in df.columns:
        raise SystemExit(f"Column '{ID_COL}' not found. Columns are: {list(df.columns)}")

    # Load cache so reruns only fetch what's missing
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            cache = json.load(f)

    unique_ids = df[ID_COL].dropna().unique().tolist()
    todo = [i for i in unique_ids if i not in cache]
    print(f"{len(unique_ids)} unique tracks, {len(todo)} to fetch")

    token, expires = get_token()

    try:
        for n, tid in enumerate(tqdm(todo)):
            if time.time() > expires:
                token, expires = get_token()

            t = fetch_track(tid, token)
            if t is None:  # ID no longer exists on Spotify
                cache[tid] = {"album_release_date": None,
                              "release_date_precision": None,
                              "album_id": None, "album_type": None}
            else:
                album = t["album"]
                cache[tid] = {
                    "album_release_date": album["release_date"],
                    "release_date_precision": album["release_date_precision"],
                    "album_id": album["id"],
                    "album_type": album["album_type"],
                }

            # checkpoint every 50 tracks
            if n % 50 == 0:
                with open(CACHE_FILE, "w") as f:
                    json.dump(cache, f)
            time.sleep(0.05)
    finally:
        # always save progress, even on Ctrl+C or a crash, so a rerun resumes here
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f)

    # ---------- Join ----------
    release_df = (
        pd.DataFrame.from_dict(cache, orient="index")
        .rename_axis(ID_COL)
        .reset_index()
    )
    merged = df.merge(release_df, on=ID_COL, how="left")

    # Spotify dates can be "YYYY", "YYYY-MM", or "YYYY-MM-DD".
    # Keep the raw string, plus a parsed date (missing parts default to Jan / 1st) and a year.
    merged["release_year"] = pd.to_numeric(
        merged["album_release_date"].str[:4], errors="coerce"
    ).astype("Int64")
    merged["album_release_date_parsed"] = pd.to_datetime(
        merged["album_release_date"], errors="coerce", format="mixed"
    )

    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {OUTPUT_CSV}")
    print(f"Missing release dates: {merged['album_release_date'].isna().sum()}")


if __name__ == "__main__":
    main()