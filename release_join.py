"""Add album release dates to largeSpotify.csv without calling the Spotify API.

Source: the Hugging Face dataset alpha31476/spotify-metadata-parquet, a Parquet export
of a large Spotify metadata dump. Its `tracks` table maps a Spotify track ID to an
album, and its `albums` table holds Spotify's own `release_date` and
`release_date_precision` (the same fields the Spotify API returns).

Each track ID is one specific release of a song (single, album, compilation, remaster),
so its album date can be years after the song first came out. To find the original
date we use the track's ISRC, a code that identifies the recording itself and stays the
same across re-releases: we collect every release in the catalog with the same ISRC
and take the earliest valid date.

DuckDB reads the Parquet files straight from Hugging Face and only downloads the
columns we ask for, so the full ~17 GB never lands on disk. Each step's matches are
saved under cache/, so a rerun skips steps that already finished.
"""

import os
import json
import time
import duckdb
import pandas as pd

# ---------- CONFIG ----------
INPUT_CSV = "largeSpotify.csv"
OUTPUT_CSV = "spotify_tracks_with_release.csv"
ID_COL = "track_id"
HF_BASE = "hf://datasets/alpha31476/spotify-metadata-parquet/spotify_clean"
CACHE_DIR = "cache"
TRACK_MATCHES = os.path.join(CACHE_DIR, "track_matches.parquet")     # our tracks -> album, ISRC
VERSION_MATCHES = os.path.join(CACHE_DIR, "version_matches.parquet")  # all releases sharing those ISRCs
ALBUM_MATCHES = os.path.join(CACHE_DIR, "album_matches.parquet")
API_CACHE = "release_cache.json"  # dates joiner.py already fetched from the API, used as a check
CHUNK_SIZE = 10                   # Parquet files per query; progress is saved after each chunk
RATE_LIMIT_WAIT = 200             # seconds; Hugging Face's anonymous limit resets every 5 minutes
MAX_DURATION_DIFF_MS = 10_000     # a "same ISRC" release must also be within 10s in length
# -----------------------------


def timed(label, fn):
    start = time.time()
    print(f"{label}...", flush=True)
    result = fn()
    print(f"  done in {time.time() - start:.0f}s", flush=True)
    return result


def scan_in_chunks(con, table, select_sql, out_file, name):
    """Run `select_sql` over a Hugging Face table a chunk of files at a time.

    Anonymous Hugging Face downloads are limited to ~3000 requests per 5 minutes, so
    scanning everything in one query gets cut off partway. Each chunk's matches are
    saved as soon as it finishes, and a rate-limit error just waits and retries
    that chunk. At the end the chunks are combined into `out_file`.
    """
    files = sorted(f[0] for f in con.execute(f"SELECT file FROM glob('{HF_BASE}/{table}/*.parquet')").fetchall())
    chunk_dir = os.path.join(CACHE_DIR, f"{name}_chunks")
    os.makedirs(chunk_dir, exist_ok=True)

    for start in range(0, len(files), CHUNK_SIZE):
        chunk_file = os.path.join(chunk_dir, f"{start:06d}.parquet")
        if os.path.exists(chunk_file):
            continue
        chunk = files[start:start + CHUNK_SIZE]
        failures = 0
        while True:
            try:
                con.execute(f"COPY ({select_sql.format(files=chunk)}) TO '{chunk_file}.tmp' (FORMAT parquet)")
                break
            except duckdb.Error as e:
                msg = str(e)
                if "429" in msg:
                    print(f"  Hugging Face rate limit hit; waiting {RATE_LIMIT_WAIT}s, then retrying", flush=True)
                    time.sleep(RATE_LIMIT_WAIT)
                    continue
                # Mistakes in the SQL itself won't fix themselves, so stop right away
                if isinstance(e, (duckdb.ParserException, duckdb.BinderException, duckdb.CatalogException)):
                    raise
                # Anything else here is a dropped connection / DNS hiccup: retry with growing waits
                failures += 1
                if failures > 6:
                    raise
                print(f"  Network error ({msg[:80]}...); retry {failures}/6", flush=True)
                time.sleep(30 * failures)
        os.replace(f"{chunk_file}.tmp", chunk_file)  # only a finished chunk counts as done
        print(f"  {name}: {min(start + CHUNK_SIZE, len(files))}/{len(files)} files", flush=True)

    con.execute(f"COPY (SELECT * FROM '{chunk_dir}/*.parquet') TO '{out_file}' (FORMAT parquet)")


def find_tracks(con):
    """Step 1: scan the ~256M-row tracks table for our track IDs."""
    scan_in_chunks(con, "tracks", """
        SELECT t.id AS track_id, t.album_rowid, t.external_id_isrc AS isrc, t.duration_ms
        FROM read_parquet({files}) t
        SEMI JOIN wanted w ON t.id = w.track_id
    """, TRACK_MATCHES, "tracks")


def find_versions(con):
    """Step 2: scan the tracks table again for every release that shares one of our ISRCs."""
    con.execute(f"""
        CREATE OR REPLACE TABLE wanted_isrcs AS
        SELECT DISTINCT isrc FROM '{TRACK_MATCHES}' WHERE isrc IS NOT NULL AND isrc <> ''
    """)
    scan_in_chunks(con, "tracks", """
        SELECT t.external_id_isrc AS isrc, t.album_rowid, t.duration_ms
        FROM read_parquet({files}) t
        SEMI JOIN wanted_isrcs w ON t.external_id_isrc = w.isrc
    """, VERSION_MATCHES, "versions")


def find_albums(con):
    """Step 3: look up the albums for our tracks and for all of their other releases."""
    con.execute(f"""
        CREATE OR REPLACE TABLE wanted_albums AS
        SELECT album_rowid FROM '{TRACK_MATCHES}'
        UNION SELECT album_rowid FROM '{VERSION_MATCHES}'
    """)
    scan_in_chunks(con, "albums", """
        SELECT a.rowid AS album_rowid, a.id AS album_id, a.name AS album_name, a.album_type,
               a.release_date, a.release_date_precision
        FROM read_parquet({files}) a
        SEMI JOIN wanted_albums m ON a.rowid = m.album_rowid
    """, ALBUM_MATCHES, "albums")


def parse_dates(rows):
    """Parse Spotify date strings, flagging any that don't match their precision.

    Spotify dates come as "YYYY", "YYYY-MM" or "YYYY-MM-DD", depending on precision.
    A value counts as valid only if its format matches its stated precision and it is
    a real calendar date in a plausible range. Anything else is invalid, including
    Spotify's "unknown date" placeholders "0000" and year 1900 (e.g. "1900-01-01").
    """
    raw = rows["release_date"].astype("string")
    prec = rows["release_date_precision"].astype("string")
    expected_len = prec.map({"year": 4, "month": 7, "day": 10})
    format_ok = raw.str.len() == expected_len
    parsed = pd.to_datetime(raw.where(format_ok), errors="coerce", format="mixed")
    year = parsed.dt.year
    valid = parsed.notna() & year.between(1901, 2026)
    return raw, prec, parsed, year, valid


def normalize(release):
    """Turn Spotify's raw date string + precision into clean, typed columns.

    Invalid dates become missing rather than a made-up date.
    """
    raw, prec, parsed, year, valid = parse_dates(release)

    out = pd.DataFrame({ID_COL: release[ID_COL]})
    out["album_id"] = release["album_id"]
    out["album_type"] = release["album_type"]
    out["album_release_date"] = raw.where(valid)          # Spotify's original string
    out["release_date_precision"] = prec.where(valid)
    out["release_year"] = year.where(valid).astype("Int64")
    # Month and full date are only filled when Spotify actually knows them,
    # so a year-only release is never shown as January 1st.
    out["release_month"] = parsed.dt.month.where(valid & prec.isin(["month", "day"])).astype("Int64")
    out["release_date"] = parsed.where(valid & (prec == "day")).dt.date
    return out, (~valid & raw.notna()).sum()


def original_releases(con):
    """For each of our tracks, find the earliest release of the same recording.

    Candidates are all releases with the same ISRC and a similar length (a guard
    against the occasional ISRC that was reused for a different recording), plus the
    track's own release. The earliest year wins; within that year we prefer the most
    precise date, so a known day isn't replaced by a vaguer year-only entry.
    """
    cands = con.execute(f"""
        SELECT t.track_id, a.album_id, a.album_name, a.album_type,
               a.release_date, a.release_date_precision
        FROM '{TRACK_MATCHES}' t
        JOIN '{VERSION_MATCHES}' v
          ON v.isrc = t.isrc AND abs(v.duration_ms - t.duration_ms) <= {MAX_DURATION_DIFF_MS}
        JOIN '{ALBUM_MATCHES}' a ON a.album_rowid = v.album_rowid
        UNION
        SELECT t.track_id, a.album_id, a.album_name, a.album_type,
               a.release_date, a.release_date_precision
        FROM '{TRACK_MATCHES}' t
        JOIN '{ALBUM_MATCHES}' a ON a.album_rowid = t.album_rowid
    """).df()

    raw, prec, parsed, year, valid = parse_dates(cands)
    cands = cands.assign(prec=prec, parsed=parsed, year=year,
                         prec_rank=prec.map({"day": 0, "month": 1, "year": 2}))[valid]
    n_releases = cands.groupby(ID_COL)["album_id"].nunique().rename("n_releases")
    best = (cands.sort_values([ID_COL, "year", "prec_rank", "parsed", "album_id"])
                 .drop_duplicates(ID_COL)
                 .set_index(ID_COL))

    # Some uploads carry made-up dates (e.g. a 2018 hit dated 1930), and nothing in the
    # data reliably tells those apart from a real early release. So instead of guessing,
    # count how many separate releases agree with the earliest year (within a year).
    # One lone release is "unconfirmed"; two or more back each other up.
    near = cands[ID_COL].map(best["year"]) + 1 >= cands["year"]
    n_agree = cands[near].groupby(ID_COL)["album_id"].nunique()

    out = pd.DataFrame(index=best.index)
    out["original_release_year"] = best["year"].astype("Int64")
    out["original_release_date"] = best["parsed"].where(best["prec"] == "day").dt.date
    out["original_release_date_precision"] = best["prec"]
    out["original_album_id"] = best["album_id"]
    out["original_album_name"] = best["album_name"]
    out["original_album_type"] = best["album_type"]
    out = out.join(n_releases)  # how many distinct releases of this recording were found
    out["n_original_agree"] = n_agree
    return out.reset_index()


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    df = pd.read_csv(INPUT_CSV)
    con = duckdb.connect()
    con.execute("SET threads TO 4")  # more parallel downloads made Hugging Face drop connections
    con.register("ids_df", df[[ID_COL]].dropna().drop_duplicates())
    con.execute("CREATE TABLE wanted AS SELECT track_id FROM ids_df")
    n_ids = con.execute("SELECT count(*) FROM wanted").fetchone()[0]
    print(f"{len(df)} rows, {n_ids} unique track IDs")

    if not os.path.exists(TRACK_MATCHES):
        timed("Step 1/3: scanning tracks table for our track IDs", lambda: find_tracks(con))
    if not os.path.exists(VERSION_MATCHES):
        timed("Step 2/3: scanning tracks table for other releases of the same recordings",
              lambda: find_versions(con))
    if not os.path.exists(ALBUM_MATCHES):
        timed("Step 3/3: scanning albums table", lambda: find_albums(con))

    # Join track -> album. A track ID should map to exactly one album; if the dump
    # has duplicates, keep one row per track so the final join can't add rows.
    release = con.execute(f"""
        SELECT t.track_id, a.album_id, a.album_type, a.release_date, a.release_date_precision
        FROM '{TRACK_MATCHES}' t
        JOIN '{ALBUM_MATCHES}' a USING (album_rowid)
        QUALIFY row_number() OVER (PARTITION BY t.track_id ORDER BY a.album_id) = 1
    """).df()
    dup_tracks = con.execute(
        f"SELECT count(*) - count(DISTINCT track_id) FROM '{TRACK_MATCHES}'").fetchone()[0]

    release, n_invalid = normalize(release)
    release = release.merge(original_releases(con), on=ID_COL, how="left", validate="one_to_one")
    # The track's own release is always a candidate, so the original can never be later
    later = (release["original_release_year"] > release["release_year"]).sum()
    assert later == 0, f"{later} tracks have an original year after their own release year"
    # Confirmed = the original is this track's own release year, or 2+ releases agree on it
    release["original_confirmed"] = (
        (release["original_release_year"] == release["release_year"])
        | (release.pop("n_original_agree") >= 2)
    ).where(release["original_release_year"].notna())

    merged = df.merge(release, on=ID_COL, how="left", validate="many_to_one")
    assert len(merged) == len(df), "join changed the row count"

    # ---------- Checks ----------
    matched = merged.drop_duplicates(ID_COL)
    print("\n--- Quality report ---")
    print(f"Tracks found in source:     {release[ID_COL].nunique()} / {n_ids} "
          f"({release[ID_COL].nunique() / n_ids:.1%})")
    print(f"Duplicate track rows in source (deduplicated): {dup_tracks}")
    print(f"Dates dropped as invalid:   {n_invalid}")
    print(f"Unique tracks with a year:  {matched['release_year'].notna().sum()}")
    print("Precision breakdown (unique tracks):")
    print(matched["release_date_precision"].value_counts(dropna=False).to_string())
    print(f"Year range: {matched['release_year'].min()} - {matched['release_year'].max()}")

    has_isrc = con.execute(f"SELECT count(*) FROM '{TRACK_MATCHES}' WHERE isrc <> ''").fetchone()[0]
    earlier = matched["original_release_year"] < matched["release_year"]
    print(f"\nTracks with an ISRC:        {has_isrc} / {n_ids}")
    print(f"Tracks with an original year: {matched['original_release_year'].notna().sum()}")
    print(f"Original year earlier than this release's year: {earlier.sum()} tracks")
    for kind in ["album", "single", "compilation"]:
        sub = matched[matched["album_type"] == kind]
        print(f"  {kind:<12} {(sub['original_release_year'] < sub['release_year']).mean():.1%} moved earlier")
    gap = (matched["release_year"] - matched["original_release_year"])[earlier]
    print(f"  median shift {gap.median():.0f} years, 90th percentile {gap.quantile(0.9):.0f} years")
    conf = matched.loc[earlier, "original_confirmed"]
    print(f"  of those, confirmed by 2+ releases: {conf.sum()} ({conf.mean():.1%}); "
          f"single-release (unconfirmed): {(~conf.astype(bool)).sum()}")
    # Same artist + title elsewhere in our own data is an independent check: its earliest
    # year should rarely be before the ISRC-based original.
    key = matched["artists"].str.lower().str.strip() + "|" + matched["track_name"].str.lower().str.strip()
    dataset_min = matched.groupby(key)["release_year"].transform("min")
    print(f"Tracks where a same-title version in our dataset is older than the ISRC original: "
          f"{(dataset_min < matched['original_release_year']).sum()}")

    # Compare against dates fetched directly from the Spotify API by joiner.py
    if os.path.exists(API_CACHE):
        with open(API_CACHE) as f:
            api = pd.DataFrame.from_dict(json.load(f), orient="index").rename_axis(ID_COL).reset_index()
        api = api.dropna(subset=["album_release_date"])
        cmp = api.merge(release, on=ID_COL, suffixes=("_api", ""))
        same_date = (cmp["album_release_date_api"] == cmp["album_release_date"]).sum()
        same_album = (cmp["album_id_api"] == cmp["album_id"]).sum()
        print(f"\nAPI check: {len(cmp)} of {len(api)} API-fetched tracks found here; "
              f"release date matches {same_date}/{len(cmp)}, album ID matches {same_album}/{len(cmp)}")

    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {OUTPUT_CSV} ({len(merged)} rows, same as {INPUT_CSV})")


if __name__ == "__main__":
    main()
