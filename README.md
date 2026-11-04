# Music Project (DS2023)

## Setup
Requires [uv](https://docs.astral.sh/uv/). From this folder:

```
uv sync
```

This creates `.venv/` with everything listed in `pyproject.toml`.

## Add release dates (recommended): `release_join.py`
```
uv run python release_join.py
```
Takes about 10-15 minutes and needs no Spotify keys. It reads the Hugging Face dataset
[`alpha31476/spotify-metadata-parquet`](https://huggingface.co/datasets/alpha31476/spotify-metadata-parquet),
a Parquet export of a large Spotify metadata dump, and downloads only the columns it needs.
Matches are cached in `cache/`, so reruns are fast. Delete `cache/` to refetch.

**Data source note:** that dump comes from an unofficial scrape of Spotify, and its uploader
assigns no license. The release dates are Spotify's own values, the same fields the API returns.

**Output:** `spotify_tracks_with_release.csv`, all rows of `largeSpotify.csv` plus:

| column | meaning |
|---|---|
| `album_id`, `album_type` | Spotify album the track belongs to (`album`, `single`, `compilation`) |
| `album_release_date` | Spotify's original string: `YYYY`, `YYYY-MM` or `YYYY-MM-DD` |
| `release_date_precision` | `year`, `month` or `day` |
| `release_year` | integer year (filled whenever a date is known) |
| `release_month` | filled only when precision is `month` or `day` |
| `release_date` | full date, filled only when precision is `day` (a year-only release is never shown as Jan 1) |

| `original_release_year` | year the **recording** first came out, across all its releases (use this for era analysis) |
| `original_release_date` | full original date, filled only when that release has day precision |
| `original_release_date_precision` | `year`, `month` or `day` for the original release |
| `original_album_id`, `original_album_name`, `original_album_type` | the release the original date came from |
| `n_releases` | how many releases of this recording were found |
| `original_confirmed` | `True` if the original is this track's own release, or 2+ separate releases agree on that year; `False` if one lone release is the only evidence for an earlier date |

The `release_*` columns describe this specific `track_id`'s album, which may be a
compilation or remaster from years later. The `original_*` columns use the track's
ISRC (a code for the recording itself, shared by its single, album, compilation and
remaster releases) and take the earliest date in Spotify's catalog. Releases must also be
within 10 seconds of the same length, to guard against reused ISRCs.
Limits: a remaster registered under a new ISRC keeps the remaster's date, and nothing
can go earlier than the oldest release on Spotify. Some uploads carry made-up dates (e.g.
Billie Eilish's 2018 "lovely" appears on one release dated 1930). These can't be told
apart automatically from real lone originals (e.g. Mazzy Star's 1993 "Fade Into You"), so
they are marked `original_confirmed = False` (about 5% of tracks) rather than guessed.
For strict analysis, filter to `original_confirmed == True`.

Dates that don't match their stated precision, or fall outside 1900 to 2026, are left blank
instead of guessed.

**Verify:** the script prints a quality report with the match rate, precision breakdown and
year range, and confirms the output has the same row count as the input. It also compares
the dates against the tracks already fetched from the Spotify API by `joiner.py`
(`release_cache.json`). Those should match.

## Alternative: Spotify API (`joiner.py`)
Needs Spotify keys in `.env` (copy `.env.example`). Spotify blocks the batch endpoint for
personal apps and rate-limits hard (one lockout lasted ~23 hours), so this is only practical
for small samples.

## Notebook
Open `musicViz.ipynb` in VS Code and choose the `.venv` Python interpreter as the kernel
(top right of the notebook, "Select Kernel"). To check it works, run a cell with
`import pandas as pd; pd.read_csv("largeSpotify.csv").shape`.
