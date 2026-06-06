# Investistan PSX Screener — Streamlit Cloud

Live PSX (KSE-100) screening dashboard. Hosted on Streamlit Community Cloud —
open the URL on any device, no Python needed.

## This is the full, current build
Includes: sector-aware + coverage-fair scoring, the session-state portfolio
(with the sell fix), the input-visibility + screener-table contrast fixes,
premium buttons, the SVG brand lockup, the rebuilt candlestick (synced volume),
and the dark/light theme toggle.

## How to update your repo with this folder
1. Unzip.
2. Upload the **contents** of this folder (the files + the `.streamlit` folder)
   to your repo root — overwriting the existing files and adding the new ones
   (`ui.py`, `.streamlit/config.toml`). Do **not** upload the wrapping
   `investistan-screener` folder itself; upload what's inside it.
3. Streamlit Cloud redeploys automatically on the new commit (~1–2 min).
4. Hard-refresh the browser (Ctrl+F5).

## How data works on the cloud
No separate background process, so the app generates its own data: it re-screens
the KSE-100 in-process every ~45 min and caches it. The included `.csv`/`.json`
files are seed data so the first load shows results instantly. (Seed scores use
the previous methodology; they refresh to the new sector-aware scoring on the
next re-screen — or delete `psx_screener_results.csv` before upload to force a
fresh scrape on first load.)

## Notes
- Free apps sleep when idle; first visit after a nap is a slow cold start.
- The hosted filesystem is ephemeral — portfolio/watchlist edits reset on
  restart (permanent persistence needs a DB, e.g. Supabase).
- The app URL is publicly viewable by default.
- Dark is the polished default theme; light mode is a first pass.
- Main file: **dashboard.py**

## Not included (desktop-only)
The desktop launcher (`investistan_app.py`), auto-update loop (`auto_update.py`),
`live_status.py`, and the `.bat`/`.spec` build files are left out — they're for
the local/packaged version and aren't used by the cloud app.
