# Investistan PSX Screener — Streamlit Cloud

A live PSX (KSE-100) screening dashboard. Runs on Streamlit Community Cloud as a
hosted web app — open the URL on any device, no Python needed.

## Deploy (3 steps)
1. Put this folder in a GitHub repo — either with **GitHub Desktop** (Add Local
   Repository → Publish), or on **github.com** (New repository → "uploading an
   existing file" → drag all these files in → Commit). A private repo is fine.
2. On **Streamlit Community Cloud**: Create app → Deploy from GitHub → pick this
   repo → branch `main` → **Main file path: `dashboard.py`** → (Advanced) Python
   3.12 → Deploy.
3. You get a URL like `https://<name>.streamlit.app`. Open it anywhere.

## How data works on the cloud
There's no separate background process here, so the app generates its own data:
it re-screens the KSE-100 **in-process every ~45 minutes** and caches it. The
included `.csv` / `.json` files are **seed data** so the first load shows results
instantly instead of waiting ~2 minutes for the first scrape.

## Things to know
- Free apps **sleep when idle** — the first visit after a nap is a slow cold start.
- The hosted filesystem is **ephemeral** — watchlist/portfolio edits made in the
  app reset when it restarts.
- The app URL is **publicly viewable** by default.
- Main file: **dashboard.py**

## What's NOT here (on purpose)
The desktop launcher (`investistan_app.py`), the auto-update loop
(`auto_update.py`), the `.bat`/`.spec` build files and dev scripts are left out —
they're for the local/packaged desktop version and aren't used by the cloud.
