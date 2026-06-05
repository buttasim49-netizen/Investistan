"""
PSX Multi-Horizon Screener v2  --  KSE-100 Edition
â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
* Auto-fetches live KSE-100 constituent list from dps.psx.com.pk
  (re-checks every INDEX_CACHE_DAYS days; alerts on recomposition)
* Scrapes fundamentals from psxterminal.com + dps.psx.com.pk
* Supplements with yfinance (.KA) for D/E, beta, P/B
* 24hr fundamental cache  ->  psx_fundamentals.json
* 7-level price fallback: yfinance -> investing.com -> psxterminal -> DPS -> sarmaaya
* Scores each stock for Swing / Long-Term / Very-Long-Term horizons
* Saves ranked results to psx_screener_results.csv
"""

import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import ta
import yfinance as yf
from bs4 import BeautifulSoup

from app_paths import DATA_DIR

# â"€â"€â"€ CONFIG â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
CACHE_FILE        = DATA_DIR / "psx_fundamentals.json"
INDEX_CACHE_FILE  = DATA_DIR / "psx_index_constituents.json"
NEWS_CACHE_FILE   = DATA_DIR / "psx_news_cache.json"
CACHE_HOURS       = 24       # Re-scrape fundamentals if older than this
INDEX_CACHE_DAYS  = 7        # Re-fetch index constituent list after this many days
NEWS_CACHE_HOURS  = 1        # Re-fetch news every hour
REQUEST_DELAY     = 0.3      # Seconds between stocks (parallelism handles per-source delay)
SCRAPE_WORKERS    = 8        # Parallel workers for fundamentals scraping
PRICE_WORKERS     = 12       # Parallel workers for price data fetching
KSESTOCKS_DELAY   = 0.2      # Seconds between ksestocks day requests
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# â"€â"€â"€ FALLBACK KSE-100 LIST (used if live fetch fails) â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
# Last verified: Feb 2026 recomposition notice (psx.com.pk)
# This is only a safety net -- the script always tries to fetch live first.
KSE100_FALLBACK = [
    'AIRLINK','AKBL','ANL','ATRL','AVN','BAFL','BAHL','BOP','BNWM',
    'CHCC','COLG','CYAN','DCR','DGKC','EFERT','EFOODS','ELSA','ENGROH',
    'ENGRO','FCCL','FFC','FFBL','GADT','GAL','GATM','GHNI','GLAXO',
    'GRAYS','GTYR','HBL','HCAR','HGFA','HINOON','HUBC','INIL','ISL',
    'JVDC','KAPCO','KOHC','LUCK','LUMS','MARI','MCB','MEBL','MLCF',
    'MTL','MUREB','NBP','NCL','NESTLE','NML','NRL','NRSL','OGDC','PAEL',
    'PAKOXY','PAKT','PIAA','PIOC','PKGS','POL','PPL','PSO','PSYL',
    'RAVI','SAZEW','SEARL','SGPL','SHEL','SIEM','SMBL','SNGP','SSGC',
    'STPL','SYS','TELE','THCCL','TRG','UBL','UNITY','WAVES','YOUW','ZIL'
]

# â"€â"€â"€ LIVE INDEX CONSTITUENT FETCHER â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

def fetch_kse100_constituents() -> list[str]:
    """
    Fetch the current KSE-100 constituent list from dps.psx.com.pk/indices/KSE100.
    Caches the result in psx_index_constituents.json for INDEX_CACHE_DAYS days.
    On recomposition (list changes), prints a diff so you know what changed.
    Falls back to KSE100_FALLBACK if the live fetch fails.
    """
    # â"€â"€ Load cached list â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    cached_entry = {}
    if INDEX_CACHE_FILE.exists():
        try:
            with open(INDEX_CACHE_FILE, "r") as f:
                cached_entry = json.load(f)
        except Exception:
            pass

    cached_tickers  = cached_entry.get("tickers", [])
    cached_at_str   = cached_entry.get("fetched_at", "")
    cache_fresh     = False
    if cached_at_str:
        try:
            age = datetime.now() - datetime.fromisoformat(cached_at_str)
            cache_fresh = age < timedelta(days=INDEX_CACHE_DAYS)
        except Exception:
            pass

    if cache_fresh and cached_tickers:
        print(f"[Index] Using cached KSE-100 list "
              f"({len(cached_tickers)} tickers, fetched {cached_at_str[:10]})")
        return cached_tickers

    # â"€â"€ Fetch live list â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    print("[Index] Fetching live KSE-100 constituent list from dps.psx.com.pk ...")
    live_tickers = _scrape_index_tickers("KSE100")

    if not live_tickers:
        print("[Index] WARNING: Live fetch failed -- falling back to hardcoded list.")
        return KSE100_FALLBACK

    # â"€â"€ Detect recomposition â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    if cached_tickers:
        added   = sorted(set(live_tickers) - set(cached_tickers))
        removed = sorted(set(cached_tickers) - set(live_tickers))
        if added or removed:
            print("\n" + "!"*60)
            print("  KSE-100 RECOMPOSITION DETECTED")
            print("!"*60)
            if added:
                print(f"  ADDED   (+{len(added)}): {', '.join(added)}")
            if removed:
                print(f"  REMOVED (-{len(removed)}): {', '.join(removed)}")
            print("  Fundamental cache entries for removed tickers kept for reference.")
            print("!"*60 + "\n")
        else:
            print(f"[Index] KSE-100 composition unchanged ({len(live_tickers)} tickers).")
    else:
        print(f"[Index] First-time fetch: {len(live_tickers)} tickers loaded.")

    # â"€â"€ Save updated cache â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    with open(INDEX_CACHE_FILE, "w") as f:
        json.dump({
            "index":      "KSE100",
            "tickers":    live_tickers,
            "fetched_at": datetime.now().isoformat(),
            "count":      len(live_tickers)
        }, f, indent=2)

    return live_tickers


def _scrape_index_tickers(index_code: str) -> list[str]:
    """
    Scrape ticker symbols from dps.psx.com.pk/indices/{index_code}.
    Returns a sorted list of uppercase ticker strings, or [] on failure.
    """
    url = f"https://dps.psx.com.pk/indices/{index_code}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        if resp.status_code != 200:
            print(f"[Index] HTTP {resp.status_code} from {url}")
            return []

        soup = BeautifulSoup(resp.text, "html.parser")

        # Strategy 1: look for ticker-style links (/company/SYMBOL)
        tickers = set()
        for a in soup.find_all("a", href=True):
            m = re.match(r"/company/([A-Z0-9]{2,10})$", a["href"], re.IGNORECASE)
            if m:
                tickers.add(m.group(1).upper())

        # Strategy 2: look for table cells that look like PSX tickers
        # (2-10 uppercase letters/digits, standalone)
        if len(tickers) < 10:
            text = soup.get_text(separator="\n")
            for line in text.splitlines():
                line = line.strip()
                if re.fullmatch(r"[A-Z]{2,10}", line):
                    tickers.add(line)

        # Strategy 3: parse any <td> or <span> with short uppercase text
        if len(tickers) < 10:
            for tag in soup.find_all(["td", "span", "div"]):
                txt = tag.get_text(strip=True)
                if re.fullmatch(r"[A-Z]{2,10}", txt):
                    tickers.add(txt)

        result = sorted(tickers)
        if len(result) >= 10:
            print(f"[Index] Scraped {len(result)} tickers from {url}")
            return result
        else:
            print(f"[Index] Only found {len(result)} tickers -- likely JS-rendered page.")
            return []

    except Exception as e:
        print(f"[Index] Scrape error: {e}")
        return []

# â"€â"€â"€ SCRAPER â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

def scrape_dps(symbol: str) -> dict:
    """
    Scrape fundamentals AND current price from dps.psx.com.pk/company/{symbol}.
    The current price is used as spot-price fallback for tickers with no yfinance history.
    """
    url = f"https://dps.psx.com.pk/company/{symbol}"
    data = {
        "symbol": symbol, "pe_ratio": None, "eps_ttm": None,
        "eps_growth": None, "gross_margin": None, "net_margin": None,
        "market_cap_000s": None, "price": None,
        "one_year_change_pct": None, "peg_ratio": None,
        "52w_high": None, "52w_low": None,
        "source": "dps.psx.com.pk", "scraped_at": datetime.now().isoformat()
    }

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            print(f"  [{symbol}] HTTP {resp.status_code} from DPS")
            return data
        soup = BeautifulSoup(resp.text, "html.parser")
        text = soup.get_text(separator="\n")

        def extract(pattern, cast=float, default=None):
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                val = m.group(1).replace(",", "").strip()
                try:
                    return cast(val)
                except (ValueError, TypeError):
                    pass
            return default

        data["pe_ratio"]           = extract(r"P/E Ratio[^\d\-]*([\d,\.]+)")
        data["eps_ttm"]            = extract(r"EPS\s*\(?\s*(?:TTM|Trailing)?\s*\)?\s*\**\s*([\d,\.]+)")
        data["eps_growth"]         = extract(r"EPS Growth[^\d\-]*([\-\d,\.]+)")
        data["gross_margin"]       = extract(r"Gross Profit Margin[^\d\-]*([\d,\.]+)")
        data["net_margin"]         = extract(r"Net Profit Margin[^\d\-]*([\d,\.]+)")
        data["market_cap_000s"]    = extract(r"Market Cap[^\d\-]*([\d,\.]+)")
        data["peg_ratio"]          = extract(r"PEG Ratio[^\d\-]*([\d,\.]+)")
        data["one_year_change_pct"]= extract(r"1-Year[^\d\-]*([\-\d,\.]+)")
        data["52w_high"]           = extract(r"52.Week High[^\d\-]*([\d,\.]+)")
        data["52w_low"]            = extract(r"52.Week Low[^\d\-]*([\d,\.]+)")

        # Extract the reporting year shown on DPS page e.g. "2025" in "Annual 2025"
        dps_fy = re.search(r"(?:Annual|FY|Year Ended)[^\d]*(\d{4})", text, re.IGNORECASE)
        if dps_fy:
            data["dps_data_year"] = f"FY{dps_fy.group(1)}"
        else:
            # Look for most recent year in financial tables
            years = re.findall(r"\b(20\d{2})\b", text)
            if years:
                data["dps_data_year"] = f"FY{max(set(years), key=lambda y: int(y))}"

        # Price: try multiple patterns used by DPS page
        for price_pat in [
            r"Rs\.\s*([\d,\.]+)",                             # Rs. 267.67
            r"PKR\s*([\d,\.]+)",                              # PKR 267.67
            r"Share Price[^\d]*([\d,\.]+)",
            r"Current Price[^\d]*([\d,\.]+)",
            r"Last Price[^\d]*([\d,\.]+)",
        ]:
            m = re.search(price_pat, text, re.IGNORECASE)
            if m:
                try:
                    val = float(m.group(1).replace(",", ""))
                    if 0.1 < val < 1_000_000:       # sanity check
                        data["price"] = val
                        break
                except ValueError:
                    pass

        print(f"  [{symbol}] DPS: Price={data['price']} P/E={data['pe_ratio']} "
              f"GM={data['gross_margin']} NM={data['net_margin']}")
    except Exception as e:
        print(f"  [{symbol}] DPS scrape error: {e}")

    return data


def scrape_sarmaaya_price(symbol: str) -> float | None:
    """
    Fallback spot-price scraper from sarmaaya.pk/stocks/{symbol}.
    Returns current price (PKR) or None on failure.
    Sarmaaya is a PSX-authorized data redistributor and covers all listed stocks
    including ENGROH, GAL, and others missing from Yahoo Finance.
    """
    url = f"https://sarmaaya.pk/stocks/{symbol}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        text = soup.get_text(separator="\n")

        # Sarmaaya shows "PKR 267.67" or plain numbers near price labels
        for pat in [
            r"PKR\s*([\d,\.]+)",
            r"Rs\.?\s*([\d,\.]+)",
            r"(?:Current Price|Last Price|Share Price)[^\d]*([\d,\.]+)",
            # Next.js hydration sometimes embeds price as a bare number after symbol
            rf"{re.escape(symbol)}\s*[\|:\-]?\s*([\d,\.]+)",
        ]:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                try:
                    val = float(m.group(1).replace(",", ""))
                    if 0.1 < val < 1_000_000:
                        return val
                except ValueError:
                    pass
    except Exception as e:
        print(f"  [{symbol}] Sarmaaya scrape error: {e}")
    return None


def scrape_psxterminal_financials(symbol: str) -> dict:
    """
    Scrape rich fundamentals from psxterminal.com/financials/{symbol}.
    Provides ROE, Dividend Yield, ROA, FCF, EPS, Revenue Growth -- fields
    that DPS PSX and yfinance rarely populate for PSX tickers.
    Also scrapes current spot price from psxterminal.com/symbol/{symbol}.
    """
    fin_data = {
        "roe": None, "roa": None, "dividend_yield": None,
        "current_ratio": None, "eps_ttm_psx": None,
        "revenue_growth": None, "earnings_growth": None,
        "fcf": None, "pe_ratio_psx": None,
        "data_as_of": None,   # e.g. "FY2025", "FY2025 Q1"
    }

    # â"€â"€ Financials page â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    fin_url = f"https://psxterminal.com/financials/{symbol}"
    try:
        resp = requests.get(fin_url, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            text = soup.get_text(separator="\n")

            def extract(pattern, cast=float, default=None):
                m = re.search(pattern, text, re.IGNORECASE)
                if m:
                    raw = m.group(1).replace(",", "").strip().rstrip("%")
                    try:
                        return cast(raw)
                    except (ValueError, TypeError):
                        pass
                return default

            fin_data["pe_ratio_psx"]   = extract(r"P/E\s*(?:\(TTM\))?[^\d\-]*([\d,\.]+)")
            fin_data["roe"]            = extract(r"ROE[^\d\-]*([\-\d,\.]+)")
            fin_data["roa"]            = extract(r"ROA[^\d\-]*([\-\d,\.]+)")
            fin_data["dividend_yield"] = extract(r"Dividend Yield[^\d\-]*([\-\d,\.]+)")
            fin_data["current_ratio"]  = extract(r"Current Ratio[^\d\-]*([\d,\.]+)")
            fin_data["eps_ttm_psx"]    = extract(r"EPS[^\d\-]*([\d,\.]+)")

            # Revenue growth: compare most recent two years from text numbers
            # Look for "Revenue" followed by a series of numbers
            rev_matches = re.findall(
                r"(?:Revenue|Net Sales)[^\n]*\n\s*([\d,\.]+)[^\n]*\n\s*([\d,\.]+)",
                text, re.IGNORECASE
            )
            if rev_matches:
                try:
                    rev_new = float(rev_matches[0][0].replace(",", ""))
                    rev_old = float(rev_matches[0][1].replace(",", ""))
                    if rev_old > 0:
                        fin_data["revenue_growth"] = round(
                            (rev_new - rev_old) / rev_old * 100, 1
                        )
                except (ValueError, ZeroDivisionError):
                    pass

            # Earnings growth: YoY EPS change
            eps_matches = re.findall(
                r"EPS[^\n]*\n\s*([\-\d,\.]+)[^\n]*\n\s*([\-\d,\.]+)",
                text, re.IGNORECASE
            )
            if eps_matches:
                try:
                    eps_new = float(eps_matches[0][0].replace(",", ""))
                    eps_old = float(eps_matches[0][1].replace(",", ""))
                    if eps_old and eps_old != 0:
                        fin_data["earnings_growth"] = round(
                            (eps_new - eps_old) / abs(eps_old) * 100, 1
                        )
                except (ValueError, ZeroDivisionError):
                    pass

            # Extract fiscal year from page (PSXTerminal shows e.g. "FY2025" or "2025")
            fy_match = re.search(r"FY\s*(\d{4})", text, re.IGNORECASE)
            if not fy_match:
                fy_match = re.search(r"(?:Annual|Year)[^\d]*(\d{4})", text, re.IGNORECASE)
            if fy_match:
                fin_data["data_as_of"] = f"FY{fy_match.group(1)}"

            # Also look for quarterly period
            qtr_match = re.search(r"Q([1-4])\s*(\d{4})", text, re.IGNORECASE)
            if qtr_match:
                fin_data["data_as_of"] = f"FY{qtr_match.group(2)} Q{qtr_match.group(1)} (TTM)"

            populated = sum(1 for k,v in fin_data.items() if v is not None and k != "data_as_of")
            print(f"  [{symbol}] PSXTerminal financials: {populated}/9 fields "
                  f"| ROE={fin_data['roe']} DY={fin_data['dividend_yield']}% "
                  f"ROA={fin_data['roa']} | As of: {fin_data['data_as_of']}")
        else:
            print(f"  [{symbol}] PSXTerminal HTTP {resp.status_code}")
    except Exception as e:
        print(f"  [{symbol}] PSXTerminal financials error: {e}")

    return fin_data


def scrape_psxterminal_price(symbol: str) -> float | None:
    """
    Spot price from psxterminal.com/symbol/{symbol}.
    Covers ALL PSX-listed stocks including ENGROH and GAL.
    Parses price from labelled fields only -- avoids picking up dividend amounts.
    """
    url = f"https://psxterminal.com/symbol/{symbol}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        text = soup.get_text(separator="\n")

        # Only match explicitly labelled price fields to avoid dividend/EPS confusion
        for pat in [
            r"(?:Current Price|Last Price|Share Price|Close Price|Closing Price)[^\d\n]{0,10}([\d,\.]+)",
            r"(?:^|\n)\s*Price\s*[:\-]?\s*([\d,\.]+)",
            # Match price with change: "267.67 +7.54"  -- first number after symbol header
            rf"(?:{re.escape(symbol)})[^\d\n]{{0,30}}([\d,\.]+)",
        ]:
            for m in re.finditer(pat, text, re.IGNORECASE | re.MULTILINE):
                try:
                    val = float(m.group(1).replace(",", ""))
                    # Stock prices on PSX are generally > 5 PKR and < 100,000
                    if 5 < val < 100_000:
                        return val
                except ValueError:
                    pass
    except Exception as e:
        print(f"  [{symbol}] PSXTerminal price error: {e}")
    return None


def fetch_yfinance_fundamentals(symbol: str) -> dict:
    """Fetch supplementary fundamentals from yfinance (.KA suffix)"""
    yf_data = {
        "debt_equity": None, "beta": None, "pb_ratio": None,
    }
    try:
        ticker = yf.Ticker(f"{symbol}.KA")
        info   = ticker.info or {}
        yf_data["debt_equity"] = info.get("debtToEquity")
        yf_data["beta"]        = info.get("beta")
        yf_data["pb_ratio"]    = info.get("priceToBook")
        populated = sum(1 for v in yf_data.values() if v is not None)
        print(f"  [{symbol}] yfinance: {populated}/3 fields populated")
    except Exception as e:
        print(f"  [{symbol}] yfinance error: {e}")
    return yf_data


# â"€â"€â"€ CACHE â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

def load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_cache(data: dict):
    with open(CACHE_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\n[Cache] Saved to {CACHE_FILE}")


def is_stale(record: dict) -> bool:
    scraped = record.get("scraped_at")
    if not scraped:
        return True
    try:
        age = datetime.now() - datetime.fromisoformat(scraped)
        return age > timedelta(hours=CACHE_HOURS)
    except Exception:
        return True


def _fetch_one_symbol_fundamentals(sym: str) -> tuple[str, dict]:
    """Fetch and merge all fundamental sources for a single symbol. Thread-safe."""
    psx_t = scrape_psxterminal_financials(sym)
    dps   = scrape_dps(sym)
    scs   = scrape_scstrade_fundamentals(sym)
    yf_f  = fetch_yfinance_fundamentals(sym)

    merged = {**yf_f, **dps, **psx_t}

    # Fill gaps from scstrade
    for field, scs_key in [
        ("debt_equity",   "debt_equity_scs"),
        ("dividend_yield","dividend_yield_scs"),
        ("roe",           "roe_scs"),
        ("pb_ratio",      "pb_ratio_scs"),
    ]:
        if not merged.get(field) and scs.get(scs_key) is not None:
            merged[field] = scs[scs_key]

    if not merged.get("price") and dps.get("price"):
        merged["price"] = dps["price"]
    if psx_t.get("pe_ratio_psx") and (
        not dps.get("pe_ratio") or dps.get("pe_ratio", 0) > 500
    ):
        merged["pe_ratio"] = psx_t["pe_ratio_psx"]

    merged["scraped_at"] = datetime.now().isoformat()

    # Consolidate reporting period from all sources (SCS is most granular)
    period = (
        merged.get("scs_data_period") or
        merged.get("data_as_of") or
        merged.get("dps_data_year") or
        "Unknown"
    )
    merged["data_as_of"] = period

    return sym, merged


def get_fundamentals(symbols: list) -> dict:
    """
    Return merged fundamentals for all symbols, using cache where fresh.
    Stale symbols are fetched in parallel (SCRAPE_WORKERS threads) so
    100 stocks complete in ~2 min instead of ~12 min.

    Data priority (highest wins on conflict):
      1. psxterminal.com  -- ROE, ROA, DivYield, CurrentRatio, EPS, Rev/EPS growth
      2. dps.psx.com.pk   -- P/E, GrossMargin, NetMargin, Price, 52w range
      3. scstrade.com     -- D/E, DivYield fallback, ROE fallback
      4. yfinance (.KA)   -- Beta, P/B
    """
    cache  = load_cache()
    result = {}
    to_fetch = []

    for sym in symbols:
        cached = cache.get(sym, {})
        if cached and not is_stale(cached):
            result[sym] = cached
        else:
            to_fetch.append(sym)

    if to_fetch:
        print(f"  Scraping {len(to_fetch)} stocks in parallel "
              f"({SCRAPE_WORKERS} workers)...")

        with ThreadPoolExecutor(max_workers=SCRAPE_WORKERS) as ex:
            futures = {ex.submit(_fetch_one_symbol_fundamentals, sym): sym
                       for sym in to_fetch}
            done = 0
            for fut in as_completed(futures):
                sym, merged = fut.result()
                result[sym]  = merged
                cache[sym]   = merged
                done += 1
                print(f"  [{sym}] done ({done}/{len(to_fetch)}) "
                      f"ROE={merged.get('roe')} D/E={merged.get('debt_equity')} "
                      f"DY={merged.get('dividend_yield')}")
    else:
        print(f"  All {len(symbols)} stocks loaded from cache (< {CACHE_HOURS}h old).")

    save_cache(cache)
    return result


# --- YAHOO FINANCE TICKER ALIASES --------------------------------------------
# Some PSX tickers are listed under a different symbol on Yahoo Finance.
# ENGROH (Engro Holdings) was formerly Dawood Hercules > Yahoo uses DAWH.KA
# Add any other mismatches here as you discover them.
YAHOO_ALIASES = {
    'ENGROH': 'DAWH',   # Engro Holdings Ltd (formerly Dawood Hercules Corp)
}


# --- INVESTING.COM SLUG MAP (PSX ticker -> investing.com URL slug) ------------
# Used as fallback when yfinance has no data for a ticker.
# Extend this dict as needed for other PSX tickers.
INVESTING_SLUGS = {
    'AIRLINK': 'air-link-communication',
    'ATRL':    'attock-refiner',
    'BOP':     'bank-of-punjab',
    'DGKC':    'd.g.-khan-ceme',
    'EFERT':   'engro-fertilizers-ltd',
    'ENGROH':  'dawood-hercule',       # Engro Holdings (formerly Dawood Hercules)
    'FCCL':    'fauji-cement',
    'FFC':     'fauji-fertiliz',
    'GAL':     'ghandara-n.-lt',       # Ghandhara Automobiles
    'GHNI':    'ghandhara-ind',
    'HBL':     'habib-bank-ltd',
    'HUBC':    'hub-power-co-l',
    'LUCK':    'lucky-cement-l',
    'MARI':    'mari-gas',
    'MCB':     'mcb-bank',
    'MEBL':    'meezan-bank',
    'MLCF':    'maple-leaf-cmn',
    'NBP':     'national-bank',
    'NRL':     'attock-refiner',       # National Refinery -- verify slug if needed
    'OGDC':    'oil---gas-dev',
    'PAEL':    'pak-electron',
    'POL':     'pak-oilfields',
    'PPL':     'pak-petroleum',
    'PSO':     'pak-state-oil',
    'SAZEW':   'sazgar-enginee',
    'SEARL':   'searl-company',
    'SNGP':    'sui-northern-g',
    'SSGC':    'sui-southern-g',
    'SYS':     'systems-ltd',
    'UBL':     'united-bank-lt',
}


def _scrape_investing_history(slug: str, months: int = 24) -> pd.DataFrame | None:
    """
    Scrape OHLCV history from investing.com/{slug}-historical-data.
    Returns a DataFrame with columns [Open, High, Low, Close, Volume]
    indexed by date, or None on failure.

    Note: investing.com serves only ~1 month of data in the static HTML.
    We make multiple monthly requests using their time-filter form to build
    up to `months` months of history, sleeping between requests.
    """
    from datetime import date
    import calendar

    base_url = f"https://www.investing.com/equities/{slug}-historical-data"
    all_rows = []
    session  = requests.Session()
    session.headers.update(HEADERS)

    # First: get the page to grab any hidden form tokens (if needed)
    try:
        resp = session.get(base_url, timeout=15)
        if resp.status_code == 403:
            print(f"    investing.com blocked request (403) for {slug}")
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        rows = _parse_investing_table(soup)
        all_rows.extend(rows)
    except Exception as e:
        print(f"    investing.com fetch error for {slug}: {e}")
        return None

    if not all_rows:
        return None

    df = pd.DataFrame(all_rows)
    if df.empty:
        return None

    df["Date"]   = pd.to_datetime(df["Date"], dayfirst=False, errors="coerce")
    df           = df.dropna(subset=["Date"])
    df           = df.set_index("Date").sort_index()
    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ""), errors="coerce"
            )
    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(
            df["Volume"].astype(str).str.replace(",", "")
                        .str.replace("K", "e3").str.replace("M", "e6")
                        .str.replace("B", "e9").str.replace("-", "0"),
            errors="coerce"
        ).fillna(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    print(f"    investing.com: {len(df)} rows scraped for {slug}")
    return df if not df.empty else None


def _parse_investing_table(soup: BeautifulSoup) -> list:
    """Parse the historical data table from an investing.com page."""
    rows = []
    # investing.com uses a <table> with id="curr_table" or class containing "genTbl"
    table = (
        soup.find("table", {"id": "curr_table"}) or
        soup.find("table", class_=lambda c: c and "historicalTbl" in c) or
        soup.find("table", class_=lambda c: c and "genTbl" in c)
    )
    if table is None:
        # Try any table with Date/Price headers
        for t in soup.find_all("table"):
            headers = [th.get_text(strip=True) for th in t.find_all("th")]
            if "Date" in headers and ("Price" in headers or "Close" in headers):
                table = t
                break

    if table is None:
        return rows

    headers = [th.get_text(strip=True) for th in table.find_all("th")]
    # Normalize header names
    col_map = {}
    for i, h in enumerate(headers):
        hl = h.lower()
        if "date"   in hl: col_map[i] = "Date"
        elif "price" in hl or "close" in hl: col_map[i] = "Close"
        elif "open"  in hl: col_map[i] = "Open"
        elif "high"  in hl: col_map[i] = "High"
        elif "low"   in hl: col_map[i] = "Low"
        elif "vol"   in hl: col_map[i] = "Volume"

    for tr in table.find_all("tr")[1:]:
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if not cells:
            continue
        row = {}
        for i, val in enumerate(cells):
            if i in col_map:
                row[col_map[i]] = val
        if "Date" in row and "Close" in row:
            # Fill missing OHLCV with Close
            for k in ["Open", "High", "Low"]:
                row.setdefault(k, row["Close"])
            row.setdefault("Volume", "0")
            rows.append(row)
    return rows


def _get_investing_spot_price(slug: str) -> float | None:
    """Get only the current price from investing.com (no history needed)."""
    try:
        url  = f"https://www.investing.com/equities/{slug}"
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        text = soup.get_text()
        # investing.com embeds price in multiple ways; try common patterns
        m = re.search(
            r'"last[_\s]?(?:price|value)"[:\s]+"?([\d,\.]+)"?', text, re.IGNORECASE
        )
        if not m:
            m = re.search(r'(?:Last Price|Current Price)[^\d]*([\d,\.]+)', text)
        if m:
            return float(m.group(1).replace(",", ""))
    except Exception:
        pass
    return None


# --- NEWS MODULE -----------------------------------------------------------------
# Sources:
#   1. Dawn Business  -- RSS feed (clean XML)
#   2. Mettis Global  -- HTML scrape (/Equity + /latest)
#   3. Profit PK      -- HTML scrape (fallback if unblocked)
#
# News is cached for NEWS_CACHE_HOURS (1hr default).
# Each stock gets matched against headlines by ticker + company name keywords.
# Sentiment: +1 positive word, -1 negative word per headline.
# ---------------------------------------------------------------------------------

# Keywords that map sectors/themes to likely stock tickers
SECTOR_KEYWORDS = {
    "fertilizer": ["FFC", "EFERT", "FATIMA", "FFBL"],
    "cement":     ["LUCK", "DGKC", "CHCC", "MLCF", "FCCL", "PIOC", "KOHC"],
    "bank":       ["HBL", "MCB", "UBL", "MEBL", "BAHL", "NBP", "ABL", "BAFL"],
    "oil":        ["PSO", "APL", "HASCOL"],
    "gas":        ["SNGP", "SSGC"],
    "power":      ["HUBC", "KAPCO", "NPML", "NCPL", "KEL"],
    "refinery":   ["ATRL", "NRL", "CNERGY"],
    "pharma":     ["SEARL", "GLAXO", "ABOT", "AGP", "HINOON"],
    "tech":       ["SYS", "TRG", "AIRLINK", "AVN"],
    "auto":       ["INDU", "HCAR", "ATLH", "SAZEW", "GAL"],
    "steel":      ["ISL", "INIL", "MUGHAL"],
    "textile":    ["NML", "GATM", "KTML"],
    "exploration":["OGDC", "PPL", "POL", "MARI"],
    "sbp":        ["HBL", "MCB", "UBL", "MEBL", "NBP"],
    "imf":        ["HBL", "MCB", "UBL", "NBP", "PSO", "OGDC"],
    "budget":     ["FFC", "EFERT", "FATIMA", "PSO", "OGDC", "PPL"],
    "dividend":   [],   # matched by ticker only
    "kse":        [],   # general market
    "psx":        [],   # general market
}

POSITIVE_WORDS = {
    "profit", "growth", "record", "rise", "surge", "rally", "gain", "upgrade",
    "dividend", "expansion", "strong", "beat", "outperform", "buy", "bullish",
    "increase", "approval", "contract", "deal", "acquisition", "award",
    "recovery", "rebound", "high", "peak", "milestone", "positive", "improve",
}

NEGATIVE_WORDS = {
    "loss", "decline", "fall", "drop", "slide", "default", "penalty", "fine",
    "investigation", "downgrade", "sell", "bearish", "cut", "reduce", "halt",
    "suspension", "delisted", "deficit", "debt", "inflation", "tariff", "tax",
    "miss", "weak", "low", "crash", "plunge", "concern", "risk", "warning",
}

# Company name -> ticker mapping for news matching
COMPANY_TO_TICKER = {
    "fauji fertilizer": "FFC", "engro fertilizers": "EFERT",
    "fatima fertilizer": "FATIMA", "hub power": "HUBC",
    "habib bank": "HBL", "mcb bank": "MCB", "united bank": "UBL",
    "meezan bank": "MEBL", "bank al habib": "BAHL",
    "national bank": "NBP", "allied bank": "ABL",
    "oil and gas development": "OGDC", "pakistan petroleum": "PPL",
    "pakistan oilfields": "POL", "mari petroleum": "MARI",
    "lucky cement": "LUCK", "d.g. khan cement": "DGKC",
    "cherat cement": "CHCC", "maple leaf": "MLCF",
    "fauji cement": "FCCL", "kohat cement": "KOHC",
    "pakistan state oil": "PSO", "attock petroleum": "APL",
    "systems limited": "SYS", "trg pakistan": "TRG",
    "air link": "AIRLINK", "avanceon": "AVN",
    "indus motor": "INDU", "honda atlas": "HCAR",
    "atlas honda": "ATLH", "sazgar": "SAZEW",
    "ghandhara automobiles": "GAL",
    "searle": "SEARL", "glaxo": "GLAXO", "abbott": "ABOT",
    "sui northern": "SNGP", "sui southern": "SSGC",
    "k-electric": "KEL", "kot addu": "KAPCO",
    "international steels": "ISL", "mughal": "MUGHAL",
    "nishat mills": "NML", "gul ahmed": "GATM",
    "nestle": "NESTLE", "colgate": "COLG",
    "engro holdings": "ENGROH", "dawood hercules": "ENGROH",
}


def _sentiment_score(text: str) -> int:
    """Return +1 (positive), -1 (negative), or 0 (neutral) for a headline."""
    words = set(re.findall(r"\w+", text.lower()))
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    if pos > neg:   return 1
    if neg > pos:   return -1
    return 0


def _match_tickers(text: str, all_symbols: list) -> set:
    """Return set of ticker symbols mentioned or implied in a headline."""
    matched = set()
    tl = text.lower()

    # Direct ticker mention (e.g. "FFC", "HUBC")
    for sym in all_symbols:
        if re.search(r"\b" + sym + r"\b", text, re.IGNORECASE):
            matched.add(sym)

    # Company name match
    for name, ticker in COMPANY_TO_TICKER.items():
        if name in tl:
            matched.add(ticker)

    # Sector/theme keywords
    for keyword, tickers in SECTOR_KEYWORDS.items():
        if keyword in tl:
            matched.update(tickers)

    return matched


def fetch_news(all_symbols: list = None) -> list:
    """
    Fetch latest Pakistan financial news from:
      - Dawn Business RSS feed
      - Mettis Global Equity + Latest pages

    Returns list of dicts:
      {title, url, source, date, sentiment, tickers}

    Cached for NEWS_CACHE_HOURS to avoid hammering sources on every run.
    """
    if all_symbols is None:
        all_symbols = []

    # --- Check cache ---
    if NEWS_CACHE_FILE.exists():
        try:
            with open(NEWS_CACHE_FILE, "r", encoding="utf-8") as f:
                cached = json.load(f)
            age = datetime.now() - datetime.fromisoformat(cached.get("fetched_at", "2000-01-01"))
            if age < timedelta(hours=NEWS_CACHE_HOURS):
                print(f"  [News] Using cached news ({len(cached['articles'])} articles, "
                      f"{int(age.total_seconds()/60)}m old)")
                return cached["articles"]
        except Exception:
            pass

    articles = []

    # --- 1. Dawn Business RSS ---
    try:
        resp = requests.get("https://www.dawn.com/feeds/business",
                            headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            root = ET.fromstring(resp.text)
            ns = {"atom": "http://www.w3.org/2005/Atom",
                  "media": "http://search.yahoo.com/mrss/"}
            channel = root.find("channel")
            items   = channel.findall("item") if channel else root.findall(".//item")
            for item in items[:20]:
                title = (item.findtext("title") or "").strip()
                url   = (item.findtext("link")  or "").strip()
                date  = (item.findtext("pubDate") or "").strip()
                if title:
                    articles.append({
                        "title":     title,
                        "url":       url,
                        "source":    "Dawn",
                        "date":      date,
                        "sentiment": _sentiment_score(title),
                        "tickers":   list(_match_tickers(title, all_symbols)),
                    })
            print(f"  [News] Dawn: {len([a for a in articles if a['source']=='Dawn'])} articles")
    except Exception as e:
        print(f"  [News] Dawn error: {e}")

    # --- 2. Mettis Global ---
    # Article URLs on Mettis always end in a numeric ID like "-60767"
    # Navigation/footer links don't — this is the key filter.
    METTIS_ARTICLE_PATTERN = re.compile(r"-\d{4,6}$")
    METTIS_SKIP = ["login","register","subscribe","cookie","portfolio management",
                   "reach your target","contact our","editorial board","advertise",
                   "about us","privacy policy","terms","careers","follow us"]

    for url, label in [
        ("https://mettisglobal.news/Equity",     "Mettis"),
        ("https://mettisglobal.news/PSXRoundup", "Mettis"),
        ("https://mettisglobal.news/latest",     "Mettis"),
        ("https://mettisglobal.news/Economy",    "Mettis"),
    ]:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            if resp.status_code == 200:
                soup  = BeautifulSoup(resp.text, "html.parser")
                count = 0
                for a_tag in soup.find_all("a", href=True):
                    href  = a_tag["href"]
                    full_url = href if href.startswith("http") else f"https://mettisglobal.news{href}"

                    # Must match Mettis article URL pattern (ends in -NNNNN)
                    path = full_url.rstrip("/").split("?")[0]
                    if not METTIS_ARTICLE_PATTERN.search(path):
                        continue

                    # Clean up title -- take only first sentence if duplicated
                    raw_title = a_tag.get_text(separator=" ", strip=True)
                    # Mettis sometimes doubles the text: "Title Title" -> take first half
                    half = len(raw_title) // 2
                    if half > 10 and raw_title[:half].strip() == raw_title[half:].strip():
                        raw_title = raw_title[:half].strip()
                    title = raw_title[:150]  # cap length

                    if len(title) < 15:
                        continue
                    if any(skip in title.lower() for skip in METTIS_SKIP):
                        continue

                    articles.append({
                        "title":     title,
                        "url":       full_url,
                        "source":    "Mettis Global",
                        "date":      datetime.now().strftime("%Y-%m-%d"),
                        "sentiment": _sentiment_score(title),
                        "tickers":   list(_match_tickers(title, all_symbols)),
                    })
                    count += 1
                    if count >= 25:
                        break
                if count:
                    print(f"  [News] Mettis ({url.split('/')[-1]}): {count} articles")
        except Exception as e:
            print(f"  [News] Mettis error ({url}): {e}")

    # --- 3. Profit Pakistan Today (try multiple entry points) ---
    profit_urls = [
        "https://profit.pakistantoday.com.pk/category/business/",
        "https://profit.pakistantoday.com.pk/category/stocks/",
        "https://profit.pakistantoday.com.pk",
    ]
    profit_headers = {**HEADERS,
                      "Referer": "https://www.google.com/",
                      "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}
    for profit_url in profit_urls:
        try:
            resp = requests.get(profit_url, headers=profit_headers, timeout=12)
            if resp.status_code == 200:
                soup  = BeautifulSoup(resp.text, "html.parser")
                count = 0
                for a_tag in soup.find_all("a", href=True):
                    title = a_tag.get_text(strip=True)
                    href  = a_tag["href"]
                    if (len(title) > 20 and
                            "profit.pakistantoday.com.pk" in href and
                            re.search(r"/20\d\d/", href)):   # URL contains /2024/, /2025/ etc
                        articles.append({
                            "title":     title[:150],
                            "url":       href,
                            "source":    "Profit PK",
                            "date":      datetime.now().strftime("%Y-%m-%d"),
                            "sentiment": _sentiment_score(title),
                            "tickers":   list(_match_tickers(title, all_symbols)),
                        })
                        count += 1
                        if count >= 20:
                            break
                if count:
                    print(f"  [News] Profit-PK ({profit_url.split('/')[-2]}): {count} articles")
                    break   # stop trying other URLs if we got articles
            elif resp.status_code == 403:
                continue    # try next URL
        except Exception as e:
            print(f"  [News] Profit-PK error: {e}")
            continue

    # Deduplicate by title
    seen   = set()
    unique = []
    for a in articles:
        key = a["title"][:60].lower()
        if key not in seen:
            seen.add(key)
            unique.append(a)

    # Save cache
    try:
        with open(NEWS_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({"fetched_at": datetime.now().isoformat(),
                       "articles":   unique}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    print(f"  [News] Total: {len(unique)} unique articles fetched")
    return unique


def get_stock_news(symbol: str, company_name: str, all_news: list) -> list:
    """Return news articles relevant to a specific stock."""
    relevant = []
    sym_lower = symbol.lower()
    name_lower = company_name.lower() if company_name else ""

    for article in all_news:
        # Direct ticker match in pre-computed tickers list
        if symbol in article.get("tickers", []):
            relevant.append(article)
            continue
        # Fallback: check title text
        title_lower = article["title"].lower()
        if sym_lower in title_lower or (name_lower and name_lower[:8] in title_lower):
            relevant.append(article)

    return relevant


def market_sentiment(all_news: list) -> dict:
    """
    Compute overall market sentiment from today's headlines.
    Returns {score, label, positive_count, negative_count, key_headlines}
    """
    today = datetime.now().strftime("%Y-%m-%d")
    scores = [a["sentiment"] for a in all_news]
    pos    = scores.count(1)
    neg    = scores.count(-1)
    net    = pos - neg

    if   net >= 5:  label = "BULLISH"
    elif net >= 2:  label = "MILDLY BULLISH"
    elif net <= -5: label = "BEARISH"
    elif net <= -2: label = "MILDLY BEARISH"
    else:           label = "NEUTRAL"

    # Most impactful headlines (highest absolute sentiment or ticker-specific)
    key = sorted(all_news, key=lambda a: abs(a["sentiment"]) * len(a["tickers"]) + abs(a["sentiment"]),
                 reverse=True)[:5]

    return {
        "score":           net,
        "label":           label,
        "positive_count":  pos,
        "negative_count":  neg,
        "neutral_count":   scores.count(0),
        "key_headlines":   key,
    }


# --- KSESTOCKS HISTORY (full OHLCV for any PSX ticker) -----------------------

PRICE_CACHE_FILE = DATA_DIR / "psx_price_cache.json"

def _load_price_cache() -> dict:
    if PRICE_CACHE_FILE.exists():
        try:
            with open(PRICE_CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def _save_price_cache(cache: dict):
    with open(PRICE_CACHE_FILE, "w") as f:
        json.dump(cache, f)

def _ksestocks_day(date_str: str) -> dict:
    """
    Fetch one day's market summary from ksestocks.com/MarketSummary/{date}.
    Returns {symbol: {open,high,low,close,volume}} for all stocks that day.
    Uses a short timeout (5s) so failures don't hang the build.
    """
    url = f"https://ksestocks.com/MarketSummary/{date_str}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=5)  # short timeout
        if resp.status_code != 200:
            return {}
        soup = BeautifulSoup(resp.text, "html.parser")

        # Table 2 is the main data table (0-indexed)
        tables = soup.find_all("table")
        if len(tables) < 3:
            return {}
        table = tables[2]

        # Find the header row -- scan first 5 rows for "Symbol" header
        rows = table.find_all("tr")
        header_row_idx = None
        col_map = {}
        for ri, tr in enumerate(rows[:5]):
            cells = [td.get_text(strip=True).lower()
                     for td in tr.find_all(["th", "td"])]
            if "symbol" in cells and "open" in cells:
                header_row_idx = ri
                for i, h in enumerate(cells):
                    if h == "symbol":          col_map[i] = "symbol"
                    elif h == "company name":  col_map[i] = "company"
                    elif h == "open":          col_map[i] = "open"
                    elif h == "high":          col_map[i] = "high"
                    elif h == "low":           col_map[i] = "low"
                    elif h == "close":         col_map[i] = "close"
                    elif h == "volume":        col_map[i] = "volume"
                break

        if header_row_idx is None or "symbol" not in col_map.values():
            return {}

        # Build reverse map: field_name -> column_index
        field_idx = {v: k for k, v in col_map.items()}
        i_sym = field_idx.get("symbol", 0)
        i_opn = field_idx.get("open",   2)
        i_hgh = field_idx.get("high",   3)
        i_low = field_idx.get("low",    4)
        i_cls = field_idx.get("close",  5)
        i_vol = field_idx.get("volume", 7)

        def _f(cells, idx):
            try:
                return float(cells[idx].replace(",", "") or 0)
            except (IndexError, ValueError):
                return 0.0

        result = {}
        for tr in rows[header_row_idx + 1:]:
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(cells) < 6:
                continue  # sector header rows have fewer cells

            sym = cells[i_sym].strip().upper()
            if not sym or "-" in sym:
                continue  # skip futures (ENGROH-MAY, GAL-JUN, etc.)
            if not re.fullmatch(r"[A-Z][A-Z0-9]{1,11}", sym):
                continue  # skip non-ticker strings

            close_val = _f(cells, i_cls)
            if close_val <= 0:
                continue   # skip rows with no close price

            result[sym] = {
                "open":   _f(cells, i_opn),
                "high":   _f(cells, i_hgh),
                "low":    _f(cells, i_low),
                "close":  close_val,
                "volume": _f(cells, i_vol),
            }
        return result
    except Exception as e:
        return {}


def fetch_ksestocks_history(symbol: str, days: int = 500) -> pd.DataFrame | None:
    """
    Build historical OHLCV for any PSX ticker using ksestocks.com market summaries.
    Caches results in psx_price_cache.json -- only fetches dates not yet stored.
    On first call for a missing ticker: fetches ~500 trading days (~4 min).
    On subsequent calls: fetches only today (instant).
    """
    cache     = _load_price_cache()
    sym_cache = cache.get(symbol, {})   # {date_str: {o,h,l,c,v}}

    # Only go back as far as needed for technical analysis (250 trading days = 1yr)
    # We work BACKWARDS from today so the most recent days are fetched first --
    # if we get enough we stop early rather than wasting time on old dates.
    TARGET_DAYS = 250
    end_date    = pd.Timestamp.now().normalize()
    start_date  = end_date - pd.Timedelta(days=int(TARGET_DAYS * 1.6))
    bdays       = list(reversed(pd.bdate_range(start_date, end_date).tolist()))
    needed      = [d.strftime("%Y-%m-%d") for d in bdays
                   if d.strftime("%Y-%m-%d") not in sym_cache]

    if needed:
        already = len(sym_cache)
        print(f"  [{symbol}] ksestocks: {already} days cached, "
              f"fetching up to {len(needed)} more (target {TARGET_DAYS} days)...")
        fetched       = 0
        consecutive_empty = 0
        for date_str in needed:
            # Stop once we have enough data for all indicators
            if len(sym_cache) >= TARGET_DAYS:
                break
            # Give up if ksestocks keeps returning nothing (likely rate-limited)
            if consecutive_empty >= 10:
                print(f"  [{symbol}] ksestocks: 10 consecutive empty responses -- stopping")
                break

            day_data = _ksestocks_day(date_str)
            if symbol in day_data and day_data[symbol]["close"] > 0:
                sym_cache[date_str] = day_data[symbol]
                fetched += 1
                consecutive_empty = 0
                # Save progress every 25 days so work isn't lost if interrupted
                if fetched % 25 == 0:
                    cache[symbol] = sym_cache
                    _save_price_cache(cache)
                    print(f"  [{symbol}] ksestocks: {len(sym_cache)} days so far...")
            else:
                consecutive_empty += 1
            time.sleep(KSESTOCKS_DELAY)

        cache[symbol] = sym_cache
        _save_price_cache(cache)
        print(f"  [{symbol}] ksestocks: done -- {len(sym_cache)} total days cached")

    if not sym_cache:
        return None

    # Build DataFrame
    rows = []
    for date_str, bar in sorted(sym_cache.items()):
        rows.append({
            "Date":   pd.Timestamp(date_str),
            "Open":   bar["open"],
            "High":   bar["high"],
            "Low":    bar["low"],
            "Close":  bar["close"],
            "Volume": bar["volume"],
        })

    df = pd.DataFrame(rows).set_index("Date").sort_index()
    df = df[df["Close"] > 0]
    return df if len(df) >= 30 else None


# â"€â"€â"€ SCSTRADE FUNDAMENTALS (fills D/E, dividend yield, ROE gaps) â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

def scrape_scstrade_fundamentals(symbol: str) -> dict:
    """
    Scrape fundamental ratios from scstrade.com company snapshot page.
    Fills fields that PSXTerminal/DPS/yfinance miss: D/E, dividend yield, ROE.
    """
    url = f"https://scstrade.com/stockscreening/SS_CompanySnapShot.aspx?symbol={symbol}"
    data = {
        "roe_scs": None, "debt_equity_scs": None,
        "dividend_yield_scs": None, "pb_ratio_scs": None,
        "eps_scs": None, "pe_ratio_scs": None,
        "book_value_scs": None,
    }
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return data
        soup = BeautifulSoup(resp.text, "html.parser")
        text = soup.get_text(separator="\n")

        def extract(pattern):
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                try:
                    return float(m.group(1).replace(",", "").strip().rstrip("%x"))
                except (ValueError, TypeError):
                    pass
            return None

        # SCS page format: "Return On Equity Upto 2025 4Q  ...  31.41 %"
        # Strategy: find the label, then in the following 400 chars look for the
        # FIRST number with a decimal point followed by the unit (% or x or Rs).
        # Requiring a decimal rules out years (2025, 2026) and quarter codes (4Q).

        def extract_scs_near(label: str, unit: str = r"%", max_chars: int = 400) -> float | None:
            m = re.search(label, text, re.IGNORECASE)
            if not m:
                return None
            snippet = text[m.end(): m.end() + max_chars]
            # Find decimal number followed by unit
            hit = re.search(r"(\d{1,5}\.\d{1,4})\s*" + unit, snippet, re.IGNORECASE)
            if hit:
                try:
                    return float(hit.group(1).replace(",", ""))
                except (ValueError, TypeError):
                    pass
            return None

        data["roe_scs"]            = extract_scs_near(r"Return On Equity",    r"%")
        data["debt_equity_scs"]    = extract_scs_near(r"Total Debt To Equity", r"%")
        data["dividend_yield_scs"] = extract_scs_near(r"Dividend Yield",      r"%")
        data["pe_ratio_scs"]       = extract_scs_near(r"Price To Earning",    r"x")
        data["book_value_scs"]     = extract_scs_near(r"Book Value",          r"")
        data["eps_scs"]            = extract_scs_near(r"Latest EPS",          r"")
        data["pb_ratio_scs"]       = extract_scs_near(r"Price[/\s]Book",      r"x")

        # SCS shows period as "Upto 2025 4Q" or "Upto 2026 1Q"
        period_m = re.search(r"Upto\s+(\d{4})\s+(\d)Q", text, re.IGNORECASE)
        if period_m:
            data["scs_data_period"] = f"FY{period_m.group(1)} Q{period_m.group(2)}"
        else:
            yr_m = re.search(r"Upto\s+(\d{4})", text, re.IGNORECASE)
            if yr_m:
                data["scs_data_period"] = f"FY{yr_m.group(1)}"

        populated = sum(1 for k,v in data.items()
                        if v is not None and k not in ("scs_data_period",))
        if populated:
            print(f"  [{symbol}] SCS Trade: {populated}/7 fields "
                  f"| ROE={data['roe_scs']} D/E={data['debt_equity_scs']} "
                  f"DY={data['dividend_yield_scs']} | Period: {data.get('scs_data_period')}")
    except Exception as e:
        print(f"  [{symbol}] SCS Trade error: {e}")
    return data


# â"€â"€â"€ TECHNICALS â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

def _append_live_price(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Top up a historical daily DataFrame with the most recent trading day's data.
    Fixes the yfinance 1-2 day lag on PSX .KA tickers.

    Fetches the last 5 days at daily interval -- reliable for PSX .KA tickers
    (fast_info.last_price is unreliable on .KA suffix).
    Appends any days newer than the last bar in df.
    """
    if df is None or df.empty:
        return df

    base        = YAHOO_ALIASES.get(symbol, symbol)
    last_in_df  = df.index[-1].normalize()

    for sfx in [".KA", ".KX"]:
        try:
            recent = yf.download(f"{base}{sfx}", period="5d",
                                  progress=False, auto_adjust=True)
            if isinstance(recent.columns, pd.MultiIndex):
                recent.columns = recent.columns.get_level_values(0)
            if recent.empty:
                continue
            new_rows = recent[recent.index.normalize() > last_in_df]
            if not new_rows.empty:
                df = pd.concat([df, new_rows[~new_rows.index.isin(df.index)]]).sort_index()
            break   # got the recent fetch even if no new rows
        except Exception:
            pass

    # Fallback: use DPS PSX scraped price to update the last bar's Close
    # (DPS always shows today's live market price, no lag)
    try:
        fund_cache = load_cache()
        dps_price  = fund_cache.get(symbol, {}).get("price")
        if dps_price and dps_price > 0:
            last_close = float(df["Close"].iloc[-1])
            # Only update if DPS price is within 25% of last known close (sanity check)
            if abs(dps_price - last_close) / last_close < 0.25:
                today = pd.Timestamp.now().normalize()
                if today in df.index:
                    df.loc[today, "Close"] = dps_price
                else:
                    # Add today as a new row with DPS price
                    prev = df.iloc[-1]
                    new_row = pd.DataFrame({
                        "Open":   [dps_price],
                        "High":   [max(float(prev["High"]), dps_price)],
                        "Low":    [min(float(prev["Low"]),  dps_price)],
                        "Close":  [dps_price],
                        "Volume": [int(prev.get("Volume", 0))],
                    }, index=[today])
                    df = pd.concat([df, new_row]).sort_index()
    except Exception:
        pass

    return df


def fetch_price_data(symbol: str, period: str = "5y") -> tuple[pd.DataFrame | None, str]:
    """
    Fetch historical OHLCV -- 7-level fallback chain:
      1. yfinance (.KA)
      2. yfinance (.KX)
      3. ksestocks.com daily market summaries  â† NEW (covers all PSX tickers)
      4. investing.com full history scrape
      5. investing.com spot price
      6. psxterminal.com spot price
      7. dps.psx.com.pk spot price (from cache)
      8. sarmaaya.pk spot price
    Returns (DataFrame | None, source_label)
    """
    # Level 1 & 2: yfinance
    for suffix in [".KA", ".KX"]:
        try:
            df = yf.download(f"{symbol}{suffix}", period=period,
                             progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty:
                df = _append_live_price(df, symbol)   # top up to today
                return df, f"yfinance({symbol}{suffix})"
        except Exception:
            pass

    # Level 3: ksestocks.com -- full OHLCV history for any PSX ticker
    print(f"  [{symbol}] yfinance unavailable -- trying ksestocks.com...")
    df = fetch_ksestocks_history(symbol)
    if df is not None and not df.empty:
        return df, "ksestocks.com"

    # Level 4: investing.com (full history)
    slug = INVESTING_SLUGS.get(symbol)
    if slug:
        print(f"  [{symbol}] ksestocks failed -- trying investing.com ({slug})...")
        df = _scrape_investing_history(slug)
        if df is not None and not df.empty:
            return df, f"investing.com({slug})"

    # Level 5: investing.com spot price
    if slug:
        spot = _get_investing_spot_price(slug)
        if spot:
            return _spot_df(spot), f"investing.com-spot({slug})"

    # Level 6: PSXTerminal spot price (covers ALL PSX stocks)
    print(f"  [{symbol}] Trying psxterminal.com for spot price...")
    psx_price = scrape_psxterminal_price(symbol)
    if psx_price:
        print(f"  [{symbol}] psxterminal.com spot price: PKR {psx_price:.2f}")
        return _spot_df(psx_price), f"psxterminal.com-spot"

    # Level 7: DPS PSX spot price (already scraped -- read from cache)
    cache = load_cache()
    dps_price = cache.get(symbol, {}).get("price")
    if dps_price and dps_price > 0:
        print(f"  [{symbol}] Using DPS spot price: PKR {dps_price:.2f} (fundamentals-only mode)")
        return _spot_df(dps_price), f"dps.psx.com.pk-spot"

    # Level 8: sarmaaya.pk spot price
    print(f"  [{symbol}] Trying sarmaaya.pk for spot price...")
    sarmaaya_price = scrape_sarmaaya_price(symbol)
    if sarmaaya_price:
        print(f"  [{symbol}] sarmaaya.pk spot price: PKR {sarmaaya_price:.2f}")
        return _spot_df(sarmaaya_price), f"sarmaaya.pk-spot"

    print(f"  [{symbol}] No price data found from any source")
    return None, "unavailable"


def _spot_df(price: float) -> pd.DataFrame:
    """Build a minimal single-row DataFrame for spot-price-only mode."""
    idx = pd.DatetimeIndex([pd.Timestamp.now()])
    return pd.DataFrame(
        {"Open": price, "High": price, "Low": price, "Close": price, "Volume": 0},
        index=idx
    )


def calculate_technical_scores(df: pd.DataFrame | None) -> dict:
    empty = {"trend_score": 0, "momentum_score": 0,
             "volume_score": 0, "volatility_score": 0, "total_technical": 0,
             "rsi": None, "atr": None, "price": None,
             "sma20": None, "sma50": None, "sma200": None}

    if df is None or len(df) < 50:
        return empty

    close  = df["Close"]
    latest = df.iloc[-1]

    sma20  = close.rolling(20).mean().iloc[-1]
    sma50  = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1] if len(df) >= 200 else None
    price  = float(latest["Close"])

    trend_score = (
        (20 if price > sma20  else 0) +
        (15 if sma20 > sma50  else 0) +
        (15 if (sma200 is not None and price > sma200) else 0)
    )

    rsi       = ta.momentum.rsi(close, window=14).iloc[-1]
    macd_ser  = ta.trend.macd_diff(close)
    macd_now  = macd_ser.iloc[-1]
    macd_prev = macd_ser.iloc[-2]
    momentum_score = (
        (15 if 40 <= rsi <= 65 else (10 if 35 <= rsi < 40 or 65 < rsi <= 70 else 0)) +
        (10 if macd_now > 0 and macd_now > macd_prev else 0)
    )

    vol_avg = df["Volume"].rolling(20).mean().iloc[-1]
    vol     = float(latest["Volume"])
    volume_score = (15 if vol > 1.3 * vol_avg else (10 if vol > vol_avg else 5))

    atr     = ta.volatility.average_true_range(
        df["High"], df["Low"], df["Close"], window=14).iloc[-1]
    atr_pct = (atr / price) * 100
    volatility_score = (
        10 if 2 <= atr_pct <= 8 else
        (5 if 1.5 <= atr_pct < 2 or 8 < atr_pct <= 12 else 0)
    )

    return {
        "trend_score":      trend_score,
        "momentum_score":   momentum_score,
        "volume_score":     volume_score,
        "volatility_score": volatility_score,
        "total_technical":  trend_score + momentum_score + volume_score + volatility_score,
        "rsi":   round(float(rsi), 1),
        "atr":   round(float(atr), 2),
        "price": round(price, 2),
        "sma20": round(float(sma20), 2),
        "sma50": round(float(sma50), 2),
        "sma200":round(float(sma200), 2) if sma200 is not None else None,
    }


# â"€â"€â"€ SCORING â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

HORIZON_CONFIG = {
    "swing":          {"tech_w": 0.70, "fund_w": 0.30, "threshold": 45},
    "long_term":      {"tech_w": 0.40, "fund_w": 0.60, "threshold": 45},
    "very_long_term": {"tech_w": 0.20, "fund_w": 0.80, "threshold": 45},
}


# ─── SECTOR MAP (editable) ────────────────────────────────────────────────────
# Maps KSE-100 tickers to a scoring profile. Any ticker NOT listed falls back to
# the "general" profile — i.e. exactly the original behaviour. Extend freely.
#   financial -> Debt/Equity test skipped (leverage is structural for banks/insurers)
#   tech      -> higher P/E tolerated (growth premium)
SECTORS = {
    # Banks & insurers
    "ABL": "financial", "AKBL": "financial", "BAFL": "financial", "BAHL": "financial",
    "BOP": "financial", "FABL": "financial", "HBL": "financial", "HMB": "financial",
    "MCB": "financial", "MEBL": "financial", "NBP": "financial", "SCBPL": "financial",
    "BIPL": "financial", "JSBL": "financial", "AICL": "financial", "EFUG": "financial",
    "JGICL": "financial", "ATLH": "financial",
    # Technology / IT
    "SYS": "tech", "AVN": "tech", "TRG": "tech", "NETSOL": "tech",
    "AIRLINK": "tech", "OCTOPUS": "tech", "PTC": "tech",
}


def _sector(symbol: str) -> str:
    return SECTORS.get((symbol or "").upper(), "general")


def fundamental_score(f: dict, symbol: str | None = None) -> tuple[float, list, str]:
    """
    Return (score 0-100, reasons, data_quality).

    Sector-aware and coverage-fair:
      • Financials skip the Debt/Equity test — leverage is structural for banks,
        so they're no longer penalised for a metric that doesn't apply to them.
      • Tech names tolerate higher P/E (growth premium).
      • The score is normalised by the points achievable from the fields we
        ACTUALLY have, so a strong company with missing data isn't unfairly
        dragged down — but a name with fewer than 3 fields is capped at 50 so
        thin data can't masquerade as a high-conviction call.
    """
    sector  = _sector(symbol or f.get("symbol") or "")
    score    = 0.0
    possible = 0.0
    reasons  = []

    def add(points, max_points, reason):
        nonlocal score, possible
        score    += points
        possible += max_points
        if reason:
            reasons.append(reason)

    # Profitability — ROE
    roe = f.get("roe")
    if roe is not None:
        if roe > 20:   add(20, 20, f"ROE {roe:.1f}% (excellent)")
        elif roe > 15: add(15, 20, f"ROE {roe:.1f}% (good)")
        elif roe > 10: add(8,  20, f"ROE {roe:.1f}% (fair)")
        else:          add(0,  20, f"ROE {roe:.1f}% (weak)")

    # Leverage — Debt/Equity  (skipped for financials)
    de = f.get("debt_equity")
    if de is not None and sector != "financial":
        if de < 50:    add(20, 20, f"D/E {de:.0f} (low leverage)")
        elif de < 100: add(13, 20, f"D/E {de:.0f} (moderate)")
        elif de < 200: add(5,  20, f"D/E {de:.0f} (high)")
        else:          add(0,  20, f"D/E {de:.0f} (very high leverage)")

    # Valuation — P/E  (tech tolerates higher multiples)
    pe = f.get("pe_ratio")
    if pe and pe > 0:
        if sector == "tech":
            if pe < 15:   add(15, 15, f"P/E {pe:.1f} (cheap for tech)")
            elif pe < 25: add(12, 15, f"P/E {pe:.1f} (fair for tech)")
            elif pe < 40: add(7,  15, f"P/E {pe:.1f} (growth premium)")
            else:         add(2,  15, f"P/E {pe:.1f} (expensive)")
        else:
            if pe < 8:    add(15, 15, f"P/E {pe:.1f} (deep value)")
            elif pe < 12: add(12, 15, f"P/E {pe:.1f} (value)")
            elif pe < 18: add(8,  15, f"P/E {pe:.1f} (fair)")
            elif pe < 25: add(4,  15, f"P/E {pe:.1f} (slightly rich)")
            else:         add(0,  15, f"P/E {pe:.1f} (expensive)")

    # Dividend
    dy = f.get("dividend_yield")
    if dy is not None and dy > 0:
        if dy > 8:    add(15, 15, f"Div yield {dy:.1f}% (high income)")
        elif dy > 5:  add(10, 15, f"Div yield {dy:.1f}% (solid)")
        elif dy > 3:  add(6,  15, f"Div yield {dy:.1f}% (moderate)")
        else:         add(2,  15, f"Div yield {dy:.1f}% (low)")

    # Net margin
    nm = f.get("net_margin")
    if nm is not None:
        if nm > 20:   add(10, 10, f"Net margin {nm:.1f}% (excellent)")
        elif nm > 12: add(7,  10, f"Net margin {nm:.1f}% (good)")
        elif nm > 6:  add(4,  10, f"Net margin {nm:.1f}% (fair)")
        else:         add(0,  10, f"Net margin {nm:.1f}% (thin)")

    # Earnings / revenue growth  (0 is a valid value, so check None explicitly)
    eg = f.get("eps_growth")
    if eg is None:
        eg = f.get("earnings_growth")
    if eg is not None:
        if eg > 20:   add(10, 10, f"EPS growth {eg:.1f}% (strong)")
        elif eg > 10: add(6,  10, f"EPS growth {eg:.1f}% (solid)")
        elif eg > 0:  add(3,  10, f"EPS growth {eg:.1f}% (positive)")
        else:         add(0,  10, f"EPS growth {eg:.1f}% (declining)")

    # Stability — Beta
    beta = f.get("beta")
    if beta is not None:
        if beta < 0.8:   add(5, 5, f"Beta {beta:.2f} (defensive)")
        elif beta < 1.2: add(3, 5, f"Beta {beta:.2f} (market-like)")
        else:            add(0, 5, f"Beta {beta:.2f} (volatile)")

    populated = sum(1 for k in ["roe", "debt_equity", "pe_ratio", "dividend_yield",
                                 "net_margin", "eps_growth", "beta"]
                    if f.get(k) is not None)

    # Coverage-fair: scale by the points achievable from the fields we have.
    norm = (score / possible * 100) if possible > 0 else 0.0
    if populated < 3:                 # thin data can't earn a high-conviction grade
        norm = min(norm, 50.0)
    final = round(min(norm, 100), 1)

    data_quality = "LIVE" if populated >= 4 else f"PARTIAL ({populated}/7 fields)"
    return final, reasons, data_quality


def recommendation(score: float, horizon: str) -> str:
    if horizon == "swing":
        if score >= 82: return "STRONG BUY"
        if score >= 68: return "BUY"
        if score >= 52: return "WATCH"
        return "AVOID"
    else:
        if score >= 82: return "CORE HOLDING"
        if score >= 68: return "ACCUMULATE"
        if score >= 52: return "HOLD"
        return "AVOID"


# â"€â"€â"€ TRADE PLAN â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

def generate_trade_plan(symbol: str, horizon: str,
                        tech: dict, fund: dict) -> dict:
    price  = tech.get("price")
    atr    = tech.get("atr", 0) or 0
    sma20  = tech.get("sma20")
    sma50  = tech.get("sma50")
    sma200 = tech.get("sma200")

    if price is None:
        return {"symbol": symbol, "horizon": horizon, "error": "No price data"}

    is_bearish = sma200 and price < sma200

    # ── Bearish: no long-side targets, show wait signal ───────────────────────
    if is_bearish and horizon in ("long_term", "very_long_term"):
        return {
            "symbol":        symbol,
            "horizon":       horizon,
            "current_price": f"PKR {price:,.2f}",
            "trend":         "BEARISH",
            "entry_zone":    "WAIT -- price below SMA-200",
            "stop_loss":     "N/A",
            "take_profit_1": "N/A",
            "take_profit_2": "N/A",
            "tp1_pct":       "N/A",
            "tp2_pct":       "N/A",
            "risk_reward":   "N/A",
            "position_size": "0% -- avoid",
            "max_weight":    "0%",
            "rebalance":     "N/A",
            "note":          "Wait for price to reclaim SMA-200 before entering long-term position.",
        }

    # ── Horizon-specific entry, stop, targets ────────────────────────────────
    if horizon == "swing":
        # Entry: current price +/-1% (trade now or wait for minor pullback)
        entry_low  = round(price * 0.99, 2)
        entry_high = round(price * 1.01, 2)
        stop       = round(price - 1.8 * atr, 2) if atr else round(price * 0.93, 2)
        tp1        = round(price + 1.5 * atr, 2) if atr else round(price * 1.05, 2)
        tp2        = round(price + 2.5 * atr, 2) if atr else round(price * 1.08, 2)
        pos_pct = 2.0; max_wt = 5.0; rebalance = "Weekly"

    elif horizon == "long_term":
        # Entry: wait for pullback to SMA-20 (or -5% if SMA-20 not available)
        sma20_val  = sma20 if sma20 else round(price * 0.95, 2)
        entry_low  = round(min(sma20_val, price * 0.97), 2)
        entry_high = round(price * 0.99, 2)
        stop       = round(price * 0.88, 2)
        tp1        = round(price * 1.20, 2)
        tp2        = round(price * 1.40, 2)
        pos_pct = 5.0; max_wt = 15.0; rebalance = "Quarterly"

    else:  # very_long_term
        # Entry: wait for deeper pullback to SMA-50 (or -10%)
        sma50_val  = sma50 if sma50 else round(price * 0.90, 2)
        entry_low  = round(min(sma50_val, price * 0.92), 2)
        entry_high = round(price * 0.96, 2)
        stop       = round(price * 0.80, 2)
        tp1        = round(price * 1.30, 2)
        tp2        = round(price * 1.80, 2)
        pos_pct = 8.0; max_wt = 25.0; rebalance = "Annually"

    rr1 = round((tp1 - price) / (price - stop), 2) if price > stop else 0
    rr2 = round((tp2 - price) / (price - stop), 2) if price > stop else 0

    # Percentage gains from current price
    tp1_pct = round((tp1 - price) / price * 100, 1) if price else 0
    tp2_pct = round((tp2 - price) / price * 100, 1) if price else 0
    sl_pct  = round((stop - price) / price * 100, 1) if price else 0

    # Position size explanation
    pos_note = (f"Allocate {pos_pct}% of portfolio to this trade "
                f"(risk 1% max = Entry-Stop x Shares)")

    return {
        "symbol":          symbol,
        "horizon":         horizon,
        "current_price":   f"PKR {price:,.2f}",
        "trend":           "Bullish" if not is_bearish else "Bearish",
        "entry_zone":      f"PKR {entry_low:,.2f} - {entry_high:,.2f}",
        "stop_loss":       f"PKR {stop:,.2f} ({sl_pct:+.1f}%)",
        "take_profit_1":   f"PKR {tp1:,.2f} (+{tp1_pct:.1f}%)",
        "take_profit_2":   f"PKR {tp2:,.2f} (+{tp2_pct:.1f}%)",
        "tp1_pct":         f"+{tp1_pct:.1f}%",
        "tp2_pct":         f"+{tp2_pct:.1f}%",
        "sl_pct":          f"{sl_pct:+.1f}%",
        "risk_reward":     f"1 : {rr1}  (to TP2: 1 : {rr2})",
        "position_size":   pos_note,
        "max_weight":      f"{max_wt}% of portfolio",
        "rebalance":       rebalance,
        "rsi":             tech.get("rsi"),
    }


# â"€â"€â"€ MAIN SCREENER â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

def run_screener(symbols: list) -> pd.DataFrame:
    print("\n" + "="*60)
    print(" PSX MULTI-HORIZON SCREENER v2")
    print("="*60)

    # Step 1: Fetch all fundamentals (with caching)
    print("\n[1/3] Fetching fundamentals (live scrape + yfinance)...")
    all_funds = get_fundamentals(symbols)

    # Step 2: Fetch price data and calculate technicals (parallel)
    print(f"\n[2/3] Fetching price data...")
    all_tech   = {}
    all_price  = {}
    all_source = {}

    # -- Bulk yfinance download (single batched request -- no rate limits) ------
    # Uses YAHOO_ALIASES to map PSX tickers to their correct Yahoo Finance symbols
    # e.g. ENGROH > DAWH (Engro Holdings = formerly Dawood Hercules on Yahoo)
    print(f"  Bulk downloading {len(symbols)} tickers via yfinance...")
    yf_results = {}

    # Build download list using aliases where available
    def yf_ticker(sym, suffix):
        base = YAHOO_ALIASES.get(sym, sym)
        return f"{base}{suffix}"

    for suffix in [".KA", ".KX"]:
        tickers_try = [yf_ticker(s, suffix) for s in symbols if s not in yf_results]
        if not tickers_try:
            break
        try:
            bulk = yf.download(
                tickers_try, period="5y", progress=False,
                auto_adjust=True, group_by="ticker", threads=True
            )
            if isinstance(bulk.columns, pd.MultiIndex):
                lvl0 = bulk.columns.get_level_values(0)
                for sym in symbols:
                    if sym in yf_results:
                        continue
                    tk = yf_ticker(sym, suffix)
                    if tk in lvl0:
                        df = bulk[tk].dropna(how="all")
                        if not df.empty:
                            df    = _append_live_price(df, sym)   # top up to today
                            label = f"yfinance({tk})"
                            if sym in YAHOO_ALIASES:
                                label += f" [alias for {sym}]"
                            yf_results[sym] = (df, label)
        except Exception as e:
            print(f"  Bulk yfinance error: {e}")

    print(f"  yfinance: {len(yf_results)}/{len(symbols)} tickers returned data")

    # -- investing.com history for remaining missing tickers -------------------
    # GAL (Ghandhara Automobiles) confirmed available at investing.com/equities/ghandara-n.-lt
    missing = [s for s in symbols if s not in yf_results]
    if missing:
        inv_missing = [s for s in missing if s in INVESTING_SLUGS]
        if inv_missing:
            print(f"  Trying investing.com history for: {inv_missing}")
            for sym in inv_missing:
                slug = INVESTING_SLUGS[sym]
                df   = _scrape_investing_history(slug)
                if df is not None and not df.empty:
                    yf_results[sym] = (df, f"investing.com({slug})")
                    print(f"  [{sym}] investing.com: {len(df)} days")

    # -- Spot-price fallback for anything still missing ------------------------
    missing = [s for s in symbols if s not in yf_results]
    if missing:
        print(f"  No price history for: {missing} -- fundamentals-only scoring")
        fund_cache = load_cache()
        for sym in missing:
            spot = fund_cache.get(sym, {}).get("price")
            if spot:
                yf_results[sym] = (_spot_df(spot), "dps.psx.com.pk-spot")
                print(f"  [{sym}] spot price: PKR {spot} [FUND-ONLY]")
            else:
                print(f"  [{sym}] no price available")

    # â"€â"€ Calculate technicals â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    for sym in symbols:
        if sym in yf_results:
            df, src = yf_results[sym]
        else:
            df, src = None, "unavailable"
        all_price[sym]  = df
        all_source[sym] = src
        tech = calculate_technical_scores(df)
        all_tech[sym]   = tech
        print(f"  [{sym}] [{src[:20]}] Price:{tech.get('price')} RSI:{tech.get('rsi')} Tech:{tech.get('total_technical')}/100")

    # Step 3: Fetch news
    print("\n[3/4] Fetching news...")
    all_news   = fetch_news(all_symbols=symbols)
    mkt_sent   = market_sentiment(all_news)
    print(f"  Market sentiment: {mkt_sent['label']} "
          f"(+{mkt_sent['positive_count']} / -{mkt_sent['negative_count']})")

    # Step 4: Score each stock for each horizon
    print("\n[4/4] Scoring stocks...")
    rows = []
    for sym in symbols:
        f        = all_funds.get(sym, {})
        tech     = all_tech.get(sym, {})
        src      = all_source.get(sym, "unavailable")
        f_score, reasons, quality = fundamental_score(f, sym)

        # News for this stock
        stock_news  = get_stock_news(sym, f.get("symbol", sym), all_news)
        news_titles = " | ".join(a["title"] for a in stock_news[:3]) if stock_news else ""
        news_sentiment = sum(a["sentiment"] for a in stock_news)

        # Nudge score slightly based on news sentiment (+/-2 max)
        news_nudge = max(-2, min(2, news_sentiment))

        spot_only = src.startswith("dps.psx") or src.startswith("sarmaaya") or (
            all_price.get(sym) is not None and len(all_price[sym]) <= 2
        )

        for horizon, cfg in HORIZON_CONFIG.items():
            t_score = tech.get("total_technical", 0)

            if spot_only:
                weighted = round(min(100, f_score * 1.0 + news_nudge), 1)
                rec      = recommendation(weighted, horizon) + " [FUND-ONLY]"
            else:
                weighted = round(min(100,
                    t_score * cfg["tech_w"] + f_score * cfg["fund_w"] + news_nudge
                ), 1)
                rec = recommendation(weighted, horizon)

            plan = generate_trade_plan(sym, horizon, tech, f)

            rows.append({
                "symbol":            sym,
                "horizon":           horizon,
                "weighted_score":    weighted,
                "technical_score":   t_score if not spot_only else "N/A",
                "fundamental_score": f_score,
                "news_sentiment":    news_sentiment,
                "news_headlines":    news_titles,
                "data_quality":      quality,
                "price_source":      src,
                "recommendation":    rec,
                "reasons":           " | ".join(reasons[:3]),
                "plan":              plan,
            })

    df_out = pd.DataFrame(rows).sort_values("weighted_score", ascending=False)
    return df_out, all_price, mkt_sent, all_news


# â"€â"€â"€ DISPLAY â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

HORIZON_LABELS = {
    "swing":          "SWING  (days-weeks)   | Stop: -1.8x ATR | Pos: 2% portfolio",
    "long_term":      "LONG TERM  (3-12 mo)  | Stop: -12%      | Pos: 5% portfolio",
    "very_long_term": "VERY LONG TERM (1-3y) | Stop: -20%      | Pos: 8% portfolio",
}

def flatten_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unpack the 'plan' dict column into flat columns so Excel shows
    entry zone, stop loss, TP1, TP2, R/R, % gains etc as proper columns.
    """
    plan_fields = [
        "trend", "current_price", "entry_zone", "stop_loss",
        "take_profit_1", "take_profit_2", "tp1_pct", "tp2_pct",
        "risk_reward", "position_size", "rebalance", "rsi",
    ]
    out = df.drop(columns=["plan"]).copy()
    for field in plan_fields:
        out[field] = df["plan"].apply(
            lambda p: p.get(field, "") if isinstance(p, dict) else ""
        )
    core = [
        "symbol", "horizon", "weighted_score", "technical_score",
        "fundamental_score", "news_sentiment", "data_quality",
        "price_source", "recommendation", "reasons", "news_headlines",
    ]
    cols = [c for c in core if c in out.columns] + plan_fields
    return out[cols]


def print_results(df: pd.DataFrame, mkt_sent: dict = None, all_news: list = None):
    print("\n" + "="*90)
    print("  PSX SCREENER RESULTS -- TOP PICKS BY HORIZON")
    print("="*90)

    # Market sentiment banner
    if mkt_sent:
        label = mkt_sent["label"]
        score = mkt_sent["score"]
        pos   = mkt_sent["positive_count"]
        neg   = mkt_sent["negative_count"]
        print(f"\n  MARKET SENTIMENT: {label}  (score: {score:+d} | "
              f"+{pos} positive / -{neg} negative headlines)")
        if mkt_sent.get("key_headlines"):
            print("  Key news today:")
            for a in mkt_sent["key_headlines"][:3]:
                sent_icon = "+" if a["sentiment"] > 0 else ("-" if a["sentiment"] < 0 else "~")
                print(f"    [{sent_icon}] {a['title'][:80]}  [{a['source']}]")
        print()

    for horizon in ["swing", "long_term", "very_long_term"]:
        sub = df[df["horizon"] == horizon].head(5)
        print("\n" + "-"*90)
        print(f"  {HORIZON_LABELS[horizon]}")
        print("-"*90)

        # Header
        print(f"  {'SYMBOL':<8} {'SCORE':>6} {'TECH':>5} {'FUND':>5} "
              f"{'PRICE':>8} {'ENTRY ZONE':>22} {'STOP':>8} "
              f"{'TP1':>8} {'TP2':>8} {'R/R':>5}  SIGNAL")
        print("  " + "-"*88)

        for _, row in sub.iterrows():
            plan = row.get("plan", {})
            tech_disp  = str(row['technical_score']) if str(row['technical_score']) != "N/A" else "N/A"
            ns         = row.get("news_sentiment", 0)
            ns_disp    = f"+{ns}" if ns > 0 else str(ns)

            def sp(s):
                return str(s).replace("PKR ", "").replace(",", "")

            trend = plan.get("trend", "")
            trend_icon = "^" if "Bull" in str(trend) else ("v" if "Bear" in str(trend) else "~")

            print(f"  {row['symbol']:<8} {row['weighted_score']:>6.1f} "
                  f"{tech_disp:>5} {row['fundamental_score']:>5.1f} "
                  f"news:{ns_disp:<3} {trend_icon} "
                  f"{sp(plan.get('current_price','N/A')):>8} "
                  f"{sp(plan.get('entry_zone','N/A')):>22} "
                  f"{sp(plan.get('stop_loss','N/A')):>12} "
                  f"{sp(plan.get('take_profit_1','N/A')):>14} "
                  f"{sp(plan.get('take_profit_2','N/A')):>14} "
                  f"R/R:{str(plan.get('risk_reward','N/A')).replace('1 : ','')}  "
                  f"{row['recommendation']}")
            if row["reasons"]:
                print(f"           Fundamentals: {row['reasons']}")
            if row.get("news_headlines"):
                # Show first relevant headline
                first = row["news_headlines"].split(" | ")[0]
                print(f"           News: {first[:85]}")

    # â"€â"€ Position sizing reminder â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    print("\n" + "="*90)
    print("  POSITION SIZING  (1% portfolio risk per trade, adjust to your capital)")
    print("="*90)
    print(f"  {'SYMBOL':<8} {'HORIZON':<15} {'PRICE':>8} {'STOP':>8} {'RISK/SHR':>9} "
          f"{'MAX SHARES @ 1M PKR':>20}  RECOMMENDATION")
    print("  " + "-"*88)

    shown = set()
    for horizon in ["swing", "long_term", "very_long_term"]:
        sub = df[df["horizon"] == horizon].head(3)
        for _, row in sub.iterrows():
            key = (row['symbol'], horizon)
            if key in shown:
                continue
            shown.add(key)
            plan = row.get("plan", {})
            try:
                price = float(re.sub(r"[^0-9.]", "", str(plan.get("current_price","0")).split()[1] if "PKR" in str(plan.get("current_price","")) else str(plan.get("current_price","0"))))
                # stop_loss now includes "(-X.X%)" -- strip everything after the space
                stop_raw = str(plan.get("stop_loss","0")).replace("PKR","").strip()
                stop_raw = stop_raw.split()[0] if stop_raw else "0"
                stop  = float(stop_raw.replace(",",""))
                risk_per_share = price - stop
                if risk_per_share > 0:
                    max_shares = int((1_000_000 * 0.01) / risk_per_share)
                    capital    = max_shares * price
                    print(f"  {row['symbol']:<8} {horizon:<15} {price:>8,.0f} {stop:>8,.0f} "
                          f"{risk_per_share:>9,.1f} {max_shares:>12,} shrs / "
                          f"PKR {capital:>10,.0f}  {row['recommendation']}")
            except Exception:
                pass

    print(f"\n[Note] Stocks tagged [FUND-ONLY] have no price history on Yahoo Finance.")
    print(f"       Entry/TP levels use current spot price only -- no ATR available.")
    print(f"       Re-run daily; cache refreshes fundamentals every 24h, index every 7 days.")


# â"€â"€â"€ SINGLE-TICKER DEEP DIVE â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

def analyse_ticker(symbol: str):
    """
    Full analysis for a single user-specified ticker.
    Works for ANY PSX stock -- not just KSE-100.
    Usage:  python psx_screener_v2.py ENGRO
            python psx_screener_v2.py FFC
    """
    symbol = symbol.strip().upper()
    print("\n" + "="*70)
    print(f"  DEEP DIVE: {symbol}")
    print("="*70)

    # â"€â"€ Fundamentals â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    print("\n[1/3] Fetching fundamentals...")
    _, merged = _fetch_one_symbol_fundamentals(symbol)

    data_period = merged.get("data_as_of", "Unknown")
    print(f"\n  FUNDAMENTALS SUMMARY  [Data as of: {data_period}]")
    print("  " + "-"*50)
    fields = [
        ("P/E Ratio",       merged.get("pe_ratio")),
        ("ROE",             f"{merged.get('roe'):.1f}%" if merged.get("roe") else None),
        ("ROA",             f"{merged.get('roa'):.1f}%" if merged.get("roa") else None),
        ("Debt / Equity",   f"{merged.get('debt_equity'):.1f}%" if merged.get("debt_equity") else None),
        ("Net Margin",      f"{merged.get('net_margin'):.1f}%" if merged.get("net_margin") else None),
        ("Gross Margin",    f"{merged.get('gross_margin'):.1f}%" if merged.get("gross_margin") else None),
        ("Dividend Yield",  f"{merged.get('dividend_yield'):.2f}%" if merged.get("dividend_yield") else None),
        ("EPS Growth",      f"{merged.get('eps_growth'):.1f}%" if merged.get("eps_growth") else None),
        ("Current Ratio",   merged.get("current_ratio")),
        ("Beta",            merged.get("beta")),
        ("Market Cap",      f"PKR {merged.get('market_cap_000s'):,.0f}k" if merged.get("market_cap_000s") else None),
        ("52w High",        f"PKR {merged.get('52w_high'):,.2f}" if merged.get("52w_high") else None),
        ("52w Low",         f"PKR {merged.get('52w_low'):,.2f}" if merged.get("52w_low") else None),
    ]
    for label, val in fields:
        if val is not None:
            print(f"  {label:<20}: {val}")

    # â"€â"€ Price & Technicals â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    print("\n[2/3] Fetching price history...")
    df, src = fetch_price_data(symbol)
    tech    = calculate_technical_scores(df)

    print(f"\n  TECHNICALS  (source: {src})")
    print("  " + "-"*50)
    tech_fields = [
        ("Current Price",   f"PKR {tech.get('price'):,.2f}" if tech.get("price") else "N/A"),
        ("RSI (14)",        tech.get("rsi")),
        ("ATR (14)",        tech.get("atr")),
        ("SMA-20",          tech.get("sma20")),
        ("SMA-50",          tech.get("sma50")),
        ("SMA-200",         tech.get("sma200")),
        ("Trend",           "Bullish" if (tech.get("price") or 0) > (tech.get("sma200") or 9e9) else "Bearish"),
        ("Tech Score",      f"{tech.get('total_technical')}/100"),
    ]
    for label, val in tech_fields:
        if val is not None:
            print(f"  {label:<20}: {val}")

    # â"€â"€ Scores & Plans â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    print("\n[3/3] Scoring & generating trade plans...")
    f_score, reasons, quality = fundamental_score(merged)

    spot_only = src.startswith("dps.psx") or src.startswith("sarmaaya") or \
                src.startswith("psxterminal") or (df is not None and len(df) <= 2)

    print(f"\n  SCORES  (fundamental data quality: {quality})")
    print("  " + "-"*50)
    for horizon, cfg in HORIZON_CONFIG.items():
        t_score = tech.get("total_technical", 0)
        if spot_only:
            weighted = round(f_score * 1.0, 1)
            rec      = recommendation(weighted, horizon) + " [FUND-ONLY]"
        else:
            weighted = round(t_score * cfg["tech_w"] + f_score * cfg["fund_w"], 1)
            rec      = recommendation(weighted, horizon)
        label = horizon.replace("_", " ").title()
        print(f"  {label:<18}: {weighted:>5.1f}/100  ->  {rec}")

    print(f"\n  Key reasons: {' | '.join(reasons[:4])}")

    print("\n  TRADE PLANS")
    print("  " + "-"*70)
    print(f"  {'HORIZON':<15} {'PRICE':>8} {'ENTRY ZONE':>22} {'STOP':>8} {'TP1':>8} {'TP2':>8} {'R/R':>5}")
    print("  " + "-"*70)
    for horizon in HORIZON_CONFIG:
        plan = generate_trade_plan(symbol, horizon, tech, merged)
        if "error" in plan:
            print(f"  {horizon:<15} No price data available")
            continue
        def sp(s): return str(s).replace("PKR ","").replace(",","")
        print(f"  {horizon:<15} {sp(plan['current_price']):>8} "
              f"{sp(plan['entry_zone']):>22} "
              f"{sp(plan['stop_loss']):>8} "
              f"{sp(plan['take_profit_1']):>8} "
              f"{sp(plan['take_profit_2']):>8} "
              f"{str(plan['risk_reward']).replace('1 : ',''):>5}")

    print(f"\n  Position sizing (1% risk on PKR 1,000,000 portfolio):")
    for horizon in HORIZON_CONFIG:
        plan = generate_trade_plan(symbol, horizon, tech, merged)
        if "error" in plan:
            continue
        try:
            price = float(str(plan["current_price"]).replace("PKR","").replace(",",""))
            stop  = float(str(plan["stop_loss"]).replace("PKR","").replace(",",""))
            rps   = price - stop
            if rps > 0:
                shares  = int(10000 / rps)
                capital = shares * price
                print(f"  {horizon:<15}: {shares:>6,} shares  /  PKR {capital:>10,.0f}  capital used")
        except Exception:
            pass

    print(f"\n  Exit rules: Trailing stop (1.5x ATR) after TP1. "
          f"Exit if weekly close < SMA-200.")
    print("="*70)


# â"€â"€â"€ ENTRY POINT â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

def run_full_screen_and_save():
    """
    Run the full KSE-100 screen and save results to CSV.

    Importable entry point used by auto_update.py and the packaged app
    (investistan_app.py) so a screen can run IN-PROCESS without shelling out to
    this script -- which is required when the app is frozen into a single .exe
    (there is no python.exe / loose .py to call). Returns the results DataFrame.
    """
    constituents = fetch_kse100_constituents()
    print(f"[Index] Running screener on {len(constituents)} KSE-100 stocks.\n")
    results_df, price_data, mkt_sent, all_news = run_screener(constituents)
    print_results(results_df, mkt_sent, all_news)
    csv_path = DATA_DIR / "psx_screener_results.csv"
    try:
        flatten_for_csv(results_df).to_csv(csv_path, index=False)
        print(f"\n[Saved] Full results -> {csv_path}")
    except PermissionError:
        alt = csv_path.with_name("psx_screener_results_new.csv")
        flatten_for_csv(results_df).to_csv(alt, index=False)
        print(f"\n[Saved] -> {alt}  (close Excel to overwrite main file)")
    return results_df


if __name__ == "__main__":
    import sys

    # â"€â"€ Single-ticker mode: python psx_screener_v2.py TICKER â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    if len(sys.argv) > 1:
        ticker_input = sys.argv[1].strip().upper()
        analyse_ticker(ticker_input)

    # â"€â"€ Interactive mode: no argument -> ask â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    elif sys.stdin.isatty():
        print("PSX Screener -- what would you like to do?")
        print("  1. Screen full KSE-100")
        print("  2. Analyse a specific ticker")
        choice = input("\nEnter 1 or 2 (default=1): ").strip()
        if choice == "2":
            ticker_input = input("Enter PSX ticker symbol (e.g. FFC, ENGRO, MEBL): ").strip().upper()
            analyse_ticker(ticker_input)
        else:
            constituents = fetch_kse100_constituents()
            print(f"[Index] Running screener on {len(constituents)} KSE-100 stocks.\n")
            results_df, price_data, mkt_sent, all_news = run_screener(constituents)
            print_results(results_df, mkt_sent, all_news)
            csv_path = Path(__file__).parent / "psx_screener_results.csv"
            try:
                flatten_for_csv(results_df).to_csv(csv_path, index=False)
                print(f"\n[Saved] Full results -> {csv_path}")
            except PermissionError:
                alt = csv_path.with_name("psx_screener_results_new.csv")
                flatten_for_csv(results_df).to_csv(alt, index=False)
                print(f"\n[Saved] -> {alt}  (close Excel to overwrite main file)")

    # â"€â"€ Default: full KSE-100 screen (non-interactive / scheduled run) â"€â"€â"€â"€â"€â"€â"€â"€
    else:
        constituents = fetch_kse100_constituents()
        print(f"[Index] Running screener on {len(constituents)} KSE-100 stocks.\n")
        results_df, price_data, mkt_sent, all_news = run_screener(constituents)
        print_results(results_df, mkt_sent, all_news)
        csv_path = Path(__file__).parent / "psx_screener_results.csv"
        try:
            flatten_for_csv(results_df).to_csv(csv_path, index=False)
            print(f"\n[Saved] Full results -> {csv_path}")
        except PermissionError:
            alt = csv_path.with_name("psx_screener_results_new.csv")
            flatten_for_csv(results_df).to_csv(alt, index=False)
            print(f"\n[Saved] -> {alt}  (close Excel to overwrite main file)")

