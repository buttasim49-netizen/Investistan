"""
Value & Reversal Screener
==========================
1. DEEP VALUE    -- fundamentally cheap stocks using Graham/value criteria
2. REVERSAL      -- technically oversold stocks with bottoming signals
3. HIDDEN GEMS   -- combination: cheap + showing reversal = highest conviction

Run standalone:   python value_reversal.py
Or import:        from value_reversal import screen_value, screen_reversal
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent))

FUND_CACHE    = Path(__file__).parent / "psx_fundamentals.json"
SCREENER_CSV  = Path(__file__).parent / "psx_screener_results.csv"
YAHOO_ALIASES = {"ENGROH": "DAWH"}


# ─── LOAD DATA ────────────────────────────────────────────────────────────────

def load_data():
    fund = {}
    if FUND_CACHE.exists():
        with open(FUND_CACHE, "r", encoding="utf-8") as f:
            fund = json.load(f)

    screener = pd.DataFrame()
    if SCREENER_CSV.exists():
        screener = pd.read_csv(SCREENER_CSV)

    return fund, screener


# ─── DEEP VALUE SCREEN ────────────────────────────────────────────────────────

def value_score(f: dict) -> tuple[float, list]:
    """
    Score a stock on value investing criteria (0-100).
    Based on Graham Number, Buffett metrics, and income generation.
    """
    score   = 0
    reasons = []

    pe  = f.get("pe_ratio")
    roe = f.get("roe")
    de  = f.get("debt_equity")
    dy  = f.get("dividend_yield")
    nm  = f.get("net_margin")
    gm  = f.get("gross_margin")
    cr  = f.get("current_ratio")
    eg  = f.get("eps_growth") or f.get("earnings_growth")
    pb  = f.get("pb_ratio")
    roa = f.get("roa")

    # 1. P/E Ratio (25 pts) -- lower is better, but must be positive
    if pe and 0 < pe < 50:
        if pe < 5:
            score += 25; reasons.append(f"P/E {pe:.1f} (extreme deep value)")
        elif pe < 8:
            score += 20; reasons.append(f"P/E {pe:.1f} (deep value)")
        elif pe < 12:
            score += 15; reasons.append(f"P/E {pe:.1f} (value zone)")
        elif pe < 15:
            score += 8;  reasons.append(f"P/E {pe:.1f} (fair value)")
        else:
            score += 3;  reasons.append(f"P/E {pe:.1f} (slightly rich)")

    # 2. Return on Equity (20 pts) -- quality of the business
    if roe is not None:
        if roe > 25:
            score += 20; reasons.append(f"ROE {roe:.1f}% (exceptional)")
        elif roe > 18:
            score += 15; reasons.append(f"ROE {roe:.1f}% (strong)")
        elif roe > 12:
            score += 10; reasons.append(f"ROE {roe:.1f}% (good)")
        elif roe > 0:
            score += 4;  reasons.append(f"ROE {roe:.1f}% (positive)")
        else:
            reasons.append(f"ROE {roe:.1f}% (negative -- loss-making)")

    # 3. Dividend Yield (15 pts) -- income while you wait
    if dy and dy > 0:
        if dy > 10:
            score += 15; reasons.append(f"Div yield {dy:.1f}% (exceptional income)")
        elif dy > 7:
            score += 12; reasons.append(f"Div yield {dy:.1f}% (high income)")
        elif dy > 5:
            score += 8;  reasons.append(f"Div yield {dy:.1f}% (solid income)")
        elif dy > 3:
            score += 4;  reasons.append(f"Div yield {dy:.1f}% (moderate income)")

    # 4. Balance Sheet (15 pts) -- safety
    if de is not None:
        if de < 20:
            score += 15; reasons.append(f"D/E {de:.0f}% (very clean balance sheet)")
        elif de < 50:
            score += 10; reasons.append(f"D/E {de:.0f}% (conservative leverage)")
        elif de < 100:
            score += 5;  reasons.append(f"D/E {de:.0f}% (manageable debt)")
        else:
            reasons.append(f"D/E {de:.0f}% (high debt -- caution)")

    # 5. Profitability (10 pts)
    if nm and nm > 0:
        if nm > 20:
            score += 10; reasons.append(f"Net margin {nm:.1f}% (highly profitable)")
        elif nm > 12:
            score += 7;  reasons.append(f"Net margin {nm:.1f}% (profitable)")
        elif nm > 5:
            score += 4;  reasons.append(f"Net margin {nm:.1f}% (breakeven+)")

    # 6. Growth (10 pts) -- cheap AND growing is ideal
    if eg is not None and eg > 0:
        if eg > 25:
            score += 10; reasons.append(f"EPS growth {eg:.1f}% (fast growth at value price)")
        elif eg > 10:
            score += 7;  reasons.append(f"EPS growth {eg:.1f}% (growing cheaply)")
        elif eg > 0:
            score += 3;  reasons.append(f"EPS growth {eg:.1f}% (slow but positive)")

    # 7. Price-to-Book (5 pts) -- Graham classic
    if pb and 0 < pb < 5:
        if pb < 0.8:
            score += 5;  reasons.append(f"P/B {pb:.2f} (trading below book value!)")
        elif pb < 1.5:
            score += 4;  reasons.append(f"P/B {pb:.2f} (near book value)")
        elif pb < 2.5:
            score += 2;  reasons.append(f"P/B {pb:.2f} (reasonable)")

    return round(min(score, 100), 1), reasons


def screen_value(min_score: float = 55) -> pd.DataFrame:
    """
    Screen all stocks for deep value opportunities.
    Returns DataFrame sorted by value score descending.
    """
    fund, screener = load_data()

    rows = []
    for symbol, f in fund.items():
        if symbol in ("holdings", "updated_at"):
            continue
        if not isinstance(f, dict):
            continue

        v_score, reasons = value_score(f)
        if v_score < min_score:
            continue

        # Get current price and tech score from screener
        price      = f.get("price", 0)
        tech_score = 0
        rec        = "N/A"
        entry      = "N/A"
        stop       = "N/A"
        tp2        = "N/A"

        if not screener.empty:
            match = screener[(screener["symbol"] == symbol) &
                             (screener["horizon"] == "swing")]
            if not match.empty:
                tech_score = match.iloc[0].get("technical_score", 0)
                rec        = match.iloc[0].get("recommendation", "N/A")
                entry      = str(match.iloc[0].get("entry_zone", "N/A")).replace("PKR","").strip()
                stop       = str(match.iloc[0].get("stop_loss",  "N/A")).replace("PKR","").strip()
                tp2        = str(match.iloc[0].get("take_profit_2","N/A")).replace("PKR","").strip()

        # Graham Number = sqrt(22.5 * EPS * Book Value per share)
        graham_num = None
        eps = f.get("eps_ttm") or f.get("eps_ttm_psx")
        bv  = f.get("book_value_scs")
        if eps and bv and eps > 0 and bv > 0:
            graham_num = round((22.5 * eps * bv) ** 0.5, 2)

        rows.append({
            "symbol":       symbol,
            "value_score":  v_score,
            "tech_score":   tech_score,
            "price":        price,
            "graham_number":graham_num,
            "upside_to_graham": round((graham_num - price) / price * 100, 1) if (graham_num and price and price > 0) else None,
            "pe_ratio":     f.get("pe_ratio"),
            "roe":          f.get("roe"),
            "div_yield":    f.get("dividend_yield"),
            "debt_equity":  f.get("debt_equity"),
            "net_margin":   f.get("net_margin"),
            "eps_growth":   f.get("eps_growth") or f.get("earnings_growth"),
            "data_as_of":   f.get("data_as_of", "Unknown"),
            "recommendation": rec,
            "entry_zone":   entry,
            "stop_loss":    stop,
            "take_profit_2": tp2,
            "reasons":      " | ".join(reasons[:4]),
        })

    df = pd.DataFrame(rows).sort_values("value_score", ascending=False)
    return df


# ─── REVERSAL / BOTTOM-FISHING SCREEN ────────────────────────────────────────

def fetch_technicals(symbol: str) -> dict | None:
    """Fetch 1yr price history and compute reversal indicators."""
    base = YAHOO_ALIASES.get(symbol, symbol)
    df   = None

    for sfx in [".KA", ".KX"]:
        try:
            tmp = yf.download(f"{base}{sfx}", period="1y",
                              progress=False, auto_adjust=True)
            if isinstance(tmp.columns, pd.MultiIndex):
                tmp.columns = tmp.columns.get_level_values(0)
            if not tmp.empty:
                df = tmp
                break
        except Exception:
            pass

    if df is None or len(df) < 30:
        return None

    try:
        import ta, re as _re
        from bs4 import BeautifulSoup as _BS
        close  = df["Close"]
        high   = df["High"]
        low    = df["Low"]
        vol    = df["Volume"]

        # Always use live DPS price — yfinance lags 1-2 days on PSX tickers
        _live = None
        try:
            import requests as _rq
            _resp = _rq.get(f"https://dps.psx.com.pk/company/{symbol}",
                            headers={"User-Agent": "Mozilla/5.0"}, timeout=6)
            if _resp.status_code == 200:
                _text = _BS(_resp.text, "html.parser").get_text()
                for _pat in [r"Rs\.\s*([\d,\.]+)", r"PKR\s*([\d,\.]+)"]:
                    _m = _re.search(_pat, _text, _re.IGNORECASE)
                    if _m:
                        _p = float(_m.group(1).replace(",",""))
                        if 0.1 < _p < 1_000_000:
                            _live = _p
                            break
        except Exception:
            pass

        price = _live if _live else float(close.iloc[-1])

        rsi        = float(ta.momentum.rsi(close, window=14).iloc[-1])
        macd_hist  = ta.trend.macd_diff(close)
        macd_now   = float(macd_hist.iloc[-1])
        macd_prev  = float(macd_hist.iloc[-2])
        macd_rising = macd_now > macd_prev

        sma20  = float(close.rolling(20).mean().iloc[-1])
        sma50  = float(close.rolling(50).mean().iloc[-1])
        sma200 = float(close.rolling(200).mean().iloc[-1])
        atr    = float(ta.volatility.average_true_range(high, low, close, window=14).iloc[-1])

        wk52_high = float(high.max())
        wk52_low  = float(low.min())
        pct_from_low  = (price - wk52_low)  / wk52_low  * 100
        pct_from_high = (price - wk52_high) / wk52_high * 100

        vol_avg   = float(vol.rolling(20).mean().iloc[-1])
        vol_ratio = float(vol.iloc[-1]) / vol_avg if vol_avg > 0 else 0

        # Volume spike in last 5 days (Selling Climax signal)
        max_vol_5d  = float(vol.iloc[-5:].max())
        vol_climax  = max_vol_5d > vol_avg * 2.0

        # Price recovering from near 52w low
        near_52w_low = pct_from_low < 25     # within 25% of 52-week low
        recovering   = close.iloc[-1] > close.iloc[-5]  # up over last week

        return {
            "price":          price,
            "rsi":            round(rsi, 1),
            "macd_rising":    macd_rising,
            "macd_hist":      round(macd_now, 3),
            "sma20":          round(sma20, 2),
            "sma50":          round(sma50, 2),
            "sma200":         round(sma200, 2),
            "atr":            round(atr, 2),
            "atr_pct":        round(atr / price * 100, 1),
            "wk52_high":      round(wk52_high, 2),
            "wk52_low":       round(wk52_low, 2),
            "pct_from_low":   round(pct_from_low, 1),
            "pct_from_high":  round(pct_from_high, 1),
            "vol_ratio":      round(vol_ratio, 2),
            "vol_climax":     vol_climax,
            "near_52w_low":   near_52w_low,
            "recovering":     recovering,
            "above_sma20":    price > sma20,
            "above_sma50":    price > sma50,
            "above_sma200":   price > sma200,
        }
    except Exception:
        return None


def reversal_score(tech: dict, fund: dict) -> tuple[float, list]:
    """
    Score a stock as a reversal / bottom-fishing candidate (0-100).
    Looks for technically oversold + fundamentally sound + showing early recovery.
    """
    score   = 0
    reasons = []
    if tech is None:
        return 0, ["No technical data"]

    rsi           = tech.get("rsi", 50)
    macd_rising   = tech.get("macd_rising", False)
    near_low      = tech.get("near_52w_low", False)
    recovering    = tech.get("recovering", False)
    vol_climax    = tech.get("vol_climax", False)
    above_sma20   = tech.get("above_sma20", False)
    pct_from_low  = tech.get("pct_from_low", 100)
    pct_from_high = tech.get("pct_from_high", 0)
    vol_ratio     = tech.get("vol_ratio", 1)

    # 1. RSI zone scoring
    # Standard definitions: <30 = oversold, 30-40 = weak/bearish, 40-50 = below midline
    if rsi < 30:
        score += 30; reasons.append(f"RSI {rsi:.0f} -- oversold (rebound zone)")
    elif rsi < 40:
        score += 18; reasons.append(f"RSI {rsi:.0f} -- weak momentum, approaching oversold")
    elif rsi < 50:
        score += 10; reasons.append(f"RSI {rsi:.0f} -- below midline, bearish bias")

    # 2. MACD histogram turning up (20 pts)
    if macd_rising and tech.get("macd_hist", 0) < 0:
        score += 20; reasons.append("MACD histogram turning up from below zero (classic reversal)")
    elif macd_rising:
        score += 10; reasons.append("MACD histogram rising (momentum building)")

    # 3. Near 52-week low + recovering (20 pts)
    if near_low and recovering:
        score += 20; reasons.append(f"Near 52w low ({pct_from_low:.0f}% up) + recovering -- base forming")
    elif near_low:
        score += 10; reasons.append(f"Near 52w low ({pct_from_low:.0f}% above) -- watch for bounce")
    elif pct_from_high < -40:
        score += 8;  reasons.append(f"Down {abs(pct_from_high):.0f}% from 52w high -- deep correction")

    # 4. Volume climax (Wyckoff Selling Climax signal) (15 pts)
    if vol_climax:
        score += 15; reasons.append("Volume spike (2x avg) recently -- possible Selling Climax")

    # 5. Price crossing above SMA-20 (10 pts) -- early breakout
    if above_sma20 and not tech.get("above_sma50", True):
        score += 10; reasons.append("Crossed above SMA-20 while still below SMA-50 -- early signal")

    # 6. Fundamental quality (5 pts) -- avoid garbage
    roe = fund.get("roe", 0) or 0
    pe  = fund.get("pe_ratio", 99) or 99
    if roe > 10 and pe < 15:
        score += 5; reasons.append(f"Fundamentally sound (ROE {roe:.0f}%, P/E {pe:.1f})")

    return round(min(score, 100), 1), reasons


def screen_reversal(symbols: list = None, min_score: float = 45) -> pd.DataFrame:
    """
    Screen for reversal / bottom-fishing candidates.
    Fetches live technical data for each symbol.
    Returns DataFrame sorted by reversal score descending.
    """
    fund, screener = load_data()

    if symbols is None:
        # Use all symbols that have fundamental data
        symbols = [k for k in fund.keys()
                   if k not in ("holdings", "updated_at") and isinstance(fund.get(k), dict)]

    rows   = []
    total  = len(symbols)

    for i, symbol in enumerate(symbols):
        print(f"  [{symbol}] ({i+1}/{total})", end="\r")
        f    = fund.get(symbol, {})
        tech = fetch_technicals(symbol)

        if tech is None:
            continue

        r_score, reasons = reversal_score(tech, f)
        if r_score < min_score:
            continue

        # Graham Number
        graham_num = None
        eps = f.get("eps_ttm") or f.get("eps_ttm_psx")
        bv  = f.get("book_value_scs")
        if eps and bv and eps > 0 and bv > 0:
            graham_num = round((22.5 * eps * bv) ** 0.5, 2)

        # Swing entry/TP from screener
        entry = stop = tp1 = tp2 = "N/A"
        if not screener.empty:
            m = screener[(screener["symbol"]==symbol) & (screener["horizon"]=="swing")]
            if not m.empty:
                entry = str(m.iloc[0].get("entry_zone","N/A")).replace("PKR","").strip()
                stop  = str(m.iloc[0].get("stop_loss","N/A")).replace("PKR","").strip()
                tp1   = str(m.iloc[0].get("take_profit_1","N/A")).replace("PKR","").strip()
                tp2   = str(m.iloc[0].get("take_profit_2","N/A")).replace("PKR","").strip()

        rows.append({
            "symbol":         symbol,
            "reversal_score": r_score,
            "price":          tech["price"],
            "rsi":            tech["rsi"],
            "52w_low":        tech["wk52_low"],
            "52w_high":       tech["wk52_high"],
            "pct_from_low":   f"+{tech['pct_from_low']:.1f}%",
            "pct_from_high":  f"{tech['pct_from_high']:.1f}%",
            "vol_ratio":      tech["vol_ratio"],
            "vol_climax":     "YES" if tech["vol_climax"] else "no",
            "macd_rising":    "YES" if tech["macd_rising"] else "no",
            "above_sma20":    "YES" if tech["above_sma20"] else "no",
            "above_sma200":   "YES" if tech["above_sma200"] else "no",
            "graham_number":  graham_num,
            "pe_ratio":       f.get("pe_ratio"),
            "roe":            f.get("roe"),
            "data_as_of":     f.get("data_as_of", "Unknown"),
            "entry_zone":     entry,
            "stop_loss":      stop,
            "take_profit_1":  tp1,
            "take_profit_2":  tp2,
            "reasons":        " | ".join(reasons[:4]),
        })

    print()
    df = pd.DataFrame(rows).sort_values("reversal_score", ascending=False)
    return df


# ─── HIDDEN GEMS (Value + Reversal combined) ──────────────────────────────────

def screen_hidden_gems(value_min: float = 50, reversal_min: float = 40) -> pd.DataFrame:
    """
    Find stocks that are BOTH undervalued AND showing reversal signals.
    These are the highest conviction opportunities.
    """
    fund, screener = load_data()

    symbols = [k for k in fund.keys()
               if k not in ("holdings","updated_at") and isinstance(fund.get(k), dict)]

    rows  = []
    total = len(symbols)

    for i, symbol in enumerate(symbols):
        print(f"  [{symbol}] ({i+1}/{total})", end="\r")
        f = fund.get(symbol, {})

        v_score, v_reasons = value_score(f)
        if v_score < value_min:
            continue

        tech = fetch_technicals(symbol)
        if tech is None:
            continue

        r_score, r_reasons = reversal_score(tech, f)
        if r_score < reversal_min:
            continue

        combined = round(v_score * 0.5 + r_score * 0.5, 1)

        rows.append({
            "symbol":          symbol,
            "combined_score":  combined,
            "value_score":     v_score,
            "reversal_score":  r_score,
            "price":           tech["price"],
            "rsi":             tech["rsi"],
            "pe_ratio":        f.get("pe_ratio"),
            "roe":             f.get("roe"),
            "div_yield":       f.get("dividend_yield"),
            "pct_from_high":   f"{tech['pct_from_high']:.1f}%",
            "pct_from_low":    f"+{tech['pct_from_low']:.1f}%",
            "vol_climax":      "YES" if tech["vol_climax"] else "no",
            "macd_rising":     "YES" if tech["macd_rising"] else "no",
            "data_as_of":      f.get("data_as_of", "Unknown"),
            "value_reasons":   " | ".join(v_reasons[:3]),
            "reversal_reasons":  " | ".join(r_reasons[:3]),
        })

    print()
    df = pd.DataFrame(rows).sort_values("combined_score", ascending=False)
    return df


# ─── STANDALONE OUTPUT ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*75)
    print("  PSX VALUE & REVERSAL SCREENER")
    print("="*75)

    print("\n[1/3] Deep Value Screen...")
    value_df = screen_value(min_score=55)
    print(f"\n  {'SYMBOL':<8} {'V.SCORE':>8} {'PRICE':>8} {'P/E':>6} {'ROE':>6} {'DY':>5} {'GRAHAM':>8} {'UPSIDE':>8} {'Period'}")
    print("  " + "-"*75)
    for _, r in value_df.head(10).iterrows():
        graham = f"PKR {r['graham_number']:.0f}" if r['graham_number'] else "N/A"
        upside = f"+{r['upside_to_graham']:.0f}%" if r['upside_to_graham'] and r['upside_to_graham'] > 0 else "N/A"
        pe  = f"{r['pe_ratio']:.1f}" if r['pe_ratio'] else "N/A"
        roe = f"{r['roe']:.1f}%" if r['roe'] else "N/A"
        dy  = f"{r['div_yield']:.1f}%" if r['div_yield'] else "N/A"
        print(f"  {r['symbol']:<8} {r['value_score']:>8.1f} {r['price']:>8.2f} {pe:>6} {roe:>6} {dy:>5} {graham:>8} {upside:>8}  {r['data_as_of']}")
        print(f"           {r['reasons']}")

    print(f"\n[2/3] Reversal / Bottom-Fishing Screen (fetching live technicals)...")
    fund, screener = load_data()
    all_syms = [k for k in fund if k not in ("holdings","updated_at") and isinstance(fund.get(k),dict)]
    reversal_df = screen_reversal(all_syms, min_score=45)
    print(f"\n  {'SYMBOL':<8} {'R.SCORE':>8} {'PRICE':>8} {'RSI':>5} {'FROM LOW':>9} {'FROM HIGH':>10} {'VOL SPIKE':>10} {'MACD':>6}")
    print("  " + "-"*70)
    for _, r in reversal_df.head(10).iterrows():
        print(f"  {r['symbol']:<8} {r['reversal_score']:>8.1f} {r['price']:>8.2f} {r['rsi']:>5} {r['pct_from_low']:>9} {r['pct_from_high']:>10} {r['vol_climax']:>10} {r['macd_rising']:>6}")
        print(f"           {r['reasons']}")

    if not reversal_df.empty and not value_df.empty:
        print(f"\n[3/3] Hidden Gems (Value + Reversal combined)...")
        gems_df = screen_hidden_gems(value_min=50, reversal_min=40)
        if not gems_df.empty:
            print(f"\n  {'SYMBOL':<8} {'COMBINED':>9} {'VALUE':>7} {'REVERSAL':>9} {'PRICE':>8} {'RSI':>5} {'P/E':>6} {'ROE':>6}")
            print("  " + "-"*65)
            for _, r in gems_df.head(8).iterrows():
                pe  = f"{r['pe_ratio']:.1f}" if r['pe_ratio'] else "N/A"
                roe = f"{r['roe']:.1f}%" if r['roe'] else "N/A"
                print(f"  {r['symbol']:<8} {r['combined_score']:>9.1f} {r['value_score']:>7.1f} {r['reversal_score']:>9.1f} {r['price']:>8.2f} {r['rsi']:>5} {pe:>6} {roe:>6}")
                print(f"           Value:    {r['value_reasons']}")
                print(f"           Reversal: {r['reversal_reasons']}")
        else:
            print("  No hidden gems found at current thresholds.")
