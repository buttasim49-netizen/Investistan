"""
Portfolio Manager
=================
Stores holdings in portfolio.json.
Provides P&L, hold/add/exit signals, and exit alerts per position.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

PORTFOLIO_FILE = Path(__file__).parent / "portfolio.json"
YAHOO_ALIASES  = {"ENGROH": "DAWH"}


def load_portfolio() -> dict:
    if PORTFOLIO_FILE.exists():
        try:
            with open(PORTFOLIO_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"holdings": [], "updated_at": ""}


def save_portfolio(data: dict):
    data["updated_at"] = datetime.now().isoformat()
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(data, f, indent=2)


def add_holding(symbol, quantity, avg_price, stop_loss=None, target=None):
    port = load_portfolio()
    symbol = symbol.upper().strip()
    for h in port["holdings"]:
        if h["symbol"] == symbol:
            h["quantity"] = quantity
            h["avg_price"] = avg_price
            h["stop_loss"] = stop_loss
            h["target"] = target
            h["added_at"] = datetime.now().isoformat()
            save_portfolio(port)
            return
    port["holdings"].append({
        "symbol": symbol, "quantity": quantity, "avg_price": avg_price,
        "stop_loss": stop_loss, "target": target,
        "added_at": datetime.now().isoformat(),
    })
    save_portfolio(port)


def remove_holding(symbol):
    port = load_portfolio()
    port["holdings"] = [h for h in port["holdings"] if h["symbol"] != symbol.upper()]
    save_portfolio(port)


# ── Trade Log ─────────────────────────────────────────────────────────────────
TRADE_LOG_FILE = Path(__file__).parent / "trade_log.json"

def load_trade_log() -> list:
    if TRADE_LOG_FILE.exists():
        try:
            with open(TRADE_LOG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return []

def _save_trade_log(log: list):
    with open(TRADE_LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)

def record_trade(symbol: str, action: str, quantity: float,
                 price: float, avg_buy_price: float = None, notes: str = ""):
    """
    Record a trade in the permanent log.
    action: 'BUY' or 'SELL'
    """
    log = load_trade_log()
    entry = {
        "date":          datetime.now().strftime("%Y-%m-%d %H:%M"),
        "symbol":        symbol.upper(),
        "action":        action,
        "quantity":      quantity,
        "price":         price,
        "value":         round(quantity * price, 2),
        "avg_buy_price": avg_buy_price,
        "realized_pnl":  None,
        "realized_pnl_pct": None,
        "notes":         notes,
    }
    if action == "SELL" and avg_buy_price:
        pnl     = (price - avg_buy_price) * quantity
        pnl_pct = (price - avg_buy_price) / avg_buy_price * 100
        entry["realized_pnl"]     = round(pnl, 2)
        entry["realized_pnl_pct"] = round(pnl_pct, 2)
    log.insert(0, entry)   # newest first
    _save_trade_log(log)
    return entry

def sell_holding(symbol: str, sell_qty: float, sell_price: float, notes: str = ""):
    """
    Sell a position (fully or partially).
    Records the trade in the log, reduces/removes from holdings.
    Returns the trade log entry.
    """
    port = load_portfolio()
    symbol = symbol.upper()

    holding = next((h for h in port["holdings"] if h["symbol"] == symbol), None)
    if not holding:
        return None

    avg_buy = holding.get("avg_price")
    hold_qty = holding["quantity"]

    # Record the sale
    entry = record_trade(symbol, "SELL", sell_qty, sell_price, avg_buy, notes)

    # Update holding
    remaining = hold_qty - sell_qty
    if remaining <= 0:
        # Fully sold — remove from holdings
        port["holdings"] = [h for h in port["holdings"] if h["symbol"] != symbol]
    else:
        # Partial sell — reduce quantity
        for h in port["holdings"]:
            if h["symbol"] == symbol:
                h["quantity"] = remaining
                break

    save_portfolio(port)
    return entry


def get_current_price(symbol: str) -> float | None:
    """
    Fetch current price with 3-source fallback:
    1. yfinance (.KA / .KX)  — fast, covers most KSE stocks
    2. DPS PSX company page  — covers everything including newer listings
    3. PSXTerminal            — last resort
    """
    import re, requests
    from bs4 import BeautifulSoup

    base    = YAHOO_ALIASES.get(symbol, symbol)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    # 1. yfinance
    for sfx in [".KA", ".KX"]:
        try:
            df = yf.download(f"{base}{sfx}", period="2d",
                             progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty:
                p = float(df["Close"].dropna().iloc[-1])
                if p > 0:
                    return round(p, 2)
        except Exception:
            pass

    # 2. DPS PSX company page (covers all listed stocks including new ones)
    try:
        resp = requests.get(f"https://dps.psx.com.pk/company/{symbol}",
                            headers=headers, timeout=8)
        if resp.status_code == 200:
            text = BeautifulSoup(resp.text, "html.parser").get_text()
            for pat in [r"Rs\.\s*([\d,\.]+)", r"PKR\s*([\d,\.]+)"]:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    p = float(m.group(1).replace(",", ""))
                    if 0.1 < p < 1_000_000:
                        return round(p, 2)
    except Exception:
        pass

    # 3. PSXTerminal symbol page
    try:
        resp = requests.get(f"https://psxterminal.com/symbol/{symbol}",
                            headers=headers, timeout=8)
        if resp.status_code == 200:
            text = BeautifulSoup(resp.text, "html.parser").get_text()
            m = re.search(r"(?:Current Price|Last Price|PKR)\s*([\d,\.]+)",
                          text, re.IGNORECASE)
            if m:
                p = float(m.group(1).replace(",", ""))
                if 5 < p < 100_000:
                    return round(p, 2)
    except Exception:
        pass

    return None


def _fetch_all_prices_dps(symbols: list) -> dict:
    """
    Fetch current prices for multiple symbols in ONE request from DPS PSX screener.
    Returns {symbol: price}. Falls back to individual DPS pages for any missing.
    """
    import re, requests
    from bs4 import BeautifulSoup

    prices  = {}
    sym_set = set(symbols)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    try:
        resp = requests.get("https://dps.psx.com.pk/screener",
                            headers=headers, timeout=15)
        if resp.status_code == 200:
            soup  = BeautifulSoup(resp.text, "html.parser")
            table = soup.find("table")
            if table:
                hdrs      = [th.get_text(strip=True).upper() for th in table.find_all("th")]
                sym_col   = next((i for i,h in enumerate(hdrs) if h=="SYMBOL"), 0)
                price_col = next((i for i,h in enumerate(hdrs) if h=="PRICE"),  4)
                for tr in table.find_all("tr")[1:]:
                    cells = [td.get_text(strip=True) for td in tr.find_all("td")]
                    if len(cells) > price_col:
                        sym = cells[sym_col].strip().upper()
                        if sym in sym_set:
                            try:
                                p = float(cells[price_col].replace(",",""))
                                if p > 0:
                                    prices[sym] = round(p, 2)
                            except Exception:
                                pass
    except Exception:
        pass

    # Per-stock fallback for anything still missing
    for sym in [s for s in symbols if s not in prices]:
        try:
            resp = requests.get(f"https://dps.psx.com.pk/company/{sym}",
                                headers=headers, timeout=8)
            if resp.status_code == 200:
                text = BeautifulSoup(resp.text, "html.parser").get_text()
                for pat in [r"Rs\.\s*([\d,\.]+)", r"PKR\s*([\d,\.]+)"]:
                    m = re.search(pat, text, re.IGNORECASE)
                    if m:
                        p = float(m.group(1).replace(",",""))
                        if 0.1 < p < 1_000_000:
                            prices[sym] = round(p, 2)
                            break
        except Exception:
            pass

    return prices


def analyse_portfolio(screener_results=None, holdings=None):
    if holdings is None:
        holdings = load_portfolio().get("holdings", [])
    results  = []

    # Batch-fetch all prices in ONE request (fast, correct, uses DPS screener)
    symbols    = [h["symbol"] for h in holdings]
    all_prices = _fetch_all_prices_dps(symbols)

    for h in holdings:
        sym       = h["symbol"]
        qty       = h["quantity"]
        avg_price = h["avg_price"]
        stop_loss = h.get("stop_loss")
        target    = h.get("target")

        # Use batch DPS price — most accurate source
        cur_price = all_prices.get(sym)

        if not cur_price:
            results.append({"symbol": sym, "quantity": qty, "avg_price": avg_price,
                            "current_price": None, "pnl_pct": None, "pnl_pkr": None,
                            "value": None, "cost": avg_price*qty, "alerts": ["No price data"],
                            "action": "UNKNOWN", "screener_rec": "N/A",
                            "stop_loss": stop_loss, "target": target})
            continue
        pnl_pct = (cur_price - avg_price) / avg_price * 100
        pnl_pkr = (cur_price - avg_price) * qty
        alerts  = []

        # ── Pull screener data ────────────────────────────────────────────────
        screener_rec   = "N/A"
        tech_score     = 0
        fund_score     = 0
        swing_score    = 0
        trend_bullish  = False
        rsi_val        = None

        if screener_results is not None and not screener_results.empty:
            m = screener_results[(screener_results["symbol"]==sym) &
                                  (screener_results["horizon"]=="swing")]
            if not m.empty:
                r            = m.iloc[0]
                screener_rec = str(r.get("recommendation","N/A"))
                swing_score  = float(r.get("weighted_score", 0) or 0)
                tech_score   = float(str(r.get("technical_score","0")).replace("N/A","0") or 0)
                fund_score   = float(r.get("fundamental_score", 0) or 0)
                trend_col    = str(r.get("trend","")).lower()
                trend_bullish= "bull" in trend_col

        # ── Smart recommendation engine ───────────────────────────────────────
        alerts_add = []
        avoid_flag = "AVOID" in screener_rec.upper()

        # Hard exits
        if stop_loss and cur_price <= stop_loss:
            action = "EXIT -- Stop Loss Hit"
            alerts_add.append(f"Stop loss hit: PKR {cur_price:.2f} <= PKR {stop_loss:.2f}")

        elif target and cur_price >= target:
            action = "EXIT -- Target Reached"
            alerts_add.append(f"Target hit: PKR {cur_price:.2f} >= PKR {target:.2f}")

        # Strong sell signals
        elif pnl_pct <= -25 and (avoid_flag or not trend_bullish):
            action = "CUT LOSS -- Deploy Elsewhere"
            alerts_add.append(
                f"Down {pnl_pct:.1f}% with {'bearish trend' if not trend_bullish else 'weak screener signal'}. "
                f"Capital may be better deployed in a stronger opportunity."
            )

        elif pnl_pct <= -20 and avoid_flag:
            action = "SELL -- Screener AVOID + Large Loss"
            alerts_add.append(
                f"Down {pnl_pct:.1f}% AND screener signals AVOID. "
                f"Both price action and fundamentals working against you."
            )

        elif pnl_pct <= -15 and tech_score < 40 and not trend_bullish:
            action = "REDUCE -- Technically Weak"
            alerts_add.append(
                f"Down {pnl_pct:.1f}%, tech score only {tech_score:.0f}/100, bearish trend. "
                f"Consider cutting 50% and deploying into a stronger name."
            )

        elif pnl_pct <= -15 and avoid_flag:
            action = "REDUCE / EXIT"
            alerts_add.append(f"Down {pnl_pct:.1f}% — screener says AVOID. Review your thesis.")

        elif pnl_pct <= -10 and tech_score < 35:
            action = "REVIEW -- Weak Technicals"
            alerts_add.append(
                f"Down {pnl_pct:.1f}% with very weak technical score ({tech_score:.0f}/100). "
                f"Set a stop loss if you don't have one."
            )

        # Profit taking
        elif pnl_pct >= 50:
            action = "TAKE PROFITS -- Large Gain"
            alerts_add.append(
                f"Up {pnl_pct:.1f}% — excellent gain. Consider taking 50-75% off the table "
                f"and letting the rest run with a trailing stop."
            )

        elif pnl_pct >= 30 and (avoid_flag or tech_score < 50):
            action = "TAKE PROFITS -- Momentum Fading"
            alerts_add.append(
                f"Up {pnl_pct:.1f}% but {'screener turning AVOID' if avoid_flag else f'tech score falling to {tech_score:.0f}/100'}. "
                f"Good time to lock in gains."
            )

        elif pnl_pct >= 20:
            action = "TAKE PARTIAL PROFITS"
            alerts_add.append(
                f"Up {pnl_pct:.1f}% — consider taking 30-50% profits. "
                f"Move stop loss to breakeven on remainder."
            )

        # Positive signals
        elif swing_score >= 75 and trend_bullish and pnl_pct > -10:
            action = "HOLD / ADD MORE"
            if pnl_pct < 0:
                alerts_add.append(
                    f"Down {pnl_pct:.1f}% but score is {swing_score:.0f}/100 and trend is bullish. "
                    f"This is a potential averaging opportunity."
                )

        elif swing_score >= 60 and trend_bullish:
            action = "HOLD -- Strong Setup"

        elif swing_score >= 45:
            action = "HOLD"

        elif pnl_pct <= -15:
            action = "REVIEW -- Reassess Position"
            alerts_add.append(f"Down {pnl_pct:.1f}% — reassess your original thesis for holding.")

        else:
            action = "HOLD -- Monitor"

        alerts.extend(alerts_add)

        results.append({
            "symbol":       sym,
            "quantity":     qty,
            "avg_price":    avg_price,
            "current_price":cur_price,
            "pnl_pct":      round(pnl_pct, 2),
            "pnl_pkr":      round(pnl_pkr, 2),
            "value":        round(cur_price * qty, 2),
            "cost":         round(avg_price * qty, 2),
            "alerts":       alerts,
            "action":       action,
            "screener_rec": screener_rec,
            "tech_score":   tech_score,
            "swing_score":  swing_score,
            "trend_bullish":trend_bullish,
            "stop_loss":    stop_loss,
            "target":       target,
        })
    return results


def portfolio_summary(holdings_analysis):
    total_cost  = sum(h["cost"]  for h in holdings_analysis if h["cost"])
    total_value = sum(h["value"] for h in holdings_analysis if h["value"])
    total_pnl   = total_value - total_cost if total_cost else 0
    pnl_pct     = total_pnl / total_cost * 100 if total_cost else 0
    return {
        "total_cost":    round(total_cost,2),
        "total_value":   round(total_value,2),
        "total_pnl_pkr": round(total_pnl,2),
        "total_pnl_pct": round(pnl_pct,2),
        "num_holdings":  len(holdings_analysis),
        "winners":       sum(1 for h in holdings_analysis if (h["pnl_pct"] or 0) > 0),
        "losers":        sum(1 for h in holdings_analysis if (h["pnl_pct"] or 0) < 0),
        "alerts_count":  sum(len(h["alerts"]) for h in holdings_analysis),
        "exit_alerts":   sum(1 for h in holdings_analysis if "EXIT" in str(h["action"]).upper()),
    }
