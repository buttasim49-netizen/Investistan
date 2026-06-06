"""
Elliott Wave & Wyckoff Analysis Module
=======================================
Elliott Wave  -- detects swing points, counts waves, gives Fibonacci targets
Wyckoff       -- identifies accumulation/distribution phases and key events
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


# ============================================================
# ELLIOTT WAVE
# ============================================================

def find_swing_points(close: pd.Series, order: int = 5):
    """
    Find local swing highs and lows.
    order = how many bars each side must be lower/higher to qualify.
    Returns (highs_idx, lows_idx).
    """
    arr = close.values
    highs = argrelextrema(arr, np.greater_equal, order=order)[0]
    lows  = argrelextrema(arr, np.less_equal,    order=order)[0]
    return highs, lows


def fibonacci_levels(low: float, high: float, direction: str = "up"):
    """
    Return key Fibonacci retracement and extension levels.
    direction='up'  -> retracements below high, extensions above high
    direction='down'-> retracements above low,  extensions below low
    """
    diff = high - low
    retrace_ratios   = [0.236, 0.382, 0.500, 0.618, 0.786]
    extension_ratios = [1.000, 1.272, 1.618, 2.000, 2.618]

    if direction == "up":
        retracements = {f"{int(r*100)}%": round(high - diff * r, 2) for r in retrace_ratios}
        extensions   = {f"{int(r*100)}%": round(low  + diff * r, 2) for r in extension_ratios}
    else:
        retracements = {f"{int(r*100)}%": round(low  + diff * r, 2) for r in retrace_ratios}
        extensions   = {f"{int(r*100)}%": round(high - diff * r, 2) for r in extension_ratios}

    return retracements, extensions


def _pct_retrace(wave_start, wave_end, retrace_end):
    """How much of wave_start->wave_end did the retracement cover?"""
    wave_len = abs(wave_end - wave_start)
    if wave_len == 0:
        return 0
    return abs(retrace_end - wave_end) / wave_len


def detect_elliott_wave(df: pd.DataFrame, order: int = 5):
    """
    Attempt to identify the current Elliott Wave position in recent price action.

    Returns a dict with:
      wave_count    -- best guess label e.g. 'Wave 3 of Impulse'
      current_wave  -- integer 1-5 or string A/B/C
      pattern       -- 'Impulse' or 'Corrective'
      key_levels    -- dict of Fibonacci support/resistance
      description   -- plain English explanation
      confidence    -- Low / Medium / High
      swing_points  -- list of recent swing high/low prices with index
    """
    if df is None or len(df) < 50:
        return {"wave_count": "Insufficient data", "description": "Need 50+ bars", "confidence": "N/A"}

    close  = df["Close"]
    high_idx, low_idx = find_swing_points(close, order=order)

    # Build merged, time-sorted swing point list
    swings = []
    for i in high_idx:
        swings.append({"idx": i, "price": float(close.iloc[i]), "type": "H"})
    for i in low_idx:
        swings.append({"idx": i, "price": float(close.iloc[i]), "type": "L"})
    swings.sort(key=lambda x: x["idx"])

    # Deduplicate: remove duplicate consecutive H or L, keep extreme
    deduped = []
    for s in swings:
        if deduped and deduped[-1]["type"] == s["type"]:
            # Keep the more extreme one
            if s["type"] == "H" and s["price"] > deduped[-1]["price"]:
                deduped[-1] = s
            elif s["type"] == "L" and s["price"] < deduped[-1]["price"]:
                deduped[-1] = s
        else:
            deduped.append(s)

    current_price = float(close.iloc[-1])

    if len(deduped) < 4:
        return {
            "wave_count":   "Pattern unclear",
            "current_wave": "?",
            "pattern":      "Unknown",
            "key_levels":   {},
            "description":  "Not enough distinct swing points to count waves reliably.",
            "confidence":   "Low",
            "swing_points": deduped,
        }

    # Use the last 6 swing points for wave counting
    recent = deduped[-6:]
    prices = [s["price"] for s in recent]
    types  = [s["type"]  for s in recent]

    # --- Impulse wave detection (5-wave structure: L-H-L-H-L-H or H-L-H-L-H-L) ---
    impulse_up   = False
    impulse_down = False
    wave_num     = 0
    pattern      = "Corrective"
    confidence   = "Low"

    # Check for upward impulse pattern: L H L H L H
    if len(prices) >= 6:
        p = prices[-6:]
        t = types[-6:]
        if t == ["L","H","L","H","L","H"]:
            # Validate Elliott rules for upward impulse
            w1_low, w1_high = p[0], p[1]
            w2_low          = p[2]
            w3_high         = p[3]
            w4_low          = p[4]
            w5_high         = p[5]

            w1_len = w1_high - w1_low
            w3_len = w3_high - w2_low
            w2_ret = _pct_retrace(w1_low, w1_high, w2_low)
            w4_ret = _pct_retrace(w2_low, w3_high, w4_low)

            rules_ok = (
                w2_low > w1_low and          # W2 doesn't go below W1 start
                w3_high > w1_high and        # W3 makes new high
                w4_low > w1_high and         # W4 doesn't enter W1 territory
                w3_len >= w1_len * 1.0 and   # W3 not shorter than W1
                0.236 <= w2_ret <= 0.786 and # W2 retraces 23.6-78.6% of W1
                0.236 <= w4_ret <= 0.500      # W4 retraces 23.6-50% of W3
            )
            if rules_ok:
                impulse_up = True
                pattern    = "Impulse (Upward)"
                confidence = "High" if w2_ret >= 0.382 and w4_ret <= 0.382 else "Medium"
                # Determine current position: after W5 high, could be start of correction
                if current_price < w5_high * 0.97:
                    wave_num = "A (start of correction after Wave 5)"
                else:
                    wave_num = "5 (near completion of impulse)"

        # Check for downward impulse: H L H L H L
        elif t == ["H","L","H","L","H","L"]:
            w1_high, w1_low = p[0], p[1]
            w2_high         = p[2]
            w3_low          = p[3]
            w4_high         = p[4]
            w5_low          = p[5]
            w1_len = w1_high - w1_low
            w3_len = w2_high - w3_low
            rules_ok = (
                w2_high < w1_high and
                w3_low  < w1_low  and
                w4_high < w1_low  and
                w3_len  >= w1_len
            )
            if rules_ok:
                impulse_down = True
                pattern      = "Impulse (Downward)"
                confidence   = "Medium"
                wave_num     = "5 (downward near completion)"

    # --- Partial impulse check (3-4 swing points) ---
    if not impulse_up and not impulse_down and len(recent) >= 4:
        p = prices[-4:]
        t = types[-4:]

        if t == ["L","H","L","H"]:
            w1_low, w1_high, w2_low, w3_high = p
            w2_ret = _pct_retrace(w1_low, w1_high, w2_low)
            if (w2_low > w1_low and w3_high > w1_high and
                    0.236 <= w2_ret <= 0.786):
                pattern    = "Impulse (Upward) - Early"
                confidence = "Medium"
                wave_num   = 3
                if current_price < w3_high * 0.98:
                    wave_num = "4 (pullback expected before Wave 5)"
                else:
                    wave_num = "3 (strongest wave -- momentum phase)"

        elif t == ["H","L","H","L"]:
            # Possible wave 1-2 of downward impulse OR corrective ABC
            w1_high, w1_low, w2_high, w3_low = p
            abc_ratio = (w2_high - w1_low) / (w1_high - w1_low)
            if 0.382 <= abc_ratio <= 0.786:
                pattern    = "Corrective A-B-C"
                confidence = "Medium"
                wave_num   = "C (final leg of correction)"
            else:
                pattern    = "Impulse (Downward) - Early"
                wave_num   = 3
                confidence = "Low"

        elif t == ["L","H","L"] or t[-3:] == ["L","H","L"]:
            pattern    = "Impulse - Wave 1 or A"
            wave_num   = "1 or A (early -- watching for Wave 2 pullback)"
            confidence = "Low"

        elif t == ["H","L","H"] or t[-3:] == ["H","L","H"]:
            pattern    = "Corrective B or Wave 2"
            wave_num   = "B or 2 (rebound -- watch for next leg down)"
            confidence = "Low"

    # --- Fibonacci levels from most recent significant swing ---
    fib_levels = {}
    if len(deduped) >= 2:
        last_two = deduped[-2:]
        if last_two[0]["type"] == "L" and last_two[1]["type"] == "H":
            ret, ext = fibonacci_levels(last_two[0]["price"], last_two[1]["price"], "up")
            fib_levels = {"Retracements": ret, "Extensions": ext,
                          "Swing Low": last_two[0]["price"],
                          "Swing High": last_two[1]["price"]}
        elif last_two[0]["type"] == "H" and last_two[1]["type"] == "L":
            ret, ext = fibonacci_levels(last_two[1]["price"], last_two[0]["price"], "down")
            fib_levels = {"Retracements": ret, "Extensions": ext,
                          "Swing High": last_two[0]["price"],
                          "Swing Low":  last_two[1]["price"]}

    # --- Generate description ---
    wn = str(wave_num)
    if "3" in wn and "Impulse" in pattern:
        desc = (f"Possible Wave 3 of an upward impulse -- typically the strongest and "
                f"longest wave. Momentum should be building. Watch for volume expansion "
                f"and RSI staying above 50. Fibonacci extension target: "
                f"1.618x Wave 1 from Wave 2 low.")
    elif "4" in wn:
        desc = (f"Possible Wave 4 pullback -- corrective, should hold above Wave 1 high. "
                f"Typical retracement 23.6-38.2% of Wave 3. Good re-entry zone if "
                f"holding above support. Wave 5 push to new highs expected after.")
    elif "5" in wn and "completion" in wn:
        desc = (f"Possible near Wave 5 completion -- final push of the impulse. "
                f"Watch for momentum divergence (RSI making lower high while price "
                f"makes higher high). Start tightening stops. Corrective A-B-C likely to follow.")
    elif "A" in wn and "correction" in wn:
        desc = ("Post-impulse correction underway (Wave A). Expect 3-wave A-B-C "
                "correction. Wave A typically retraces 38.2-61.8% of full impulse. "
                "Avoid new longs until correction completes.")
    elif "C" in wn:
        desc = ("Wave C of corrective A-B-C -- final corrective leg. Often matches "
                "Wave A in length. Good accumulation zone if C reaches 61.8-78.6% "
                "retracement of the prior impulse.")
    elif "1" in wn:
        desc = ("Possible Wave 1 start -- early impulse, often unrecognized. "
                "Watch for Wave 2 pullback (38.2-61.8% retrace) as the better entry "
                "before Wave 3 acceleration.")
    else:
        desc = ("Wave pattern unclear -- not enough distinct swings or pattern doesn't "
                "satisfy Elliott rules. Use Fibonacci levels as support/resistance guides.")

    if not wave_num:
        wave_num = "?"
        pattern  = "Unclear"
        confidence = "Low"

    return {
        "wave_count":   f"Wave {wave_num}",
        "current_wave": wave_num,
        "pattern":      pattern,
        "key_levels":   fib_levels,
        "description":  desc,
        "confidence":   confidence,
        "swing_points": deduped[-8:],   # last 8 for charting
    }


# ============================================================
# WYCKOFF METHOD
# ============================================================

def detect_wyckoff(df: pd.DataFrame, lookback: int = 60):
    """
    Detect Wyckoff market phases and key events.

    Phases:
      Phase A -- Stopping the prior trend (SC, AR, ST)
      Phase B -- Building a cause (test of extremes)
      Phase C -- Spring/Upthrust (final shakeout or test)
      Phase D -- Trend emerges (SOS/LPS or SOW/LPSY)
      Phase E -- Trend in full swing (Markup or Markdown)

    Returns dict with phase, events, description, bias (Bullish/Bearish).
    """
    if df is None or len(df) < 30:
        return {"phase": "Insufficient data", "bias": "Unknown", "events": [], "description": ""}

    # Use last `lookback` bars
    sub   = df.tail(lookback).copy()
    close = sub["Close"].values
    high  = sub["High"].values
    low   = sub["Low"].values
    vol   = sub["Volume"].values.astype(float)

    n = len(close)
    if n < 20:
        return {"phase": "Insufficient data", "bias": "Unknown", "events": [], "description": ""}

    # --- Utility ---
    vol_avg     = np.mean(vol)
    vol_std     = np.std(vol)
    price_range = np.max(high) - np.min(low)
    price_mid   = (np.max(high) + np.min(low)) / 2
    current     = float(close[-1])
    recent_high = float(np.max(high[-20:]))
    recent_low  = float(np.min(low[-20:]))

    # Bar spreads (high-low range per bar)
    spreads = high - low
    spread_avg = np.mean(spreads)

    # Volume MA
    vol_ma20 = np.convolve(vol, np.ones(20)/20, mode='valid')

    events = []

    # --- Event detection ---

    # Selling Climax (SC): Very high volume + wide spread + close near low (bearish bar)
    for i in range(5, n-2):
        if (vol[i] > vol_avg + 1.5*vol_std and
                spreads[i] > spread_avg * 1.5 and
                close[i] < (high[i] + low[i]) / 2 and
                close[i] < close[i-1]):
            events.append({
                "event": "Selling Climax (SC)",
                "bar":   i,
                "price": round(float(low[i]), 2),
                "meaning": "Panic selling exhaustion -- potential bottom forming"
            })

    # Buying Climax (BC): Very high volume + wide spread + close near high
    for i in range(5, n-2):
        if (vol[i] > vol_avg + 1.5*vol_std and
                spreads[i] > spread_avg * 1.5 and
                close[i] > (high[i] + low[i]) / 2 and
                close[i] > close[i-1]):
            events.append({
                "event": "Buying Climax (BC)",
                "bar":   i,
                "price": round(float(high[i]), 2),
                "meaning": "Buying exhaustion -- potential top forming"
            })

    # Sign of Strength (SOS): Strong up bar on above-average volume
    for i in range(3, n):
        if (close[i] > close[i-1] * 1.02 and
                vol[i] > vol_avg * 1.3 and
                spreads[i] > spread_avg):
            events.append({
                "event": "Sign of Strength (SOS)",
                "bar":   i,
                "price": round(float(close[i]), 2),
                "meaning": "Institutional buying -- markup likely beginning"
            })

    # Sign of Weakness (SOW): Strong down bar on above-average volume
    for i in range(3, n):
        if (close[i] < close[i-1] * 0.98 and
                vol[i] > vol_avg * 1.3 and
                spreads[i] > spread_avg):
            events.append({
                "event": "Sign of Weakness (SOW)",
                "bar":   i,
                "price": round(float(close[i]), 2),
                "meaning": "Distribution -- markdown may be starting"
            })

    # Spring: Dip below recent support on low volume, quick recovery
    support = float(np.min(low[-lookback//2:]))
    for i in range(5, n-3):
        if (low[i] < support * 1.01 and
                close[i] > low[i] * 1.01 and
                vol[i] < vol_avg * 0.8):
            events.append({
                "event": "Spring",
                "bar":   i,
                "price": round(float(low[i]), 2),
                "meaning": "False breakdown below support -- strong buy signal if volume confirms next bar"
            })

    # Upthrust (UT): Poke above resistance on high vol, fails to hold
    resistance = float(np.max(high[-lookback//2:]))
    for i in range(5, n-3):
        if (high[i] > resistance * 0.99 and
                close[i] < resistance * 0.99 and
                vol[i] > vol_avg * 1.2):
            events.append({
                "event": "Upthrust (UT)",
                "bar":   i,
                "price": round(float(high[i]), 2),
                "meaning": "False breakout above resistance -- bearish; distribution likely"
            })

    # Last Point of Support (LPS): Higher low on declining volume before SOS
    for i in range(5, n-2):
        if (low[i] > recent_low * 1.01 and
                close[i] > close[i-1] and
                vol[i] < vol_ma20[min(i, len(vol_ma20)-1)] * 0.7):
            events.append({
                "event": "Last Point of Support (LPS)",
                "bar":   i,
                "price": round(float(low[i]), 2),
                "meaning": "Quiet pullback to support -- low-risk re-entry before markup"
            })

    # Keep only most recent of each event type
    seen_types = {}
    for e in reversed(events):
        if e["event"] not in seen_types:
            seen_types[e["event"]] = e
    recent_events = list(seen_types.values())
    recent_events.sort(key=lambda x: x["bar"], reverse=True)

    # --- Phase determination ---
    # Score bullish vs bearish events
    bullish_events = {"Selling Climax (SC)", "Sign of Strength (SOS)", "Spring",
                      "Last Point of Support (LPS)"}
    bearish_events = {"Buying Climax (BC)", "Sign of Weakness (SOW)", "Upthrust (UT)"}

    bull_score = sum(1 for e in recent_events if e["event"] in bullish_events)
    bear_score = sum(1 for e in recent_events if e["event"] in bearish_events)

    # Price position relative to range
    price_position = (current - float(np.min(low))) / price_range if price_range > 0 else 0.5

    # Volume trend (is volume expanding on up or down moves?)
    recent_up_vol   = np.mean([vol[i] for i in range(max(0,n-10), n) if close[i] > close[i-1]] or [0])
    recent_down_vol = np.mean([vol[i] for i in range(max(0,n-10), n) if close[i] < close[i-1]] or [0])
    vol_bias = "Bullish" if recent_up_vol > recent_down_vol else "Bearish"

    # Determine phase
    has_sc  = any(e["event"] == "Selling Climax (SC)"    for e in recent_events)
    has_bc  = any(e["event"] == "Buying Climax (BC)"     for e in recent_events)
    has_sos = any(e["event"] == "Sign of Strength (SOS)" for e in recent_events)
    has_sow = any(e["event"] == "Sign of Weakness (SOW)" for e in recent_events)
    has_sp  = any(e["event"] == "Spring"                 for e in recent_events)
    has_ut  = any(e["event"] == "Upthrust (UT)"          for e in recent_events)
    has_lps = any(e["event"] == "Last Point of Support (LPS)" for e in recent_events)

    if has_sos and has_lps and price_position > 0.5:
        phase = "Phase D/E -- Markup (Uptrend in progress)"
        bias  = "Bullish"
        desc  = ("Wyckoff markup phase detected. Institutions have accumulated and price "
                 "is trending up. Sign of Strength (SOS) confirms buying. Look for "
                 "Last Point of Support (LPS) pullbacks as low-risk entries. Hold longs "
                 "until Signs of Weakness appear.")

    elif has_sc and has_sp:
        phase = "Phase C -- Spring/Test (Accumulation near completion)"
        bias  = "Bullish"
        desc  = ("Classic Wyckoff Spring detected -- price briefly dipped below support "
                 "on low volume then recovered. This is a bullish shakeout designed to "
                 "shake out weak holders before markup. Watch for a Sign of Strength "
                 "(SOS) breakout on high volume to confirm entry.")

    elif has_sc and not has_sos:
        phase = "Phase A/B -- Accumulation (Base building)"
        bias  = "Neutral/Bullish"
        desc  = ("Selling Climax suggests the prior downtrend has exhausted. Price is "
                 "now base-building in a trading range (Phases A-B). Expect sideways "
                 "chop while institutions accumulate. Buy near the low of the range, "
                 "avoid chasing breakouts until volume confirms.")

    elif has_bc and has_ut:
        phase = "Phase C -- Upthrust (Distribution near completion)"
        bias  = "Bearish"
        desc  = ("Wyckoff Upthrust detected -- price broke above resistance briefly "
                 "on high volume but failed to hold. This is a bearish false breakout "
                 "designed to trap late buyers before markdown. Watch for Sign of "
                 "Weakness (SOW) to confirm distribution is complete.")

    elif has_bc and not has_sow:
        phase = "Phase A/B -- Distribution (Top forming)"
        bias  = "Neutral/Bearish"
        desc  = ("Buying Climax suggests the prior uptrend may be exhausting. Price "
                 "is forming a distribution range at the top. Institutions are selling "
                 "into strength. Reduce longs near the top of the range. Watch for "
                 "Signs of Weakness (SOW) to confirm markdown is beginning.")

    elif has_sow and price_position < 0.4:
        phase = "Phase D/E -- Markdown (Downtrend in progress)"
        bias  = "Bearish"
        desc  = ("Wyckoff markdown phase -- distribution is complete and price is "
                 "declining. Signs of Weakness (SOW) on high volume confirm selling "
                 "pressure. Avoid longs. Short-sellers look for Last Point of Supply "
                 "(LPSY) rallies as entry points.")

    elif vol_bias == "Bullish" and price_position > 0.6:
        phase = "Possible Markup (no climax events detected)"
        bias  = "Bullish"
        desc  = ("Volume is expanding on up-days and price is in the upper part of "
                 "its recent range. While no classic Wyckoff climax events are clearly "
                 "identified, the volume/price relationship is bullish. "
                 "Trend-following is appropriate.")

    elif vol_bias == "Bearish" and price_position < 0.4:
        phase = "Possible Markdown (no climax events detected)"
        bias  = "Bearish"
        desc  = ("Volume is expanding on down-days and price is in the lower part of "
                 "its recent range. The volume/price relationship is bearish. "
                 "Avoid new longs; wait for a Selling Climax to signal exhaustion.")

    else:
        phase = "Trading Range (Phase B -- Cause building)"
        bias  = "Neutral"
        desc  = ("Price is moving sideways in a trading range without clear climax "
                 "events. This is typical of Wyckoff Phase B where the market is "
                 "building a cause (either accumulation or distribution). "
                 "Trade the range: buy support, sell resistance. Wait for breakout "
                 "with volume confirmation before taking directional positions.")

    return {
        "phase":         phase,
        "bias":          bias,
        "events":        recent_events[:5],   # Top 5 most recent
        "description":   desc,
        "bull_score":    bull_score,
        "bear_score":    bear_score,
        "vol_bias":      vol_bias,
        "price_position": round(price_position * 100, 1),
        "support":       round(support, 2),
        "resistance":    round(resistance, 2),
        "trading_range": f"PKR {round(float(np.min(low)),2)} - PKR {round(float(np.max(high)),2)}",
    }
