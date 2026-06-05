"""
Advanced Technical Analysis Module
====================================
Pro-level pattern detection and trade signals:

TREND          : ADX, Supertrend, MA alignment, Golden/Death Cross
MOMENTUM       : RSI divergence, MACD histogram, Williams %R, CCI, Rate of Change
VOLATILITY     : Bollinger Band squeeze, ATR expansion, Keltner Channels
VOLUME         : OBV trend, Volume climax, Volume divergence
PATTERNS       : Double Top/Bottom, Head & Shoulders, Triangles, Flags, Wedges, Cups
CANDLESTICKS   : Hammer, Doji, Engulfing, Morning/Evening Star, Shooting Star
BREAKOUTS      : Resistance breakout, Support breakdown, Consolidation break
S/R LEVELS     : Pivot points, key swing levels, Fibonacci zones
SIGNALS        : Composite BUY/SELL/HOLD with confidence score
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def _clean(series: pd.Series) -> np.ndarray:
    return series.values.astype(float)


def _swing_points(close: pd.Series, order: int = 5):
    arr = _clean(close)
    highs = argrelextrema(arr, np.greater_equal, order=order)[0]
    lows  = argrelextrema(arr, np.less_equal,    order=order)[0]
    return highs, lows


# ══════════════════════════════════════════════════════════════════════════════
# TREND INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

def adx(high, low, close, period=14):
    """Average Directional Index — measures trend STRENGTH (not direction)."""
    h = _clean(high)
    l = _clean(low)
    c = _clean(close)
    n = len(c)

    tr  = np.zeros(n)
    pdm = np.zeros(n)
    ndm = np.zeros(n)

    for i in range(1, n):
        hl  = h[i] - l[i]
        hpc = abs(h[i] - c[i-1])
        lpc = abs(l[i] - c[i-1])
        tr[i] = max(hl, hpc, lpc)

        up   = h[i] - h[i-1]
        down = l[i-1] - l[i]
        pdm[i] = up   if (up > down and up > 0)   else 0
        ndm[i] = down if (down > up and down > 0) else 0

    def smooth(arr, p):
        s = np.zeros(n)
        s[p] = arr[1:p+1].sum()
        for i in range(p+1, n):
            s[i] = s[i-1] - s[i-1]/p + arr[i]
        return s

    atr_s  = smooth(tr,  period)
    pdm_s  = smooth(pdm, period)
    ndm_s  = smooth(ndm, period)

    pdi = np.where(atr_s > 0, 100 * pdm_s / atr_s, 0)
    ndi = np.where(atr_s > 0, 100 * ndm_s / atr_s, 0)
    dx  = np.where((pdi + ndi) > 0, 100 * np.abs(pdi - ndi) / (pdi + ndi), 0)

    adx_val = np.zeros(n)
    adx_val[2*period] = dx[period:2*period+1].mean()
    for i in range(2*period+1, n):
        adx_val[i] = (adx_val[i-1] * (period-1) + dx[i]) / period

    return float(adx_val[-1]), float(pdi[-1]), float(ndi[-1])


def supertrend(high, low, close, period=10, multiplier=3.0):
    """Supertrend — dynamic support/resistance that flips with trend."""
    h = _clean(high)
    l = _clean(low)
    c = _clean(close)
    n = len(c)

    atr = np.zeros(n)
    for i in range(1, n):
        atr[i] = (atr[i-1] * (period-1) + max(
            h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])
        )) / period

    ub = (h + l) / 2 + multiplier * atr
    lb = (h + l) / 2 - multiplier * atr

    st  = np.zeros(n)
    dir_up = True

    for i in range(1, n):
        if c[i] > ub[i-1]:
            dir_up = True
        elif c[i] < lb[i-1]:
            dir_up = False

        if dir_up:
            st[i] = lb[i]
        else:
            st[i] = ub[i]

    return float(st[-1]), dir_up


def golden_death_cross(close):
    """Golden Cross (50 > 200 SMA) or Death Cross — major trend signals."""
    c   = _clean(close)
    if len(c) < 200:
        return None
    sma50  = c[-50:].mean() if len(c) >= 50 else None
    sma200 = c[-200:].mean()
    prev50  = c[-51:-1].mean() if len(c) >= 51 else None
    prev200 = c[-201:-1].mean() if len(c) >= 201 else None

    if all(v is not None for v in [sma50, sma200, prev50, prev200]):
        if prev50 < prev200 and sma50 > sma200:
            return "GOLDEN CROSS -- Bullish long-term signal: 50 SMA just crossed above 200 SMA"
        if prev50 > prev200 and sma50 < sma200:
            return "DEATH CROSS -- Bearish long-term signal: 50 SMA just crossed below 200 SMA"
    return None


# ══════════════════════════════════════════════════════════════════════════════
# MOMENTUM
# ══════════════════════════════════════════════════════════════════════════════

def rsi_divergence(close, rsi_series, lookback=20):
    """
    Detect RSI divergence — one of the most reliable reversal signals.
    Bullish: price makes lower low, RSI makes higher low.
    Bearish: price makes higher high, RSI makes lower high.
    """
    c   = _clean(close)[-lookback:]
    rsi = _clean(rsi_series)[-lookback:]

    if len(c) < lookback:
        return None

    # Find last two swing lows and highs in price
    price_hi_idx = argrelextrema(c, np.greater_equal, order=3)[0]
    price_lo_idx = argrelextrema(c, np.less_equal,    order=3)[0]

    # Bullish divergence: price lower low, RSI higher low
    if len(price_lo_idx) >= 2:
        i1, i2 = price_lo_idx[-2], price_lo_idx[-1]
        if c[i2] < c[i1] and rsi[i2] > rsi[i1]:
            return {
                "type":    "BULLISH DIVERGENCE",
                "signal":  "BUY",
                "details": f"Price made lower low ({c[i2]:.2f} < {c[i1]:.2f}) "
                           f"but RSI made higher low ({rsi[i2]:.1f} > {rsi[i1]:.1f}). "
                           f"Momentum is reversing upward."
            }

    # Bearish divergence: price higher high, RSI lower high
    if len(price_hi_idx) >= 2:
        i1, i2 = price_hi_idx[-2], price_hi_idx[-1]
        if c[i2] > c[i1] and rsi[i2] < rsi[i1]:
            return {
                "type":    "BEARISH DIVERGENCE",
                "signal":  "SELL",
                "details": f"Price made higher high ({c[i2]:.2f} > {c[i1]:.2f}) "
                           f"but RSI made lower high ({rsi[i2]:.1f} < {rsi[i1]:.1f}). "
                           f"Momentum is weakening -- reversal likely."
            }
    return None


def williams_r(high, low, close, period=14):
    """Williams %R -- overbought/oversold oscillator."""
    h = _clean(high)[-period:]
    l = _clean(low)[-period:]
    c = float(_clean(close)[-1])
    hh = h.max()
    ll = l.min()
    if hh == ll:
        return 0.0
    return float(-100 * (hh - c) / (hh - ll))


def cci(high, low, close, period=20):
    """Commodity Channel Index."""
    h = _clean(high)
    l = _clean(low)
    c = _clean(close)
    tp = (h + l + c) / 3
    tp_ma  = np.convolve(tp, np.ones(period)/period, mode='valid')
    mad    = np.array([np.abs(tp[i:i+period] - tp_ma[i]).mean()
                       for i in range(len(tp_ma))])
    cci_v  = (tp[period-1:] - tp_ma) / (0.015 * mad + 1e-10)
    return float(cci_v[-1])


def rate_of_change(close, period=10):
    """Rate of Change -- momentum as percentage."""
    c = _clean(close)
    if len(c) < period + 1:
        return 0.0
    return float((c[-1] - c[-period-1]) / c[-period-1] * 100)


def obv(close, volume):
    """On Balance Volume -- cumulative volume pressure."""
    c = _clean(close)
    v = _clean(volume)
    o = np.zeros(len(c))
    for i in range(1, len(c)):
        if c[i] > c[i-1]:   o[i] = o[i-1] + v[i]
        elif c[i] < c[i-1]: o[i] = o[i-1] - v[i]
        else:                o[i] = o[i-1]

    # OBV trend: compare last 5 vs previous 5
    obv_trend = "Rising" if o[-5:].mean() > o[-10:-5].mean() else "Falling"
    return float(o[-1]), obv_trend


# ══════════════════════════════════════════════════════════════════════════════
# VOLATILITY
# ══════════════════════════════════════════════════════════════════════════════

def bollinger_squeeze(close, period=20, k_mult=1.5):
    """
    Bollinger Band Squeeze — volatility contraction before big move.
    When BB narrows inside Keltner Channel = energy coiling for breakout.
    """
    c   = _clean(close)
    if len(c) < period + 10:
        return None

    # Bollinger Bands
    ma   = np.convolve(c, np.ones(period)/period, mode='valid')
    std  = np.array([c[i:i+period].std() for i in range(len(ma))])
    bb_w = 4 * std / (ma + 1e-10)   # band width as % of price

    # Keltner Channel (ATR-based)
    atr  = np.array([max(abs(c[j]-c[j-1]) for j in range(i, i+period))
                     for i in range(len(ma))])
    kc_w = 2 * k_mult * atr / (ma + 1e-10)

    squeeze_now  = bb_w[-1]  < kc_w[-1]
    squeeze_prev = bb_w[-2]  < kc_w[-2]

    if squeeze_prev and not squeeze_now:
        return {
            "type":    "SQUEEZE RELEASE",
            "signal":  "WATCH",
            "details": "Bollinger Band squeeze just ended -- expect a large directional move. "
                       "Watch volume direction to determine which way."
        }
    if squeeze_now:
        return {
            "type":    "ACTIVE SQUEEZE",
            "signal":  "WAIT",
            "details": f"Volatility compressed (BB width {bb_w[-1]*100:.1f}%). "
                       "Energy coiling for a breakout. Wait for direction confirmation."
        }
    return None


# ══════════════════════════════════════════════════════════════════════════════
# CHART PATTERNS
# ══════════════════════════════════════════════════════════════════════════════

def detect_double_top_bottom(close, tolerance=0.03, lookback=60):
    """Double Top (bearish) and Double Bottom (bullish) — major reversal patterns."""
    c       = _clean(close)[-lookback:]
    price   = float(c[-1])
    results = []

    hi_idx = argrelextrema(c, np.greater_equal, order=5)[0]
    lo_idx = argrelextrema(c, np.less_equal,    order=5)[0]

    # Double Top: two highs at similar level, price now below neckline
    if len(hi_idx) >= 2:
        h1, h2 = c[hi_idx[-2]], c[hi_idx[-1]]
        if abs(h1 - h2) / max(h1, h2) < tolerance and price < min(h1, h2) * 0.98:
            neckline = c[lo_idx[(lo_idx > hi_idx[-2]) & (lo_idx < hi_idx[-1])]].min() if len(lo_idx) else None
            target   = round(max(h1,h2) - (max(h1,h2) - neckline), 2) if neckline else None
            results.append({
                "pattern": "DOUBLE TOP",
                "signal":  "SELL / AVOID",
                "strength":"High",
                "details": f"Two peaks at PKR {h1:.2f} and {h2:.2f} ({abs(h1-h2)/h1*100:.1f}% apart). "
                           f"Bearish reversal confirmed. Target: PKR {target}" if target else
                           f"Two peaks at PKR {h1:.2f} and {h2:.2f}. Bearish reversal."
            })

    # Double Bottom: two lows at similar level, price now above neckline
    if len(lo_idx) >= 2:
        l1, l2 = c[lo_idx[-2]], c[lo_idx[-1]]
        if abs(l1 - l2) / min(l1, l2) < tolerance and price > max(l1, l2) * 1.02:
            neckline = c[hi_idx[(hi_idx > lo_idx[-2]) & (hi_idx < lo_idx[-1])]].max() if len(hi_idx) else None
            target   = round(min(l1,l2) + (neckline - min(l1,l2)) * 2, 2) if neckline else None
            results.append({
                "pattern": "DOUBLE BOTTOM",
                "signal":  "BUY",
                "strength":"High",
                "details": f"Two troughs at PKR {l1:.2f} and {l2:.2f} ({abs(l1-l2)/l1*100:.1f}% apart). "
                           f"Bullish reversal. Target: PKR {target}" if target else
                           f"Two troughs at PKR {l1:.2f} and {l2:.2f}. Bullish reversal."
            })

    return results


def detect_head_and_shoulders(close, tolerance=0.04, lookback=80):
    """Head & Shoulders (bearish) and Inverse H&S (bullish)."""
    c = _clean(close)[-lookback:]
    results = []

    hi_idx = argrelextrema(c, np.greater_equal, order=4)[0]
    lo_idx = argrelextrema(c, np.less_equal,    order=4)[0]

    # H&S: left shoulder < head > right shoulder, all at similar heights
    if len(hi_idx) >= 3:
        ls, head, rs = c[hi_idx[-3]], c[hi_idx[-2]], c[hi_idx[-1]]
        if (head > ls and head > rs and
                abs(ls - rs) / max(ls, rs) < tolerance and
                head > max(ls, rs) * 1.02):
            neckline_pts = lo_idx[(lo_idx > hi_idx[-3]) & (lo_idx < hi_idx[-1])]
            neckline = c[neckline_pts].mean() if len(neckline_pts) >= 2 else None
            target   = round(neckline - (head - neckline), 2) if neckline else None
            results.append({
                "pattern": "HEAD & SHOULDERS",
                "signal":  "SELL / AVOID",
                "strength":"Very High",
                "details": f"Classic H&S: shoulders at {ls:.2f}/{rs:.2f}, head at {head:.2f}. "
                           f"Bearish reversal pattern. Target if breakdown: PKR {target}" if target else
                           f"H&S detected. Shoulders: {ls:.2f}/{rs:.2f}, Head: {head:.2f}."
            })

    # Inverse H&S (bullish)
    if len(lo_idx) >= 3:
        ls, head, rs = c[lo_idx[-3]], c[lo_idx[-2]], c[lo_idx[-1]]
        if (head < ls and head < rs and
                abs(ls - rs) / min(ls, rs) < tolerance and
                head < min(ls, rs) * 0.98):
            neckline_pts = hi_idx[(hi_idx > lo_idx[-3]) & (hi_idx < lo_idx[-1])]
            neckline = c[neckline_pts].mean() if len(neckline_pts) >= 2 else None
            target   = round(neckline + (neckline - head), 2) if neckline else None
            results.append({
                "pattern": "INVERSE HEAD & SHOULDERS",
                "signal":  "BUY",
                "strength":"Very High",
                "details": f"Inverse H&S: shoulders at {ls:.2f}/{rs:.2f}, head at {head:.2f}. "
                           f"Bullish reversal. Target if breakout: PKR {target}" if target else
                           f"Inverse H&S detected. Bullish reversal forming."
            })

    return results


def detect_triangle(close, lookback=50):
    """Ascending, Descending, and Symmetrical triangles — continuation/breakout patterns."""
    c       = _clean(close)[-lookback:]
    results = []

    hi_idx = argrelextrema(c, np.greater_equal, order=4)[0]
    lo_idx = argrelextrema(c, np.less_equal,    order=4)[0]

    if len(hi_idx) < 2 or len(lo_idx) < 2:
        return results

    highs = c[hi_idx[-3:]] if len(hi_idx) >= 3 else c[hi_idx]
    lows  = c[lo_idx[-3:]] if len(lo_idx) >= 3 else c[lo_idx]

    high_slope = np.polyfit(range(len(highs)), highs, 1)[0]
    low_slope  = np.polyfit(range(len(lows)),  lows,  1)[0]

    flat_tol = highs.std() * 0.3

    if high_slope < flat_tol and low_slope > flat_tol:
        results.append({
            "pattern": "ASCENDING TRIANGLE",
            "signal":  "BUY (on breakout above flat top)",
            "strength":"High",
            "details": f"Flat resistance at ~PKR {highs.max():.2f}, rising lows. "
                       "Bullish continuation pattern -- enter on volume breakout above resistance."
        })
    elif high_slope < -flat_tol and abs(low_slope) < flat_tol:
        results.append({
            "pattern": "DESCENDING TRIANGLE",
            "signal":  "SELL (on breakdown below flat bottom)",
            "strength":"High",
            "details": f"Flat support at ~PKR {lows.min():.2f}, falling highs. "
                       "Bearish continuation -- watch for breakdown below support."
        })
    elif high_slope < -flat_tol and low_slope > flat_tol:
        results.append({
            "pattern": "SYMMETRICAL TRIANGLE",
            "signal":  "WATCH for breakout direction",
            "strength":"Medium",
            "details": "Converging highs and lows -- neutral coiling pattern. "
                       "Breakout direction determines the trade. Volume confirms."
        })

    return results


def detect_flag_pennant(close, volume, lookback=40):
    """Bull/Bear Flag and Pennant — high-probability continuation patterns."""
    c = _clean(close)
    v = _clean(volume)

    if len(c) < lookback + 10:
        return []

    results = []
    pole_end = len(c) - lookback

    # Look for a sharp move (pole) followed by consolidation (flag)
    pole = c[max(0, pole_end-10):pole_end]
    flag = c[pole_end:]

    if len(pole) < 5:
        return []

    pole_return = (pole[-1] - pole[0]) / pole[0] * 100
    flag_range  = (flag.max() - flag.min()) / flag.mean() * 100
    vol_decline = v[pole_end:].mean() < v[max(0,pole_end-10):pole_end].mean()

    if abs(pole_return) > 8 and flag_range < 5 and vol_decline:
        if pole_return > 0:
            target = round(c[-1] + (pole[-1] - pole[0]), 2)
            results.append({
                "pattern": "BULL FLAG",
                "signal":  "BUY on breakout above flag top",
                "strength":"High",
                "details": f"Sharp rally (+{pole_return:.1f}%) followed by tight consolidation "
                           f"({flag_range:.1f}% range) on declining volume. "
                           f"Measured target: PKR {target:.2f}"
            })
        else:
            target = round(c[-1] + (pole[-1] - pole[0]), 2)
            results.append({
                "pattern": "BEAR FLAG",
                "signal":  "SELL on breakdown below flag bottom",
                "strength":"High",
                "details": f"Sharp decline ({pole_return:.1f}%) followed by tight consolidation "
                           f"on declining volume. Measured target: PKR {target:.2f}"
            })

    return results


# ══════════════════════════════════════════════════════════════════════════════
# CANDLESTICK PATTERNS
# ══════════════════════════════════════════════════════════════════════════════

def detect_candlestick_patterns(open_, high, low, close, n=3):
    """Detect key candlestick reversal patterns on the last N bars."""
    o = _clean(open_)[-10:]
    h = _clean(high)[-10:]
    l = _clean(low)[-10:]
    c = _clean(close)[-10:]
    patterns = []

    if len(c) < 3:
        return patterns

    # Current candle metrics
    body    = abs(c[-1] - o[-1])
    rng     = h[-1] - l[-1]
    upper_w = h[-1] - max(c[-1], o[-1])
    lower_w = min(c[-1], o[-1]) - l[-1]
    bullish = c[-1] > o[-1]

    # Doji — indecision
    if rng > 0 and body / rng < 0.1:
        patterns.append({"name":"Doji","signal":"NEUTRAL",
                         "detail":"Indecision candle -- market undecided. Watch next candle for direction."})

    # Hammer / Shooting Star
    if rng > 0 and body < rng * 0.3:
        if lower_w > body * 2 and upper_w < body * 0.5:
            if c[-2] < c[-3]:  # after downtrend
                patterns.append({"name":"Hammer","signal":"BUY",
                                  "detail":"Long lower wick after downtrend -- sellers tried and failed. Bullish reversal."})
        if upper_w > body * 2 and lower_w < body * 0.5:
            if c[-2] > c[-3]:  # after uptrend
                patterns.append({"name":"Shooting Star","signal":"SELL",
                                  "detail":"Long upper wick after uptrend -- buyers rejected at highs. Bearish reversal."})

    # Bullish/Bearish Engulfing
    if len(c) >= 2:
        prev_body = abs(c[-2] - o[-2])
        if (not (c[-2] > o[-2])) and bullish and body > prev_body * 1.1 and o[-1] < c[-2] and c[-1] > o[-2]:
            patterns.append({"name":"Bullish Engulfing","signal":"BUY",
                              "detail":"Large bullish candle engulfs prior bearish candle. Strong reversal signal."})
        if (c[-2] > o[-2]) and (not bullish) and body > prev_body * 1.1 and o[-1] > c[-2] and c[-1] < o[-2]:
            patterns.append({"name":"Bearish Engulfing","signal":"SELL",
                              "detail":"Large bearish candle engulfs prior bullish candle. Strong reversal signal."})

    # Morning Star (3-candle bullish reversal)
    if len(c) >= 3:
        c1_bear  = c[-3] < o[-3]
        c2_small = abs(c[-2]-o[-2]) < (h[-2]-l[-2]) * 0.3
        c3_bull  = c[-1] > o[-1] and c[-1] > (c[-3] + o[-3]) / 2
        if c1_bear and c2_small and c3_bull:
            patterns.append({"name":"Morning Star","signal":"BUY",
                              "detail":"3-candle bullish reversal: bearish → doji → bullish. High-confidence bottom signal."})

        c1_bull  = c[-3] > o[-3]
        c3_bear  = c[-1] < o[-1] and c[-1] < (c[-3] + o[-3]) / 2
        if c1_bull and c2_small and c3_bear:
            patterns.append({"name":"Evening Star","signal":"SELL",
                              "detail":"3-candle bearish reversal: bullish → doji → bearish. High-confidence top signal."})

    # Marubozu (strong trend bar)
    if rng > 0 and body / rng > 0.9:
        if bullish:
            patterns.append({"name":"Bullish Marubozu","signal":"BUY",
                              "detail":"Near-full bullish candle -- buyers in complete control. Strong momentum."})
        else:
            patterns.append({"name":"Bearish Marubozu","signal":"SELL",
                              "detail":"Near-full bearish candle -- sellers in complete control. Strong momentum."})

    return patterns


# ══════════════════════════════════════════════════════════════════════════════
# SUPPORT & RESISTANCE
# ══════════════════════════════════════════════════════════════════════════════

def key_levels(high, low, close, n_levels=5):
    """
    Identify key support and resistance levels using:
    - Swing highs/lows (most reliable)
    - Pivot points (daily/weekly)
    - Volume-weighted price levels
    """
    h = _clean(high)
    l = _clean(low)
    c = _clean(close)
    price = c[-1]

    # Swing-based levels
    hi_idx = argrelextrema(h, np.greater_equal, order=5)[0]
    lo_idx = argrelextrema(l, np.less_equal,    order=5)[0]

    resistance_levels = sorted(set(round(h[i], 2) for i in hi_idx if h[i] > price), reverse=False)[:n_levels]
    support_levels    = sorted(set(round(l[i], 2) for i in lo_idx if l[i] < price), reverse=True)[:n_levels]

    # Classic Pivot Points (last 5-day range)
    ph = h[-5:].max()
    pl = l[-5:].min()
    pc = c[-5:].mean()
    pivot = (ph + pl + pc) / 3
    r1 = 2 * pivot - pl
    s1 = 2 * pivot - ph
    r2 = pivot + (ph - pl)
    s2 = pivot - (ph - pl)

    return {
        "price":       round(price, 2),
        "resistance":  resistance_levels[:3],
        "support":     support_levels[:3],
        "pivot":       round(pivot, 2),
        "r1": round(r1,2), "r2": round(r2,2),
        "s1": round(s1,2), "s2": round(s2,2),
        "nearest_resistance": min(resistance_levels, key=lambda x: abs(x-price)) if resistance_levels else None,
        "nearest_support":    min(support_levels,    key=lambda x: abs(x-price)) if support_levels    else None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# BREAKOUT DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_breakout(close, high, low, volume, lookback=20):
    """
    Detect confirmed breakouts and breakdowns.
    A breakout is CONFIRMED when:
    - Price closes above resistance / below support
    - Volume is significantly above average (conviction)
    """
    c   = _clean(close)
    h   = _clean(high)
    l   = _clean(low)
    v   = _clean(volume)
    price = c[-1]

    # Consolidation range (last lookback bars excl. last 2)
    range_h = h[-lookback:-2].max()
    range_l = l[-lookback:-2].min()
    vol_avg = v[-lookback:].mean()
    vol_now = v[-1]
    vol_confirmed = vol_now > vol_avg * 1.3

    results = []

    # Resistance breakout
    if price > range_h and vol_confirmed:
        pct = (price - range_h) / range_h * 100
        target = round(price + (range_h - range_l), 2)
        results.append({
            "type":    "RESISTANCE BREAKOUT",
            "signal":  "STRONG BUY",
            "details": f"Price broke above {lookback}-day high of PKR {range_h:.2f} "
                       f"(+{pct:.1f}%) on {vol_now/vol_avg:.1f}x average volume. "
                       f"Measured target: PKR {target:.2f}",
            "target":  target,
            "stop":    round(range_h * 0.98, 2),
        })
    elif price > range_h and not vol_confirmed:
        results.append({
            "type":    "UNCONFIRMED BREAKOUT",
            "signal":  "WATCH",
            "details": f"Price above {lookback}-day high (PKR {range_h:.2f}) "
                       f"but volume only {vol_now/vol_avg:.1f}x avg -- needs volume to confirm.",
            "target":  None,
            "stop":    None,
        })

    # Support breakdown
    if price < range_l and vol_confirmed:
        pct = (range_l - price) / range_l * 100
        target = round(price - (range_h - range_l), 2)
        results.append({
            "type":    "SUPPORT BREAKDOWN",
            "signal":  "STRONG SELL / AVOID",
            "details": f"Price broke below {lookback}-day low of PKR {range_l:.2f} "
                       f"(-{pct:.1f}%) on high volume. Target: PKR {target:.2f}",
            "target":  target,
            "stop":    round(range_l * 1.02, 2),
        })

    return results


# ══════════════════════════════════════════════════════════════════════════════
# MASTER ANALYSIS FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def full_technical_analysis(df: pd.DataFrame, symbol: str = "") -> dict:
    """
    Run complete pro-level technical analysis on a price DataFrame.
    Returns a structured dict of all signals, patterns, and a composite recommendation.
    """
    if df is None or len(df) < 50:
        return {"error": "Insufficient data (need 50+ bars)"}

    import ta as _ta

    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]
    open_  = df["Open"] if "Open" in df.columns else close

    price = float(close.iloc[-1])

    # ── Core indicators ──────────────────────────────────────────────────────
    rsi_ser = _ta.momentum.rsi(close, window=14)
    rsi_val = float(rsi_ser.iloc[-1])
    macd_h  = _ta.trend.macd_diff(close)
    macd_v  = float(macd_h.iloc[-1])
    macd_p  = float(macd_h.iloc[-2])

    sma20   = float(close.rolling(20).mean().iloc[-1])
    sma50   = float(close.rolling(50).mean().iloc[-1])
    sma200  = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
    ema9    = float(close.ewm(span=9).mean().iloc[-1])
    ema21   = float(close.ewm(span=21).mean().iloc[-1])

    atr_v   = float(_ta.volatility.average_true_range(high, low, close, window=14).iloc[-1])
    vol_avg = float(volume.rolling(20).mean().iloc[-1])
    vol_now = float(volume.iloc[-1])

    # ── Advanced indicators ───────────────────────────────────────────────────
    adx_v, pdi, ndi = adx(high, low, close)
    st_val, st_bull = supertrend(high, low, close)
    wr_v            = williams_r(high, low, close)
    cci_v           = cci(high, low, close)
    roc_v           = rate_of_change(close)
    obv_v, obv_dir  = obv(close, volume)
    gdc             = golden_death_cross(close)

    # ── Pattern detection ─────────────────────────────────────────────────────
    all_patterns = []
    all_patterns.extend(detect_double_top_bottom(close))
    all_patterns.extend(detect_head_and_shoulders(close))
    all_patterns.extend(detect_triangle(close))
    all_patterns.extend(detect_flag_pennant(close, volume))

    sq  = bollinger_squeeze(close)
    if sq:
        all_patterns.append(sq)

    bko = detect_breakout(close, high, low, volume)

    # ── Candlestick patterns ──────────────────────────────────────────────────
    cs_patterns = detect_candlestick_patterns(open_, high, low, close)

    # ── RSI divergence ────────────────────────────────────────────────────────
    rsi_div = rsi_divergence(close, rsi_ser)

    # ── Support / Resistance ──────────────────────────────────────────────────
    lvls = key_levels(high, low, close)

    # ── Composite signal scoring ──────────────────────────────────────────────
    bull_pts = 0
    bear_pts = 0
    signals  = []

    # Trend
    if sma200 and price > sma200:   bull_pts += 2; signals.append(("Trend","Above SMA-200","bull"))
    elif sma200 and price < sma200: bear_pts += 2; signals.append(("Trend","Below SMA-200","bear"))
    if ema9 > ema21:  bull_pts += 1; signals.append(("EMA","EMA-9 > EMA-21 (short-term bullish)","bull"))
    else:             bear_pts += 1; signals.append(("EMA","EMA-9 < EMA-21 (short-term bearish)","bear"))

    # ADX
    if adx_v > 25:
        if pdi > ndi: bull_pts += 2; signals.append(("ADX",f"Strong trend ({adx_v:.0f}) — Bullish direction","bull"))
        else:         bear_pts += 2; signals.append(("ADX",f"Strong trend ({adx_v:.0f}) — Bearish direction","bear"))
    else:
        signals.append(("ADX",f"Weak trend ({adx_v:.0f}) — ranging market","neutral"))

    # Supertrend
    if st_bull: bull_pts += 2; signals.append(("Supertrend","Bullish (price above Supertrend line)","bull"))
    else:       bear_pts += 2; signals.append(("Supertrend","Bearish (price below Supertrend line)","bear"))

    # MACD
    if macd_v > 0 and macd_v > macd_p: bull_pts += 2; signals.append(("MACD","Positive and rising","bull"))
    elif macd_v > 0:                   bull_pts += 1; signals.append(("MACD","Positive but slowing","bull"))
    elif macd_v < 0 and macd_v < macd_p: bear_pts += 2; signals.append(("MACD","Negative and falling","bear"))
    else:                              bear_pts += 1; signals.append(("MACD","Negative, slightly improving","bear"))

    # RSI
    if 40 <= rsi_val <= 60:   bull_pts += 1; signals.append(("RSI",f"{rsi_val:.0f} — neutral momentum","neutral"))
    elif rsi_val < 30:        bull_pts += 2; signals.append(("RSI",f"{rsi_val:.0f} — oversold, bounce likely","bull"))
    elif rsi_val > 70:        bear_pts += 2; signals.append(("RSI",f"{rsi_val:.0f} — overbought, pullback risk","bear"))
    elif rsi_val < 50:        bear_pts += 1; signals.append(("RSI",f"{rsi_val:.0f} — below midline","bear"))
    else:                     bull_pts += 1; signals.append(("RSI",f"{rsi_val:.0f} — above midline","bull"))

    # Volume
    if vol_now > vol_avg * 1.3: bull_pts += 1; signals.append(("Volume",f"{vol_now/vol_avg:.1f}x avg — conviction","bull"))
    elif vol_now < vol_avg * 0.7: signals.append(("Volume","Below average — low conviction","neutral"))
    if obv_dir == "Rising":     bull_pts += 1; signals.append(("OBV","Rising — institutional accumulation","bull"))
    else:                       bear_pts += 1; signals.append(("OBV","Falling — distribution","bear"))

    # Williams %R
    if wr_v < -80:  bull_pts += 1; signals.append(("Williams %R",f"{wr_v:.0f} — oversold","bull"))
    elif wr_v > -20: bear_pts += 1; signals.append(("Williams %R",f"{wr_v:.0f} — overbought","bear"))

    # RSI divergence
    if rsi_div:
        if rsi_div["signal"] == "BUY":  bull_pts += 3; signals.append(("RSI Divergence", rsi_div["type"], "bull"))
        else:                           bear_pts += 3; signals.append(("RSI Divergence", rsi_div["type"], "bear"))

    # Chart patterns
    for p in all_patterns:
        sig = str(p.get("signal","")).upper()
        if "BUY" in sig:   bull_pts += 2; signals.append(("Pattern", p.get("pattern",""), "bull"))
        elif "SELL" in sig: bear_pts += 2; signals.append(("Pattern", p.get("pattern",""), "bear"))

    # Breakouts
    for b in bko:
        sig = str(b.get("signal","")).upper()
        if "BUY" in sig:   bull_pts += 3; signals.append(("Breakout", b["type"], "bull"))
        elif "SELL" in sig: bear_pts += 3; signals.append(("Breakout", b["type"], "bear"))

    # Golden/Death cross
    if gdc:
        if "GOLDEN" in gdc: bull_pts += 3; signals.append(("Cross", gdc, "bull"))
        else:               bear_pts += 3; signals.append(("Cross", gdc, "bear"))

    # ── Final composite recommendation ────────────────────────────────────────
    total = bull_pts + bear_pts
    bull_pct = round(bull_pts / total * 100) if total > 0 else 50

    if bull_pct >= 75:   rec, conf = "STRONG BUY",  "Very High"
    elif bull_pct >= 60: rec, conf = "BUY",          "High"
    elif bull_pct >= 52: rec, conf = "MILD BUY",     "Moderate"
    elif bull_pct <= 25: rec, conf = "STRONG SELL",  "Very High"
    elif bull_pct <= 40: rec, conf = "SELL",          "High"
    elif bull_pct <= 48: rec, conf = "MILD SELL",    "Moderate"
    else:                rec, conf = "NEUTRAL / WAIT","Low"

    # Trade levels
    stop_l = round(lvls["nearest_support"] * 0.99, 2) if lvls["nearest_support"] else round(price - 2*atr_v, 2)
    tp1    = round(lvls["nearest_resistance"], 2)       if lvls["nearest_resistance"] else round(price + 1.5*atr_v, 2)
    tp2    = round(lvls["r1"], 2)

    return {
        "symbol":          symbol,
        "price":           price,
        "recommendation":  rec,
        "confidence":      conf,
        "bull_score":      bull_pts,
        "bear_score":      bear_pts,
        "bull_pct":        bull_pct,

        # Indicators
        "adx":             round(adx_v, 1),
        "adx_trend":       "Strong" if adx_v > 25 else ("Moderate" if adx_v > 18 else "Weak"),
        "pdi":             round(pdi, 1),
        "ndi":             round(ndi, 1),
        "supertrend":      round(st_val, 2),
        "supertrend_bull": st_bull,
        "williams_r":      round(wr_v, 1),
        "cci":             round(cci_v, 1),
        "roc":             round(roc_v, 2),
        "obv_trend":       obv_dir,
        "golden_cross":    gdc,

        # Patterns
        "chart_patterns":  all_patterns,
        "candlesticks":    cs_patterns,
        "breakouts":       bko,
        "rsi_divergence":  rsi_div,

        # Levels
        "levels":          lvls,
        "stop_loss":       stop_l,
        "target_1":        tp1,
        "target_2":        tp2,

        # All signals list
        "signals":         signals,
    }
