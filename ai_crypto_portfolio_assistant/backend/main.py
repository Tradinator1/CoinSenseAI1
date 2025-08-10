# backend/main.py
"""
Full backend for AI Crypto Portfolio Assistant (LIVE PRICES ONLY).
- Drop in place of your previous backend/main.py
- Exposes process_portfolio(user_input) -> (market_summary, risk_report, trend_summary, rebalance, suggestions, reallocation)
- Requires: requests, pandas, numpy
"""

import requests
import time
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np

# ----------------------
# Configuration
# ----------------------
COINGECKO_COINS_URL = "https://api.coingecko.com/api/v3/coins/list"
COINGECKO_PRICE_URL = "https://api.coingecko.com/api/v3/simple/price"
COINGECKO_MARKET_CHART = "https://api.coingecko.com/api/v3/coins/{id}/market_chart"

# Cache settings
_COINS_CACHE: Dict[str, Any] = {"data": None, "ts": 0}
_COINS_CACHE_TTL = 24 * 60 * 60  # 24 hours
_PRICE_CACHE: Dict[str, Any] = {"data": {}, "ts": 0}
_PRICE_CACHE_TTL = 10  # seconds (short window)

# Manual overrides for common tokens (ensure deterministic resolution)
_MANUAL_OVERRIDES = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "LINK": "chainlink",
    "ALGO": "algorand",
    "SUI": "sui",
    "ADA": "cardano",
    "DOGE": "dogecoin",
    "XRP": "ripple",
    "LTC": "litecoin",
    "DOT": "polkadot",
    "NEAR": "near",
    "AVAX": "avalanche-2",
    "MATIC": "matic-network",
    "BCH": "bitcoin-cash",
    "ATOM": "cosmos",
    "TRX": "tron",
    "UNI": "uniswap",
    "USDC": "usd-coin",
    "USDT": "tether",
}

# ----------------------
# Utilities: Coin list & resolution
# ----------------------
def load_coin_map(force_refresh: bool = False) -> Dict[str, List[Dict[str, str]]]:
    """Load CoinGecko coin list and build mapping SYMBOL -> list of coin dicts."""
    now = time.time()
    if not force_refresh and _COINS_CACHE["data"] and now - _COINS_CACHE["ts"] < _COINS_CACHE_TTL:
        return _COINS_CACHE["data"]

    resp = requests.get(COINGECKO_COINS_URL, timeout=15)
    resp.raise_for_status()
    coins = resp.json()  # list of {id, symbol, name}
    mapping: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for c in coins:
        sym = (c.get("symbol") or "").upper()
        mapping[sym].append({"id": c.get("id"), "name": c.get("name")})
    _COINS_CACHE["data"] = mapping
    _COINS_CACHE["ts"] = now
    return mapping

def resolve_symbol_to_id(symbol: str, coin_map: Dict[str, List[Dict[str, str]]]) -> Optional[str]:
    """Resolve symbol to a single CoinGecko id using overrides and heuristics."""
    s = symbol.upper()
    if s in _MANUAL_OVERRIDES:
        return _MANUAL_OVERRIDES[s]
    candidates = coin_map.get(s)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]["id"]
    # prefer id == symbol.lower()
    for c in candidates:
        if c["id"].lower() == s.lower():
            return c["id"]
    # prefer name contains symbol
    for c in candidates:
        if s.lower() in (c.get("name") or "").lower().split():
            return c["id"]
    # fallback to first
    return candidates[0]["id"]

# ----------------------
# Parsing user input
# ----------------------
_INPUT_RE = re.compile(r'^([A-Za-z0-9\-_\.]+)\s*\$?\s*([0-9,]*\.?[0-9]+)$')

def parse_portfolio_input(text: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse horizontal input like: "ETH 0.5, SOL $250, SUI 300"
    Returns dict: SYMBOL -> {"amount": float, "is_usd": bool, "raw": str}
    """
    parsed: Dict[str, Dict[str, Any]] = {}
    if not text:
        return parsed
    tokens = re.split(r'\s*,\s*|\n+', text.strip())
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        m = _INPUT_RE.match(tok)
        if m:
            sym = m.group(1).upper()
            rawval = m.group(2).replace(",", "")
            try:
                val = float(rawval)
            except:
                continue
            is_usd = "$" in tok
            parsed[sym] = {"amount": val, "is_usd": is_usd, "raw": tok}
            continue
        # fallback splitting
        parts = tok.split()
        if len(parts) >= 2:
            sym = parts[0].upper()
            rawval = parts[1].replace("$", "").replace(",", "")
            try:
                val = float(rawval)
            except:
                continue
            is_usd = "$" in parts[1]
            parsed[sym] = {"amount": val, "is_usd": is_usd, "raw": tok}
    return parsed

# ----------------------
# Price fetching & caching
# ----------------------
def _fetch_prices_by_ids(ids: List[str]) -> Dict[str, float]:
    """Fetch live prices for coin ids from CoinGecko and return {id: price}."""
    if not ids:
        return {}
    # Merge cache window
    now = time.time()
    # Check price cache: if every id exists in cache and within TTL, return subset
    cache = _PRICE_CACHE["data"]
    if cache and now - _PRICE_CACHE["ts"] < _PRICE_CACHE_TTL:
        found = {i: cache[i] for i in ids if i in cache}
        missing = [i for i in ids if i not in found]
        if not missing:
            return found
        # fetch missing
        ids_to_fetch = missing
    else:
        ids_to_fetch = ids

    params = {"ids": ",".join(ids_to_fetch), "vs_currencies": "usd"}
    r = requests.get(COINGECKO_PRICE_URL, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    prices = {}
    for cid in ids_to_fetch:
        if cid in data and "usd" in data[cid]:
            prices[cid] = float(data[cid]["usd"])
    # update cache
    cache_data = _PRICE_CACHE["data"] or {}
    cache_data.update(prices)
    _PRICE_CACHE["data"] = cache_data
    _PRICE_CACHE["ts"] = now
    # return for requested ids
    return {cid: _PRICE_CACHE["data"].get(cid) for cid in ids if cid in _PRICE_CACHE["data"]}

# ----------------------
# Historical data & indicators
# ----------------------
def fetch_historical_prices(coingecko_id: str, days: int = 60) -> Optional[pd.DataFrame]:
    """
    Fetch historical prices (daily) for the last `days` days.
    Returns DataFrame with columns ['timestamp', 'price'] where timestamp is ms since epoch.
    """
    try:
        params = {"vs_currency": "usd", "days": days}
        r = requests.get(COINGECKO_MARKET_CHART.format(id=coingecko_id), params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        prices = data.get("prices", [])
        if not prices:
            return None
        df = pd.DataFrame(prices, columns=["timestamp", "price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except Exception:
        return None

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi(series: pd.Series, period: int = 14) -> Optional[float]:
    try:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        val = float(rsi.iloc[-1])
        return round(val, 1)
    except Exception:
        return None

def calculate_macd(series: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    try:
        ema12 = ema(series, span=12)
        ema26 = ema(series, span=26)
        macd_line = ema12 - ema26
        signal = macd_line.ewm(span=9, adjust=False).mean()
        macd_val = float(macd_line.iloc[-1])
        macd_signal = float(signal.iloc[-1])
        return round(macd_val, 4), round(macd_signal, 4)
    except Exception:
        return None, None

# ----------------------
# Enrichment: compute USD & quantities
# ----------------------
def enrich_portfolio(parsed: Dict[str, Dict[str, Any]], coin_map: Dict[str, List[Dict[str, str]]]):
    """
    Resolve ids, fetch prices, and compute usd/qty.
    Returns enriched dict and list of unknown symbols.
    enriched: {SYMBOL: {id, price, quantity, usd, is_usd}}
    """
    symbol_to_id = {}
    unknown = []
    for sym in parsed.keys():
        cid = resolve_symbol_to_id(sym, coin_map)
        if cid is None:
            unknown.append(sym)
        symbol_to_id[sym] = cid

    unique_ids = sorted(set([cid for cid in symbol_to_id.values() if cid]))
    prices_by_id = _fetch_prices_by_ids(unique_ids) if unique_ids else {}

    enriched = {}
    for sym, info in parsed.items():
        cid = symbol_to_id.get(sym)
        price = prices_by_id.get(cid) if cid else None
        amount = info["amount"]
        is_usd = info["is_usd"]

        if price is not None:
            if is_usd:
                usd_val = amount
                qty = usd_val / price if price > 0 else 0.0
            else:
                qty = amount
                usd_val = qty * price
        else:
            # unknown price: set usd/qty appropriately to None so UI can show unknown asset
            if is_usd:
                usd_val = amount
                qty = None
            else:
                qty = amount
                usd_val = None

        enriched[sym] = {
            "id": cid,
            "price": price,
            "quantity": qty,
            "usd": usd_val,
            "is_usd": is_usd,
        }
    return enriched, unknown

# ----------------------
# Allocation, risk, rebalance helpers
# ----------------------
def compute_allocations(enriched: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    total = 0.0
    for v in enriched.values():
        if v["usd"] is not None:
            total += v["usd"]
    allocations = {}
    for sym, v in enriched.items():
        usd = v["usd"] or 0.0
        pct = (usd / total * 100) if total > 0 else 0.0
        allocations[sym] = {
            "usd": round(usd, 2),
            "pct": round(pct, 2),
            "quantity": v["quantity"],
            "price": v["price"],
            "id": v["id"],
        }
    return total, allocations

def diversification_score(allocations: Dict[str, Any]) -> float:
    hhi = 0.0
    for v in allocations.values():
        p = v["pct"] / 100.0
        hhi += p * p
    score = max(0.0, min(100.0, round((1 - hhi) * 100, 1)))
    return score

def risk_label(pct: float) -> str:
    if pct >= 50:
        return "Very High (single-asset concentration)"
    if pct >= 30:
        return "High"
    if pct >= 15:
        return "Medium"
    return "Low"

# ----------------------
# Output builders (market, risk, trend, rebalance, suggestions, reallocation)
# ----------------------
def build_market_summary(total: float, allocations: Dict[str, Any], unknown: List[str]) -> str:
    lines = []
    lines.append(f"Total portfolio value (live estimate): **${total:,.2f}**.")
    if unknown:
        lines.append(f"⚠️ Unknown assets: {', '.join(unknown)} — not found on CoinGecko.")
    if total > 0:
        sorted_by_value = sorted(allocations.items(), key=lambda x: x[1]["usd"], reverse=True)
        top_sym, top_info = sorted_by_value[0]
        lines.append(f"Top holding: **{top_sym}** — ${top_info['usd']:,.2f} ({top_info['pct']}% of portfolio).")
        snapshot = ", ".join([f"{s}: {info['pct']}%" for s, info in sorted_by_value[:6]])
        lines.append(f"Allocation snapshot: {snapshot}.")
    else:
        lines.append("No USD-valued positions found (all unknown or zero).")
    lines.append("Market note: prices pulled live from CoinGecko.")
    return "\n\n".join(lines)

def build_risk_report(allocations: Dict[str, Any], diversification: float) -> str:
    lines = []
    lines.append(f"**Diversification Score:** {diversification}/100.")
    for sym, info in allocations.items():
        lines.append(f"- {sym}: ${info['usd']:,.2f} ({info['pct']}%) → Risk: *{risk_label(info['pct'])}*.")
    if diversification < 40:
        lines.append("Overall: portfolio is concentrated. Consider trimming largest positions and adding stablecoins.")
    elif diversification < 70:
        lines.append("Overall: moderately diversified. Consider small rebalances.")
    else:
        lines.append("Overall: well diversified.")
    return "\n\n".join(lines)

def build_trend_summary(enriched: Dict[str, Any]) -> str:
    """
    For each coin produce 2-3 lines:
      1) current situation with price
      2) short technical hint (RSI/MACD/MA)
      3) short outlook / catalyst note
    Keep concise.
    """
    lines = []
    for sym, data in enriched.items():
        cid = data.get("id")
        price = data.get("price")
        # heading
        if price is None:
            lines.append(f"- **{sym}**: price unknown — cannot compute trend.")
            continue

        # fetch historical data (short window)
        df = fetch_historical_prices(cid, days=60)
        if df is None or df.empty:
            lines.append(f"- **{sym}** ({price:.4f}): insufficient history for indicators.")
            continue

        series = df["price"]
        rsi_val = calculate_rsi(series) or None
        macd_val, macd_signal = calculate_macd(series)
        ma7 = series.rolling(7).mean().iloc[-1]
        ma30 = series.rolling(30).mean().iloc[-1]

        # Compose 2-3 sentence summary
        s1 = f"**{sym}** is trading at **${price:,.2f}** (recent range shown)."
        # technical hint
        tech_parts = []
        if rsi_val is not None:
            if rsi_val > 70:
                tech_parts.append(f"RSI {rsi_val} (overbought).")
            elif rsi_val < 30:
                tech_parts.append(f"RSI {rsi_val} (oversold).")
            else:
                tech_parts.append(f"RSI {rsi_val} (neutral).")
        if macd_val is not None and macd_signal is not None:
            if macd_val > macd_signal:
                tech_parts.append("MACD bullish crossover.")
            else:
                tech_parts.append("MACD bearish/weak.")
        # moving average hint
        if ma7 is not None and ma30 is not None:
            if ma7 > ma30:
                tech_parts.append("7d MA above 30d MA (short-term bullish).")
            else:
                tech_parts.append("7d MA below 30d MA (short-term weak).")

        s2 = " ".join(tech_parts[:3])
        # outlook
        if rsi_val is not None and rsi_val > 75:
            outlook = "Short-term pullback possible; watch key supports."
        elif rsi_val is not None and rsi_val < 25:
            outlook = "Potential rebound if buying volume returns."
        else:
            outlook = "Neutral to mildly favorable near-term; watch volume and BTC behavior."

        # combine (limit to 2-3 lines worth of text)
        entry = f"- {s1} {s2} {outlook}"
        lines.append(entry)
    return "\n\n".join(lines)

def build_rebalance_plan(allocations: Dict[str, Any], total: float) -> str:
    suggestions = []
    # target preference: BTC 50, ETH 30 if present
    target = {}
    if any(s in ("BTC", "BITCOIN") for s in allocations.keys()):
        t = "BTC" if "BTC" in allocations else next((k for k in allocations if k.startswith("BTC")), None)
        if t:
            target[t] = 50.0
    if any(s in ("ETH", "ETHEREUM") for s in allocations.keys()):
        t = "ETH" if "ETH" in allocations else next((k for k in allocations if k.startswith("ETH")), None)
        if t:
            target[t] = 30.0
    remaining = 100.0 - sum(target.values())
    others = [s for s in allocations.keys() if s not in target]
    if others and remaining > 0:
        split = round(remaining / max(1, len(others)), 2)
        for s in others:
            target[s] = split

    for s, info in allocations.items():
        cur = info["pct"]
        tgt = target.get(s, 0.0)
        diff = round(tgt - cur, 2)
        if diff >= 2:
            usd_move = diff / 100.0 * total
            suggestions.append(f"Buy ${usd_move:,.2f} of {s} (~{diff}% increase).")
        elif diff <= -2:
            usd_move = abs(diff) / 100.0 * total
            suggestions.append(f"Sell ${usd_move:,.2f} of {s} (~{abs(diff)}% decrease).")
        else:
            suggestions.append(f"Hold {s} (current {cur}%, target ~{tgt}%).")
    return "\n\n".join(suggestions) if suggestions else "No rebalance suggestions."

def build_asset_suggestions(allocations: Dict[str, Any]) -> str:
    preferred = ["SOL", "LINK", "ADA", "NEAR", "DOT", "LTC", "MATIC"]
    picks = [p for p in preferred if p not in allocations]
    lines = [f"- Consider adding {p} for diversification (small allocation 1-5%)." for p in picks[:3]]
    return "\n".join(lines) if lines else "No additional suggestions."

def build_reallocation_plan(allocations: Dict[str, Any], total: float) -> str:
    # same target logic as rebalance; show before -> after percentages
    target = {}
    if any(s in ("BTC", "BITCOIN") for s in allocations.keys()):
        t = "BTC" if "BTC" in allocations else next((k for k in allocations if k.startswith("BTC")), None)
        if t:
            target[t] = 50.0
    if any(s in ("ETH", "ETHEREUM") for s in allocations.keys()):
        t = "ETH" if "ETH" in allocations else next((k for k in allocations if k.startswith("ETH")), None)
        if t:
            target[t] = 30.0
    others = [s for s in allocations.keys() if s not in target]
    remaining = 100.0 - sum(target.values())
    if others and remaining > 0:
        split = round(remaining / max(1, len(others)), 2)
        for s in others:
            target[s] = split
    lines = ["Before → Suggested (percent):"]
    for s, info in allocations.items():
        lines.append(f"- {s}: {info['pct']}% → {target.get(s, 0.0)}%")
    return "\n".join(lines)

# ----------------------
# Public entrypoint for UI
# ----------------------
def process_portfolio(user_input: str) -> Tuple[str, str, str, str, str, str]:
    """
    Main function consumed by the frontend UI.
    Returns 6 strings suitable for display in the Gradio UI.
    """
    try:
        coin_map = load_coin_map()
    except Exception as e:
        return (f"❌ Failed to load CoinGecko coin list: {e}", "", "", "", "", "")

    parsed = parse_portfolio_input(user_input)
    if not parsed:
        return ("❌ No valid assets found. Example input: ETH $0.5, SOL $250, ALGO 150, BTC 0.01",) + ("",) * 5

    enriched, unknown = enrich_portfolio(parsed, coin_map)
    total, allocations = compute_allocations(enriched)
    diversification = diversification_score(allocations)

    try:
        market_summary = build_market_summary(total, allocations, unknown)
        risk_report = build_risk_report(allocations, diversification)
        trend_summary = build_trend_summary(enriched)
        rebalance_plan = build_rebalance_plan(allocations, total)
        asset_suggestions = build_asset_suggestions(allocations)
        reallocation_plan = build_reallocation_plan(allocations, total)
    except Exception as e:
        return (f"❌ Unexpected error building outputs: {e}", "", "", "", "", "")

    return (market_summary, risk_report, trend_summary, rebalance_plan, asset_suggestions, reallocation_plan)

# End of file
