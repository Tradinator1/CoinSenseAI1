import requests
import re

# --- Fetch the coin list once and build symbol → ID map ---
def get_coin_map():
    url = "https://api.coingecko.com/api/v3/coins/list"
    r = requests.get(url)
    data = r.json()
    coin_map = {}
    for coin in data:
        symbol = coin["symbol"].lower()
        coin_map[symbol] = coin["id"]
    return coin_map

COIN_MAP = get_coin_map()

# --- Fetch live prices from CoinGecko ---
def fetch_prices(symbols):
    ids = [COIN_MAP[sym.lower()] for sym in symbols if sym.lower() in COIN_MAP]
    if not ids:
        return {}
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": ",".join(ids), "vs_currencies": "usd"}
    r = requests.get(url, params=params)
    prices_data = r.json()
    
    prices = {}
    for sym in symbols:
        sym_l = sym.lower()
        if sym_l in COIN_MAP and COIN_MAP[sym_l] in prices_data:
            prices[sym.upper()] = prices_data[COIN_MAP[sym_l]]["usd"]
    return prices

# --- Parse horizontal portfolio input ---
def parse_portfolio(portfolio_str):
    entries = re.split(r',\s*|\s{2,}', portfolio_str.strip())  # Split by commas or multiple spaces
    portfolio = {}
    for entry in entries:
        parts = entry.strip().split()
        if len(parts) == 2:
            symbol, amount = parts
            amount = amount.replace("$", "")
            try:
                amount = float(amount)
                portfolio[symbol.upper()] = amount
            except ValueError:
                pass
    return portfolio

# --- Main portfolio analysis ---
def analyze_portfolio(portfolio_str):
    portfolio = parse_portfolio(portfolio_str)
    symbols = list(portfolio.keys())
    
    prices = fetch_prices(symbols)
    unknown_assets = [sym for sym in symbols if sym not in prices]
    
    # Calculate value
    total_value = 0
    for sym, amount in portfolio.items():
        if sym in prices:
            if amount < 50 and sym != "BTC" and sym != "ETH":  # Assume it's quantity if <50 (rough)
                value = prices[sym] * amount
            else:
                value = amount if amount > 50 else prices[sym] * amount
            total_value += value
    
    # Find top holding
    if total_value > 0:
        top_holding = max(portfolio.items(), key=lambda x: (prices.get(x[0], 0) * (x[1] if x[1] < 50 else 1)))
    else:
        top_holding = ("None", 0)
    
    result = f"📊 Market Summary\n"
    result += f"• Total portfolio value (estimated): **${total_value:,.2f}**.\n"
    if unknown_assets:
        result += f"• ⚠️ Unknown assets: {', '.join(unknown_assets)} — priced at $0.00 for now.\n"
    if total_value > 0:
        result += f"• Top holding: **{top_holding[0]}**.\n"
    
    return result
