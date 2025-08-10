def analyze_market():
    """Provide general market analysis"""
    return "📈 Market shows mixed signals with cautious optimism. Institutional adoption continues while regulatory clarity improves."

def analyze_trends(prices):
    """Analyze trends for individual cryptocurrencies based on price levels"""
    trends = {}
    
    if not prices:
        return {"Error": "No price data available"}
    
    for symbol, price in prices.items():
        # Simple trend analysis based on price ranges
        if symbol == 'bitcoin':
            if price > 45000:
                trends[symbol] = "Bullish - Strong institutional support"
            elif price > 30000:
                trends[symbol] = "Sideways - Consolidation phase"
            else:
                trends[symbol] = "Bearish - Support levels tested"
        elif symbol == 'ethereum':
            if price > 3000:
                trends[symbol] = "Bullish - DeFi and staking growth"
            elif price > 2000:
                trends[symbol] = "Sideways - Awaiting next catalyst"
            else:
                trends[symbol] = "Bearish - Market uncertainty"
        else:
            # Generic analysis for other coins
            if price > 1:
                trends[symbol] = "Bullish - Positive momentum"
            elif price > 0.1:
                trends[symbol] = "Sideways - Range-bound trading"
            else:
                trends[symbol] = "Bearish - Low price levels"
    
    return trends