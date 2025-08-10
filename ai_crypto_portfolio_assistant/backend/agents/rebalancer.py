def suggest_rebalance(portfolio, risk_data, trend_data, prices):
    """Suggest portfolio rebalancing based on analysis"""
    try:
        if isinstance(portfolio, str):
            portfolio_dict = eval(portfolio)
        else:
            portfolio_dict = portfolio
        
        suggestions = []
        
        # Calculate total portfolio value
        total_value = 0
        for asset, amount in portfolio_dict.items():
            asset_key = asset.lower()
            if asset_key in prices:
                total_value += amount * prices[asset_key]
        
        if total_value == 0:
            return "❌ Cannot calculate portfolio value - check asset names"
        
        # Analyze current allocation
        for asset, amount in portfolio_dict.items():
            asset_key = asset.lower()
            if asset_key in prices:
                current_value = amount * prices[asset_key]
                allocation = (current_value / total_value) * 100
                
                if allocation > 50:
                    suggestions.append(f"🔄 {asset}: Consider reducing from {allocation:.1f}% - over-concentrated")
                elif allocation < 10 and asset_key in ['bitcoin', 'ethereum']:
                    suggestions.append(f"📈 {asset}: Consider increasing from {allocation:.1f}% - under-allocated for core asset")
        
        if not suggestions:
            suggestions.append("✅ Portfolio appears well-balanced based on current analysis")
        
        return "\n".join(suggestions)
        
    except Exception as e:
        return f"❌ Rebalancing analysis failed: {str(e)}"

def suggest_additional_assets(portfolio, risk_data, trend_data, prices):
    """Suggest additional assets to consider"""
    try:
        if isinstance(portfolio, str):
            portfolio_dict = eval(portfolio)
        else:
            portfolio_dict = portfolio
        
        current_assets = [asset.lower() for asset in portfolio_dict.keys()]
        suggestions = []
        
        # Suggest diversification assets
        diversification_assets = {
            'solana': 'High-performance blockchain with growing ecosystem',
            'cardano': 'Research-driven blockchain with sustainability focus',
            'chainlink': 'Leading oracle network for DeFi',
            'polkadot': 'Interoperability-focused parachain platform'
        }
        
        for asset, description in diversification_assets.items():
            if asset not in current_assets and asset in prices:
                suggestions.append(f"💡 Consider {asset.capitalize()}: {description} (${prices[asset]:.2f})")
        
        if not suggestions:
            suggestions.append("✅ You have good asset diversity - monitor existing holdings")
        
        return "\n".join(suggestions[:3])  # Limit to top 3 suggestions
        
    except Exception as e:
        return f"❌ Asset suggestion failed: {str(e)}"

def suggest_reallocation(portfolio, risk_data, trend_data, prices):
    """Suggest portfolio reallocation strategy"""
    try:
        if isinstance(portfolio, str):
            portfolio_dict = eval(portfolio)
        else:
            portfolio_dict = portfolio
        
        recommendations = []
        
        # Basic allocation strategy
        recommendations.append("💼 Suggested Allocation Strategy:")
        recommendations.append("• Bitcoin: 40-50% (Store of value, lower volatility)")
        recommendations.append("• Ethereum: 25-35% (DeFi ecosystem, growth potential)")
        recommendations.append("• Altcoins: 15-25% (Diversification, higher risk/reward)")
        recommendations.append("• Stablecoins: 5-10% (Liquidity, opportunity buffer)")
        
        # Market-specific advice
        market_trend = "mixed"  # Could be enhanced with actual market analysis
        if market_trend == "bullish":
            recommendations.append("\n🔥 Market Bullish: Consider increasing altcoin allocation")
        elif market_trend == "bearish":
            recommendations.append("\n🛡️ Market Bearish: Increase Bitcoin/stablecoin allocation")
        else:
            recommendations.append("\n⚖️ Mixed Market: Maintain balanced approach with periodic rebalancing")
        
        return "\n".join(recommendations)
        
    except Exception as e:
        return f"❌ Reallocation suggestion failed: {str(e)}"