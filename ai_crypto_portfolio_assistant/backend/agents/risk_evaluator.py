class RiskEvaluatorAgent:
    def __init__(self, current_prices):
        self.prices = current_prices
        
    def evaluate(self, portfolio_input):
        """Evaluate risk levels for the portfolio"""
        try:
            if isinstance(portfolio_input, str):
                portfolio = eval(portfolio_input)
            else:
                portfolio = portfolio_input
            
            risk_analysis = {}
            
            for asset, amount in portfolio.items():
                if asset.lower() in ['bitcoin', 'btc']:
                    risk_analysis[asset] = {
                        'risk_level': 'Medium',
                        'volatility': 'Moderate',
                        'recommendation': 'Core holding - maintain allocation'
                    }
                elif asset.lower() in ['ethereum', 'eth']:
                    risk_analysis[asset] = {
                        'risk_level': 'Medium-High',
                        'volatility': 'High',
                        'recommendation': 'Growth asset - monitor closely'
                    }
                else:
                    risk_analysis[asset] = {
                        'risk_level': 'High',
                        'volatility': 'Very High',
                        'recommendation': 'Speculative - limit exposure'
                    }
            
            return risk_analysis
            
        except Exception as e:
            return {"Error": f"Risk evaluation failed: {str(e)}"}