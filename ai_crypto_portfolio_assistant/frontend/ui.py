import gradio as gr
from backend.main import process_portfolio

def launch_ui():
    def analyze_portfolio(portfolio_input):
        market_summary, risk_report, trend_summary, rebalance, suggestions, reallocation = process_portfolio(portfolio_input)

        def format_section(title, text, emoji, bg_color):
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            bullet_list = "<br>".join([f"• {line}" for line in lines])
            return f"""
            <div style="
                background-color:{bg_color};
                padding:15px;
                border-radius:12px;
                margin-bottom:15px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                color:#000000;
                font-weight:500;
            ">
                <h3 style="margin-top:0; color:#000000;">{emoji} {title}</h3>
                <p style="margin:0; color:#000000;">{bullet_list}</p>
            </div>
            """

        return (
            format_section("Market Summary", market_summary, "📊", "#d6eaff"),
            format_section("Risk Report", risk_report, "⚠️", "#ffe0b2"),
            format_section("Trend Summary", trend_summary, "📈", "#c8e6c9"),
            format_section("Rebalance Plan", rebalance, "🔄", "#b9f6ca"),
            format_section("Suggestions", suggestions, "💡", "#fff59d"),
            format_section("Reallocation Plan", reallocation, "📂", "#d1c4e9")
        )

    with gr.Blocks(theme=gr.themes.Soft()) as ui:
        # Title + subtitle with colors
        gr.Markdown(
           """
    <h2 style="color:#5850ec; font-weight:600; font-family:'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; text-align:center;">
    💰 CoinSenseAI — AI Crypto Portfolio Assistant
    </h2>
    """
        )
        gr.Markdown(
            "<p style='text-align:center; font-size:16px; color:#1a237e;'>"
            "Enter your portfolio in the format: <b>BTC 0.5, ETH $1000, ADA 300</b></p>"
        )

        # Input
        portfolio_input = gr.Textbox(
            label="Your Portfolio",
            placeholder="Example: BTC 0.5, ETH $1000, ADA 300"
        )
        submit_btn = gr.Button("Analyze Portfolio", variant="primary")

        # Outputs (all HTML for full styling control)
        market_summary_box = gr.HTML()
        risk_report_box = gr.HTML()
        trend_summary_box = gr.HTML()
        rebalance_box = gr.HTML()
        suggestions_box = gr.HTML()
        reallocation_box = gr.HTML()

        submit_btn.click(
            analyze_portfolio,
            inputs=portfolio_input,
            outputs=[
                market_summary_box,
                risk_report_box,
                trend_summary_box,
                rebalance_box,
                suggestions_box,
                reallocation_box
            ]
        )

    ui.launch()

if __name__ == "__main__":
    launch_ui()
