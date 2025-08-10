CoinSenseAI
Your AI-Powered Crypto Portfolio & News Bias Analyzer

Author: Muhammad Ahmad Khattak

📌 Overview
CoinSenseAI is an intelligent web application that combines real-time crypto portfolio analysis with fake news & bias detection for cryptocurrency-related headlines.
It’s built to help investors make smarter, data-driven decisions and avoid falling for misleading information in the volatile crypto market.

🚀 Features
Crypto Portfolio Analysis:

Fetches live crypto prices from CoinGecko API

Analyzes portfolio performance

Gives market trend insights

Fake News & Bias Detection:

Uses AI to detect misinformation and bias in crypto-related news

Returns bias percentage and reasoning

Simple & Fast UI:

Clean Gradio interface

Real-time outputs for both modes

🛠️ Tech Stack
Frontend: Gradio

Backend: FastAPI

Agent Framework: CrewAI / LangChain (planned integration)

Data Source: CoinGecko API

Deployment: HuggingFace Spaces

📂 Project Structure
bash
Copy code
CoinSenseAI/
│── api_crypto.py       # Portfolio analysis backend logic
│── api_news.py         # Fake news & bias detection backend logic
│── app.py              # Main Gradio UI
│── requirements.txt    # Dependencies
│── README.md           # Project documentation
⚡ Installation
bash
Copy code
# 1. Clone this repository
  git clone https://github.com/Tradinator1/coinsenseai.git
  cd coinsenseai

# 2. Install dependencies
  pip install -r requirements.txt

# 3. Run locally
  python app.py

🎯 Usage
Portfolio Mode:

Enter your crypto holdings

Get real-time market value + trend insights

News Analysis Mode:

Paste any crypto-related news headline

Instantly detect bias and misinformation

👨‍💻 Author
Muhammad Ahmad Khattak
Data Science & AI Enthusiast | Crypto Analyst | Full-Stack AI Developer

