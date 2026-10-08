# AI-Powered Market Intelligence & Mentorship Pipeline

## Overview
This project is an AI-driven pipeline designed to analyze Indian Stock Market (NSE) news, extract actionable corporate catalysts, and generate structured educational case studies. It bridges the gap between fundamental news and technical analysis, acting as an automated "senior hedge fund portfolio manager" to mentor retail traders and investors.

## 🚀 Features & Pipeline Flow
The system processes financial data through a robust, multi-step pipeline:

1. **News Ingestion**: 
   - Scrapes the latest breaking market news using RSS feeds (Moneycontrol, Livemint) or the NewsData API.
   - Intelligently handles fallbacks if specific catalyst queries return zero results.
2. **Intelligent Extraction & Classification**: 
   - Uses a local Large Language Model (LLM) via LangChain to parse the news.
   - Classifies articles into `SINGLE_STOCK`, `MACRO_SECTOR`, or `NOISE`.
   - Identifies companies and accurately maps them to their official NSE tickers.
3. **Technical Analysis Calculation**: 
   - For identified equities, it fetches historical market data using `yfinance`.
   - Computes key technical indicators (14-Day RSI, 20-Day EMA, MACD) using `pandas_ta`.
4. **Educational Mentorship Synthesis**: 
   - Feeds the fundamental catalyst and technical data back into the LLM.
   - Generates a highly structured `MarketEducationalReport` (enforced via Pydantic) that includes:
     - Core catalyst summary and direct company impact.
     - Read-across (competitor impact) and macroeconomic ripples.
     - Key financial terms and classroom concepts applied to the news.
     - Historical precedents, common "retail traps", and a diagnostic pop quiz.
     - An actionable trader playbook synthesizing fundamentals and technicals.

## 🛠️ Tech Stack
- **Core Language**: Python
- **AI & Orchestration**: LangChain, ChatOllama (Local LLM inference, defaults to `gemma4:e4b`)
- **Data Validation**: Pydantic (for strictly typed LLM outputs)
- **Market Data & Quant Tools**: `yfinance`, `pandas`, `pandas-ta`
- **Data Ingestion**: `requests`, `feedparser`
- **Environment**: `python-dotenv`

## 💡 Usefulness & Project Value
- **Financial Education in Real-Time**: Acts as a simulated mentor for beginner/intermediate investors, teaching institutional finance concepts using *today's* news rather than textbook theory.
- **Noise Reduction**: Automatically filters out non-financial news (entertainment, sports, general noise) and focuses purely on market-moving catalysts.
- **Holistic Analysis**: Seamlessly combines fundamental news (the *why*) with technical indicators (the *when*) to provide a comprehensive market view.
- **Time-Saving Automation**: Saves hours of manual research by automatically aggregating, filtering, and synthesizing multiple news feeds into digestible case studies.

## ⚙️ Setup & Execution

1. **Environment Setup**: Ensure you have Python installed and install the required dependencies (e.g., `langchain_ollama`, `yfinance`, `pandas_ta`, `feedparser`, `pydantic`).
2. **Local LLM**: Install [Ollama](https://ollama.com/) and pull the necessary model:
   ```bash
   ollama run gemma4:e4b
   ```
   *(Note: You can swap this out for other models in the code if preferred).*
3. **API Keys**: Create a `.env` file in the root directory and add your NewsData API key (optional, used as a fallback or alternate source):
   ```
   NEWS_DATA_API=your_api_key_here
   ```
4. **Run the Pipeline**:
   ```bash
   python pipeline2.py
   ```
*(Note: `Untitled-1.py` and the `.ipynb` notebooks are included as experimental scratchpads and earlier iterations of the pipeline).*
