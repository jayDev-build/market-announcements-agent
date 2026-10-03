# %%
import os
import io
import json
import requests
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

NEWS_DATA_API = os.getenv("NEWS_DATA_API")

# Initialize LLM
model = ChatOllama(model="gemma4:e4b", temperature=0.3, num_ctx=8192)

# =====================================================================
# 1. NSE TICKER UNIVERSE LOADER
# =====================================================================

def get_valid_nse_tickers() -> set[str]:
    """Downloads official active NSE equity list with a safe fallback."""
    url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/csv,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    }
    try:
        response = requests.get(url, headers=headers, timeout=8)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        valid_symbols = set(df["SYMBOL"].str.strip().str.upper())
        print(f"[Init] Loaded {len(valid_symbols)} active NSE tickers from official exchange list.")
        return valid_symbols
    except Exception as e:
        print(f"[Init Warning] Could not fetch NSE master list ({e}). Using core fallback tickers.")
        return {
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "ITC", 
            "SBIN", "BHARTIARTL", "TATAMOTORS", "LICI", "LT", "WIPRO"
        }

VALID_NSE_TICKERS = get_valid_nse_tickers()

# =====================================================================
# 2. RESILIENT NEWS RETRIEVAL (WITH ZERO-RESULT FALLBACK)
# =====================================================================

def fetch_market_news(topic: str = "Indian Stock Market", max_results: int = 10) -> List[Dict[str, str]]:
    """
    Fetches news using flexible catalyst keywords.
    Automatically falls back to top business news if the specific query returns 0 results.
    """
    if not NEWS_DATA_API:
        print("[Error] NEWS_DATA_API environment variable is missing.")
        return []

    # Broad catalyst query avoids over-filtering
    catalyst_query = "deal OR profit OR order OR acquisition OR quarterly OR results"
    base_url = "https://newsdata.io/api/1/latest"
    
    primary_params = {
        "apikey": NEWS_DATA_API,
        "country": "in",
        "category": "business",
        "q": catalyst_query,
        "size": max_results
    }

    try:
        res = requests.get(base_url, params=primary_params, timeout=10)
        data = res.json()
        articles_raw = data.get("results", [])

        # Fallback if specific search returns zero results
        if not articles_raw or data.get("status") != "success":
            print("[News] Specific catalyst query returned 0 articles. Falling back to top business feed...")
            fallback_params = {
                "apikey": NEWS_DATA_API,
                "country": "in",
                "category": "business",
                "size": max_results
            }
            res = requests.get(base_url, params=fallback_params, timeout=10)
            data = res.json()
            articles_raw = data.get("results", [])

        clean_articles = []
        for art in articles_raw:
            title = art.get("title")
            desc = art.get("description")
            if title and desc:
                clean_articles.append({
                    "title": title.strip(),
                    "description": desc.strip(),
                    "source": art.get("source_name", "Financial Media")
                })

        print(f"[News] Successfully loaded {len(clean_articles)} articles.")
        return clean_articles

    except Exception as e:
        print(f"[News Error] Failed to retrieve news: {e}")
        return []

# =====================================================================
# 3. ROBUST TECHNICAL INDICATORS CALCULATOR
# =====================================================================

def calculate_ticker_technicals(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Fetches technical metrics for a verified NSE ticker.
    Returns None safely if yfinance has no history or fails (e.g., IPO/delisted).
    """
    base_ticker = ticker.strip().upper().replace(".NS", "").replace(".BO", "")

    if base_ticker not in VALID_NSE_TICKERS:
        return None

    ticker_ns = f"{base_ticker}.NS"
    try:
        stock = yf.Ticker(ticker_ns)
        df = stock.history(period="6mo")

        if df.empty or len(df) < 30:
            print(f"[Technicals] Incomplete trading bars (<30) for {ticker_ns}. Proceeding without technicals.")
            return None

        # Add indicators
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.ema(length=20, append=True)

        last = df.iloc[-1]
        return {
            "close": round(float(last.get("Close", 0.0)), 2),
            "rsi": round(float(last.get("RSI_14", 0.0)), 2) if not pd.isna(last.get("RSI_14")) else None,
            "ema20": round(float(last.get("EMA_20", 0.0)), 2) if not pd.isna(last.get("EMA_20")) else None,
            "macd": round(float(last.get("MACD_12_26_9", 0.0)), 2) if not pd.isna(last.get("MACD_12_26_9")) else None,
        }
    except Exception as e:
        print(f"[Technicals] Failed to compute metrics for {ticker_ns}: {e}")
        return None

# =====================================================================
# 4. STRUCTURED PYDANTIC OUTPUT MODELS
# =====================================================================

class ExtractedEntity(BaseModel):
    ticker: str = Field(description="Official NSE ticker symbol (e.g., TCS, RELIANCE) or 'NONE' if no listed stock is identified.")
    news_title: str = Field(description="Headline of the article")
    description: str = Field(description="Summary of the event")
    source: str = Field(description="Publisher/Source name")

class IdentifiedArticles(BaseModel):
    items: List[ExtractedEntity] = Field(description="List of mapped Indian equities and catalysts")

class FinancialTerm(BaseModel):
    term: str = Field(description="Financial or economic concept name")
    definition: str = Field(description="Plain English definition")
    applied_to_news: str = Field(description="How it applies directly to this specific catalyst")

class EducationalInsights(BaseModel):
    historical_precedent: str = Field(description="Historical market precedent where a similar event occurred")
    retail_trap: str = Field(description="Common psychological or execution mistake beginner retail traders make")
    pop_quiz: str = Field(description="A 1-sentence diagnostic question to test comprehension of the concept")

class MarketEducationalReport(BaseModel):
    ticker: str = Field(description="Stock ticker or NONE")
    catalyst_summary: str = Field(description="Core event analysis and market expectations")
    company_impact: str = Field(description="Balance sheet, margin, or revenue implications")
    competitors_impact: str = Field(description="Read-across impact on sector peers")
    macro_ripple: str = Field(description="Macroeconomic context (rates, commodity prices, demand shifts)")
    key_terms: List[FinancialTerm] = Field(description="1-2 financial mechanisms illustrated by the news")
    educational_insights: EducationalInsights = Field(description="Precedent, traps, and quiz")
    trader_takeaway: str = Field(description="Actionable trade setup synthesizing fundamentals and technicals (or purely fundamental if technicals are unavailable)")

# =====================================================================
# 5. PIPELINE CHAINS & PROMPTS
# =====================================================================

identification_parser = PydanticOutputParser(pydantic_object=IdentifiedArticles)

identification_prompt = PromptTemplate(
    template="""You are an equity research associate covering the National Stock Exchange of India (NSE).
Review the following articles:

{articles_data}

Instructions:
1. For each article, determine if a specific, publicly listed Indian company is the primary subject.
2. If YES, output its clean NSE ticker (e.g., RELIANCE, INFY, TATAMOTORS).
3. If NO (foreign company, index summary, or unnamed entity), set ticker strictly to "NONE".
4. Extract up to 5 relevant items.

Format strictly as JSON:
{format_instructions}
""",
    input_variables=["articles_data"],
    partial_variables={"format_instructions": identification_parser.get_format_instructions()}
)

identification_chain = identification_prompt | model | identification_parser

mentor_model = model.with_structured_output(MarketEducationalReport)

mentor_prompt = PromptTemplate(
    template="""You are a senior hedge fund portfolio manager and financial educator.
Analyze this corporate catalyst and provide structured educational mentorship.

Catalyst Details:
- Title: {title}
- Source: {source}
- Summary: {description}

Technical Market Data:
- Ticker: {ticker}
- Close Price: {close}
- 14-Day RSI: {rsi}
- 20-Day EMA: {ema20}
- MACD: {macd}

Special Rules for Incomplete Technicals:
- If technical indicators are marked "N/A" (due to low trading history, IPO, or API bounds), DO NOT abort.
- Rely purely on fundamental catalyst mechanics, margins, and valuation drivers for the Trader Playbook.

Perform your analysis and teach 1-2 institutional finance concepts from this event.
""",
    input_variables=["title", "source", "description", "ticker", "close", "rsi", "ema20", "macd"]
)

mentor_chain = mentor_prompt | mentor_model

# =====================================================================
# 6. PIPELINE EXECUTION
# =====================================================================

def run_pipeline():
    # Step 1: Ingest News
    articles = fetch_market_news(max_results=8)
    
    if not articles:
        print("\n[Pipeline Result] No news articles could be retrieved from the feed at this time. Pipeline finished.")
        return

    articles_text = "\n\n".join([
        f"Article {i+1}:\nTitle: {a['title']}\nSource: {a['source']}\nBody: {a['description']}"
        for i, a in enumerate(articles)
    ])

    # Step 2: Map to NSE Equities
    print("\n[Analysis] Extracting corporate catalysts and tickers...")
    try:
        identified = identification_chain.invoke({"articles_data": articles_text})
    except Exception as e:
        print(f"[Error] Failed to parse company extraction: {e}")
        return

    if not identified.items:
        print("\n[Pipeline Result] No actionable single-stock catalysts found in current headlines.")
        return

    # Step 3: Technicals + Educational Synthesis
    final_reports: List[MarketEducationalReport] = []

    for item in identified.items:
        ticker = item.ticker.strip().upper()
        print(f"\nProcessing: {ticker} | Headline: {item.news_title[:50]}...")

        # Safe technical fetching
        tech_data = None
        if ticker != "NONE" and ticker in VALID_NSE_TICKERS:
            tech_data = calculate_ticker_technicals(ticker)

        # Build resilient input payloads (handling null technicals cleanly)
        payload = {
            "title": item.news_title,
            "source": item.source,
            "description": item.description,
            "ticker": ticker,
            "close": f"₹{tech_data['close']}" if tech_data and tech_data.get("close") else "N/A",
            "rsi": str(tech_data["rsi"]) if tech_data and tech_data.get("rsi") else "N/A",
            "ema20": str(tech_data["ema20"]) if tech_data and tech_data.get("ema20") else "N/A",
            "macd": str(tech_data["macd"]) if tech_data and tech_data.get("macd") else "N/A",
        }

        try:
            deep_dive = mentor_chain.invoke(payload)
            final_reports.append(deep_dive)
        except Exception as e:
            print(f"[Error] Failed generating deep dive for {ticker}: {e}")

    # Step 4: Display Output
    print(f"\n{'='*70}\nGENERATED {len(final_reports)} MARKET INTELLIGENCE CASE STUDIES\n{'='*70}")
    for idx, r in enumerate(final_reports, 1):
        print(f"\nCASE STUDY #{idx}: [{r.ticker}]")
        print(f"Catalyst: {r.catalyst_summary}")
        print(f"First-Order Impact: {r.company_impact}")
        print(f"Read-Across (Peers): {r.competitors_impact}")
        print(f"Macro Link: {r.macro_ripple}")
        
        print("\n🎓 Classroom Concepts:")
        for term in r.key_terms:
            print(f"  • {term.term}: {term.definition}")
            print(f"    Applied: {term.applied_to_news}")
            
        print(f"\n🏛️ Historical Precedent:\n  {r.educational_insights.historical_precedent}")
        print(f"⚠️ The Retail Trap:\n  {r.educational_insights.retail_trap}")
        print(f"❓ Pop Quiz:\n  {r.educational_insights.pop_quiz}")
        print(f"\n💼 Trader Playbook:\n  {r.trader_takeaway}")
        print("-" * 70)

if __name__ == "__main__":
    run_pipeline()