
import sys

sys.stdout.reconfigure(encoding="utf-8")
from typing import Literal, TypedDict

import pandas as pd
import yfinance as yf
import feedparser
from urllib.parse import quote

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, START, END
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from datetime import timedelta

# ============================================================
# 1. LANGGRAPH STATE
# ============================================================

class StockState(TypedDict):
    stock: str
    exchange: str

    price: float
    price_history: object
    technical: str
    fundamental: str
    news: str
    source_quality: str
    recommendation: str
    confidence: int
    rationale: str

    decision: str
    retry_count: int


# ============================================================
# 2. STRUCTURED LLM OUTPUT
# ============================================================

class StockAnalysis(BaseModel):

    recommendation: Literal["BUY", "HOLD", "SELL"]

    confidence: int = Field(
        ge=0,
        le=100
    )

    rationale: str


# ============================================================
# 3. QWEN
# ============================================================

llm = ChatOllama(
    model="qwen3:8b",
    base_url="http://localhost:11434"
)

structured_llm = llm.with_structured_output(StockAnalysis)


# ============================================================
# 4. MARKET DATA NODE
# ============================================================

def market_data_node(state: StockState):

    print("\n---- MARKET DATA NODE ----")

    stock = state["stock"]
    exchange = state["exchange"]

    # Yahoo Finance ticker convention
    if exchange == "NSE":
        ticker_symbol = stock + ".NS"

    elif exchange == "BSE":
        ticker_symbol = stock + ".BO"

    else:
        raise ValueError(
            f"Unsupported exchange: {exchange}"
        )

    print("Fetching:", ticker_symbol)

    ticker = yf.Ticker(ticker_symbol)

    # Get approximately 1 year of daily data
    data = ticker.history(period="1y")

    if data.empty:
        raise ValueError(
            f"No market data found for {stock} on {exchange}"
        )

    current_price = float(data["Close"].iloc[-1])

    print("Current Price:", current_price)
    print("Rows retrieved:", len(data))

    return {
        "price": current_price,
        "price_history": data
    }

def validate_market_data_node(state: StockState):

    print("\n---- VALIDATE MARKET DATA ----")

    price = state["price"]

    print("Price:", price)
    print("Retry count:", state["retry_count"])

    if price is None:
        print("❌ Price is missing")
        return {"retry_count": state["retry_count"] + 1}

    if price != price:
        print("❌ Price is NaN")
        return {"retry_count": state["retry_count"] + 1}

    print("✅ Market data looks valid")

    return {}


def validate_market_data(state: StockState):

    price = state["price"]
    retry_count = state["retry_count"]

    # Valid
    if price is not None and price == price:
        return "valid"

    # Retry available
    if retry_count < 3:
        return "retry"

    # Maximum retries reached
    return "failed"


# ============================================================
# 5. TECHNICAL ANALYSIS NODE
# ============================================================

def technical_node(state: StockState):

    print("\n---- TECHNICAL ANALYSIS NODE ----")

    # Reconstruct dataframe
    data = state["price_history"].copy()

    # --------------------------------------------------------
    # Moving averages
    # --------------------------------------------------------

    data["DMA20"] = data["Close"].rolling(20).mean()

    data["DMA50"] = data["Close"].rolling(50).mean()

    data["DMA200"] = data["Close"].rolling(200).mean()

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    delta = data["Close"].diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()

    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    data["RSI"] = 100 - (
        100 / (1 + rs)
    )

    # --------------------------------------------------------
    # 52-week high / low
    # --------------------------------------------------------

    week52_high = data["Close"].max()

    week52_low = data["Close"].min()

    # --------------------------------------------------------
    # Latest values
    # --------------------------------------------------------

    latest = data.iloc[-1]

    dma20 = latest["DMA20"]

    dma50 = latest["DMA50"]

    dma200 = latest["DMA200"]

    rsi = latest["RSI"]

    # --------------------------------------------------------
    # Determine trend
    # --------------------------------------------------------

    if dma50 > dma200:
        trend = "Bullish"
    else:
        trend = "Bearish"

    technical = f"""
Current Price: ₹{state["price"]:.2f}

20 DMA: ₹{dma20:.2f}
50 DMA: ₹{dma50:.2f}
200 DMA: ₹{dma200:.2f}

RSI: {rsi:.2f}

52 Week High: ₹{week52_high:.2f}
52 Week Low: ₹{week52_low:.2f}

Technical Trend: {trend}
"""

    print(technical)

    return {
        "technical": technical
    }

def fundamental_node(state: StockState):

    print("\n---- FUNDAMENTAL ANALYSIS NODE ----")

    stock = state["stock"]
    exchange = state["exchange"]

    if exchange == "NSE":
        ticker_symbol = stock + ".NS"

    elif exchange == "BSE":
        ticker_symbol = stock + ".BO"

    else:
        raise ValueError(
            f"Unsupported exchange: {exchange}"
        )

    print("Fetching fundamental data:", ticker_symbol)

    ticker = yf.Ticker(ticker_symbol)

    info = ticker.info

    # --------------------------------------------------------
    # Extract metrics
    # --------------------------------------------------------

    pe = info.get("trailingPE")

    forward_pe = info.get("forwardPE")

    eps = info.get("trailingEps")

    roe = info.get("returnOnEquity")

    roa = info.get("returnOnAssets")

    debt_to_equity = info.get("debtToEquity")

    revenue_growth = info.get("revenueGrowth")

    earnings_growth = info.get("earningsGrowth")

    dividend_yield = info.get("dividendYield")

    market_cap = info.get("marketCap")

    # --------------------------------------------------------
    # Convert percentages
    # --------------------------------------------------------

    if roe is not None:
        roe = roe * 100

    if roa is not None:
        roa = roa * 100

    if revenue_growth is not None:
        revenue_growth = revenue_growth * 100

    if earnings_growth is not None:
        earnings_growth = earnings_growth * 100

    if dividend_yield is not None:
        dividend_yield = dividend_yield * 100

    # --------------------------------------------------------
    # Format values safely
    # --------------------------------------------------------

    def fmt(value, suffix=""):

        if value is None:
            return "Not available"

        return f"{value:.2f}{suffix}"

    # --------------------------------------------------------
    # Fundamental report
    # --------------------------------------------------------

    fundamental = f"""
Stock: {stock}
Exchange: {exchange}

P/E Ratio:
{fmt(pe)}

Forward P/E:
{fmt(forward_pe)}

EPS:
{fmt(eps)}

ROE:
{fmt(roe, "%")}

ROA:
{fmt(roa, "%")}

Debt / Equity:
{fmt(debt_to_equity)}

Revenue Growth:
{fmt(revenue_growth, "%")}

Earnings Growth:
{fmt(earnings_growth, "%")}

Dividend Yield:
{fmt(dividend_yield, "%")}

Market Capitalization:
{market_cap if market_cap is not None else "Not available"}
"""

    print(fundamental)

    return {
        "fundamental": fundamental
    }

def parse_news_date(date_string):

    if not date_string:
        return datetime.min.replace(
            tzinfo=timezone.utc
        )

    try:

        dt = parsedate_to_datetime(
            date_string
        )

        if dt.tzinfo is None:

            dt = dt.replace(
                tzinfo=timezone.utc
            )

        return dt

    except Exception:

        return datetime.min.replace(
            tzinfo=timezone.utc
        )

def news_node(state: StockState):

    print("\n---- NEWS ANALYSIS NODE ----")

    stock = state["stock"]
    exchange = state["exchange"]

    print("Collecting news for:", stock)
    print("Exchange:", exchange)

    # --------------------------------------------------
    # Multiple news searches
    # --------------------------------------------------

    queries = [
        f"{stock} latest India stock news",
        f"{stock} earnings India",
        f"{stock} results India",
        f"{stock} orders India",
        f"{stock} acquisition India",
        f"{stock} management India",
        f"{stock} analyst India",
        f"{stock} {exchange}",
    ]

    all_articles = []

    # --------------------------------------------------
    # Search Google News RSS
    # --------------------------------------------------

    for query in queries:

        print("\nSearching:", query)

        encoded_query = quote(query)

        rss_url = (
            "https://news.google.com/rss/search?"
            f"q={encoded_query}&"
            "hl=en-IN&"
            "gl=IN&"
            "ceid=IN:en"
        )

        feed = feedparser.parse(rss_url)

        print(
            "Articles found:",
            len(feed.entries)
        )

        for entry in feed.entries:

            title = entry.get(
                "title",
                ""
            )

            link = entry.get(
                "link",
                ""
            )

            published = entry.get(
                "published",
                ""
            )

            source = "Unknown"

            if hasattr(entry, "source"):

                source = entry.source.get(
                    "title",
                    "Unknown"
                )

            all_articles.append({
                "title": title,
                "link": link,
                "published": published,
                "source": source
            })

    print(
        "\nTotal articles collected:",
        len(all_articles)
    )

    # --------------------------------------------------
    # Remove duplicate articles
    # --------------------------------------------------

    unique_articles = {}

    for article in all_articles:

        title = article["title"].strip().lower()

        if not title:
            continue

        if title not in unique_articles:

            unique_articles[title] = article

    articles = list(
        unique_articles.values()
    )

    print(
        "After removing duplicates:",
        len(articles)
    )

    # --------------------------------------------------
    # Keep only recent news - last 7 days
    # --------------------------------------------------

    now = datetime.now(timezone.utc)

    recent_articles = []

    for article in articles:

        published_date = parse_news_date(
            article["published"]
        )

        age = now - published_date

        if age <= timedelta(days=7):

            recent_articles.append(article)

    articles = recent_articles

    print(
        "Articles from last 7 days:",
        len(articles)
    )

    # --------------------------------------------------
    # Source Quality Score
    # --------------------------------------------------

    def get_source_score(source):

        source_lower = source.lower()

        high_quality = [
            "reuters",
            "bloomberg",
            "economic times",
            "moneycontrol",
            "business standard",
            "mint",
            "cnbc",
            "business today",
            "financial express"
        ]

        for name in high_quality:

            if name in source_lower:

                return 25

        return 12

    # --------------------------------------------------
    # Source Quality Label
    # --------------------------------------------------

    def get_source_quality(source):

        score = get_source_score(source)

        if score == 25:

            return "HIGH"

        return "NORMAL"

    # --------------------------------------------------
    # Relevance Score
    # --------------------------------------------------

    def get_relevance_score(
        title,
        stock
    ):

        title_lower = title.lower()
        stock_lower = stock.lower()

        # Stock explicitly mentioned
        # in the headline

        if stock_lower in title_lower:

            return 20

        # Otherwise consider it less relevant

        return 8

    # --------------------------------------------------
    # Materiality Score
    # --------------------------------------------------

    def get_materiality_score(title):

        title_lower = title.lower()

        high_materiality_words = [

            "earnings",
            "results",
            "profit",
            "revenue",
            "order",
            "contract",
            "acquisition",
            "merger",
            "buyback",
            "dividend",
            "guidance",
            "layoff",
            "regulatory",
            "penalty",
            "lawsuit",
            "fraud",
            "deal",
            "partnership"
        ]

        for word in high_materiality_words:

            if word in title_lower:

                return 15

        return 5

    # --------------------------------------------------
    # Recency Score
    # --------------------------------------------------

    def get_recency_score(published):

        article_date = parse_news_date(
            published
        )

        if article_date == datetime.min.replace(
            tzinfo=timezone.utc
        ):

            return 0

        now = datetime.now(
            timezone.utc
        )

        age_hours = (
            now - article_date
        ).total_seconds() / 3600

        if age_hours < 6:

            return 40

        elif age_hours < 12:

            return 35

        elif age_hours < 24:

            return 30

        elif age_hours < 48:

            return 25

        elif age_hours < 72:

            return 18

        elif age_hours < 120:

            return 10

        elif age_hours <= 168:

            return 5

        return 0

    # --------------------------------------------------
    # Calculate Article Score
    # --------------------------------------------------

    def calculate_article_score(
        article,
        stock
    ):

        recency_score = get_recency_score(
            article["published"]
        )

        source_score = get_source_score(
            article["source"]
        )

        relevance_score = get_relevance_score(
            article["title"],
            stock
        )

        materiality_score = get_materiality_score(
            article["title"]
        )

        total_score = (
            recency_score
            + source_score
            + relevance_score
            + materiality_score
        )

        return {
            "recency_score": recency_score,
            "source_score": source_score,
            "relevance_score": relevance_score,
            "materiality_score": materiality_score,
            "article_score": total_score
        }

    # --------------------------------------------------
    # Calculate scores for every article
    # --------------------------------------------------

    for article in articles:

        scores = calculate_article_score(
            article,
            stock
        )

        article.update(scores)

    # --------------------------------------------------
    # Sort by Article Score
    # Highest score first
    # --------------------------------------------------

    articles.sort(
        key=lambda x: x["article_score"],
        reverse=True
    )

    # --------------------------------------------------
    # Keep Top 20 Ranked Articles
    # --------------------------------------------------

    articles = articles[:20]

    print(
        "\nArticles selected for analysis:",
        len(articles)
    )

    # --------------------------------------------------
    # Print Article Ranking
    # --------------------------------------------------

    print("\n====================================")
    print("       NEWS ARTICLE RANKING")
    print("====================================")

    for i, article in enumerate(
        articles,
        start=1
    ):

        print(
            f"\n{i}. Score: "
            f"{article['article_score']}"
        )

        print(
            f"   Title: "
            f"{article['title']}"
        )

        print(
            f"   Source: "
            f"{article['source']}"
        )

        print(
            f"   Source Quality: "
            f"{get_source_quality(article['source'])}"
        )

        print(
            f"   Published: "
            f"{article['published']}"
        )

        print(
            f"   Recency Score: "
            f"{article['recency_score']}"
        )

        print(
            f"   Source Score: "
            f"{article['source_score']}"
        )

        print(
            f"   Relevance Score: "
            f"{article['relevance_score']}"
        )

        print(
            f"   Materiality Score: "
            f"{article['materiality_score']}"
        )

    # --------------------------------------------------
    # Build news text
    # --------------------------------------------------

    news_text = ""

    for i, article in enumerate(
        articles,
        start=1
    ):

        title = article["title"]

        source = article["source"]

        source_quality = get_source_quality(
            source
        )

        published = article["published"]

        link = article["link"]

        article_score = article[
            "article_score"
        ]

        print(
            f"\n{i}. {title}"
        )

        print(
            f"   Source: {source}"
        )

        print(
            f"   Source Quality: "
            f"{source_quality}"
        )

        print(
            f"   Published: {published}"
        )

        news_text += f"""

{i}. {title}

Source: {source}

Source Quality: {source_quality}

Published: {published}

Article Score: {article_score}/100

URL: {link}

"""

    # --------------------------------------------------
    # Simple sentiment calculation
    # --------------------------------------------------

    positive_words = [

        "profit",
        "growth",
        "strong",
        "positive",
        "upgrade",
        "surge",
        "record",
        "deal",
        "order",
        "expansion",
        "contract",
        "wins"
    ]

    negative_words = [

        "loss",
        "decline",
        "weak",
        "negative",
        "downgrade",
        "fall",
        "drop",
        "fraud",
        "layoff",
        "risk",
        "warning",
        "cut"
    ]

    text_lower = news_text.lower()

    positive_count = sum(
        text_lower.count(word)
        for word in positive_words
    )

    negative_count = sum(
        text_lower.count(word)
        for word in negative_words
    )

    # --------------------------------------------------
    # Determine sentiment
    # --------------------------------------------------

    if positive_count > negative_count:

        sentiment = "Positive"

    elif negative_count > positive_count:

        sentiment = "Negative"

    else:

        sentiment = "Neutral"

    # --------------------------------------------------
    # Final News State
    # --------------------------------------------------

    news = f"""
Stock: {stock}

Exchange: {exchange}

RECENT RANKED NEWS:

{news_text}

NEWS SENTIMENT:
{sentiment}

Positive Signals:
{positive_count}

Negative Signals:
{negative_count}
"""

    print("\n================================")
    print("NEWS SENTIMENT:", sentiment)
    print("Positive:", positive_count)
    print("Negative:", negative_count)
    print("================================")

    return {
        "news": news
    }


# ============================================================
# 6. ANALYST NODE
# ============================================================

def analyst_node(state: StockState):

    print("\n---- QWEN ANALYST ----")

    prompt = f"""
    You are a professional Indian equity market analyst.

    You analyze stocks listed on:
    - NSE
    - BSE

    Analyze ONLY the information supplied below.

    Stock:
    {state["stock"]}

    Exchange:
    {state["exchange"]}

    Current Price:
    ₹{state["price"]:.2f}

    Technical Analysis:
    {state["technical"]}

    Fundamental Analysis:
    {state["fundamental"]}

    News Analysis:
    {state["news"]}

    IMPORTANT RULES:

    1. Do NOT invent financial data.
    2. Do NOT assume missing metrics.
    3. Use only the supplied market data.
    4. Use Indian stock-market terminology.
    5. Consider the technical indicators carefully.
    6. Give a cautious analytical recommendation.
    7. Confidence must reflect the quality and completeness of the available data.

    NEWS ANALYSIS RULES:

    8. News sentiment is derived from recent news articles.
    9. Consider the actual news headlines rather than relying only on the calculated sentiment label.
    10. Do not assume that positive news means the stock should be bought.
    11. Consider whether the news is material to the company's earnings,
    valuation, operations or future growth.
    12. Do not treat old or unrelated news as a current catalyst.
    13. Give greater weight to news from established financial and
    business-news sources.
    14. Do not treat an individual news article as confirmed fact unless
    supported by reliable information.
    15. Recent news should generally receive more weight than older news.
    16. Do not infer a BUY or SELL recommendation from news sentiment alone.
    17. If recent relevant news is unavailable, explicitly consider this
    limitation in your confidence.

    DECISION RULES:

    18. Consider Technical Analysis, Fundamental Analysis and News Analysis
    together.
    19. Do not allow one positive or negative indicator to determine the
    recommendation by itself.
    20. If important data is missing, reduce confidence accordingly.
    21. Distinguish between short-term technical signals and longer-term
    fundamental strength.
    22. The recommendation should represent the overall evidence available
    at the time of analysis.

    Recommendation must be:
    BUY, HOLD, or SELL.

    Confidence must be between 0 and 100.
    """

    print("Prompt sent to QWEN:")
    print(prompt)

    response = structured_llm.invoke(prompt)

    print("Recommendation:", response.recommendation)
    print("Confidence:", response.confidence)
    print("Rationale:", response.rationale)

    return {
        "recommendation": response.recommendation,
        "confidence": response.confidence,
        "rationale": response.rationale
    }

def start_analysis_node(state: StockState):
    print("\n---- STARTING PARALLEL ANALYSIS ----")
    return {}

# ============================================================
# 7. CONFIDENCE ROUTER
# ============================================================

def route_recommendation(state: StockState):

    recommendation = state["recommendation"]

    confidence = state["confidence"]

    print("\n---- CONFIDENCE ROUTER ----")

    print("Recommendation:", recommendation)

    print("Confidence:", confidence)

    if confidence >= 70:

        if recommendation == "BUY":
            return "strong_buy"

        elif recommendation == "HOLD":
            return "strong_hold"

        elif recommendation == "SELL":
            return "strong_sell"

    return "needs_more_analysis"


# ============================================================
# 8. DECISION NODES
# ============================================================

def strong_buy_node(state: StockState):

    decision = (
        f"STRONG BUY - {state['stock']} "
        f"with {state['confidence']}% confidence"
    )

    print("\n---- STRONG BUY ----")

    print(decision)

    return {
        "decision": decision
    }


def strong_hold_node(state: StockState):

    decision = (
        f"HOLD - {state['stock']} "
        f"with {state['confidence']}% confidence"
    )

    print("\n---- STRONG HOLD ----")

    print(decision)

    return {
        "decision": decision
    }


def strong_sell_node(state: StockState):

    decision = (
        f"STRONG SELL - {state['stock']} "
        f"with {state['confidence']}% confidence"
    )

    print("\n---- STRONG SELL ----")

    print(decision)

    return {
        "decision": decision
    }


def needs_more_analysis_node(state: StockState):

    decision = (
        f"INSUFFICIENT CONFIDENCE - "
        f"{state['stock']} "
        f"received {state['recommendation']} "
        f"with {state['confidence']}% confidence"
    )

    print("\n---- NEED MORE ANALYSIS ----")

    print(decision)

    return {
        "decision": decision
    }


# ============================================================
# 9. BUILD GRAPH
# ============================================================

builder = StateGraph(StockState)


builder.add_node(
    "market_data",
    market_data_node
)

builder.add_node(
    "validate_market",
    validate_market_data_node
)

builder.add_node(
    "start_analysis",
    start_analysis_node
)

builder.add_node(
    "technical",
    technical_node
)

builder.add_node(
    "analyst",
    analyst_node
)

builder.add_node(
    "strong_buy",
    strong_buy_node
)

builder.add_node(
    "strong_hold",
    strong_hold_node
)

builder.add_node(
    "strong_sell",
    strong_sell_node
)

builder.add_node(
    "needs_more_analysis",
    needs_more_analysis_node
)
builder.add_node(
    "fundamental", 
    fundamental_node
)

builder.add_node(
    "news",
    news_node
)


# ============================================================
# 10. GRAPH CONNECTIONS
# ============================================================

builder.add_edge(
    START,
    "market_data"
)

builder.add_edge("market_data", "validate_market")

builder.add_conditional_edges(
    "validate_market",
    validate_market_data,
    {
        "valid": "start_analysis",
        "retry": "market_data",
        "failed": END
    }
)






# Parallel branches

builder.add_edge(
    "start_analysis",
    "technical"
)

builder.add_edge(
    "start_analysis",
    "fundamental"
)

builder.add_edge(
    "start_analysis",
    "news"
)



# Fan-in
builder.add_edge("technical", "analyst")
builder.add_edge("fundamental", "analyst")
builder.add_edge("news", "analyst")


builder.add_conditional_edges(
    "analyst",
    route_recommendation,
    {
        "strong_buy": "strong_buy",
        "strong_hold": "strong_hold",
        "strong_sell": "strong_sell",
        "needs_more_analysis": "needs_more_analysis"
    }
)


builder.add_edge(
    "strong_buy",
    END
)

builder.add_edge(
    "strong_hold",
    END
)

builder.add_edge(
    "strong_sell",
    END
)

builder.add_edge(
    "needs_more_analysis",
    END
)


# ============================================================
# 11. COMPILE
# ============================================================

graph = builder.compile()

png_data = graph.get_graph().draw_mermaid_png()

with open("Stock_fundamental_analysis.png", "wb") as f:
    f.write(png_data)

# ============================================================
# 12. INITIAL STATE
# ============================================================
stock_name = input("Enter stock name: ").strip().upper()
exchange_name = input("Enter exchange (NSE/BSE): ").strip().upper()

initial_state = {

    "stock": stock_name,

    "exchange": exchange_name,

    "price": 0,

    "price_history": "",

    "technical": "",

    "fundamental": "",

    "news": "",

    "recommendation": "",

    "confidence": 0,

    "rationale": "",

    "decision": "",

    "retry_count": 0
}


# ============================================================
# 13. RUN
# ============================================================

print("\n==========================================")
print("       INDIAN STOCK ANALYST AGENT")
print("              LESSON 14")
print("==========================================")

result = graph.invoke(
    initial_state
)


# ============================================================
# 14. FINAL RESULT
# ============================================================

print("\n==========================================")
print("             FINAL RESULT")
print("==========================================")

print("Stock          :", result["stock"])
print("Exchange       :", result["exchange"])
print("Price          :", result["price"])
print("Recommendation :", result["recommendation"])
print("Confidence     :", result["confidence"])
print("Decision       :", result["decision"])
print("Rationale      :", result["rationale"])

print("==========================================")