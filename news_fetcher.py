import requests
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# Replace with your Finnhub API key
# Get free key at: finnhub.io/register
# ─────────────────────────────────────────────
FINNHUB_API_KEY = "d6g6hs9r01qt4931nlr0d6g6hs9r01qt4931nlrg"


def get_stock_news(symbol, days_back=7):
    """
    Fetch real-time company news from Finnhub.
    Works for US stocks (AAPL, TSLA) and Indian stocks (RELIANCE, TCS).
    For Indian stocks, pass just the base symbol e.g. 'TCS' not 'TCS.NS'
    """
    today     = datetime.today().strftime('%Y-%m-%d')
    from_date = (datetime.today() - timedelta(days=days_back)).strftime('%Y-%m-%d')

    # Clean symbol — remove .NS / .BO suffix for Finnhub
    clean_symbol = symbol.replace(".NS", "").replace(".BO", "").replace(".BSE", "").upper()

    url = (
        f"https://finnhub.io/api/v1/company-news"
        f"?symbol={clean_symbol}&from={from_date}&to={today}&token={FINNHUB_API_KEY}"
    )

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        articles = response.json()

        if not articles:
            return _get_general_market_news()

        headlines = [a["headline"] for a in articles if a.get("headline")]
        return headlines[:10]

    except Exception as e:
        print(f"[Finnhub Error] {e}")
        return ["Market activity remains steady amid global uncertainty"]


def get_stock_news_with_details(symbol, days_back=7):
    """
    Returns full article details including headline, summary, source, url, datetime.
    Useful for displaying full news cards in the dashboard.
    """
    today     = datetime.today().strftime('%Y-%m-%d')
    from_date = (datetime.today() - timedelta(days=days_back)).strftime('%Y-%m-%d')

    clean_symbol = symbol.replace(".NS", "").replace(".BO", "").upper()

    url = (
        f"https://finnhub.io/api/v1/company-news"
        f"?symbol={clean_symbol}&from={from_date}&to={today}&token={FINNHUB_API_KEY}"
    )

    try:
        response = requests.get(url, timeout=10)
        articles = response.json()

        result = []
        for a in articles[:10]:
            result.append({
                "headline" : a.get("headline", ""),
                "summary"  : a.get("summary",  "")[:200] + "...",
                "source"   : a.get("source",   ""),
                "url"      : a.get("url",       ""),
                "datetime" : datetime.fromtimestamp(a.get("datetime", 0)).strftime('%Y-%m-%d %H:%M')
            })
        return result

    except Exception as e:
        print(f"[Finnhub Detail Error] {e}")
        return []


def _get_general_market_news():
    """
    Fallback: fetch general market news when no company-specific news is found.
    """
    url = f"https://finnhub.io/api/v1/news?category=general&token={FINNHUB_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        articles = response.json()
        return [a["headline"] for a in articles[:10] if a.get("headline")]
    except Exception:
        return ["Market shows mixed signals today"]


def get_insider_sentiment(symbol):
    """
    BONUS: Get insider buy/sell sentiment for a stock.
    Returns 'Bullish', 'Bearish', or 'Neutral' based on insider activity.
    MSPR = Monthly Share Purchase Ratio (ranges from -1 to 1)
    """
    clean_symbol = symbol.replace(".NS", "").replace(".BO", "").upper()
    url = (
        f"https://finnhub.io/api/v1/stock/insider-sentiment"
        f"?symbol={clean_symbol}&from=2024-01-01&token={FINNHUB_API_KEY}"
    )

    try:
        response = requests.get(url, timeout=10)
        data = response.json().get("data", [])

        if not data:
            return "Neutral"

        latest = data[-1]
        mspr   = latest.get("mspr", 0)

        if mspr > 0.1:
            return "Bullish"
        elif mspr < -0.1:
            return "Bearish"
        return "Neutral"

    except Exception:
        return "Neutral"