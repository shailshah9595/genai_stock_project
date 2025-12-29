"""fetch_data.py
Utilities to fetch historical stock prices and recent news.
- Uses yfinance for price data
- Optionally uses NewsAPI (newsapi-python) for headlines (NewsAPI key required)
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_prices(ticker: str, days: int = 30) -> pd.DataFrame:
    """Fetch daily OHLCV for the past `days` trading days using yfinance."""
    end = datetime.utcnow().date()
    start = end - timedelta(days=int(days * 1.5))  # a bit extra to account for weekends
    df = yf.download(ticker, start=start.isoformat(), end=end.isoformat(), progress=False)
    if df.empty:
        raise ValueError(f"No price data fetched for {ticker}. Check ticker symbol or network.")
    df = df.dropna()
    # Keep only the last `days` rows
    df = df.tail(days).copy()
    # Reset index to have Date as a column
    df.reset_index(inplace=True)
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    cols = ['Date','Open','High','Low','Close','Volume']
    if 'Adj Close' in df.columns:
      cols.insert(5,'Adj Close')

    return df[cols]

def format_prices_for_prompt(df: pd.DataFrame) -> str:
    """Turn the price DataFrame into a simple text table for the LLM prompt."""
    lines = []
    lines.append("Date Open High Low Close Volume")

    for _, r in df.iterrows():
        open_p = float(r["Open"])
        high_p = float(r["High"])
        low_p = float(r["Low"])
        close_p = float(r["Close"])
        volume = int(r["Volume"])

        lines.append(
            f"{r['Date']} {open_p:.2f} {high_p:.2f} {low_p:.2f} {close_p:.2f} {volume}"
        )

    return "\n".join(lines)


def fetch_market_context(symbol: str) -> str:
    stock = yf.Ticker(symbol)
    spy = yf.Ticker("SPY")
    vix = yf.Ticker("^VIX")

    s = stock.history(period="2d")
    i = spy.history(period="2d")
    v = vix.history(period="2d")

    if len(s) < 2:
        return "Market context unavailable."

    today_change = ((s["Close"].iloc[-1] - s["Close"].iloc[-2]) / s["Close"].iloc[-2]) * 100
    spy_change = ((i["Close"].iloc[-1] - i["Close"].iloc[-2]) / i["Close"].iloc[-2]) * 100
    vix_level = v["Close"].iloc[-1]

    gap = ((s["Open"].iloc[-1] - s["Close"].iloc[-2]) / s["Close"].iloc[-2]) * 100

    context = f"""
TODAY'S MARKET CONTEXT:
- {symbol} change today: {today_change:.2f}%
- Gap at open: {gap:.2f}%
- S&P 500 (SPY) change: {spy_change:.2f}%
- VIX level: {vix_level:.2f}

Interpretation guidance:
- Positive SPY + falling VIX = risk-on
- Negative SPY + rising VIX = risk-off
- Large gap-ups often retrace intraday
"""
    return context

# Optional: News fetching (NewsAPI)
def fetch_news_headlines(query: str, api_key: str = None, page_size: int = 5):
    """Return a list of recent headlines about `query`. Requires NewsAPI key. If not provided, returns empty list."""
    if not api_key:
        return []
    try:
        from newsapi import NewsApiClient
    except Exception as e:
        raise RuntimeError("newsapi-python is not installed. Install with: pip install newsapi-python") from e
    client = NewsApiClient(api_key=api_key)
    res = client.get_everything(q=query, language='en', sort_by='relevancy', page_size=page_size)
    articles = res.get('articles') or []
    headlines = [a.get('title') for a in articles if a.get('title')]
    return headlines
