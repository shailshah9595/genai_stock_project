import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_prices(ticker: str, days: int = 30) -> pd.DataFrame:
    end = datetime.utcnow().date()
    start = end - timedelta(days=int(days * 1.5))

    df = yf.download(ticker, start=start.isoformat(), end=end.isoformat(), progress=False)

    if df.empty:
        raise ValueError(f"No price data fetched for {ticker}.")

    # --- Flatten multi-level columns ---
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep last N rows
    df = df.dropna().tail(days).copy()
    df.reset_index(inplace=True)
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

    # Ensure numeric columns
    numeric_cols = ['Open','High','Low','Close','Volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Compute SMA7
    df['SMA7'] = df['Close'].rolling(window=7, min_periods=1).mean().fillna(method='bfill').astype(float)

    return df[['Date','Open','High','Low','Close','Volume','SMA7']]

def format_prices_for_prompt(df: pd.DataFrame) -> str:
    """Format prices into a text table for LLM prompt, ensuring all scalars."""
    lines = []
    lines.append("Date Open High Low Close Volume SMA7")

    for _, r in df.iterrows():
        date = r['Date']
        open_p = float(r['Open'])
        high_p = float(r['High'])
        low_p = float(r['Low'])
        close_p = float(r['Close'])
        volume = int(r['Volume'])

        # Ensure SMA7 is a scalar float
        sma7 = r['SMA7']
        if hasattr(sma7, "__len__"):  # if somehow it is a Series
            sma7 = float(sma7.iloc[0])
        else:
            sma7 = float(sma7)

        line = f"{date} {open_p:.2f} {high_p:.2f} {low_p:.2f} {close_p:.2f} {volume} {sma7:.2f}"
        lines.append(line)

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

def fetch_news_headlines(query: str, api_key: str = None, page_size: int = 5):
    """Return a list of recent headlines about `query`. Requires NewsAPI key."""
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
