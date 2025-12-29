import streamlit as st
import pandas as pd
import os
from fetch_data import fetch_prices, format_prices_for_prompt, fetch_news_headlines, fetch_market_context
from genai_predict import build_prompt, call_openai, parse_response
from backtest import backtest_ticker
from config import NEWS_API_KEY

st.set_page_config(page_title='GenAI 7-Day Stock Trend Explorer', layout='wide')
st.title('GenAI 7-Day Stock Trend Explorer (Educational)')
st.info('This tool uses an LLM to provide a simple 7-day trend prediction for educational purposes. Not financial advice.')

# -----------------------------
# Sidebar settings
# -----------------------------
with st.sidebar:
    st.header('Settings')
    days = st.number_input('Days to include in prompt', min_value=3, max_value=60, value=30)
    headlines_count = st.number_input('Number of headlines to fetch (NewsAPI)', min_value=0, max_value=10, value=3)
    model = st.selectbox('Which model to call', options=['gpt-4','gpt-4o-mini','gpt-3.5-turbo'], index=0)
    run_backtest = st.checkbox('Run a small backtest (can be costly)', value=False)
    backtest_days = st.number_input('Backtest trials (sliding windows)', min_value=5, max_value=30, value=10)

# -----------------------------
# User input
# -----------------------------
ticker = st.text_input('Stock ticker (e.g., AAPL)', value='AAPL').upper().strip()

if st.button('Fetch & Predict'):
    try:
        # Fetch price data and calculate SMA7
        df = fetch_prices(ticker, days)
        df['SMA7'] = df['Close'].rolling(window=7).mean()
    except Exception as e:
        st.error(f"Failed to fetch prices: {e}")
        st.stop()

    st.subheader('Recent Prices')
    st.dataframe(df.tail(10))

    # Prepare price table for prompt
    price_table = format_prices_for_prompt(df)

    # Fetch news
    api_key = NEWS_API_KEY
    headlines = fetch_news_headlines(ticker, api_key=api_key, page_size=headlines_count) if headlines_count > 0 else []
    if headlines:
        st.subheader('Headlines')
        for h in headlines:
            st.write('-', h)

    # Fetch market context
    market_context = fetch_market_context(ticker)

    # Build 7-day prompt
    prompt = build_prompt(
        ticker=ticker,
        price_table=price_table,
        headlines=headlines,
        market_context=market_context,
        prediction_days=7  # NEW: 7-day horizon
    )
    st.subheader('Prompt sent to GenAI (for transparency)')
    st.code(prompt[:1500] + ('\n... (truncated)' if len(prompt) > 1500 else ''))

    # Call OpenAI / GenAI
    try:
        text = call_openai(prompt, model=model)
        parsed = parse_response(text, days=7)  # NEW: parse 7-day output
        st.subheader('GenAI 7-Day Prediction')

        for day in range(1, 8):
            day_key = f'Day {day}'
            if day_key in parsed:
                st.write(f"**{day_key}**: Trend: {parsed[day_key]['trend']}, Confidence: {parsed[day_key]['confidence']}%")
                st.write("Reasoning:")
                for r in parsed[day_key]['reasoning']:
                    st.write('-', r)
            else:
                st.write(f"**{day_key}**: SIDEWAYS, Confidence: 50%, Reasoning: Market unclear")
    except Exception as e:
        st.error(f"OpenAI call failed: {e}")

    # Optional backtest
    if run_backtest:
        st.warning('Running backtest — this can incur many OpenAI calls and cost money.')
        try:
            bt = backtest_ticker(ticker, days_for_prediction=days, backtest_days=int(backtest_days), model=model)
            st.subheader('Backtest Summary')
            st.write(bt['summary'])
            st.subheader('Backtest Results (most recent trials)')
            st.dataframe(bt['results'].tail(50))
        except Exception as e:
            st.error(f"Backtest failed: {e}")
