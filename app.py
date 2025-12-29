"""Streamlit App (app.py)
Simple interactive UI:
- Enter ticker
- Fetch last N days of prices + headlines
- Show price table
- Ask GenAI for a prediction
- Optional: run a short backtest (uses OpenAI API heavily)
"""
import streamlit as st
import pandas as pd
import os
from fetch_data import fetch_prices, format_prices_for_prompt, fetch_news_headlines
from fetch_data import fetch_market_context
from genai_predict import build_prompt, call_openai, parse_response
from backtest import backtest_ticker
from config import NEWS_API_KEY
st.set_page_config(page_title='GenAI Stock Trend Explorer', layout='wide')
st.title('GenAI Stock Trend Explorer (Educational)')
st.info('This tool uses an LLM to provide a simple UP/DOWN/STABLE trend prediction for educational purposes. Not financial advice.')

with st.sidebar:
    st.header('Settings')
    days = st.number_input('Days to include in prompt', min_value=3, max_value=30, value=5)
    headlines_count = st.number_input('Number of headlines to fetch (NewsAPI)', min_value=0, max_value=10, value=3)
    model = st.selectbox('Which model to call (your OpenAI plan may vary)', options=['gpt-4','gpt-4o-mini','gpt-3.5-turbo'], index=0)
    run_backtest = st.checkbox('Run a small backtest (can be costly)', value=False)
    backtest_days = st.number_input('Backtest trials (sliding windows)', min_value=5, max_value=30, value=10)

ticker = st.text_input('Stock ticker (e.g., AAPL)', value='AAPL').upper().strip()
if st.button('Fetch & Predict'):
    try:
        df = fetch_prices(ticker, days)
    except Exception as e:
        st.error(f"Failed to fetch prices: {e}")
        st.stop()
    st.subheader('Recent Prices')
    st.dataframe(df)

    price_table = format_prices_for_prompt(df)
    api_key = NEWS_API_KEY #os.environ.get('NEWSAPI_KEY') or None
    headlines = fetch_news_headlines(ticker, api_key=api_key, page_size=headlines_count) if headlines_count > 0 else []
    if headlines:
        st.subheader('Headlines')
        for h in headlines:
            st.write('-', h)

    market_context = fetch_market_context(ticker)
    prompt = build_prompt(ticker, price_table, headlines,market_context)
    st.subheader('Prompt sent to GenAI (for transparency)')
    st.code(prompt[:1500] + ('\n... (truncated)' if len(prompt) > 1500 else ''))

    # Call OpenAI
    try:
        text = call_openai(prompt, model=model)
        parsed = parse_response(text)
        st.subheader('GenAI Prediction')
        st.write('**Trend:**', parsed.get('trend'))
        st.write('**Confidence:**', parsed.get('confidence'))
        st.write('**Explanation:**', parsed.get('explanation'))
    except Exception as e:
        st.error(f"OpenAI call failed: {e}")

    if run_backtest:
        st.warning('Running backtest — this can incur many OpenAI calls and cost money. Proceed with caution.')
        try:
            bt = backtest_ticker(ticker, days_for_prediction=days, backtest_days=int(backtest_days), model=model)
            st.subheader('Backtest Summary')
            st.write(bt['summary'])
            st.subheader('Backtest Results (most recent trials)')
            st.dataframe(bt['results'].tail(50))
        except Exception as e:
            st.error(f"Backtest failed: {e}")
