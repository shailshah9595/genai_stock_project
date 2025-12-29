"""backtest.py
Simple backtester that runs the GenAI predictor over a historical window.
WARNING: This will do many API calls to OpenAI (one per prediction window). Be mindful of rate limits and cost.
Use a small number of days (e.g., 10) for experimentation.
"""
import time
from typing import List, Dict
import pandas as pd
from fetch_data import fetch_prices, format_prices_for_prompt
from genai_predict import build_prompt, call_openai, parse_response

def backtest_ticker(ticker: str, days_for_prediction: int = 5, backtest_days: int = 20, model: str = 'gpt-4') -> Dict:
    """For each day in the backtest period, use the previous `days_for_prediction` days to predict the next day's direction.
    Returns a dictionary with results and summary stats.
    """
    df = fetch_prices(ticker, days=backtest_days + days_for_prediction + 5)
    if df.shape[0] < days_for_prediction + 2:
        raise ValueError("Not enough data to backtest. Increase data window.")
    results = []
    # iterate over a sliding window ending `backtest_days` days from the end
    for i in range(-backtest_days,  -1):
        window = df.iloc[i - days_for_prediction : i].copy().reset_index(drop=True)
        next_day = df.iloc[i].copy()  # this is the 'actual' next day
        price_table = format_prices_for_prompt(window)
        prompt = build_prompt(ticker, price_table, [])
        # call the model
        try:
            resp_text = call_openai(prompt, model=model)
        except Exception as e:
            raise RuntimeError(f"OpenAI call failed: {e}")
        parsed = parse_response(resp_text)
        # determine actual direction
        prev_close = window.iloc[-1]['Close']
        actual_close = next_day['Close']
        if actual_close > prev_close * 1.002:  # >0.2% up
            actual = 'UP'
        elif actual_close < prev_close * 0.998:  # >0.2% down
            actual = 'DOWN'
        else:
            actual = 'STABLE'
        correct = (parsed.get('trend','').upper() == actual)
        results.append({
            'date': next_day['Date'],
            'predicted': parsed.get('trend'),
            'confidence': parsed.get('confidence'),
            'actual': actual,
            'correct': bool(correct)
        })
        # polite sleep to avoid rate limits
        time.sleep(1.0)
    resdf = pd.DataFrame(results)
    accuracy = resdf['correct'].mean()
    summary = {'ticker': ticker, 'accuracy': float(accuracy), 'trials': len(resdf)}
    return {'summary': summary, 'results': resdf}
