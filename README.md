# GenAI Stock Trend Project (High-school friendly)

This project demonstrates a beginner-friendly, deployable pipeline that uses a large language model (LLM)
to provide simple stock trend predictions (UP / DOWN / STABLE) using recent price history and news headlines.

**Important:** This is an *educational* project. The model is not a financial adviser and predictions are noisy.

## Project Contents
- `fetch_data.py`    : Fetch prices (yfinance) and news headlines (NewsAPI optional).
- `genai_predict.py` : Build LLM prompt and call OpenAI to get a prediction.
- `backtest.py`      : Simple backtester to evaluate LLM predictions vs actual outcomes.
- `app.py`           : Streamlit app for interactive use.
- `requirements.txt` : Python dependencies.

## Quickstart (local)
1. Clone or download this project.
2. Create a Python virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS / Linux
   venv\Scripts\activate    # Windows (PowerShell)
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set environment variables for API keys:
   ```bash
   export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
   export NEWSAPI_KEY="YOUR_NEWSAPI_KEY"  # optional, for headlines
   ```
5. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## How it works
1. The app fetches the recent `N` days of price data for a ticker (using yfinance).
2. It optionally fetches recent news headlines (NewsAPI).
3. It formats a short prompt containing price rows + headlines and asks the LLM to choose UP/DOWN/STABLE,
   provide a confidence score and a short explanation.
4. Optionally, run a backtest to evaluate how the LLM would have performed historically.

## Notes & Caveats
- Backtesting uses multiple LLM calls (one per historical trial). This can be costly — use small backtest sizes.
- LLM predictions reflect patterns in the supplied data and do not guarantee future performance.
- This is an educational tool; do not use it as investment advice.

## Suggested Extensions (for college showcase)
- Add technical indicators (SMA, RSI, MACD) and include them in the prompt.
- Create a "what-if" scenario generator where the user proposes a hypothetical news item.
- Add a PDF report generator summarizing predictions and the backtest results.
- Host the app on Streamlit Cloud or Heroku and include a demo link in the application materials.

## License
MIT
