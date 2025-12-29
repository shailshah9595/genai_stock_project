GenAI Stock Trend Project (High-school friendly)

This project demonstrates a beginner-friendly, deployable pipeline that uses a large language model (LLM)
to provide simple stock trend predictions (UP / DOWN / SIDEWAYS) using recent price history, SMA7, market context, and news headlines.

Important: This is an educational project. The model is not a financial adviser and predictions are noisy.

⸻

Project Contents
	•	fetch_data.py    : Fetch prices (yfinance), compute SMA7, market context, and news headlines (NewsAPI optional).
	•	genai_predict.py : Build LLM prompt and call OpenAI to get a prediction for next 7 days.
	•	backtest.py      : Optional backtester to evaluate LLM predictions vs actual outcomes.
	•	app.py           : Streamlit app for interactive use.
	•	config.py        : API keys for OpenAI and NewsAPI.
	•	requirements.txt : Python dependencies.

⸻

Quickstart (local)
	1.	Clone or download this project:

git clone <repo_url>
cd genai_stock_project

	2.	Create a Python virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # macOS / Linux
venv\Scripts\activate     # Windows (PowerShell)

	3.	Install dependencies:

pip install -r requirements.txt

	4.	Add your API keys in config.py:

OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
NEWS_API_KEY = "YOUR_NEWSAPI_KEY"  # optional, for headlines

	5.	Run the Streamlit app:

streamlit run app.py


⸻

How it works
	1.	The app fetches the recent N days of price data for a ticker using yfinance.
	2.	It computes 7-day SMA (SMA7) for trend context.
	3.	It optionally fetches recent news headlines using NewsAPI.
	4.	It fetches market context: SPY daily change, VIX level, and gap at open.
	5.	It formats a prompt containing price rows, SMA7, headlines, and market context and asks the LLM to:
	•	Predict UP / DOWN / SIDEWAYS for the next 7 trading days
	•	Provide a confidence score
	•	Provide a short, plain-English explanation
	6.	Optionally, run a backtest to evaluate historical performance.

⸻

Notes & Caveats
	•	SMA7 is always computed, even if fewer than 7 days of data are available.
	•	Backtesting uses multiple LLM calls (one per historical trial). This can be costly — use small backtest sizes.
	•	LLM predictions reflect patterns in the supplied data and do not guarantee future performance.
	•	Predictions are for educational purposes only — do not use as investment advice.
	•	Error handling ensures single tickers work, SMA7 always returns numeric values, and missing data is handled gracefully.

⸻

Suggested Extensions (for college showcase)
	•	Add more technical indicators (RSI, MACD) in the prompt.
	•	Plot candlestick charts with SMA7 overlay.
	•	Highlight news sentiment (positive/negative) visually.
	•	Create a “what-if” scenario generator for hypothetical news items.
	•	Animate trend predictions for 7 days with confidence shading.
	•	Generate a PDF report summarizing predictions and backtest results.
	•	Host the app on Streamlit Cloud or Heroku and include a demo link in application materials.

⸻

Sample Output

Recent Prices Table:

Date	Open	High	Low	Close	Volume	SMA7
2025-12-20	175.32	178.45	174.10	177.80	34500000	176.12
2025-12-21	178.10	180.20	176.50	179.50	28700000	176.98
…	…	…	…	…	…	
