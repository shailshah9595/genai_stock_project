"""genai_predict.py
Wrapper that builds prompts and calls the OpenAI API to get a trend prediction
The model is asked to return:
  - Trend: UP / DOWN / STABLE
  - Confidence: 0-100%
  - Short explanation suitable for a high-schooler
IMPORTANT: Add your OPENAI_API_KEY in the environment variable OPENAI_API_KEY before running.
"""
import os
import openai
from typing import List, Dict
from config import OPENAI_API_KEY
from config import NEWS_API_KEY

#OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY') or ''
if not OPENAI_API_KEY:
    # We will not error here; the caller can handle the missing key.
    pass
else:
    openai.api_key = OPENAI_API_KEY

def build_prompt(ticker: str, price_table: str, headlines: List[str], market_context: List[str]) -> str:
    """Create a clear prompt for the LLM."""
    news_text = "\n".join([f"{i+1}. {h}" for i,h in enumerate(headlines)]) if headlines else "(No headlines available)"
    prompt = f"""
You are a cautious financial market analyst.

{market_context}

HISTORICAL PRICE DATA:
{price_table}

RECENT NEWS HEADLINES:
{news_text}

INSTRUCTIONS:
- Give higher weight to today's market context
- If today's move contradicts historical trend, explain why
- If confidence is low, say "SIDEWAYS"
- Do NOT give financial advice
- Short-term outlook: next 1–5 trading days

OUTPUT FORMAT (MANDATORY):
Direction: one of [UP, DOWN, SIDEWAYS]
Confidence: integer between 0 and 100
Reasoning:
- Bullet 1
- Bullet 2
- Bullet 3

RULES:
- NEVER output "None" or empty values
- If confidence < 60%, Direction MUST be SIDEWAYS
- If information is conflicting, explain why in bullets

"""
    old_prompt = f"""You are a helpful assistant asked to give a short stock trend prediction for a high-school audience.

Here is today market context. 
{market_context}

Here are the recent daily prices for {ticker} (most recent last):
{price_table}

Recent news headlines about {ticker} (if any):
{news_text}

Using only the information above, do the following:
1) Predict whether the stock price will go UP, DOWN, or STAY STABLE tomorrow (choose one of: UP / DOWN / STABLE).
2) Give a confidence level (0-100%). Be honest and conservative.
3) Write a one-paragraph explanation in plain English that a high-schooler can understand.

Output format (strictly):
TREND: <UP / DOWN / STABLE>
CONFIDENCE: <number 0-100>%
EXPLANATION: <one paragraph>
"""
    return prompt

def call_openai(prompt: str, model: str = 'gpt-4', temperature: float = 0.4) -> str:
    """Call OpenAI chat completion. Returns the assistant's text output."""
    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured in the environment. Set environment variable OPENAI_API_KEY.")
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a careful financial assistant that gives concise, conservative answers."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=400
    )
    text = resp['choices'][0]['message']['content'].strip()
    return text

def parse_response(text: str) -> Dict[str,str]:
    """Parse the model's response into fields. This is forgiving but expects labels."""
    out = {'raw': text, 'trend': None, 'confidence': None, 'explanation': None}
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in lines:
        if line.upper().startswith('TREND:'):
            out['trend'] = line.split(':',1)[1].strip()
        elif line.upper().startswith('CONFIDENCE:'):
            out['confidence'] = line.split(':',1)[1].strip()
        elif line.upper().startswith('EXPLANATION:'):
            out['explanation'] = line.split(':',1)[1].strip()
    # If explanation spans multiple lines, assemble the remaining
    if out['explanation'] is None and len(lines) >= 1:
        # fallback: everything after first label-like line
        out['explanation'] = '\n'.join(lines[1:])
    return out
