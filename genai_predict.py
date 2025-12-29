import os
import openai
from typing import List, Dict
from config import OPENAI_API_KEY, NEWS_API_KEY

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# -----------------------------
# Prompt Builder
# -----------------------------
def build_prompt(
    ticker: str,
    price_table: str,
    headlines: List[str],
    market_context: str,
    prediction_days: int = 7
) -> str:
    """Create a prompt for the LLM to predict multiple days."""
    news_text = "\n".join([f"{i+1}. {h}" for i,h in enumerate(headlines)]) if headlines else "(No headlines available)"
    prompt = f"""
You are a careful financial market analyst.

{market_context}

HISTORICAL PRICE DATA:
{price_table}

RECENT NEWS HEADLINES:
{news_text}

INSTRUCTIONS:
- Predict the short-term price direction for the next {prediction_days} trading days.
- For each day, provide:
  Day N: Direction: UP / DOWN / SIDEWAYS, Confidence: 0-100%, Reasoning (1-3 bullets)
- Give higher weight to today's market context.
- If today's move contradicts historical trend, explain why.
- If confidence < 60%, Direction MUST be SIDEWAYS.
- Do NOT give financial advice.
- Use plain English suitable for a high-schooler.

OUTPUT FORMAT (MANDATORY):
Day 1: Direction: <UP/DOWN/SIDEWAYS>, Confidence: <number 0-100>, Reasoning:
- Bullet 1
- Bullet 2
- Bullet 3

Repeat for each day up to Day {prediction_days}.
"""
    return prompt

# -----------------------------
# Call OpenAI
# -----------------------------
def call_openai(prompt: str, model: str = 'gpt-4', temperature: float = 0.4) -> str:
    """Call OpenAI ChatCompletion API and return response text."""
    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured in the environment.")
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a careful financial assistant that gives concise, conservative answers."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=600
    )
    text = resp['choices'][0]['message']['content'].strip()
    return text

# -----------------------------
# Parse multi-day response
# -----------------------------
def parse_response(text: str, days: int = 7) -> Dict[str, Dict]:
    """
    Parse LLM output for multiple days.
    Returns: dict like {'Day 1': {'trend':..., 'confidence':..., 'reasoning':[...]} ...}
    """
    out = {}
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    current_day = None
    reasoning = []

    for line in lines:
        # Detect Day N
        if line.upper().startswith("DAY"):
            if current_day:
                out[current_day] = {"trend": trend, "confidence": confidence, "reasoning": reasoning}
            current_day = line.split(":",1)[0].strip()
            trend = "SIDEWAYS"
            confidence = 50
            reasoning = []

            # Try to extract trend and confidence from same line
            if "Direction:" in line and "Confidence:" in line:
                try:
                    trend_part = line.split("Direction:")[1].split(",")[0].strip()
                    conf_part = line.split("Confidence:")[1].split(",")[0].strip()
                    trend = trend_part
                    confidence = int(''.join(filter(str.isdigit, conf_part)))
                except:
                    pass
        elif line.startswith("-") and current_day:
            reasoning.append(line.lstrip("- ").strip())

    # Save last day
    if current_day:
        out[current_day] = {"trend": trend, "confidence": confidence, "reasoning": reasoning}

    # Fill missing days with default
    for d in range(1, days+1):
        day_key = f"Day {d}"
        if day_key not in out:
            out[day_key] = {"trend": "SIDEWAYS", "confidence": 50, "reasoning": ["Market unclear"]}

    return out
