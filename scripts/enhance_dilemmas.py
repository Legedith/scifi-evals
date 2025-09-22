import json
import os
from pathlib import Path
from typing import List

import openai
from dotenv import load_dotenv
from pydantic import BaseModel


class EnhancedDilemma(BaseModel):
    enhanced_question: str


def enhance_dilemma(client: openai.OpenAI, original_question: str, source: str, model: str = "gpt-4o") -> EnhancedDilemma:
    """Enhance a dilemma to better convey gravity and consequences."""
    
    system_msg = """You are an expert in ethical philosophy and moral dilemmas. Your task is to enhance ethical dilemmas to make them more compelling and genuinely difficult by:

1. Adding gravity, weight, and stakes to the situation with specific details
2. Making clear that any course of action will have serious consequences 
3. Including concrete numbers, timeframes, and affected parties to make impacts tangible
4. Subtly conveying the complexity without explicitly listing pros/cons
5. Ensuring there is no easy way out or obviously correct choice

The enhanced question should feel like a true moral weight where any decision path has serious ramifications. Include specific details about the scope of impact, urgency, and what's at stake. Make the reader feel the weight of the decision without making it obvious which path to choose.

Do not explicitly structure the response as "Option A vs Option B" or list positive/negative outcomes directly. Instead, weave the complexity naturally into the narrative of the dilemma."""

    user_prompt = f"""Original dilemma from "{source}":
{original_question}

Please enhance this dilemma by enriching the context and stakes to make it more compelling and genuinely difficult. Add specific details about what's at stake, who will be affected, and the scope of consequences. Make the gravity of the situation clear while keeping the moral complexity intact. The enhanced question should make any decision feel weighty and consequential."""

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_prompt},
    ]

    response = client.responses.parse(
        model=model,
        input=messages,
        text_format=EnhancedDilemma
    )

    return response.output_parsed


def main():
    root = Path(__file__).resolve().parents[1]
    in_path = root / "scifi-ethical-dilemmas.formatted.json"
    out_path = root / "scifi-ethical-dilemmas-enhanced.json"

    # Prefer the project's .env over any global environment variable.
    # If a key exists in the project's .env, it will override the existing env var.
    # This ensures the script uses the repo-local API key when present.
    existing_key = os.environ.get("OPENAI_API_KEY")
    load_dotenv(dotenv_path=root / ".env", override=True)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        # Fall back message remains the same
        raise SystemExit("set OPENAI_API_KEY environment variable")
    # Notify which source provided the key (without printing the key itself)
    if existing_key and api_key != existing_key:
        print("Using OPENAI_API_KEY from project .env (overrode global env var)")
    elif existing_key and api_key == existing_key:
        print("Using existing OPENAI_API_KEY from environment (no project .env override)")
    elif not existing_key:
        print("Using OPENAI_API_KEY from project .env")

    # Initialize official OpenAI client
    client = openai.OpenAI(api_key=api_key)

    data = json.loads(in_path.read_text(encoding="utf-8"))
    
    # For testing, limit to first 3 dilemmas
    # Remove this line to process all dilemmas
    # data = data[:3]
    
    results = []
    for i, item in enumerate(data):
        print(f"Processing dilemma {i+1}/{len(data)}: {item.get('source', 'Unknown')}")
        
        try:
            enhanced = enhance_dilemma(
                client, 
                item.get("question", ""), 
                item.get("source", "Unknown")
            )
            
            # Replace the original question with the enhanced question so the
            # output format matches the input schema (no new keys added).
            result = {
                **item,  # Keep original fields
                "question": enhanced.enhanced_question
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error processing dilemma {i+1}: {e}")
            # Keep original dilemma if enhancement fails
            results.append(item)

    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Enhanced dilemmas written to: {out_path}")


if __name__ == "__main__":
    main()
