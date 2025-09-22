#!/usr/bin/env python3
"""
Generate responses from multiple LLMs to ethical dilemmas.
Creates one JSON file per model with all responses in the same format as call_openai_responses.py.
"""

import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv

from llm_providers import OpenRouterProvider, FREE_MODELS


async def generate_responses_for_model(model_name: str, dilemmas: list, limit: int = 3):
    """Generate responses for a single model and save to JSON file"""
    
    print(f"Processing model: {model_name}")
    
    # Create provider
    provider = OpenRouterProvider(model_name)
    
    # Check availability
    if not provider.is_available():
        print(f"  Skipping {model_name}: OPENROUTER_API_KEY not available")
        return False
    
    # Limit dilemmas
    if limit:
        limited_dilemmas = dilemmas[:limit]
    else:
        limited_dilemmas = dilemmas
    print(f"  Generating {len(limited_dilemmas)} responses...")

    # Prepare output file and resume if partial results exist
    safe_filename = provider.get_safe_filename()
    output_path = Path(f"{safe_filename}_responses.json")
    results = []
    start_index = 0

    if output_path.exists():
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
                if isinstance(existing, list):
                    results = existing
                    start_index = len(results)
                    print(f"  Resuming from existing file, {start_index} responses already saved")
        except Exception:
            # If file is corrupt or unreadable, start fresh but back it up
            try:
                backup = output_path.with_suffix('.json.bak')
                output_path.rename(backup)
                print(f"  Backed up corrupt output to {backup}")
            except Exception:
                pass
            results = []
            start_index = 0

    # Process remaining dilemmas
    for idx in range(start_index, len(limited_dilemmas)):
        item = limited_dilemmas[idx]
        try:
            print(f"    {idx+1}/{len(limited_dilemmas)}: {item.get('source', 'Unknown')}")
            llm_decision = await provider.generate_response(item['question'])

            result = {
                "source": item.get('source', 'Unknown'),
                "question": item['question'],
                "llm_decision": llm_decision
            }
            if 'author' in item:
                result["author"] = item['author']

            results.append(result)

            # Persist after each successful query (atomic write)
            tmp_path = output_path.with_suffix('.tmp')
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            tmp_path.replace(output_path)

            print(f"      Saved response {len(results)} to {output_path.name}")

        except Exception as e:
            print(f"      Error: {e}")
            result = {
                "source": item.get('source', 'Unknown'),
                "question": item['question'],
                "llm_decision": {},
                "error": str(e)
            }
            if 'author' in item:
                result["author"] = item['author']
            results.append(result)

            # Persist error result as well to avoid re-querying
            try:
                tmp_path = output_path.with_suffix('.tmp')
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                tmp_path.replace(output_path)
            except Exception:
                pass

    success_count = len([r for r in results if 'error' not in r])
    print(f"  Completed: {success_count}/{len(results)} successful responses")
    print(f"  Saved to: {output_path}")

    return True


async def main():
    # Load environment variables from .env if present
    load_dotenv()
    # Load dilemmas
    input_file = "scifi-ethical-dilemmas-enhanced.json"
    if not Path(input_file).exists():
        print(f"Error: {input_file} not found")
        return
    
    print(f"Loading dilemmas from {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        dilemmas = json.load(f)
    
    print(f"Loaded {len(dilemmas)} dilemmas")
    print(f"Testing {len(FREE_MODELS)} models with 3 dilemmas each")
    print()
    
    # Process each model
    successful_models = 0
    for model in FREE_MODELS:
        try:
            success = await generate_responses_for_model(model, dilemmas, limit=None)
            if success:
                successful_models += 1
        except Exception as e:
            print(f"  Failed to process {model}: {e}")
        
        print()  # Add spacing between models
    
    print(f"Completed: {successful_models}/{len(FREE_MODELS)} models processed successfully")
    
    # List generated files
    response_files = list(Path(".").glob("*_responses.json"))
    if response_files:
        print(f"Generated files:")
        for file in sorted(response_files):
            print(f"  - {file.name}")


if __name__ == "__main__":
    asyncio.run(main())