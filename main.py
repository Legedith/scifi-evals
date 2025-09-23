def main():
    print("Sci-Fi Ethical Dilemmas Evaluation System")
    print("=========================================")
    print()
    print("This system evaluates how different LLMs respond to ethical dilemmas")
    print("inspired by classic science fiction literature.")
    print()
    print("Available commands:")
    print("  python scripts/transform_sources.py     - Transform raw dilemmas to standardized format")
    print("  python scripts/enhance_dilemmas.py      - Enhance dilemmas with GPT-4")
    print("  python scripts/generate_llm_responses.py - Generate responses from multiple LLMs")
    print("  python scripts/call_openai_responses.py - Generate OpenAI model responses")
    print()
    print("To explore results interactively:")
    print("  python -m http.server 8000 --directory docs")
    print("  Then visit: http://localhost:8000")
    print()
    print("See README.md for detailed usage instructions.")


if __name__ == "__main__":
    main()
