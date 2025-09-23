# Sci-Fi Ethical Dilemmas Evaluation System

A comprehensive system for evaluating how different Large Language Models (LLMs) respond to ethical dilemmas inspired by classic science fiction literature. This project generates, enhances, and analyzes AI responses to complex moral scenarios drawn from works by Isaac Asimov, Philip K. Dick, Arthur C. Clarke, and other renowned sci-fi authors.

## 🎯 Project Purpose

This system addresses the critical need to understand how AI systems approach ethical decision-making by:

1. **Creating standardized ethical scenarios** from classic sci-fi literature that explore fundamental questions about AI ethics, human-AI interaction, and moral reasoning
2. **Evaluating multiple LLM responses** to these dilemmas using various models (both open and proprietary) 
3. **Providing interactive exploration tools** for researchers, ethicists, and developers to analyze AI moral reasoning patterns
4. **Enabling comparative analysis** of how different AI architectures and training approaches influence ethical decision-making

## 🏗️ System Architecture

```
scifi-evals/
├── data/                           # Organized data storage
│   ├── source/                     # Original and formatted dilemma datasets
│   │   ├── original_dilemmas.json  # Raw ethical dilemmas from sci-fi sources
│   │   └── formatted_dilemmas.json # Processed and standardized format
│   ├── enhanced/                   # AI-enhanced versions
│   │   ├── enhanced_dilemmas.json  # GPT-4 enhanced dilemmas with richer context
│   │   └── dilemmas_with_gpt4_decisions.json # Enhanced dilemmas + GPT-4 responses
│   ├── responses/                  # LLM evaluation results
│   │   ├── gpt-5-nano_responses.json        # OpenAI GPT-5 Nano responses
│   │   ├── grok-4-fast_responses.json       # xAI Grok 4 Fast responses
│   │   ├── gemma-3-27b_responses.json       # Google Gemma 3 27B responses
│   │   ├── nemotron-nano-9b_responses.json  # NVIDIA Nemotron Nano responses
│   │   ├── deepseek-chat-v3.1_responses.json # DeepSeek Chat responses
│   │   └── kimi-k2_responses.json           # Moonshot AI Kimi responses
│   └── web/                        # Web interface data (subset for demo)
│       ├── scifi-ethical-dilemmas_short.json
│       └── scifi-ethical-dilemmas-with-decisions_short.json
├── scripts/                        # Processing pipeline
│   ├── transform_sources.py        # Convert raw data to standardized format
│   ├── enhance_dilemmas.py         # Use GPT-4 to add context and stakes
│   ├── generate_llm_responses.py   # Generate responses from multiple LLMs
│   ├── call_openai_responses.py    # Generate OpenAI-specific responses
│   └── llm_providers.py           # LLM provider abstractions and utilities
├── docs/                          # Interactive web interface
│   ├── index.html                 # Web app for exploring dilemmas/responses
│   ├── app.js                     # Frontend application logic
│   └── styles.css                 # Styling for web interface
├── pyproject.toml                 # Python dependencies and project config
└── uv.lock                       # Locked dependency versions
```

## 📈 System Evolution

### Phase 1: Foundation (Initial Commits)
- **Objective**: Establish basic project structure and data pipeline
- **Created**: Original dilemma collection from sci-fi literature (~550 scenarios)
- **Built**: Web interface for browsing and exploring dilemmas
- **Implemented**: Basic data transformation pipeline

### Phase 2: Enhancement & Enrichment 
- **Objective**: Improve dilemma quality and add AI-driven context
- **Enhanced**: Used GPT-4 to add gravity, stakes, and compelling details to scenarios
- **Developed**: Sophisticated prompt engineering for consistent ethical analysis format
- **Created**: Enhanced dataset with richer contexts and more challenging scenarios

### Phase 3: Multi-LLM Evaluation
- **Objective**: Evaluate diverse AI models on ethical reasoning tasks  
- **Implemented**: OpenRouter integration for accessing multiple free LLM APIs
- **Built**: Robust error handling, rate limiting, and retry logic for API calls
- **Added**: Support for structured JSON outputs with standardized response format

### Phase 4: Production & Analysis (Current)
- **Objective**: Generate comprehensive evaluation dataset
- **Generated**: Responses from 6+ different LLM architectures (OpenAI, Anthropic, Google, NVIDIA, etc.)
- **Implemented**: Resume functionality for handling long-running evaluation jobs
- **Created**: Standardized response format for comparative analysis

## 🚀 Getting Started

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager (recommended) or pip
- API keys for LLM providers (see Configuration section)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd scifi-evals

# Install dependencies using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### Configuration

Create a `.env` file in the project root:

```bash
# For OpenAI models (GPT-4, GPT-5)
OPENAI_API_KEY=your_openai_key

# For free models via OpenRouter
OPENROUTER_API_KEY=your_openrouter_key

# Optional: Rate limiting configuration
OPENROUTER_RATE_LIMIT_RPM=20
OPENROUTER_MAX_RETRIES=3
```

## 🔧 Usage

### 1. Data Processing Pipeline

Process source data through the full pipeline:

```bash
# Transform raw dilemmas to standardized format
python scripts/transform_sources.py

# Enhance dilemmas with GPT-4 (adds context and stakes)
python scripts/enhance_dilemmas.py

# Generate responses from multiple free LLMs
python scripts/generate_llm_responses.py

# Generate responses from OpenAI models
python scripts/call_openai_responses.py
```

### 2. Web Interface

Launch the interactive web interface to explore dilemmas and responses:

```bash
# Serve the docs directory (any HTTP server works)
python -m http.server 8000 --directory docs

# Navigate to http://localhost:8000
```

The web interface allows you to:
- Browse all ethical dilemmas by source/author
- Filter and search scenarios
- View detailed LLM responses and reasoning
- Download datasets in JSON format

### 3. Working with the Data

Each dilemma follows this standardized format:

```json
{
  "source": "Story Title",
  "author": "Author Name", 
  "question": "Enhanced ethical dilemma with context...",
  "llm_decision": {
    "decision": "Primary decision or recommendation",
    "considerations": {
      "in_favor": ["Argument 1", "Argument 2"],
      "against": ["Counter-argument 1", "Counter-argument 2"]
    },
    "reasoning": "Detailed explanation of ethical reasoning"
  }
}
```

## 📊 Evaluation Results

The system has successfully generated responses from multiple LLM families:

- **OpenAI**: GPT-5 Nano (latest model)
- **xAI**: Grok 4 Fast 
- **Google**: Gemma 3 27B
- **NVIDIA**: Nemotron Nano 9B
- **DeepSeek**: Chat v3.1
- **Moonshot AI**: Kimi K2

Each model's responses are stored in structured JSON format enabling:
- Comparative analysis across different AI architectures
- Research into ethical reasoning patterns
- Evaluation of consistency in moral decision-making
- Assessment of how model size/training affects ethical reasoning

## 🤝 Contributing

This project provides a foundation for AI ethics research. Potential contributions include:

- Adding new ethical dilemma sources (more sci-fi authors, contemporary scenarios)
- Implementing additional LLM providers
- Building analysis tools for comparing responses
- Improving the web interface with visualization capabilities
- Adding evaluation metrics for ethical reasoning quality

## 📝 Research Applications

This dataset and evaluation system supports research in:

- **AI Safety**: Understanding how different models approach harmful/beneficial decisions
- **Ethics in AI**: Comparative analysis of moral reasoning across AI architectures  
- **Bias Detection**: Identifying systematic biases in ethical decision-making
- **Alignment Research**: Evaluating how well AI systems align with human moral intuitions
- **Philosophy of Mind**: Exploring questions about AI consciousness and moral agency

## 📄 License

[Specify your license here]

## 🙏 Acknowledgments

- Ethical dilemmas inspired by classic science fiction literature
- OpenRouter for providing access to diverse LLM APIs
- The AI research community for ongoing work in AI safety and ethics

---

*This system demonstrates that science fiction literature provides an invaluable corpus for evaluating AI ethical reasoning - these stories have been exploring AI ethics questions for decades before they became practical concerns.*
