# Context Windows in Practice - Lab Experiments

LLM Context Window Empirical Research for AI Developer Expert Course.

## Overview

This project implements **4 rigorous experiments** to study LLM context window behaviors, producing professional visualizations, detailed analysis reports, and actionable prompt engineering rules.

| # | Experiment | Description | Duration |
|---|------------|-------------|----------|
| 1 | **Needle in Haystack** | Test accuracy by fact position (start/middle/end) | ~15 min |
| 2 | **Context Window Size Impact** | Measure accuracy/latency vs document count | ~20 min |
| 3 | **RAG Impact** | Compare Full Context vs RAG retrieval | ~25 min |
| 4 | **Context Engineering Strategies** | Test Select, Compress, Write strategies | ~30 min |

## Key Features

- **Multi-Model Support**: Run experiments with Claude, Gemini, Ollama, or mock backend
- **Professional Visualizations**: Publication-quality graphs with seaborn/matplotlib
- **Detailed Explanations**: Comprehensive analysis of each experiment's findings
- **Prompt Engineering Rules**: Actionable rules derived from empirical results
- **Statistical Analysis**: Confidence intervals, comparisons, threshold detection

## Key Phenomena Studied

1. **Lost in the Middle** - Information in the middle of context is harder to retrieve
2. **Context Accumulation Problem** - Accuracy degrades as context grows

## Setup

### Prerequisites

- Python 3.10+
- UV package manager (recommended)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd L22

# Create virtual environment with UV
uv venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows

# Install dependencies
uv pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your API keys if using API backends
```

### API Keys Setup

For Claude backend (Claude Code environment):
```bash
export ANTHROPIC_API_KEY="your-api-key"
```

For Gemini backend (Gemini CLI environment):
```bash
export GOOGLE_API_KEY="your-api-key"
```

### Using Ollama (Local LLM)

For local LLM inference:

```bash
# Install Ollama from https://ollama.ai
ollama pull llama2
```

## Usage

### Run All Experiments

```bash
python main.py
```

### Run Specific Experiment

```bash
python main.py --experiment 1  # Run only Experiment 1
python main.py --experiment 2  # Run only Experiment 2
python main.py --experiment 3  # Run only Experiment 3
python main.py --experiment 4  # Run only Experiment 4
```

### Choose Backend

```bash
python main.py --backend mock    # Use mock model (default, for testing)
python main.py --backend ollama  # Use local Ollama model
python main.py --backend claude  # Use Claude API (requires ANTHROPIC_API_KEY)
python main.py --backend gemini  # Use Gemini API (requires GOOGLE_API_KEY)
```

### Verbose Output

```bash
python main.py --verbose  # Enable detailed logging
```

### Run Tests

```bash
pytest tests/ -v
```

## Project Structure

```
L22/
├── main.py                    # Entry point
├── requirements.txt           # Dependencies
├── .env.example              # Environment template
├── .gitignore                # Git ignore rules
│
├── src/                      # Source code
│   ├── experiments/          # 4 experiment implementations
│   │   ├── exp1_needle_in_haystack.py
│   │   ├── exp2_context_size.py
│   │   ├── exp3_rag_impact.py
│   │   └── exp4_strategies.py
│   ├── generators/           # Test data generators
│   ├── models/               # LLM interfaces (Claude, Gemini, Ollama)
│   ├── rag/                  # RAG components (ChromaDB)
│   ├── analysis/             # Statistics, visualization, explanations
│   └── utils/                # Config, logging, helpers
│
├── results/                  # Output files
│   ├── experiments/          # JSON results + full_report.txt
│   └── graphs/              # PNG visualizations
│
├── tests/                   # Unit tests
└── docs/                    # Documentation
    └── requirements/        # PRD and tasks
```

## Supported Models

| Backend | Model | Environment | API Key |
|---------|-------|-------------|---------|
| `mock` | Mock Model | Any | None |
| `ollama` | Llama 2 (default) | Local | None |
| `claude` | Claude 3 Haiku | Claude Code | ANTHROPIC_API_KEY |
| `gemini` | Gemini 1.5 Flash | Gemini CLI | GOOGLE_API_KEY |

## Expected Results

### Experiment 1: Needle in Haystack
- **Start position**: ~95% accuracy
- **Middle position**: ~60% accuracy (Lost in the Middle effect)
- **End position**: ~95% accuracy

### Experiment 2: Context Size Impact
- Accuracy decreases linearly with document count
- Latency increases with context size
- Optimal context: 5-10 documents

### Experiment 3: RAG Impact
- **RAG**: Higher accuracy, lower latency
- **Full Context**: Lower accuracy, higher latency
- RAG wins at >5 documents

### Experiment 4: Strategies
- **Select**: Best for document Q&A
- **Compress**: Best for long conversations
- **Write**: Best for multi-step agents

## Output Files

After running experiments:

- `results/experiments/exp1_*.json` - Raw experiment data
- `results/experiments/full_report.txt` - Detailed analysis report
- `results/graphs/exp1_accuracy_by_position.png` - Position accuracy chart
- `results/graphs/exp2_accuracy_vs_size.png` - Size impact chart
- `results/graphs/exp3_performance_comparison.png` - RAG comparison
- `results/graphs/exp4_strategy_performance.png` - Strategy comparison

## Prompt Engineering Rules (Derived from Experiments)

Based on empirical findings, these rules optimize LLM interactions:

### Rule 1: Position Critical Information Strategically
**Source**: Experiment 1 (Needle in Haystack)
- Place key facts at START or END of context
- Avoid burying important information in the middle
- Use explicit markers for critical sections

### Rule 2: Limit Context Size for Accuracy
**Source**: Experiment 2 (Context Size Impact)
- More context ≠ better results
- Filter to relevant documents only
- Monitor token usage vs accuracy tradeoff

### Rule 3: Use RAG for Large Document Sets
**Source**: Experiment 3 (RAG Impact)
- Implement vector search for >5 documents
- Pre-filter before adding to prompt
- Balance retrieval depth vs accuracy

### Rule 4: Match Strategy to Task Type
**Source**: Experiment 4 (Strategies)
- Q&A tasks: Use SELECT strategy
- Summaries: Use COMPRESS strategy
- Agents: Use WRITE strategy

### Rule 5: Monitor and Optimize Token Usage
**Source**: All experiments
- Track tokens per request
- Set appropriate max_tokens limits
- Consider cost vs quality tradeoffs

## Code Files Summary

| File | Lines | Description |
|------|-------|-------------|
| `main.py` | ~150 | Main entry point |
| `src/experiments/exp1_*.py` | ~95 | Experiment 1 |
| `src/experiments/exp2_*.py` | ~100 | Experiment 2 |
| `src/experiments/exp3_*.py` | ~115 | Experiment 3 |
| `src/experiments/exp4_*.py` | ~130 | Experiment 4 |
| `src/models/claude_model.py` | ~95 | Claude API interface |
| `src/models/gemini_model.py` | ~95 | Gemini API interface |
| `src/generators/document_generator.py` | ~140 | Document generation |
| `src/rag/vector_store.py` | ~100 | ChromaDB integration |
| `src/analysis/statistics.py` | ~110 | Statistical functions |
| `src/analysis/visualizations.py` | ~130 | Graph generation |
| `src/analysis/explanations.py` | ~150 | Detailed explanations |

All files are under 150 lines as per guidelines.

## License

This project is part of the AI Developer Expert Course.

## Author

Context Windows Lab - December 2025
