"""
Central configuration for Context Windows Lab experiments.

All experiment parameters are centralized here for easy modification.
"""

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# ==============================================
# Directory Paths
# ==============================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
RESULTS_DIR = PROJECT_ROOT / "results"
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"
GRAPHS_DIR = RESULTS_DIR / "graphs"
LOGS_DIR = PROJECT_ROOT / "logs"


# ==============================================
# Experiment 1: Needle in Haystack
# ==============================================

EXP1_NUM_DOCS = 5
EXP1_WORDS_PER_DOC = 2000  # Increased for Lost in Middle effect
EXP1_POSITIONS: List[str] = ["start", "middle", "end"]


# ==============================================
# Experiment 2: Context Window Size Impact
# ==============================================

EXP2_DOCUMENT_COUNTS: List[int] = [2, 5, 10, 20, 50]
EXP2_WORDS_PER_DOC = 500  # Words per document


# ==============================================
# Experiment 3: RAG Impact
# ==============================================

EXP3_NUM_DOCS = 20
EXP3_TOPICS: List[str] = ["technology", "law", "medicine"]
EXP3_CHUNK_SIZE = 500
EXP3_RAG_TOP_K = 3


# ==============================================
# Experiment 4: Context Engineering Strategies
# ==============================================

EXP4_NUM_ACTIONS = 10
EXP4_STRATEGIES: List[str] = ["select", "compress", "write"]
EXP4_MAX_TOKENS = 4000


# ==============================================
# General Experiment Settings
# ==============================================

TRIALS_PER_CONDITION = int(os.getenv("TRIALS_PER_CONDITION", "10"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


# ==============================================
# LLM Configuration
# ==============================================

# Ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")


# ==============================================
# Visualization Settings
# ==============================================

GRAPH_DPI = 300
GRAPH_FORMAT = "png"
GRAPH_STYLE = "seaborn-v0_8-whitegrid"


def ensure_directories() -> None:
    """Create all required directories if they don't exist."""
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
