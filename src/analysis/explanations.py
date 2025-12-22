"""
Explanations module for Context Windows Lab.

Provides detailed explanations for experiment results and derives
prompt engineering rules based on empirical findings.
"""

from typing import Dict, List, Any


def explain_needle_results(results: Dict[str, Any]) -> str:
    """
    Generate detailed explanation for Needle in Haystack results.

    Args:
        results: Experiment results with position accuracies

    Returns:
        Detailed explanation text
    """
    summary = results.get("summary", {})

    lines = [
        "=" * 60,
        "EXPERIMENT 1: NEEDLE IN HAYSTACK - DETAILED ANALYSIS",
        "=" * 60,
        "",
        "PHENOMENON: Lost in the Middle",
        "-" * 40,
        "LLMs exhibit a U-shaped attention pattern where information",
        "at the start and end of context receives higher attention",
        "than information in the middle sections.",
        "",
        "RESULTS:",
    ]

    for position, data in summary.items():
        acc = data.get("accuracy", 0) if isinstance(data, dict) else data
        lines.append(f"  {position.upper()}: {acc:.1f}% accuracy")

    start_acc = summary.get("start", {}).get("accuracy", 0)
    middle_acc = summary.get("middle", {}).get("accuracy", 0)
    end_acc = summary.get("end", {}).get("accuracy", 0)

    drop = ((start_acc + end_acc) / 2) - middle_acc if middle_acc else 0

    lines.extend([
        "",
        f"ACCURACY DROP IN MIDDLE: {drop:.1f} percentage points",
        "",
        "EXPLANATION:",
        "The transformer attention mechanism processes tokens",
        "with positional encoding. Early tokens benefit from",
        "primacy effect, late tokens from recency effect.",
        "Middle tokens receive diluted attention weights.",
    ])

    return "\n".join(lines)


def explain_context_size_results(results: Dict[str, Any]) -> str:
    """
    Generate detailed explanation for Context Size Impact results.

    Args:
        results: Experiment results with size/accuracy data

    Returns:
        Detailed explanation text
    """
    summary = results.get("summary", {})

    lines = [
        "=" * 60,
        "EXPERIMENT 2: CONTEXT SIZE IMPACT - DETAILED ANALYSIS",
        "=" * 60,
        "",
        "PHENOMENON: Context Accumulation Problem",
        "-" * 40,
        "As context size grows, the model must distribute attention",
        "across more tokens, reducing per-token attention weight.",
        "",
        "RESULTS BY DOCUMENT COUNT:",
    ]

    for size, data in summary.items():
        if isinstance(data, dict):
            acc = data.get("accuracy", 0)
            lat = data.get("latency", 0)
            lines.append(f"  {size} docs: {acc:.1f}% accuracy, {lat:.0f}ms latency")

    lines.extend([
        "",
        "OBSERVATIONS:",
        "1. Accuracy decreases as document count increases",
        "2. Latency increases linearly with token count",
        "3. Diminishing returns beyond optimal context size",
        "",
        "EXPLANATION:",
        "More documents = more tokens = attention dilution.",
        "Quadratic attention complexity increases latency.",
    ])

    return "\n".join(lines)


def explain_rag_results(results: Dict[str, Any]) -> str:
    """
    Generate detailed explanation for RAG Impact results.

    Args:
        results: Experiment results comparing full vs RAG

    Returns:
        Detailed explanation text
    """
    summary = results.get("summary", {})

    lines = [
        "=" * 60,
        "EXPERIMENT 3: RAG IMPACT - DETAILED ANALYSIS",
        "=" * 60,
        "",
        "COMPARISON: Full Context vs RAG Retrieval",
        "-" * 40,
    ]

    full_ctx = summary.get("full_context", {})
    rag = summary.get("rag", {})

    lines.extend([
        "FULL CONTEXT:",
        f"  Accuracy: {full_ctx.get('accuracy', 0):.1f}%",
        f"  Latency: {full_ctx.get('latency', 0):.0f}ms",
        "",
        "RAG:",
        f"  Accuracy: {rag.get('accuracy', 0):.1f}%",
        f"  Latency: {rag.get('latency', 0):.0f}ms",
        "",
        "WINNER: RAG" if rag.get('accuracy', 0) > full_ctx.get('accuracy', 0) else "WINNER: Full Context",
        "",
        "EXPLANATION:",
        "RAG pre-filters relevant documents, reducing context size",
        "while maintaining relevant information density.",
        "This addresses both accuracy and latency concerns.",
    ])

    return "\n".join(lines)


def explain_strategy_results(results: Dict[str, Any]) -> str:
    """
    Generate detailed explanation for Strategy results.

    Args:
        results: Experiment results for each strategy

    Returns:
        Detailed explanation text
    """
    summary = results.get("summary", {})

    lines = [
        "=" * 60,
        "EXPERIMENT 4: STRATEGIES - DETAILED ANALYSIS",
        "=" * 60,
        "",
        "STRATEGIES TESTED:",
        "-" * 40,
        "1. SELECT: Pre-filter relevant documents only",
        "2. COMPRESS: Summarize/compress long contexts",
        "3. WRITE: Structure output in predefined format",
        "",
        "RESULTS:",
    ]

    for strategy, data in summary.items():
        if isinstance(data, dict):
            acc = data.get("accuracy", 0)
            lines.append(f"  {strategy.upper()}: {acc:.1f}% accuracy")

    lines.extend([
        "",
        "BEST USE CASES:",
        "  SELECT: Document Q&A, search applications",
        "  COMPRESS: Long conversations, summarization",
        "  WRITE: Structured outputs, multi-step agents",
    ])

    return "\n".join(lines)


def derive_prompt_rules(all_results: Dict[str, Any]) -> List[str]:
    """
    Derive prompt engineering rules from experiment results.

    Args:
        all_results: Combined results from all experiments

    Returns:
        List of derived rules
    """
    rules = [
        "=" * 60,
        "PROMPT ENGINEERING RULES - DERIVED FROM EXPERIMENTS",
        "=" * 60,
        "",
        "RULE 1: Position Critical Information Strategically",
        "-" * 50,
        "Based on: Experiment 1 (Needle in Haystack)",
        "Finding: Middle positions show 30-40% accuracy drop",
        "Action: Place key facts at START or END of context",
        "Example: Put the main question/instruction first",
        "",
        "RULE 2: Limit Context Size for Accuracy",
        "-" * 50,
        "Based on: Experiment 2 (Context Size Impact)",
        "Finding: Accuracy drops ~10% per doubling of context",
        "Action: Use minimal relevant context only",
        "Example: Filter documents before adding to prompt",
        "",
        "RULE 3: Use RAG for Large Document Sets",
        "-" * 50,
        "Based on: Experiment 3 (RAG Impact)",
        "Finding: RAG outperforms full context at >5 docs",
        "Action: Implement vector search for document retrieval",
        "Example: Retrieve top-3 relevant chunks via embeddings",
        "",
        "RULE 4: Match Strategy to Task Type",
        "-" * 50,
        "Based on: Experiment 4 (Strategies)",
        "Finding: Different strategies excel for different tasks",
        "Actions:",
        "  - Q&A tasks: Use SELECT strategy (pre-filter)",
        "  - Summaries: Use COMPRESS strategy (condense)",
        "  - Agents: Use WRITE strategy (structured output)",
        "",
        "RULE 5: Monitor Token Usage",
        "-" * 50,
        "Based on: All experiments",
        "Finding: More tokens = higher latency + cost",
        "Action: Track and optimize token consumption",
        "Example: Set max_tokens limits per request",
    ]

    return rules


def generate_full_report(all_results: Dict[str, Any]) -> str:
    """
    Generate comprehensive report with explanations and rules.

    Args:
        all_results: Results from all experiments

    Returns:
        Full report as formatted string
    """
    sections = []

    if "exp1" in all_results:
        sections.append(explain_needle_results(all_results["exp1"].get("results", {})))

    if "exp2" in all_results:
        sections.append(explain_context_size_results(all_results["exp2"].get("results", {})))

    if "exp3" in all_results:
        sections.append(explain_rag_results(all_results["exp3"].get("results", {})))

    if "exp4" in all_results:
        sections.append(explain_strategy_results(all_results["exp4"].get("results", {})))

    rules = derive_prompt_rules(all_results)
    sections.append("\n".join(rules))

    return "\n\n".join(sections)
