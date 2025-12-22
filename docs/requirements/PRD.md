# PRD.md - Product Requirements Document
# Context Windows in Practice - Lab Experiments

**Project Name:** LLM Context Window Empirical Research
**Course:** AI Developer Expert - Context Windows Lab (L22)
**Author:** Hadar Wayn
**Version:** 3.0
**Date:** December 2025
**Status:** Phase 1 - Awaiting Approval

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Project Overview](#2-project-overview)
3. [Background & Motivation](#3-background--motivation)
4. [Target Users & Applications](#4-target-users--applications)
5. [Research Goals & Hypotheses](#5-research-goals--hypotheses)
6. [Experimental Design](#6-experimental-design)
7. [Technical Requirements](#7-technical-requirements)
8. [Functional Requirements](#8-functional-requirements)
9. [Metrics & Measurement](#9-metrics--measurement)
10. [Visualization Requirements](#10-visualization-requirements)
11. [Success Criteria](#11-success-criteria)
12. [Constraints & Assumptions](#12-constraints--assumptions)
13. [Security Considerations](#13-security-considerations)
14. [Deliverables](#14-deliverables)
15. [Learning Objectives](#15-learning-objectives)
16. [Glossary](#16-glossary)

---

## 1. Executive Summary

### 1.1 One-Line Description

A controlled empirical research project that validates LLM context window behaviors through four experiments, testing position-based retrieval, context size impact, RAG effectiveness, and context management strategies.

### 1.2 The Problem (Explained Simply)

Imagine you're having a conversation with someone who can only remember a limited amount of what you said. If you tell them something important in the middle of a long story, they might forget it. Now imagine that "someone" is an AI that costs money every time it forgets and makes mistakes.

**Large Language Models (LLMs)** have a similar limitation called the **"context window"** - a limited memory space where they hold all the text from a conversation. When this space fills up or contains too much information, the AI starts making mistakes in predictable ways.

This project scientifically proves *when* and *why* these mistakes happen, and provides practical rules to prevent them.

### 1.3 Why This Matters

| Problem | Business Impact | Our Solution |
|---------|-----------------|--------------|
| AI forgets middle content | Wrong answers, lost productivity | Evidence-based positioning rules |
| More documents = lower accuracy | Unreliable outputs at scale | Optimal document count guidelines |
| Irrelevant content wastes resources | Higher costs, slower responses | RAG-based retrieval strategies |
| No context management | Degraded long-term performance | Select/Compress/Write strategies |

### 1.4 Expected Outcomes

1. **Quantified accuracy by position** - Start vs Middle vs End retrieval rates
2. **Context size impact curves** - Accuracy/latency vs document count
3. **RAG vs Full Context comparison** - Performance benchmarks
4. **Strategy effectiveness matrix** - Select, Compress, Write comparison
5. **Visual proof** via research graphs
6. **Practical rulebook** for prompt engineering

### 1.5 Four Experiments (As Per Requirements)

| Experiment | Topic | Duration | Difficulty |
|------------|-------|----------|------------|
| 1 | Needle in Haystack (Lost in the Middle) | ~15 min | Basic |
| 2 | Context Window Size Impact | ~20 min | Medium |
| 3 | RAG Impact | ~25 min | Medium+ |
| 4 | Context Engineering Strategies | ~30 min | Advanced |

---

## 2. Project Overview

### 2.1 What We're Building

A research framework that:

1. **Generates synthetic test data** with embedded facts at different positions
2. **Runs controlled experiments** on LLM models
3. **Measures accuracy and latency** across different context configurations
4. **Compares retrieval strategies** (Full Context vs RAG)
5. **Tests mitigation strategies** (Select, Compress, Write)
6. **Produces visual analysis** with statistical validation
7. **Derives practical rules** for production prompt engineering

### 2.2 What We're NOT Building

- A production application or API
- A fine-tuned model
- A general-purpose testing framework
- Real-time monitoring system

### 2.3 Project Scope

```
+---------------------------------------------------------+
|                    IN SCOPE                              |
+---------------------------------------------------------+
|  * 4 Experiments as specified in requirements            |
|  * Experiment 1: Needle in Haystack (Lost in Middle)    |
|  * Experiment 2: Context Window Size Impact              |
|  * Experiment 3: RAG vs Full Context                     |
|  * Experiment 4: Context Engineering Strategies          |
|  * Statistical analysis with visualizations              |
|  * Evidence-based prompt engineering rulebook            |
+---------------------------------------------------------+

+---------------------------------------------------------+
|                   OUT OF SCOPE                           |
+---------------------------------------------------------+
|  * Model fine-tuning or training                         |
|  * Real-time production deployment                       |
|  * Multi-modal (image/audio) testing                     |
|  * Automated prompt optimization                         |
+---------------------------------------------------------+
```

---

## 3. Background & Motivation

### 3.1 The Context Window Explained (For a 15-Year-Old)

Think of the context window like a whiteboard in a classroom:

- **Fixed size**: You can only write so much before running out of space
- **Everything visible**: The AI can "see" everything on the whiteboard at once
- **No long-term memory**: When a new conversation starts, the whiteboard is erased
- **Costs money**: Bigger whiteboards (larger context windows) cost more

### 3.2 The Two Key Phenomena

#### 3.2.1 Lost in the Middle

**What it is:** Information in the middle of a long context is harder for the AI to find and use.

**Real-world analogy:** Trying to remember what happened in the middle of a 3-hour movie - you remember the beginning and the end better.

**Research shows a "U-shaped" attention pattern:**

```
Attention Level
    |
100%| *                                             *
    |   *                                         *
    |     *                                     *
 50%|       *                                 *
    |         *       *   *   *   *       *
    |           * * *               * * *
    +----------------------------------------------
         Start          Middle            End
                   Position in Context
```

#### 3.2.2 Context Accumulation Problem

**What it is:** As more data accumulates in the context window (especially in multi-step agents), accuracy degrades.

**Real-world analogy:** Trying to find one specific email in an inbox with 10,000 unread messages - the more clutter, the harder the search.

### 3.3 The Four Mitigation Strategies

| Strategy | How It Works | When to Use |
|----------|-------------|-------------|
| **Write** | External memory/scratchpad for key facts | Long conversations |
| **Select** | Use RAG to only include relevant documents | Large document sets |
| **Compress** | Summarize history periodically | Multi-step agents |
| **Isolate** | Separate contexts for different tasks | Multi-agent systems |

---

## 4. Target Users & Applications

### 4.1 Who Will Benefit

| User Type | How They'll Use This Research |
|-----------|------------------------------|
| **AI Developers** | Design better prompt architectures |
| **Prompt Engineers** | Write more effective prompts |
| **Product Managers** | Set realistic AI capabilities |
| **Researchers** | Extend methodology to new models |
| **Students** | Learn empirical AI research methods |

### 4.2 Real-World Applications

#### 4.2.1 Customer Support Chatbots
**Problem:** Long conversation histories cause the bot to forget early context.
**Our solution:** Rules for conversation summarization frequency.

#### 4.2.2 Document Analysis Systems
**Problem:** Analyzing large documents loses information in the middle.
**Our solution:** Optimal document chunking and positioning strategies.

#### 4.2.3 Multi-Agent AI Systems
**Problem:** Agents accumulate data and lose track of important information.
**Our solution:** Select/Compress/Write strategies for context management.

#### 4.2.4 RAG (Retrieval-Augmented Generation) Systems
**Problem:** Full context retrieval can hurt more than help.
**Our solution:** Quality thresholds and focused retrieval guidelines.

---

## 5. Research Goals & Hypotheses

### 5.1 Primary Research Goal

**Empirically validate context window behaviors and evaluate mitigation strategies, producing actionable prompt engineering guidelines.**

### 5.2 Research Questions

| ID | Research Question | Related Experiment |
|----|------------------|-------------------|
| RQ1 | Does information position affect retrieval accuracy? | Experiment 1 |
| RQ2 | How much does accuracy degrade as context size grows? | Experiment 2 |
| RQ3 | Does RAG improve accuracy compared to full context? | Experiment 3 |
| RQ4 | Which management strategy works best? | Experiment 4 |

### 5.3 Research Hypotheses

| ID | Hypothesis | Test Method |
|----|-----------|-------------|
| H1 | Middle-positioned information has lower retrieval accuracy than start/end | Position analysis |
| H2 | Accuracy decreases as document count increases | Correlation analysis |
| H3 | Response latency increases with context size | Latency measurement |
| H4 | RAG retrieval outperforms full context in accuracy | Comparison test |
| H5 | RAG retrieval is faster than full context | Latency comparison |
| H6 | Select/Compress/Write strategies improve over baseline | Strategy comparison |

### 5.4 Expected Outcomes Matrix

| Experiment | Expected Finding | Confidence |
|------------|-----------------|------------|
| Needle in Haystack | 30-50% accuracy drop at middle positions | High |
| Context Size Impact | Linear degradation with document count | Medium |
| RAG Impact | RAG faster and more accurate than full context | High |
| Strategies | All strategies outperform baseline | Medium |

---

## 6. Experimental Design

### 6.1 Overview

```
+---------------------------------------------------------+
|                    EXPERIMENTAL FRAMEWORK                |
+---------------------------------------------------------+
|                                                          |
|   +-------------+    +-------------+    +-------------+  |
|   | Experiment 1|    | Experiment 2|    | Experiment 3|  |
|   | Needle in   |    | Context     |    | RAG         |  |
|   | Haystack    |    | Size Impact |    | Impact      |  |
|   +------+------+    +------+------+    +------+------+  |
|          |                  |                  |         |
|          +--------+---------+--------+---------+         |
|                   |                                      |
|            +------+------+                               |
|            | Experiment 4|                               |
|            | Strategies  |                               |
|            +-------------+                               |
|                                                          |
+---------------------------------------------------------+
```

### 6.2 Experiment 1: Needle in Haystack (Lost in the Middle)

#### 6.2.1 Objective
Demonstrate that information position affects retrieval accuracy, validating the "Lost in the Middle" phenomenon.

#### 6.2.2 Experiment Details
- **Duration:** ~15 minutes
- **Difficulty:** Basic
- **Goal:** Show high accuracy at start/end, low at middle

#### 6.2.3 Data Specification
- **Synthetic text:** 5 documents, each with 200 words
- **Critical fact:** One fact per document (e.g., "The company CEO is David Cohen")
- **Positions tested:** Start / Middle / End

#### 6.2.4 Methodology

```python
# Pseudocode for Experiment 1

def experiment_1_needle_in_haystack():
    """
    Test if LLMs retrieve information differently based on position.

    Variables:
    - Independent: Position (start, middle, end)
    - Dependent: Retrieval accuracy (0-100%)
    """

    positions = ['start', 'middle', 'end']
    documents = create_documents(num_docs=5, words_per_doc=200)

    results = {'start': [], 'middle': [], 'end': []}

    for doc in documents:
        for position in positions:
            # Embed fact at specified position
            test_doc = embed_critical_fact(doc, fact, position)

            # Query the model
            response = query_llm(test_doc, "Who is the company CEO?")

            # Score the response
            accuracy = evaluate_response(response, expected_answer)
            results[position].append(accuracy)

    return calculate_averages(results)

# Expected: High accuracy at start/end, low at middle
```

#### 6.2.5 Expected Results

| Position | Expected Accuracy |
|----------|-------------------|
| Start | 90-98% |
| Middle | 50-70% |
| End | 90-98% |

### 6.3 Experiment 2: Context Window Size Impact

#### 6.3.1 Objective
Measure how accuracy and latency change as context window size grows (more documents).

#### 6.3.2 Experiment Details
- **Duration:** ~20 minutes
- **Difficulty:** Medium
- **Goal:** Show accuracy degradation and latency increase with larger contexts

#### 6.3.3 Data Specification
- **Document counts:** 2, 5, 10, 20, 50
- **Measurements per size:** Response time + Accuracy + Actual context length

#### 6.3.4 Methodology

```python
# Pseudocode for Experiment 2

def experiment_2_context_size_impact():
    """
    Test how context window size affects accuracy and latency.

    Variables:
    - Independent: Number of documents (2, 5, 10, 20, 50)
    - Dependent: Accuracy, Latency, Tokens used
    """

    doc_counts = [2, 5, 10, 20, 50]
    results = []

    for num_docs in doc_counts:
        documents = load_documents(num_docs)
        context = concatenate_documents(documents)

        start_time = time.time()
        response = query_llm(context, query)
        latency = time.time() - start_time

        results.append({
            'num_docs': num_docs,
            'tokens_used': count_tokens(context),
            'latency': latency,
            'accuracy': evaluate_accuracy(response)
        })

    return results

# Expected: Accuracy decreases, latency increases as window grows
```

#### 6.3.5 Expected Results

| Document Count | Expected Accuracy | Expected Latency |
|----------------|-------------------|------------------|
| 2 | 95%+ | Low |
| 5 | 90% | Low-Medium |
| 10 | 80% | Medium |
| 20 | 70% | Medium-High |
| 50 | 50-60% | High |

### 6.4 Experiment 3: RAG Impact

#### 6.4.1 Objective
Compare two retrieval strategies: Full Context (all documents) vs RAG (only relevant documents via similarity search).

#### 6.4.2 Experiment Details
- **Duration:** ~25 minutes
- **Difficulty:** Medium+
- **Goal:** Demonstrate RAG superiority in accuracy and speed

#### 6.4.3 Data Specification
- **Document pool:** 20 documents in Hebrew (topics: technology, law, medicine)
- **Sample query:** "What are the side effects of drug X?"

#### 6.4.4 Methodology

```python
# Pseudocode for Experiment 3

def experiment_3_rag_impact():
    """
    Compare Full Context vs RAG retrieval.

    Strategies:
    - Mode A: Full context (all documents in window)
    - Mode B: RAG (only similar documents via similarity search)
    """

    # Step 1: Chunking - split documents into chunks
    chunks = split_documents(documents, chunk_size=500)

    # Step 2: Embedding - convert to vectors
    embeddings = embed_text(chunks)

    # Step 3: Store in vector database
    vector_store = create_vector_store()
    vector_store.add(chunks, embeddings)

    # Step 4: Compare two retrieval modes
    def compare_modes(query):
        # Mode A: Full context (all documents)
        full_response = query_with_full_context(all_documents, query)

        # Mode B: RAG (only similar documents)
        relevant_docs = vector_store.similarity_search(query, k=3)
        rag_response = query_with_context(relevant_docs, query)

        return {
            'full_accuracy': evaluate(full_response),
            'rag_accuracy': evaluate(rag_response),
            'full_latency': full_response.latency,
            'rag_latency': rag_response.latency
        }

    return compare_modes(query)

# Expected: RAG = accurate & fast, Full = noisy & slow
```

#### 6.4.5 Expected Results

| Mode | Expected Accuracy | Expected Latency |
|------|-------------------|------------------|
| Full Context | 60-70% | High |
| RAG (k=3) | 85-95% | Low |

### 6.5 Experiment 4: Context Engineering Strategies

#### 6.5.1 Objective
Compare effectiveness of Select, Compress, and Write strategies for context management.

#### 6.5.2 Experiment Details
- **Duration:** ~30 minutes
- **Difficulty:** Advanced
- **Goal:** Identify best strategy for long-running agents

#### 6.5.3 Data Specification
- **Simulation:** Multi-step agent performing 10 sequential actions
- **Each action:** Creates output that is added to context
- **Strategies tested:** Select, Compress, Write

#### 6.5.4 Methodology

```python
# Pseudocode for Experiment 4

def experiment_4_context_engineering():
    """
    Compare context management strategies.

    Strategies:
    1. SELECT - Use RAG for relevant retrieval only
    2. COMPRESS - Automatic history summarization
    3. WRITE - External memory (scratchpad)
    """

    # Strategy 1: SELECT - Use RAG for relevant retrieval
    def select_strategy(history, query):
        relevant = rag_search(history, query, k=5)
        return query_llm(relevant, query)

    # Strategy 2: COMPRESS - Automatic history summarization
    def compress_strategy(history, query):
        if len(history) > MAX_TOKENS:
            history = summarize(history)
        return query_llm(history, query)

    # Strategy 3: WRITE - External memory (scratchpad)
    def write_strategy(history, query, scratchpad):
        key_facts = extract_key_facts(history)
        scratchpad.store(key_facts)
        return query_llm(scratchpad.retrieve(query), query)

    # Compare all strategies across 10 sequential actions
    def benchmark_strategies(num_actions=10):
        results = {'select': [], 'compress': [], 'write': []}

        for action in range(num_actions):
            output = agent.execute(action)
            history.append(output)

            for strategy in ['select', 'compress', 'write']:
                result = evaluate_strategy(strategy, history)
                results[strategy].append(result)

        return results

    return benchmark_strategies()
```

#### 6.5.5 Expected Results

| Strategy | Expected Accuracy | Memory Usage | Best For |
|----------|-------------------|--------------|----------|
| Baseline | Degrading over time | High | N/A |
| Select | High, stable | Medium | Document Q&A |
| Compress | Medium-High | Low | Long conversations |
| Write | High, stable | Low | Multi-step agents |

---

## 7. Technical Requirements

### 7.1 Environment

| Component | Requirement | Notes |
|-----------|-------------|-------|
| **Python** | 3.10+ | Required for type hints |
| **Virtual Environment** | UV (mandatory) | Fast parallel installs |
| **OS** | Windows 11 (WSL) / Linux / macOS | Cross-platform |
| **RAM** | 8GB minimum | For large prompt processing |
| **Storage** | 2GB free | For results and logs |

### 7.2 Core Dependencies

```txt
# requirements.txt (exact versions)
numpy==1.24.3          # Vectorized operations
pandas==2.0.2          # Data analysis
matplotlib==3.7.1      # Visualization
seaborn==0.12.2        # Statistical plots
scipy==1.10.1          # Statistical tests
chromadb==0.4.0        # Vector database for RAG
sentence-transformers==2.2.2  # Embeddings
requests==2.31.0       # HTTP requests
python-dotenv==1.0.0   # Environment variables
tiktoken==0.5.1        # Token counting
tqdm==4.66.1           # Progress bars
pytest==7.4.0          # Testing
```

### 7.3 LLM Interface

The experiments will use a local LLM via Ollama or API-based models. The interface should be abstracted to allow switching between:
- Ollama (local models)
- OpenAI API
- Anthropic API
- Google Gemini API

### 7.4 Performance Requirements

| Metric | Target | Maximum |
|--------|--------|---------|
| Single experiment run | < 5 minutes | 15 minutes |
| Full experiment suite | < 2 hours | 4 hours |
| Memory usage | < 4GB | 8GB |

### 7.5 Code Architecture

```
project-root/
+-- main.py                    # Entry point
+-- requirements.txt           # Dependencies
+-- .env.example              # API key template
+-- .gitignore                # Secrets protection
|
+-- src/                      # Source code
|   +-- __init__.py
|   +-- experiments/          # Experiment implementations
|   |   +-- __init__.py
|   |   +-- exp1_needle_in_haystack.py
|   |   +-- exp2_context_size.py
|   |   +-- exp3_rag_impact.py
|   |   +-- exp4_strategies.py
|   +-- generators/           # Test data generators
|   |   +-- __init__.py
|   |   +-- document_generator.py
|   +-- models/               # LLM interfaces
|   |   +-- __init__.py
|   |   +-- base_model.py
|   |   +-- llm_interface.py
|   +-- rag/                  # RAG components
|   |   +-- __init__.py
|   |   +-- vector_store.py
|   |   +-- embeddings.py
|   +-- analysis/             # Statistical analysis
|   |   +-- __init__.py
|   |   +-- statistics.py
|   |   +-- visualizations.py
|   +-- utils/                # Utilities
|       +-- __init__.py
|       +-- logger.py
|       +-- config.py
|       +-- helpers.py
|
+-- docs/                     # Documentation
|   +-- requirements/
|       +-- PRD.md
|       +-- tasks.json
|
+-- results/                  # Output files
|   +-- experiments/          # Raw experiment data
|   +-- graphs/              # Visualizations
|   +-- reports/             # Analysis reports
|
+-- logs/                    # Ring buffer logs
|   +-- config/
|
+-- tests/                   # Unit tests
    +-- __init__.py
    +-- test_generators.py
```

---

## 8. Functional Requirements

### 8.1 Data Generation

| ID | Requirement | Priority |
|----|------------|----------|
| FR-1 | Generate synthetic documents with 200 words each | Critical |
| FR-2 | Embed critical facts at specified positions (start/middle/end) | Critical |
| FR-3 | Generate semantically neutral filler text | High |
| FR-4 | Create document sets of varying sizes (2, 5, 10, 20, 50) | Critical |
| FR-5 | Support Hebrew documents for RAG experiment | High |

### 8.2 Model Interface

| ID | Requirement | Priority |
|----|------------|----------|
| FR-6 | Unified LLM interface supporting multiple providers | Critical |
| FR-7 | Handle rate limiting and retries | Critical |
| FR-8 | Measure response latency | High |
| FR-9 | Count tokens used | High |

### 8.3 RAG Components

| ID | Requirement | Priority |
|----|------------|----------|
| FR-10 | Document chunking with configurable chunk size | Critical |
| FR-11 | Text embedding using sentence transformers | Critical |
| FR-12 | Vector store (ChromaDB) for similarity search | Critical |
| FR-13 | Configurable k for top-k retrieval | High |

### 8.4 Experiment Execution

| ID | Requirement | Priority |
|----|------------|----------|
| FR-14 | Run all 4 experiments automatically | Critical |
| FR-15 | Support experiment resume after failure | High |
| FR-16 | Run minimum 3 trials per condition | Critical |
| FR-17 | Log all results with timestamps | Critical |

### 8.5 Analysis & Visualization

| ID | Requirement | Priority |
|----|------------|----------|
| FR-18 | Generate accuracy by position graph (Exp 1) | Critical |
| FR-19 | Generate accuracy/latency vs size graph (Exp 2) | Critical |
| FR-20 | Generate RAG vs Full Context comparison (Exp 3) | Critical |
| FR-21 | Generate strategy performance table (Exp 4) | Critical |
| FR-22 | Export results to CSV/JSON | High |

---

## 9. Metrics & Measurement

### 9.1 Primary Metrics

| Metric | Definition | Measurement Method |
|--------|-----------|-------------------|
| **Retrieval Accuracy** | Percentage of correct retrievals | Exact match or semantic similarity |
| **Response Latency** | Time from query to response | Wall clock time |
| **Token Usage** | Tokens consumed per query | Token counter |

### 9.2 Per-Experiment Metrics

| Experiment | Key Metrics |
|------------|-------------|
| 1. Needle in Haystack | Accuracy by position |
| 2. Context Size | Accuracy vs doc count, Latency vs doc count |
| 3. RAG Impact | Full vs RAG accuracy, Full vs RAG latency |
| 4. Strategies | Strategy accuracy over time, Memory usage |

---

## 10. Visualization Requirements

### 10.1 Required Graphs (Minimum 4)

#### Experiment 1: Needle in Haystack
1. **Accuracy by Position** - Bar chart showing start/middle/end accuracy

#### Experiment 2: Context Window Size
2. **Accuracy vs Document Count** - Line chart showing degradation
3. **Latency vs Document Count** - Line chart showing increase

#### Experiment 3: RAG Impact
4. **Performance Comparison** - Bar chart comparing Full vs RAG (accuracy & latency)

#### Experiment 4: Strategies
5. **Strategy Performance Table** - Table showing Select/Compress/Write results

### 10.2 Graph Standards

All graphs must include:
- Clear title explaining what it shows
- Labeled axes with units
- Legend (if multiple series)
- High resolution (300 DPI)
- Consistent color scheme

---

## 11. Success Criteria

### 11.1 Quantitative Success Criteria

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Experiments completed | 4/4 (100%) | Checklist |
| Trials per condition | >= 3 | Count |
| Graphs generated | >= 4 | Count |
| Rules derived | >= 4 (one per experiment) | Count |

### 11.2 Qualitative Success Criteria

| Criterion | Standard |
|-----------|----------|
| Reproducibility | Another researcher can replicate results |
| Clarity | A 15-year-old can understand the conclusions |
| Actionability | Rules can be immediately applied to prompts |

---

## 12. Constraints & Assumptions

### 12.1 Constraints

| Type | Constraint | Mitigation |
|------|-----------|------------|
| **Time** | ~90 minutes total for all experiments | Efficient implementation |
| **API Limits** | Rate limiting on providers | Implement backoff, use local models |
| **Context Limits** | Model-specific token limits | Document limits per experiment |

### 12.2 Assumptions

| Assumption | Impact if Wrong |
|-----------|-----------------|
| LLM behavior is consistent | May need to re-run experiments |
| Synthetic data represents real use | Validate with real examples |
| Position effect is significant | May not see expected U-curve |

---

## 13. Security Considerations

### 13.1 API Key Protection

```gitignore
# .gitignore - MUST include
.env
*api_key*
*secret*
*.key
```

### 13.2 .env.example Template

```bash
# Copy to .env and fill with real values
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

### 13.3 Data Privacy

- No personal data in test documents
- No real company information
- Synthetic data only

---

## 14. Deliverables

### 14.1 Phase 1 Deliverables (Planning)

| Deliverable | Format | Status |
|------------|--------|--------|
| PRD.md | Markdown | Complete |
| tasks.json | JSON | Pending |

### 14.2 Phase 2 Deliverables (Implementation)

| Deliverable | Format | Description |
|------------|--------|-------------|
| Source code | Python | All experiments + utilities |
| Raw results | JSON/CSV | All experiment data |
| Graphs | PNG (300 DPI) | Minimum 4 visualizations |
| README.md | Markdown | Complete documentation |
| Conclusions | Markdown | Findings and rules |

---

## 15. Learning Objectives

### 15.1 Technical Skills

| Skill | How Demonstrated |
|-------|-----------------|
| Empirical research design | Controlled experiments |
| Statistical analysis | Hypothesis testing |
| RAG implementation | ChromaDB + embeddings |
| Data visualization | Professional research graphs |
| Python best practices | Modular, documented code |

### 15.2 Domain Knowledge

| Knowledge Area | What You'll Learn |
|---------------|-------------------|
| LLM context windows | How they work and fail |
| Lost in the Middle | Position-based accuracy |
| RAG systems | Retrieval vs full context |
| Context management | Select/Compress/Write strategies |

---

## 16. Glossary

| Term | Definition |
|------|-----------|
| **Context Window** | The maximum amount of text an LLM can process at once |
| **Token** | The basic unit of text processing (~0.75 words in English) |
| **Lost in the Middle** | Phenomenon where LLMs poorly recall middle-positioned information |
| **RAG** | Retrieval-Augmented Generation - adding relevant documents to prompts |
| **Prompt Engineering** | The art of writing effective prompts for LLMs |
| **Vector Store** | Database storing text as numerical vectors for similarity search |
| **Embedding** | Converting text to numerical vectors |
| **ChromaDB** | Open-source vector database |

---

## Summary Table

| Experiment | Topic | Tools | Time | Output |
|------------|-------|-------|------|--------|
| 1 | Lost in Middle | Ollama + Python | 15 min | Accuracy by position graph |
| 2 | Context Size | Ollama + LangChain | 20 min | Latency vs size graph |
| 3 | RAG Impact | Ollama + Chroma | 25 min | Performance comparison |
| 4 | Engineering | LangChain + Memory | 30 min | Strategy performance table |

---

**Document Status:** Phase 1 Complete - Awaiting Approval
**Next Step:** Create tasks.json
**Blocked Until:** PRD approval received

---

*This PRD follows the AI Developer Expert Course PROJECT_GUIDELINES.md v4.0*
*Aligned with context-windows-lab.pdf requirements (4 experiments)*
