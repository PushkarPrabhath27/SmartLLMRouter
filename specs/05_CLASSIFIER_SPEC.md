# SmartRoute — Classifier Specification

## Overview

The classifier is a **heuristic, rule-based system** that extracts 9 features from a raw prompt string and produces a `ClassificationResult` with task type, complexity score, and confidence.

**V1 Constraint:** No ML model, no embeddings, no external API calls during classification. Everything is computed from the prompt string using regex, tokenization, and simple statistics.

## Feature Vector (9 Features)

### F1: Token Count
- **Method:** `tiktoken.encoding_for_model("gpt-4o-mini").encode(prompt)`
- **Output:** Integer
- **Complexity contribution:** Linear scaling. <50 tokens = low weight, 50-500 = medium, >500 = high.

### F2: Code Block Ratio
- **Method:** Count tokens inside triple-backtick fences (``` ... ```) divided by total tokens.
- **Regex:** `` /```[\s\S]*?```/g ``
- **Output:** Float [0.0, 1.0]
- **Complexity contribution:** Code blocks strongly weight toward `task_type=CODE`. Ratio >0.5 is a strong signal.

### F3: Question vs. Instruction
- **Method:**
  - `is_question`: prompt.strip().endswith("?") OR starts with wh-word (what, why, how, when, where, who, which)
  - `is_instruction`: Contains imperative verbs (create, build, write, refactor, fix, implement, add, remove, update, delete)
- **Output:** `{is_question: bool, is_instruction: bool, instruction_verb_count: int}`
- **Complexity contribution:** Instructions typically higher complexity than questions. Pure questions often route to cheaper models.

### F4: Multi-Step Markers
- **Method:** Count occurrences of transition/sequence words.
- **Keywords:** "then", "after that", "next", "step", "first", "second", "third", "finally", "subsequently", "followed by"
- **Output:** Integer
- **Complexity contribution:** Each marker adds 0.05 to complexity, capped at 0.3.

### F5: Ambiguity Score
- **Method:** Count hedge/uncertainty words divided by total word count.
- **Keywords:** "maybe", "perhaps", "possibly", "might", "could", "not sure", "uncertain", "unclear", "vague", "either...or", "or something", "I think", "probably", "sort of", "kind of"
- **Output:** Float [0.0, 1.0]
- **Complexity contribution:** Ambiguous prompts need stronger models to "fill gaps." Score >0.1 adds 0.15 to complexity.

### F6: Domain Hint
- **Method:** Keyword matching against 5 domain dictionaries.
- **Dictionaries:**
  - **CODE:** function, class, variable, debug, refactor, compile, runtime, syntax, error, stack trace, import, module, package, API, endpoint, database, query, SQL, regex, algorithm, data structure, OOP, functional, async, await, decorator, lambda, list comprehension, git, commit, branch, merge, pull request, unit test, CI/CD, Docker, Kubernetes
  - **CREATIVE:** story, poem, essay, blog, write, draft, creative, imaginative, fiction, narrative, character, plot, setting, dialogue, tone, style, metaphor, simile, analogy, brainstorm, ideate, concept, slogan, tagline, brand, marketing copy, ad, campaign, social media, tweet, thread, email, newsletter
  - **REASONING:** analyze, compare, contrast, evaluate, assess, pros and cons, trade-off, decision, strategy, plan, framework, model, theory, hypothesis, evidence, proof, logical, rational, deduce, infer, conclude, solve, puzzle, riddle, math, equation, calculate, statistics, probability, optimize, maximize, minimize
  - **SUMMARIZATION:** summarize, summary, TL;DR, key points, main ideas, condense, shorten, brief, overview, recap, digest, abstract, extract, highlight, bullet points, executive summary
  - **TRANSLATION:** translate, translation, language, English, Spanish, French, German, Chinese, Japanese, Korean, Arabic, Hindi, Portuguese, Russian, Italian, Dutch, Swedish, Polish, Turkish, Vietnamese, Thai, Indonesian, Malay, Filipino, Hebrew, Greek, Czech, Romanian, Hungarian, Finnish, Norwegian, Danish
- **Output:** `{domain: str, match_count: int, match_ratio: float}`
- **Task type assignment:** Highest match_ratio wins. If tie, use match_count. If no match >0.05, default to GENERAL.

### F7: File Path References
- **Method:** Regex detection of file paths.
- **Patterns:**
  - `/path/to/file.ext`
  - `C:\Users\file.ext`
  - `file.py`, `file.js`, `file.ts`, `file.tsx`, `file.jsx`, `file.go`, `file.rs`, `file.java`, `file.kt`, `file.swift`, `file.cpp`, `file.c`, `file.h`, `file.rb`, `file.php`, `file.md`, `file.json`, `file.yaml`, `file.yml`, `file.toml`, `file.xml`, `file.sql`, `file.sh`, `file.bat`, `file.ps1`
  - `src/`, `lib/`, `app/`, `components/`, `models/`, `tests/`, `docs/`
- **Output:** Integer (count of unique file references)
- **Complexity contribution:** >0 file references strongly suggest CODE task. Each reference adds 0.02 to complexity (capped at 0.1).

### F8: Urgency Score
- **Method:** Count urgency keywords.
- **Keywords:** "urgent", "ASAP", "immediately", "critical", "blocking", "broken", "down", "outage", "emergency", "hotfix", "production", "prod", "live", "customer waiting", "deadline", "due today", "due tomorrow", "EOD", "end of day", "priority", "P0", "P1", "severity", "high severity", "regression", "rollback", "revert"
- **Output:** Integer
- **Complexity contribution:** Does NOT affect complexity. Used only in explainability ("Urgency detected but not routed higher -- complexity was low"). V1 does not route on urgency alone.

### F9: Instruction Verb Density
- **Method:** Count of imperative verbs / total word count.
- **Imperative verbs:** create, build, write, generate, produce, make, construct, develop, design, implement, code, program, script, automate, configure, set up, deploy, publish, release, fix, debug, refactor, rewrite, optimize, improve, enhance, upgrade, update, modify, change, adjust, tweak, edit, correct, resolve, solve, handle, manage, process, parse, validate, verify, check, test, run, execute, add, remove, delete, insert, merge, split, join, extract, filter, sort, search, find, organize, format, convert, transform, translate, explain, describe, document, log, track, analyze, evaluate, compare, calculate, estimate, predict, simulate, model, scaffold, import, export, define, declare, initialize, allocate, read, write, save, load, fetch, send, upload, download, sync, backup, migrate, install, package, encrypt, decrypt, hash, encode, decode, serialize, parse, compile, lint, prettify
- **Output:** Float [0.0, 1.0]
- **Complexity contribution:** Density >0.1 adds 0.1 to complexity (high instruction density = more work).

## Complexity Scoring Algorithm

```python
def compute_complexity(features: FeatureVector) -> float:
    score = 0.0

    # Base from token count
    tokens = features.token_count
    if tokens < 50:
        score += 0.1
    elif tokens < 200:
        score += 0.3
    elif tokens < 500:
        score += 0.5
    else:
        score += 0.7

    # Code block ratio
    score += features.code_block_ratio * 0.2

    # Multi-step markers
    score += min(features.multi_step_count * 0.05, 0.3)

    # Ambiguity
    if features.ambiguity_score > 0.1:
        score += 0.15

    # File references
    score += min(features.file_path_count * 0.02, 0.1)

    # Instruction density
    if features.instruction_verb_density > 0.1:
        score += 0.1

    # Question penalty (questions are usually simpler)
    if features.is_question and not features.is_instruction:
        score -= 0.15

    return clamp(score, 0.0, 1.0)
```

## Confidence Scoring

Confidence measures how "sure" the classifier is about its output. Low confidence means the prompt is ambiguous or mixed-domain.

```python
def compute_confidence(features: FeatureVector, task_type: TaskType) -> float:
    confidence = 0.5  # base

    # Domain clarity
    if features.domain_hint.match_ratio > 0.3:
        confidence += 0.3
    elif features.domain_hint.match_ratio > 0.1:
        confidence += 0.15
    else:
        confidence -= 0.2  # unclear domain

    # Strong code signals
    if features.code_block_ratio > 0.5:
        confidence += 0.1

    # Strong question signal (clear intent)
    if features.is_question and not features.is_instruction:
        confidence += 0.1

    # Mixed signals reduce confidence
    if features.is_question and features.is_instruction:
        confidence -= 0.15

    return clamp(confidence, 0.0, 1.0)
```

## ClassificationResult Construction

```python
@dataclass
class ClassificationResult:
    task_type: TaskType
    complexity: float
    confidence: float
    features: FeatureVector

    @property
    def complexity_bucket(self) -> ComplexityBucket:
        if self.complexity < 0.33:
            return ComplexityBucket.LOW
        elif self.complexity < 0.66:
            return ComplexityBucket.MEDIUM
        else:
            return ComplexityBucket.HIGH
```

## Fail-Open Behavior

If any feature extraction step crashes (e.g., tiktoken not installed):
1. Log warning
2. Use default feature vector (all zeros, `token_count=len(prompt.split())`)
3. Return `ClassificationResult(task_type=GENERAL, complexity=0.5, confidence=0.3)`
4. Routing engine will use default tier (medium)

## Performance Requirements

- Classification must complete in <10ms for prompts <1000 tokens
- No network calls during classification
- No disk I/O during classification
- Thread-safe (no mutable shared state)

## Testing Requirements

- Unit test each feature extractor independently
- Property-based test: complexity is monotonic with token count (all else equal)
- Hand-labeled test set of 50 prompts: accuracy >70% on task_type, >60% on complexity_bucket
- Edge cases: empty string, only code, only question, mixed language, very long prompt (>10k tokens)
