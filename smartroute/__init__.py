"""SmartRoute: local-first, embeddable LLM prompt routing with per-project adaptive learning.

SmartRoute classifies LLM prompts by task type and complexity, routes them to the
optimal model tier, and learns from implicit user feedback to improve routing
decisions per project over time. All learning happens in a local SQLite file.

Public API exports arrive with the Router implementation (Phase 6).
"""

__version__ = "0.1.0"
