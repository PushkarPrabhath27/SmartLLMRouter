"""SQL DDL for the SmartRoute SQLite schema (spec 08).

This module holds only DDL constants. The connection layer
(``smartroute.storage.connection``) executes them and manages schema
versioning via ``PRAGMA user_version``.
"""

from __future__ import annotations

SCHEMA_VERSION = 1

DECISIONS_DDL = """
CREATE TABLE decisions (
    id TEXT PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    prompt_hash TEXT NOT NULL,
    prompt_preview TEXT,
    task_type TEXT NOT NULL,
    complexity REAL NOT NULL,
    complexity_bucket TEXT NOT NULL,
    confidence REAL NOT NULL,
    model_used TEXT NOT NULL,
    provider_key TEXT NOT NULL,
    estimated_cost_usd REAL,
    actual_cost_usd REAL,
    latency_ms INTEGER,
    was_adapted BOOLEAN DEFAULT FALSE,
    override_applied TEXT,
    reason TEXT NOT NULL,
    conversation_id TEXT,
    turn_number INTEGER,
    features_json TEXT
)
"""

SIGNALS_DDL = """
CREATE TABLE signals (
    id TEXT PRIMARY KEY,
    decision_id TEXT NOT NULL,
    signal_type TEXT NOT NULL
        CHECK (signal_type IN ('hard_regen', 'soft_regen', 'explicit_correction', 'acceptance')),
    signal_value REAL NOT NULL,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    detection_method TEXT,
    FOREIGN KEY (decision_id) REFERENCES decisions(id) ON DELETE CASCADE
)
"""

REPUTATION_DDL = """
CREATE TABLE reputation (
    id TEXT PRIMARY KEY,
    bucket_key TEXT NOT NULL,
    model_tier TEXT NOT NULL,
    ema_score REAL NOT NULL DEFAULT 0.5,
    call_count INTEGER NOT NULL DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_bumped_at TIMESTAMP,
    UNIQUE(bucket_key, model_tier)
)
"""

ADAPTATIONS_DDL = """
CREATE TABLE adaptations (
    id TEXT PRIMARY KEY,
    bucket_key TEXT NOT NULL,
    old_tier TEXT NOT NULL,
    new_tier TEXT NOT NULL,
    triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ema_at_bump REAL NOT NULL
)
"""

CONFIG_OVERRIDES_DDL = """
CREATE TABLE config_overrides (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    set_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

INDEX_DDL: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON decisions(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_decisions_task_type ON decisions(task_type)",
    "CREATE INDEX IF NOT EXISTS idx_decisions_conversation ON decisions(conversation_id)",
    "CREATE INDEX IF NOT EXISTS idx_decisions_prompt_hash ON decisions(prompt_hash)",
    "CREATE INDEX IF NOT EXISTS idx_signals_decision_id ON signals(decision_id)",
    "CREATE INDEX IF NOT EXISTS idx_signals_type ON signals(signal_type)",
    "CREATE INDEX IF NOT EXISTS idx_signals_detected_at ON signals(detected_at)",
    "CREATE INDEX IF NOT EXISTS idx_reputation_bucket ON reputation(bucket_key)",
    "CREATE INDEX IF NOT EXISTS idx_adaptations_bucket ON adaptations(bucket_key)",
    "CREATE INDEX IF NOT EXISTS idx_adaptations_triggered ON adaptations(triggered_at)",
)

TABLE_DDL: tuple[str, ...] = (
    DECISIONS_DDL,
    SIGNALS_DDL,
    REPUTATION_DDL,
    ADAPTATIONS_DDL,
    CONFIG_OVERRIDES_DDL,
)

SCHEMA_STATEMENTS: tuple[str, ...] = TABLE_DDL + INDEX_DDL
