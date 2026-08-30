# SmartRoute — Storage Specification

## Overview

All persistent state lives in a local SQLite database. The design is **local-first, privacy-preserving, zero-cloud**. No telemetry, no external storage, no analytics pipeline.

## Database Location

1. `storage_path` argument to `Router()`
2. `SMARTROUTE_STORAGE` environment variable
3. `./.smartroute/db.sqlite` (current working directory)
4. `~/.smartroute/db.sqlite`

The directory is created automatically on first run.

## Technology

- **Driver:** `aiosqlite` (async wrapper around sqlite3)
- **Schema version:** Stored in `PRAGMA user_version`
- **Migration strategy:** Simple drop-and-recreate on schema mismatch (v1 only; v2 will have real migrations)
- **Concurrency:** SQLite WAL mode enabled for better concurrent read/write

## Schema

### Table: decisions

Stores every routing decision.

```sql
CREATE TABLE decisions (
    id TEXT PRIMARY KEY,                    -- UUID v4
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    prompt_hash TEXT NOT NULL,               -- SHA-256 of prompt (for privacy, not full text)
    prompt_preview TEXT,                     -- First 100 chars (for debugging, optional)
    task_type TEXT NOT NULL,
    complexity REAL NOT NULL,
    complexity_bucket TEXT NOT NULL,
    confidence REAL NOT NULL,
    model_used TEXT NOT NULL,                -- e.g., "openai/gpt-4o-mini"
    provider_key TEXT NOT NULL,              -- e.g., "openai"
    estimated_cost_usd REAL,
    actual_cost_usd REAL,                    -- if provider returns usage
    latency_ms INTEGER,
    was_adapted BOOLEAN DEFAULT FALSE,
    override_applied TEXT,
    reason TEXT NOT NULL,
    conversation_id TEXT,
    turn_number INTEGER,
    features_json TEXT                       -- JSON blob of full feature vector
);

CREATE INDEX idx_decisions_timestamp ON decisions(timestamp);
CREATE INDEX idx_decisions_task_type ON decisions(task_type);
CREATE INDEX idx_decisions_conversation ON decisions(conversation_id);
CREATE INDEX idx_decisions_prompt_hash ON decisions(prompt_hash);
```

### Table: signals

Stores detected implicit feedback signals.

```sql
CREATE TABLE signals (
    id TEXT PRIMARY KEY,
    decision_id TEXT NOT NULL,
    signal_type TEXT NOT NULL,
    signal_value REAL NOT NULL,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    detection_method TEXT,                   -- "auto" or "manual"
    FOREIGN KEY (decision_id) REFERENCES decisions(id) ON DELETE CASCADE
);

CREATE INDEX idx_signals_decision_id ON signals(decision_id);
CREATE INDEX idx_signals_type ON signals(signal_type);
CREATE INDEX idx_signals_detected_at ON signals(detected_at);
```

### Table: reputation

Stores EMA reputation scores per bucket.

```sql
CREATE TABLE reputation (
    id TEXT PRIMARY KEY,
    bucket_key TEXT NOT NULL,                -- "code_low", "creative_medium", etc.
    model_tier TEXT NOT NULL,                -- "low", "medium", "high"
    ema_score REAL NOT NULL DEFAULT 0.5,
    call_count INTEGER NOT NULL DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_bumped_at TIMESTAMP,
    UNIQUE(bucket_key, model_tier)
);

CREATE INDEX idx_reputation_bucket ON reputation(bucket_key);
```

### Table: adaptations

Records when a bucket was auto-bumped.

```sql
CREATE TABLE adaptations (
    id TEXT PRIMARY KEY,
    bucket_key TEXT NOT NULL,
    old_tier TEXT NOT NULL,
    new_tier TEXT NOT NULL,
    triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ema_at_bump REAL NOT NULL
);

CREATE INDEX idx_adaptations_bucket ON adaptations(bucket_key);
CREATE INDEX idx_adaptations_triggered ON adaptations(triggered_at);
```

### Table: config_overrides

Runtime config overrides (set via API, not YAML).

```sql
CREATE TABLE config_overrides (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    set_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Data Access Layer

```python
class Storage:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._connection: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        """Initialize connection and run migrations."""
        ...

    async def close(self) -> None:
        """Close connection."""
        ...

    # Decisions
    async def store_decision(self, decision: DecisionRecord) -> str:
        """Store a decision, return its ID."""
        ...

    async def get_recent_decisions(self, limit: int = 10) -> list[DecisionRecord]:
        ...

    async def get_decisions_by_conversation(self, conversation_id: str) -> list[DecisionRecord]:
        ...

    async def get_decision_by_prompt_hash(self, prompt_hash: str, window_seconds: int = 60) -> Optional[DecisionRecord]:
        """For signal detection: find recent decision with same prompt."""
        ...

    # Signals
    async def store_signal(self, signal: SignalRecord) -> None:
        ...

    async def get_signals_for_decision(self, decision_id: str) -> list[SignalRecord]:
        ...

    # Reputation
    async def get_reputation(self, bucket_key: str, model_tier: str) -> Optional[ReputationRecord]:
        ...

    async def update_reputation(self, bucket_key: str, model_tier: str, ema: float, call_count: int) -> None:
        ...

    async def get_all_reputation(self) -> list[ReputationRecord]:
        ...

    # Adaptations
    async def record_adaptation(self, bucket_key: str, old_tier: str, new_tier: str, ema_at_bump: float) -> None:
        ...

    async def get_adaptations(self, bucket_key: Optional[str] = None) -> list[AdaptationRecord]:
        ...

    # Reports
    async def get_project_stats(self) -> dict:
        """Aggregates for ProjectReport."""
        ...

    async def reset_reputation(self, bucket_key: Optional[str] = None) -> None:
        ...
```

## Privacy Guarantees

- **Prompt hashing:** Full prompts are never stored. Only SHA-256 hashes and 100-char previews.
- **No cloud sync:** Database never leaves the local filesystem.
- **No telemetry:** No network calls from the storage layer.
- **User control:** `router.reset_reputation()` clears all learned data.
- **Transparent schema:** All tables are inspectable with any SQLite viewer.

## Performance

- All queries must complete in <50ms for a database with <100k rows
- Use indexes on all foreign keys and search columns
- WAL mode prevents read blocking on writes
- Connection pooling: single connection per Router instance (SQLite limitation)

## Backup & Reset

```python
# Reset all learned data (keep schema)
await router.reset_reputation()

# Full reset (drop and recreate)
import shutil
shutil.remove(".smartroute/db.sqlite")
# Next Router() init will recreate with defaults
```

## Testing

- Use `:memory:` SQLite for unit tests
- Test migration from empty database
- Test migration from v0 schema (if exists)
- Test concurrent reads/writes (aiosqlite handles this, but verify)
- Test with 100k synthetic rows (performance benchmark)
