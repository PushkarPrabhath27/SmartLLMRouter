# SmartRoute — Signal Detection & Reputation Specification

## Overview

Signal detection captures implicit feedback from user behavior after a routing decision. It converts observable patterns (regeneration, correction, acceptance) into numerical signals that update the reputation EMA for each bucket.

## Signal Types

### Signal 1: Hard Regeneration
- **Definition:** The exact same prompt string is sent again within 30 seconds of the previous response.
- **Detection:**
  ```python
  def detect_hard_regen(prompt: str, history: list[DecisionRecord]) -> Optional[Signal]:
      for record in reversed(history[-5:]):  # check last 5 decisions
          if record.prompt == prompt:
              if (now - record.timestamp) < timedelta(seconds=30):
                  return Signal(type="hard_regen", value=-0.3, decision_id=record.id)
      return None
  ```
- **False Positive Mitigation:**
  - Only check last 5 decisions (not infinite history)
  - 30-second window is tight enough to catch frustration but not normal iteration
  - Ignore if the new prompt has additional context (e.g., "and also..." appended)
- **Value:** -0.3 (strong negative)

### Signal 2: Soft Regeneration
- **Definition:** A very similar prompt is sent within 60 seconds. "Similar" means >80% token overlap.
- **Detection:**
  ```python
  def detect_soft_regen(prompt: str, history: list[DecisionRecord]) -> Optional[Signal]:
      prompt_tokens = set(tokenize(prompt))
      for record in reversed(history[-5:]):
          record_tokens = set(tokenize(record.prompt))
          if len(prompt_tokens) == 0 or len(record_tokens) == 0:
              continue
          overlap = len(prompt_tokens & record_tokens) / max(len(prompt_tokens), len(record_tokens))
          if overlap > 0.8 and (now - record.timestamp) < timedelta(seconds=60):
              return Signal(type="soft_regen", value=-0.1, decision_id=record.id)
      return None
  ```
- **Tokenization:** Simple whitespace split + lowercase. No stemming, no embeddings.
- **False Positive Mitigation:**
  - 80% threshold is high enough to require substantial similarity
  - 60-second window catches quick retries
  - Only check last 5 decisions
- **Value:** -0.1 (weak negative)

### Signal 3: Explicit Correction
- **Definition:** The next user message in a conversation contains explicit negative feedback.
- **Detection:**
  ```python
  def detect_explicit_correction(next_message: str) -> Optional[Signal]:
      # English
      english_negative = [
          "no", "wrong", "incorrect", "bad", "terrible", "awful", "sucks",
          "not right", "doesn't work", "didn't work", "not working",
          "redo", "do it again", "try again", "fix this", "that's wrong",
          "you're wrong", "not what I asked", "missed the point",
          "off topic", "irrelevant", "useless", "garbage", "nonsense",
          "can you fix", "please fix", "needs fixing", "correction:",
          "actually,", "wait,", "hold on,", "scratch that",
      ]
      # Spanish
      spanish_negative = [
          "no", "incorrecto", "malo", "horrible", "no funciona", "no sirve",
          "intenta de nuevo", "hazlo de nuevo", "corrige", "estas equivocado",
          "no es lo que pedi", "fuera de tema", "inutil", "basura",
      ]
      # French
      french_negative = [
          "non", "faux", "incorrect", "mauvais", "nul", "ca ne marche pas",
          "refais", "corrige", "tu te trompes", "ce n'est pas ce que j'ai demande",
          "inutile", "a cote de la plaque",
      ]
      # German
      german_negative = [
          "nein", "falsch", "inkorrekt", "schlecht", "funktioniert nicht",
          "mach nochmal", "korrigiere", "das ist falsch", "nicht was ich wollte",
          "nutzlos", "muell", "quatsch",
      ]
      # Chinese (simplified)
      chinese_negative = [
          "不对", "错了", "不好", "不行", "没用", "垃圾", "重做", "再试一次",
          "修正", "你错了", "不是我想要的", "跑题了", "毫无意义",
      ]
      # Japanese
      japanese_negative = [
          "違う", "間違い", "悪い", "ダメ", "役に立たない", "ゴミ",
          "やり直し", "修正して", "違います", "求めたものではない",
      ]

      all_negative = (
          english_negative + spanish_negative + french_negative +
          german_negative + chinese_negative + japanese_negative
      )

      next_lower = next_message.lower().strip()
      for phrase in all_negative:
          if phrase in next_lower:
              return Signal(type="explicit_correction", value=-0.2, decision_id=...)
      return None
  ```
- **False Positive Mitigation:**
  - Phrase-level matching (not word-level) reduces false positives
  - Only triggers if the next message is short (<200 tokens) -- long follow-ups are probably new questions, not corrections
  - Requires conversation context (previous decision_id must be known)
- **Value:** -0.2 (medium negative)

### Signal 4: Acceptance
- **Definition:** The user continues the conversation naturally without negative signals, OR a significant time passes (>10 minutes) without follow-up.
- **Detection:**
  ```python
  def detect_acceptance(conversation_history: list, last_decision_id: str) -> Optional[Signal]:
      # If next message exists and is NOT a correction/regeneration -> acceptance
      # If no next message after 10 minutes -> acceptance (implicit)
      # If conversation_id has new turns with different topics -> acceptance
      return Signal(type="acceptance", value=+0.05, decision_id=last_decision_id)
  ```
- **Caveat:** This is the noisiest signal. The +0.05 value is small so false positives don't swamp the system.
- **Value:** +0.05 (weak positive)

## Signal Priority

If multiple signals apply to the same decision, apply the strongest (most negative) signal only. Do not stack signals.

```python
SIGNAL_PRIORITY = {
    "hard_regen": 4,      # strongest
    "explicit_correction": 3,
    "soft_regen": 2,
    "acceptance": 1,      # weakest
}
```

## EMA Update Procedure

```python
async def apply_signal(bucket_key: str, model_tier: str, signal: Signal, storage: Storage):
    # 1. Fetch current reputation
    rep = await storage.get_reputation(bucket_key, model_tier)
    if rep is None:
        rep = ReputationRecord(bucket_key=bucket_key, model_tier=model_tier, ema=0.5, call_count=0)

    # 2. Update EMA
    new_ema = 0.3 * signal.value + 0.7 * rep.ema
    new_count = rep.call_count + 1

    # 3. Store
    await storage.update_reputation(bucket_key, model_tier, new_ema, new_count)

    # 4. Check for bump
    if should_bump(new_ema, new_count, threshold=0.3, cooldown=rep.last_bumped_at):
        await storage.record_adaptation(bucket_key, old_tier=model_tier, new_tier=next_tier(model_tier))
        logger.info(f"Bucket {bucket_key} bumped from {model_tier} to {next_tier(model_tier)}")
```

## Storage Schema for Signals

```sql
CREATE TABLE signals (
    id TEXT PRIMARY KEY,
    decision_id TEXT NOT NULL,
    signal_type TEXT NOT NULL CHECK(signal_type IN ('hard_regen', 'soft_regen', 'explicit_correction', 'acceptance')),
    signal_value REAL NOT NULL,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (decision_id) REFERENCES decisions(id)
);

CREATE INDEX idx_signals_decision_id ON signals(decision_id);
CREATE INDEX idx_signals_type ON signals(signal_type);
```

## Storage Schema for Reputation

```sql
CREATE TABLE reputation (
    id TEXT PRIMARY KEY,
    bucket_key TEXT NOT NULL,        -- e.g., "code_low"
    model_tier TEXT NOT NULL,          -- e.g., "low", "medium", "high"
    ema_score REAL NOT NULL DEFAULT 0.5,
    call_count INTEGER NOT NULL DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_bumped_at TIMESTAMP,
    UNIQUE(bucket_key, model_tier)
);

CREATE INDEX idx_reputation_bucket ON reputation(bucket_key);
```

## Storage Schema for Adaptations

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
```

## Signal Collection API

```python
class SignalCollector:
    def __init__(self, storage: Storage):
        self.storage = storage

    async def on_new_prompt(
        self,
        prompt: str,
        context: Optional[ConversationContext]
    ) -> Optional[Signal]:
        """Called before routing. Detects hard/soft regeneration."""
        ...

    async def on_conversation_turn(
        self,
        conversation_id: str,
        turn_number: int,
        last_decision_id: str,
        next_message: str,
    ) -> Optional[Signal]:
        """Called when a new turn arrives. Detects explicit correction."""
        ...

    async def on_conversation_close(
        self,
        conversation_id: str,
        last_decision_id: str,
    ) -> Optional[Signal]:
        """Called when conversation ends or times out. Detects acceptance."""
        ...

    async def report_manual_signal(
        self,
        decision_id: str,
        signal_type: str,
    ) -> None:
        """Explicit API for apps with thumbs up/down UI."""
        ...
```

## Fire-and-Forget Guarantee

All signal storage operations are wrapped in `asyncio.create_task()` with error handling. They never block the response to the user. If storage fails, the signal is dropped and a warning is logged.

```python
async def _store_signal_safe(signal: Signal):
    try:
        await self.storage.store_signal(signal)
    except Exception as e:
        logger.warning(f"Failed to store signal: {e}")

# Usage:
asyncio.create_task(_store_signal_safe(signal))
```

## Testing Requirements

- Unit test each detector with 10 positive and 10 negative examples
- Test EMA convergence: after 10 hard_regens, EMA should be <0.3
- Test cooldown: bump should not fire twice within 5 minutes
- Test signal priority: hard_regen overrides acceptance for same decision
- Test fire-and-forget: storage failure does not raise
- Edge case: empty prompt history (no signals possible)
- Edge case: decision_id not found in storage (graceful skip)
