"""Reputation updater: applies detected signals to the reputation EMA (spec 07).

Fetches existing reputation scores, computes EMA updates, persists signals and
reputation rows, and triggers auto-bump adaptations when thresholds are crossed.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from smartroute.config.schema import Config
from smartroute.routing.reputation import (
    get_next_tier,
    should_bump,
    update_ema,
)
from smartroute.storage.connection import Storage
from smartroute.types import Signal

logger = logging.getLogger(__name__)


async def apply_signal(
    bucket_key: str,
    model_tier: str,
    signal: Signal,
    storage: Storage,
    config: Config,
) -> None:
    """Apply an implicit or manual feedback signal to a bucket's reputation.

    Args:
        bucket_key: Bucket key (e.g. 'code_low').
        model_tier: Complexity tier (e.g. 'low').
        signal: Feedback Signal containing signal_type and numeric value.
        storage: Storage facade for SQLite queries.
        config: Validated configuration containing adaptation settings.
    """
    if not config.adaptation.enabled:
        return

    # 2. Fetch existing reputation or initialize baseline
    rep = await storage.get_reputation(bucket_key, model_tier)
    if rep is not None:
        old_ema = rep.ema_score
        call_count = rep.call_count
        last_bumped_at = rep.last_bumped_at
    else:
        old_ema = 0.5
        call_count = 0
        last_bumped_at = None

    # 3. Compute new EMA and increment call count
    alpha = config.adaptation.ema_alpha
    new_ema = update_ema(old_ema, signal.value, alpha=alpha)
    new_count = call_count + 1

    # 4. Save updated reputation
    await storage.update_reputation(
        bucket_key,
        model_tier,
        ema=new_ema,
        call_count=new_count,
        last_bumped_at=last_bumped_at,
    )

    # 5. Check if reputation has dropped below threshold to trigger an auto-bump
    bump = should_bump(
        ema=new_ema,
        call_count=new_count,
        threshold=config.adaptation.bump_threshold,
        last_bumped_at=last_bumped_at,
        cooldown_minutes=config.adaptation.cooldown_minutes,
        min_calls=config.adaptation.min_calls_before_bump,
    )

    if bump:
        new_tier = get_next_tier(model_tier)
        now = datetime.now(timezone.utc)
        await storage.record_adaptation(
            bucket_key, old_tier=model_tier, new_tier=new_tier, ema_at_bump=new_ema
        )
        await storage.update_reputation(
            bucket_key,
            model_tier,
            ema=new_ema,
            call_count=new_count,
            last_bumped_at=now,
        )
        logger.info(
            "Bucket %s bumped from %s to %s (EMA: %.2f)",
            bucket_key,
            model_tier,
            new_tier,
            new_ema,
        )
