"""Routing engine: decision hierarchy, reputation integration, and explainability (spec 06).

Evaluates prompt classifications through a strict 4-level decision hierarchy:
Level 1: Programmatic override hook
Level 2: YAML override rules (exact, contains, regex, path)
Level 3: Adaptive reputation score (SQLite EMA lookup and auto-bump)
Level 4: Default complexity tier mapping
"""

from __future__ import annotations

import fnmatch
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from smartroute.config.schema import Config, OverrideRule
from smartroute.routing.explainability import (
    format_default_routing_reason,
    format_override_hook_reason,
    format_reputation_bump_reason,
    format_yaml_rule_reason,
)
from smartroute.routing.reputation import (
    get_bucket_key,
    get_next_tier,
    should_bump,
)
from smartroute.storage.connection import Storage
from smartroute.types import (
    ClassificationResult,
    ConversationContext,
    OverrideHook,
)

logger = logging.getLogger(__name__)

# Hardcoded v1 pricing table (per 1M tokens) per spec 06.
PRICING: dict[str, dict[str, float]] = {
    "groq/llama-3.1-8b": {"input": 0.05, "output": 0.08},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "anthropic/claude-3-sonnet": {"input": 3.00, "output": 15.00},
}


def estimate_cost(prompt_tokens: int, model: str, avg_output_ratio: float = 0.5) -> float:
    """Estimate pre-flight USD cost for a model call (spec 06).

    Args:
        prompt_tokens: Number of input tokens in prompt.
        model: Fully qualified model ID (e.g. 'openai/gpt-4o-mini') or bare model name.
        avg_output_ratio: Estimated ratio of output tokens to prompt tokens (default 0.5).

    Returns:
        Estimated USD cost as a float; 0.0 if model is unknown.
    """
    pricing = PRICING.get(model)
    if pricing is None:
        for key, p in PRICING.items():
            if key.endswith(f"/{model}"):
                pricing = p
                break
    if pricing is None:
        logger.debug("no pricing found for model '%s', defaulting cost to 0.0", model)
        return 0.0

    estimated_output = int(prompt_tokens * avg_output_ratio)
    input_cost = (prompt_tokens / 1_000_000.0) * pricing["input"]
    output_cost = (estimated_output / 1_000_000.0) * pricing["output"]
    return input_cost + output_cost


def _fully_qualified_model(provider_key: str, configured_model: str) -> str:
    """Ensure a model name is fully qualified with its provider prefix."""
    if "/" in configured_model:
        return configured_model
    return f"{provider_key}/{configured_model}"


def _matches_path_rule(rule_pattern: str, prompt: str) -> bool:
    """Check if a path rule pattern matches any path reference or token in prompt."""
    if rule_pattern in prompt:
        return True
    for word in prompt.split():
        clean_word = word.strip("`'\",:;()[]{}")
        if fnmatch.fnmatch(clean_word, rule_pattern):
            return True
    return False


def _matches_rule(rule: OverrideRule, prompt: str) -> bool:
    """Evaluate whether an OverrideRule matches the prompt."""
    if rule.match_type == "exact":
        return prompt == rule.match
    if rule.match_type == "contains":
        return rule.match.lower() in prompt.lower()
    if rule.match_type == "regex":
        return bool(re.search(rule.match, prompt, re.IGNORECASE))
    if rule.match_type == "path":
        return _matches_path_rule(rule.match, prompt)
    return False


@dataclass(frozen=True)
class RoutingDecision:
    """Internal decision result produced by RoutingEngine (spec 06).

    Attributes:
        provider_key: Resolved provider key (e.g. 'openai').
        model: Fully qualified model name (e.g. 'openai/gpt-4o-mini').
        fallback_chain: Ordered list of provider keys for failover.
        reason: Human-readable explanation string.
        reputation_score: Current EMA score for the bucket at decision time.
        was_adapted: True if reputation auto-bump escalated the tier.
        override_applied: Override identifier string, or None.
        estimated_cost_usd: Pre-flight USD cost estimate.
    """

    provider_key: str
    model: str
    fallback_chain: list[str]
    reason: str
    reputation_score: float
    was_adapted: bool
    override_applied: str | None
    estimated_cost_usd: float


class RoutingEngine:
    """Decision-making engine for routing LLM prompts (spec 06).

    Evaluates rules in strict priority order:
    1. Programmatic override hook
    2. YAML rule match (exact, contains, regex, path)
    3. Adaptive reputation score (auto-bump)
    4. Default complexity tier mapping
    """

    def __init__(
        self,
        config: Config,
        storage: Storage | None = None,
        override_hook: OverrideHook | None = None,
    ) -> None:
        """Initialize the routing engine.

        Args:
            config: Validated SmartRoute configuration.
            storage: Optional Storage facade for reputation lookups.
            override_hook: Optional callable for programmatic overrides.
        """
        self.config = config
        self.storage = storage
        self.override_hook = override_hook

    def _build_fallback_chain(self, primary_provider: str, tier: str | None = None) -> list[str]:
        """Construct the ordered fallback chain for a resolved tier/provider."""
        chain: list[str] = [primary_provider]
        if tier and self.config.routing.fallback:
            configured_chain = self.config.routing.fallback.get(tier)
            if configured_chain:
                for p in configured_chain:
                    if p in self.config.providers and p not in chain:
                        chain.append(p)
        return chain

    async def route(
        self,
        prompt: str,
        classification: ClassificationResult,
        context: ConversationContext | None = None,
    ) -> RoutingDecision:
        """Route a classified prompt through the 4-level decision hierarchy.

        Args:
            prompt: The raw prompt string.
            classification: Output from the classifier.
            context: Optional conversation context.

        Returns:
            A frozen RoutingDecision with full explainability metadata.
        """
        tokens = classification.features.token_count

        # --------------------------------------------------------------
        # Level 1: Programmatic Override Hook
        # --------------------------------------------------------------
        if self.override_hook is not None:
            hook_provider: str | None = None
            try:
                hook_provider = self.override_hook(prompt, context)
            except Exception as exc:
                logger.warning("programmatic override_hook failed: %s", exc)

            if hook_provider:
                if hook_provider in self.config.providers:
                    model_name = _fully_qualified_model(
                        hook_provider, self.config.providers[hook_provider].model
                    )
                    chain = self._build_fallback_chain(hook_provider)
                    cost = estimate_cost(tokens, model_name)
                    return RoutingDecision(
                        provider_key=hook_provider,
                        model=model_name,
                        fallback_chain=chain,
                        reason=format_override_hook_reason(hook_provider),
                        reputation_score=0.5,
                        was_adapted=False,
                        override_applied="programmatic_hook",
                        estimated_cost_usd=cost,
                    )
                logger.warning(
                    "override_hook returned unknown provider '%s'; ignoring",
                    hook_provider,
                )

        # --------------------------------------------------------------
        # Level 2: YAML Rule Overrides
        # --------------------------------------------------------------
        for rule in self.config.routing.overrides:
            if _matches_rule(rule, prompt):
                provider_key = rule.model
                model_name = _fully_qualified_model(
                    provider_key, self.config.providers[provider_key].model
                )
                chain = self._build_fallback_chain(provider_key)
                cost = estimate_cost(tokens, model_name)
                override_tag = (
                    f"yaml_rule:{rule.description}"
                    if rule.description
                    else f"yaml_rule:{rule.match_type}"
                )
                return RoutingDecision(
                    provider_key=provider_key,
                    model=model_name,
                    fallback_chain=chain,
                    reason=format_yaml_rule_reason(rule.match_type, rule.match, provider_key),
                    reputation_score=0.5,
                    was_adapted=False,
                    override_applied=override_tag,
                    estimated_cost_usd=cost,
                )

        # --------------------------------------------------------------
        # Level 3: Adaptive Reputation Score
        # --------------------------------------------------------------
        default_tier = classification.complexity_bucket.value
        reputation_score = 0.5

        if self.config.adaptation.enabled and self.storage is not None:
            bucket_key = get_bucket_key(classification.task_type, classification.complexity_bucket)
            try:
                record = await self.storage.get_reputation(bucket_key, default_tier)
            except Exception as exc:
                logger.warning(
                    "failed to read reputation for %s/%s: %s",
                    bucket_key,
                    default_tier,
                    exc,
                )
                record = None

            if record is not None:
                reputation_score = record.ema_score
                call_count = record.call_count
                last_bumped_at = record.last_bumped_at
            else:
                reputation_score = 0.5
                call_count = 0
                last_bumped_at = None

            bump = should_bump(
                ema=reputation_score,
                call_count=call_count,
                threshold=self.config.adaptation.bump_threshold,
                last_bumped_at=last_bumped_at,
                cooldown_minutes=self.config.adaptation.cooldown_minutes,
                min_calls=self.config.adaptation.min_calls_before_bump,
            )

            if bump:
                new_tier = get_next_tier(default_tier)
                now = datetime.now(timezone.utc)
                try:
                    await self.storage.record_adaptation(
                        bucket_key, default_tier, new_tier, reputation_score
                    )
                    await self.storage.update_reputation(
                        bucket_key,
                        default_tier,
                        ema=reputation_score,
                        call_count=call_count,
                        last_bumped_at=now,
                    )
                except Exception as exc:
                    logger.warning("failed to record adaptation event in storage: %s", exc)

                provider_key = getattr(self.config.routing, f"{new_tier}_complexity")
                model_name = _fully_qualified_model(
                    provider_key, self.config.providers[provider_key].model
                )
                chain = self._build_fallback_chain(provider_key, tier=new_tier)
                cost = estimate_cost(tokens, model_name)
                reason = format_reputation_bump_reason(reputation_score, default_tier, new_tier)
                return RoutingDecision(
                    provider_key=provider_key,
                    model=model_name,
                    fallback_chain=chain,
                    reason=reason,
                    reputation_score=reputation_score,
                    was_adapted=True,
                    override_applied=None,
                    estimated_cost_usd=cost,
                )

        # --------------------------------------------------------------
        # Level 4: Default Complexity Tier Mapping
        # --------------------------------------------------------------
        provider_key = getattr(self.config.routing, f"{default_tier}_complexity")
        model_name = _fully_qualified_model(provider_key, self.config.providers[provider_key].model)
        chain = self._build_fallback_chain(provider_key, tier=default_tier)
        cost = estimate_cost(tokens, model_name)
        reason = format_default_routing_reason(
            classification.task_type.value,
            classification.complexity_bucket.value,
            classification.complexity,
            provider_key,
        )
        return RoutingDecision(
            provider_key=provider_key,
            model=model_name,
            fallback_chain=chain,
            reason=reason,
            reputation_score=reputation_score,
            was_adapted=False,
            override_applied=None,
            estimated_cost_usd=cost,
        )
