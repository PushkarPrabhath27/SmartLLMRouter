"""Integration and unit tests for SignalCollector and reputation updater (Phase 5)."""

from __future__ import annotations

import pytest

from smartroute.config.schema import (
    AdaptationConfig,
    Config,
    ProviderConfig,
    RoutingConfig,
)
from smartroute.signals.collector import SignalCollector
from smartroute.signals.reputation_updater import apply_signal
from smartroute.storage.connection import Storage
from smartroute.types import DecisionRecord, Signal


@pytest.fixture
def test_config() -> Config:
    providers = {
        "groq": ProviderConfig(api_key="k1", model="llama-3.1-8b"),
        "openai": ProviderConfig(api_key="k2", model="gpt-4o-mini"),
        "anthropic": ProviderConfig(api_key="k3", model="claude-3-sonnet"),
    }
    routing = RoutingConfig(
        low_complexity="groq",
        medium_complexity="openai",
        high_complexity="anthropic",
    )
    adaptation = AdaptationConfig(
        enabled=True,
        bump_threshold=0.3,
        cooldown_minutes=5,
        ema_alpha=0.3,
        min_calls_before_bump=10,
    )
    return Config(
        providers=providers,
        routing=routing,
        adaptation=adaptation,
    )


@pytest.fixture
async def in_memory_storage() -> Storage:
    storage = Storage(":memory:")
    await storage.connect()
    try:
        yield storage
    finally:
        await storage.close()


async def _store_dummy_decision(
    storage: Storage,
    decision_id: str,
    prompt: str = "test",
    task_type: str = "general",
    complexity_bucket: str = "low",
) -> DecisionRecord:
    dec = DecisionRecord(
        id=decision_id,
        prompt_hash=f"hash_{decision_id}",
        task_type=task_type,
        complexity=0.2,
        complexity_bucket=complexity_bucket,
        confidence=0.9,
        model_used="groq/llama-3.1-8b",
        provider_key="groq",
        reason="default",
    )
    await storage.store_decision(dec)
    return dec


class TestReputationUpdater:
    @pytest.mark.asyncio
    async def test_ema_convergence_after_10_hard_regens(
        self, in_memory_storage: Storage, test_config: Config
    ) -> None:
        """Verify EMA converges to < 0.3 after 10 hard regenerations (-0.3)."""
        bucket_key = "code_low"
        model_tier = "low"

        # Apply 10 negative signals
        for i in range(10):
            sig = Signal(signal_type="hard_regen", value=-0.3, decision_id=f"d_{i}")
            await apply_signal(bucket_key, model_tier, sig, in_memory_storage, test_config)

        rep = await in_memory_storage.get_reputation(bucket_key, model_tier)
        assert rep is not None
        assert rep.call_count == 10
        # EMA starts at 0.5 and converges toward -0.3 (clamped at 0.0)
        # mathematically: 0.5 * 0.7^10 + (-0.3) * (1 - 0.7^10) < 0.0 -> clamped to 0.0
        assert rep.ema_score < 0.3

        # Auto-bump should have triggered because EMA < 0.3 and call_count >= 10
        adaptations = await in_memory_storage.get_adaptations(bucket_key=bucket_key)
        assert len(adaptations) == 1
        assert adaptations[0].old_tier == "low"
        assert adaptations[0].new_tier == "medium"

    @pytest.mark.asyncio
    async def test_acceptance_signal_increases_ema(
        self, in_memory_storage: Storage, test_config: Config
    ) -> None:
        bucket_key = "code_low"
        model_tier = "low"
        sig = Signal(signal_type="acceptance", value=0.05, decision_id="d_acc")
        await apply_signal(bucket_key, model_tier, sig, in_memory_storage, test_config)

        rep = await in_memory_storage.get_reputation(bucket_key, model_tier)
        assert rep is not None
        assert rep.call_count == 1
        # 0.3 * 0.05 + 0.7 * 0.5 = 0.015 + 0.35 = 0.365
        assert pytest.approx(rep.ema_score, abs=1e-4) == 0.365


class TestSignalCollector:
    @pytest.mark.asyncio
    async def test_on_new_prompt_detects_hard_regen(
        self, in_memory_storage: Storage, test_config: Config
    ) -> None:
        await _store_dummy_decision(
            in_memory_storage, "dec_1", "write quicksort in python", "code", "low"
        )
        collector = SignalCollector(storage=in_memory_storage, config=test_config)
        collector.register_prompt(
            decision_id="dec_1",
            prompt="write quicksort in python",
            task_type="code",
            complexity_bucket="low",
        )

        signal = await collector.on_new_prompt("write quicksort in python")
        assert signal is not None
        assert signal.signal_type == "hard_regen"
        assert signal.decision_id == "dec_1"

        # Wait for fire-and-forget task to complete
        await collector.wait_pending()

        # Check reputation was updated
        rep = await in_memory_storage.get_reputation("code_low", "low")
        assert rep is not None
        assert rep.call_count == 1

        # Check signal was persisted
        signals = await in_memory_storage.get_signals_for_decision("dec_1")
        assert len(signals) == 1
        assert signals[0].signal_type == "hard_regen"

    @pytest.mark.asyncio
    async def test_on_conversation_turn_explicit_correction(
        self, in_memory_storage: Storage, test_config: Config
    ) -> None:
        await _store_dummy_decision(in_memory_storage, "dec_turn", "hello world", "general", "low")
        collector = SignalCollector(storage=in_memory_storage, config=test_config)
        collector.register_prompt(
            decision_id="dec_turn",
            prompt="hello world",
            task_type="general",
            complexity_bucket="low",
        )

        signal = await collector.on_conversation_turn(
            conversation_id="c1",
            turn_number=1,
            last_decision_id="dec_turn",
            next_message="No, that's completely wrong and doesn't work",
        )
        assert signal is not None
        assert signal.signal_type == "explicit_correction"

        await collector.wait_pending()
        signals = await in_memory_storage.get_signals_for_decision("dec_turn")
        assert len(signals) == 1
        assert signals[0].signal_type == "explicit_correction"

    @pytest.mark.asyncio
    async def test_on_conversation_turn_acceptance(
        self, in_memory_storage: Storage, test_config: Config
    ) -> None:
        await _store_dummy_decision(in_memory_storage, "dec_turn2", "hello", "general", "low")
        collector = SignalCollector(storage=in_memory_storage, config=test_config)
        collector.register_prompt(
            decision_id="dec_turn2",
            prompt="hello",
            task_type="general",
            complexity_bucket="low",
        )

        signal = await collector.on_conversation_turn(
            conversation_id="c1",
            turn_number=1,
            last_decision_id="dec_turn2",
            next_message="Great, thank you! Now can you explain how it works?",
        )
        assert signal is not None
        assert signal.signal_type == "acceptance"

        await collector.wait_pending()
        signals = await in_memory_storage.get_signals_for_decision("dec_turn2")
        assert len(signals) == 1
        assert signals[0].signal_type == "acceptance"

    @pytest.mark.asyncio
    async def test_on_conversation_close_records_acceptance(
        self, in_memory_storage: Storage, test_config: Config
    ) -> None:
        await _store_dummy_decision(in_memory_storage, "dec_close", "hello", "general", "low")
        collector = SignalCollector(storage=in_memory_storage, config=test_config)
        collector.register_prompt(
            decision_id="dec_close",
            prompt="hello",
            task_type="general",
            complexity_bucket="low",
        )

        signal = await collector.on_conversation_close("c1", "dec_close")
        assert signal is not None
        assert signal.signal_type == "acceptance"

        await collector.wait_pending()
        signals = await in_memory_storage.get_signals_for_decision("dec_close")
        assert len(signals) == 1

    @pytest.mark.asyncio
    async def test_report_manual_signal_thumbs_up_and_down(
        self, in_memory_storage: Storage, test_config: Config
    ) -> None:
        # Pre-store a decision in SQLite
        dec = DecisionRecord(
            id="dec_manual",
            prompt_hash="dummy_hash",
            task_type="code",
            complexity=0.2,
            complexity_bucket="low",
            confidence=0.9,
            model_used="groq/llama-3.1-8b",
            provider_key="groq",
            reason="default",
        )
        await in_memory_storage.store_decision(dec)

        collector = SignalCollector(storage=in_memory_storage, config=test_config)
        await collector.report_manual_signal(decision_id="dec_manual", signal_type="thumbs_down")
        await collector.wait_pending()

        signals = await in_memory_storage.get_signals_for_decision("dec_manual")
        assert len(signals) == 1
        assert signals[0].signal_type == "explicit_correction"
        assert signals[0].signal_value == -0.2
        assert signals[0].detection_method == "manual"

    @pytest.mark.asyncio
    async def test_fire_and_forget_resilience_to_storage_errors(
        self, in_memory_storage: Storage, test_config: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Storage failure in background task must not raise to caller."""

        async def broken_store(*args: object, **kwargs: object) -> None:
            raise RuntimeError("disk exploded")

        monkeypatch.setattr(in_memory_storage, "store_signal", broken_store)
        collector = SignalCollector(storage=in_memory_storage, config=test_config)
        collector.register_prompt(
            decision_id="dec_fail",
            prompt="test",
            task_type="code",
            complexity_bucket="low",
        )

        # Calling on_new_prompt must return successfully without throwing
        signal = await collector.on_new_prompt("test")
        assert signal is not None
        await collector.wait_pending()

    @pytest.mark.asyncio
    async def test_adaptation_disabled_only_persists_signal(
        self, in_memory_storage: Storage, test_config: Config
    ) -> None:
        await _store_dummy_decision(
            in_memory_storage, "dec_disabled", "do something", "code", "low"
        )
        disabled_config = Config(
            providers=test_config.providers,
            routing=test_config.routing,
            adaptation=AdaptationConfig(enabled=False),
        )
        collector = SignalCollector(storage=in_memory_storage, config=disabled_config)
        collector.register_prompt(
            decision_id="dec_disabled",
            prompt="do something",
            task_type="code",
            complexity_bucket="low",
        )
        await collector.report_manual_signal("dec_disabled", "thumbs_down")
        await collector.wait_pending()

        # Signal stored
        signals = await in_memory_storage.get_signals_for_decision("dec_disabled")
        assert len(signals) == 1

        # But reputation untouched
        rep = await in_memory_storage.get_reputation("code_low", "low")
        assert rep is None

    @pytest.mark.asyncio
    async def test_auto_bump_cooldown_prevents_immediate_second_bump(
        self, in_memory_storage: Storage, test_config: Config
    ) -> None:
        bucket_key = "code_low"
        model_tier = "low"

        # Apply 10 negative signals to trigger first bump
        for i in range(10):
            sig = Signal("hard_regen", -0.3, f"d_{i}")
            await apply_signal(bucket_key, model_tier, sig, in_memory_storage, test_config)

        adaptations = await in_memory_storage.get_adaptations(bucket_key=bucket_key)
        assert len(adaptations) == 1

        # Apply 5 more negative signals immediately within cooldown (< 5 min)
        for i in range(10, 15):
            sig = Signal("hard_regen", -0.3, f"d_{i}")
            await apply_signal(bucket_key, model_tier, sig, in_memory_storage, test_config)

        # Still only 1 adaptation record because in cooldown
        adaptations2 = await in_memory_storage.get_adaptations(bucket_key=bucket_key)
        assert len(adaptations2) == 1

    @pytest.mark.asyncio
    async def test_unknown_decision_id_gracefully_skipped(
        self, in_memory_storage: Storage, test_config: Config
    ) -> None:
        collector = SignalCollector(storage=in_memory_storage, config=test_config)
        # Not registered in memory and not in storage
        await collector.report_manual_signal("non_existent_id", "thumbs_up")
        await collector.wait_pending()

        # Per spec 07 line 270: decision_id not found in storage is gracefully skipped
        signals = await in_memory_storage.get_signals_for_decision("non_existent_id")
        assert len(signals) == 0

    def test_history_trimmed_to_20_entries(
        self, in_memory_storage: Storage, test_config: Config
    ) -> None:
        collector = SignalCollector(storage=in_memory_storage, config=test_config)
        for i in range(25):
            collector.register_prompt(f"d_{i}", f"prompt {i}", "general", "low")

        assert len(collector._history) == 20
        assert collector._history[-1].decision_id == "d_24"
        assert collector._history[0].decision_id == "d_5"
