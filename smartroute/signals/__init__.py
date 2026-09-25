"""Implicit feedback signal collection and reputation updates."""

from smartroute.signals.collector import SignalCollector
from smartroute.signals.detectors import (
    ACCEPTANCE_VALUE,
    EXPLICIT_CORRECTION_VALUE,
    HARD_REGEN_VALUE,
    SIGNAL_PRIORITY,
    SIGNAL_VALUES,
    SOFT_REGEN_VALUE,
    HistoryItem,
    detect_acceptance,
    detect_explicit_correction,
    detect_hard_regen,
    detect_soft_regen,
    resolve_strongest_signal,
)
from smartroute.signals.reputation_updater import apply_signal

__all__ = [
    "ACCEPTANCE_VALUE",
    "EXPLICIT_CORRECTION_VALUE",
    "HARD_REGEN_VALUE",
    "SIGNAL_PRIORITY",
    "SIGNAL_VALUES",
    "SOFT_REGEN_VALUE",
    "HistoryItem",
    "SignalCollector",
    "apply_signal",
    "detect_acceptance",
    "detect_explicit_correction",
    "detect_hard_regen",
    "detect_soft_regen",
    "resolve_strongest_signal",
]
