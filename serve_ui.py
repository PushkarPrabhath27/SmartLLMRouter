"""SmartRoute Presentation Server & API Bridge.

Serves the production-grade inspection UI and provides live REST endpoints
bridging the frontend dashboard to the embedded SmartRoute library.

Usage:
    uv run python serve_ui.py [--port 8765]
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import sys
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# Ensure UTF-8 console output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Point to project root
ROOT_DIR = Path(__file__).resolve().parent
UI_DIR = ROOT_DIR / "ui"

# Import SmartRoute library
from smartroute import ConversationContext, Router
from smartroute.classifier.classifier import HeuristicClassifier
from smartroute.config.loader import ConfigLoader

logger = logging.getLogger("smartroute_ui")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class SmartRouteBridge:
    """Manages an active Router instance and SQLite introspection."""

    def __init__(self, config_path: str | None = None) -> None:
        self.config_path = config_path or self._detect_config()
        self.router = Router(config_path=self.config_path)
        self.classifier = HeuristicClassifier()
        self.contexts: dict[str, ConversationContext] = {}

    def _detect_config(self) -> str | None:
        for c in ["smartroute.local.yaml", "smartroute.yaml"]:
            p = ROOT_DIR / c
            if p.is_file():
                return str(p)
        return None

    def get_context(self, conversation_id: str) -> ConversationContext:
        if conversation_id not in self.contexts:
            self.contexts[conversation_id] = ConversationContext(conversation_id=conversation_id)
        return self.contexts[conversation_id]

    def get_sqlite_path(self) -> Path:
        raw = self.router.storage_path
        if raw == ":memory:":
            return ROOT_DIR / ".smartroute" / "db.sqlite"
        return Path(raw)

    def fetch_database_records(self) -> dict[str, Any]:
        """Fetch decisions, reputation, and signals directly from SQLite."""
        db_path = self.get_sqlite_path()
        if not db_path.is_file():
            return {"decisions": [], "reputation": [], "signals": []}

        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            # 15 most recent decisions
            cur.execute(
                """SELECT id, timestamp, prompt_hash, prompt_preview, task_type,
                          complexity, complexity_bucket, confidence, model_used,
                          provider_key, estimated_cost_usd, latency_ms, was_adapted,
                          override_applied, reason
                   FROM decisions ORDER BY timestamp DESC LIMIT 15"""
            )
            decisions = [dict(r) for r in cur.fetchall()]

            # Active reputation scores
            cur.execute(
                """SELECT bucket_key, model_tier, ema_score, call_count,
                          last_updated, last_bumped_at
                   FROM reputation ORDER BY bucket_key ASC"""
            )
            reputation = [dict(r) for r in cur.fetchall()]

            # Recent feedback signals
            cur.execute(
                """SELECT id, decision_id, signal_type, signal_value,
                          detected_at, detection_method
                   FROM signals ORDER BY detected_at DESC LIMIT 15"""
            )
            signals = [dict(r) for r in cur.fetchall()]

            conn.close()
            return {
                "decisions": decisions,
                "reputation": reputation,
                "signals": signals,
            }
        except Exception as exc:
            logger.warning("failed to read SQLite records: %s", exc)
            return {"decisions": [], "reputation": [], "signals": []}


# Global bridge singleton
BRIDGE = SmartRouteBridge()


class PresentationRequestHandler(SimpleHTTPRequestHandler):
    """Serves static dashboard files and handles JSON API calls."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(UI_DIR), **kwargs)

    def end_headers(self) -> None:
        # Prevent aggressive browser caching of live prototype assets
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/stats":
            self._handle_get_stats()
            return
        # Fall back to serving static frontend from ui/
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        payload_bytes = self.rfile.read(length) if length > 0 else b"{}"

        try:
            payload = json.loads(payload_bytes.decode("utf-8")) if payload_bytes else {}
        except json.JSONDecodeError:
            self._send_json({"error": "invalid json payload"}, status=400)
            return

        if parsed.path == "/api/classify":
            self._handle_classify(payload)
        elif parsed.path == "/api/route":
            self._handle_route(payload)
        elif parsed.path == "/api/feedback":
            self._handle_feedback(payload)
        elif parsed.path == "/api/reset":
            self._handle_reset(payload)
        else:
            self._send_json({"error": "endpoint not found"}, status=404)

    # ------------------------------------------------------------------
    # Endpoint Handlers
    # ------------------------------------------------------------------

    def _handle_get_stats(self) -> None:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            report = loop.run_until_complete(BRIDGE.router.report())
            db_records = BRIDGE.fetch_database_records()

            data = {
                "report": {
                    "total_decisions": report.total_decisions,
                    "total_cost_usd": report.total_cost_usd,
                    "average_latency_ms": report.average_latency_ms,
                    "model_distribution": report.model_distribution,
                    "bucket_distribution": report.bucket_distribution,
                    "adapted_buckets": report.adapted_buckets,
                },
                "decisions": db_records["decisions"],
                "reputation": db_records["reputation"],
                "signals": db_records["signals"],
                "config_loaded": bool(BRIDGE.config_path),
                "config_file": BRIDGE.config_path,
            }
            self._send_json(data)
        except Exception as exc:
            logger.exception("stats error: %s", exc)
            self._send_json({"error": str(exc)}, status=500)
        finally:
            loop.close()

    def _handle_classify(self, payload: dict[str, Any]) -> None:
        prompt = payload.get("prompt", "")
        t0 = time.perf_counter()
        res = BRIDGE.classifier.classify(prompt)
        dt_ms = (time.perf_counter() - t0) * 1000

        feats = res.features
        data = {
            "task_type": res.task_type.value,
            "complexity": round(res.complexity, 3),
            "complexity_bucket": res.complexity_bucket.value,
            "confidence": round(res.confidence, 3),
            "classification_latency_ms": round(dt_ms, 3),
            "features": {
                "token_count": feats.token_count,
                "code_block_ratio": round(feats.code_block_ratio, 3),
                "is_question": feats.is_question,
                "is_instruction": feats.is_instruction,
                "instruction_verb_count": feats.instruction_verb_count,
                "multi_step_count": feats.multi_step_count,
                "ambiguity_score": round(feats.ambiguity_score, 3),
                "domain": feats.domain_hint.domain or "general",
                "domain_match_ratio": round(feats.domain_hint.match_ratio, 3),
                "file_path_count": feats.file_path_count,
                "urgency_count": feats.urgency_count,
            },
        }
        self._send_json(data)

    def _handle_route(self, payload: dict[str, Any]) -> None:
        prompt = payload.get("prompt", "")
        conv_id = payload.get("conversation_id", "demo_session")
        simulate_override = payload.get("override_provider")

        context = BRIDGE.get_context(conv_id)

        # Pre-classify for step visualizer
        t_class_start = time.perf_counter()
        classification = BRIDGE.classifier.classify(prompt)
        class_latency_ms = (time.perf_counter() - t_class_start) * 1000

        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)

            # Optional programmatic override simulation
            orig_hook = BRIDGE.router._engine.override_hook
            if simulate_override:
                BRIDGE.router._engine.override_hook = lambda p, c: simulate_override

            t0 = time.perf_counter()
            routing_res = loop.run_until_complete(
                BRIDGE.router.complete(prompt, context=context)
            )
            total_latency_ms = (time.perf_counter() - t0) * 1000

            # Restore original hook
            BRIDGE.router._engine.override_hook = orig_hook

            meta = routing_res.meta
            feats = classification.features

            # Fetch updated DB records
            db_records = BRIDGE.fetch_database_records()

            data = {
                "text": routing_res.text,
                "meta": {
                    "model": meta.model,
                    "task_type": meta.task_type,
                    "complexity": round(meta.complexity, 3),
                    "complexity_bucket": meta.complexity_bucket,
                    "confidence": round(meta.confidence, 3),
                    "reason": meta.reason,
                    "reputation_score": round(meta.reputation_score, 3) if meta.reputation_score is not None else 1.0,
                    "was_adapted": meta.was_adapted,
                    "override_applied": meta.override_applied,
                    "estimated_cost_usd": meta.estimated_cost_usd,
                    "provider_latency_ms": meta.latency_ms,
                    "total_latency_ms": round(total_latency_ms, 1),
                    "classification_latency_ms": round(class_latency_ms, 2),
                    "decision_id": meta.decision_id,
                },
                "features": {
                    "token_count": feats.token_count,
                    "code_block_ratio": round(feats.code_block_ratio, 3),
                    "is_question": feats.is_question,
                    "is_instruction": feats.is_instruction,
                    "instruction_verb_count": feats.instruction_verb_count,
                    "multi_step_count": feats.multi_step_count,
                    "ambiguity_score": round(feats.ambiguity_score, 3),
                    "domain": feats.domain_hint.domain or "general",
                    "domain_match_ratio": round(feats.domain_hint.match_ratio, 3),
                },
                "db_records": db_records,
            }
            self._send_json(data)
        except Exception as exc:
            logger.exception("route error: %s", exc)
            self._send_json({"error": str(exc)}, status=500)
        finally:
            loop.close()

    def _handle_feedback(self, payload: dict[str, Any]) -> None:
        decision_id = payload.get("decision_id", "")
        signal_type = payload.get("signal_type", "thumbs_down")
        bucket_key = payload.get("bucket_key", "general_low")
        model_tier = payload.get("model_tier", "low")

        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(
                BRIDGE.router.report_signal(decision_id=decision_id, signal_type=signal_type)
            )
            # Flush background signal processor
            loop.run_until_complete(BRIDGE.router._collector.wait_pending())

            db_records = BRIDGE.fetch_database_records()
            self._send_json({
                "status": "success",
                "signal_type": signal_type,
                "decision_id": decision_id,
                "db_records": db_records,
            })
        except Exception as exc:
            logger.exception("feedback error: %s", exc)
            self._send_json({"error": str(exc)}, status=500)
        finally:
            loop.close()

    def _handle_reset(self, payload: dict[str, Any]) -> None:
        bucket_key = payload.get("bucket_key")
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(BRIDGE.router.reset_reputation(bucket_key))
            db_records = BRIDGE.fetch_database_records()
            self._send_json({"status": "reset", "db_records": db_records})
        except Exception as exc:
            logger.exception("reset error: %s", exc)
            self._send_json({"error": str(exc)}, status=500)
        finally:
            loop.close()


def run_server(port: int = 8765) -> None:
    server_address = ("127.0.0.1", port)
    httpd = ThreadingHTTPServer(server_address, PresentationRequestHandler)
    print("\n" + "=" * 65)
    print("  SmartRoute Production Presentation Dashboard")
    print(f"  Live UI: http://127.0.0.1:{port}/")
    print(f"  Under-the-Hood API: http://127.0.0.1:{port}/api/stats")
    print(f"  Config Loaded: {BRIDGE.config_path or 'Default In-Memory/Environment'}")
    print("=" * 65 + "\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping presentation server...")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    port = 8765
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    run_server(port)
