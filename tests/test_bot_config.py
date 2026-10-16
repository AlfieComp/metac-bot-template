"""Unit tests for the ensemble, the spend cap, and the research guards.

These tests do not call OpenRouter, AskNews, or Metaculus.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest import mock

import bot_helpers
from bot_helpers import (
    DEEPSEEK_ONLY_FORECASTER_MODELS,
    DEEPSEEK_V4_FLASH_ID,
    DEEPSEEK_V4_PRO_ID,
    GEMINI_36_FLASH_ID,
    KIMI_K3_ID,
    MIXED_FORECASTER_MODELS,
    OPENROUTER_DEEPSEEK_V4_FLASH,
    OPENROUTER_DEEPSEEK_V4_PRO,
    OPENROUTER_GEMINI_36_FLASH,
    OPENROUTER_KIMI_K3,
    SPEND_GAP_CAP_USD,
    SPEND_HARD_CAP_USD,
    SPEND_SWITCH_USD,
    SpendTracker,
    asknews_configured,
    call_would_pass_hard_cap,
    clip_and_renormalize,
    clip_probability,
    forecaster_models,
    median_binary,
    median_option_probabilities,
    openrouter_model_id,
    redact_secrets,
    resolution_checklist_prompt,
    strip_market_and_crowd_odds,
    use_deepseek_only,
)


class ModelRosterTests(unittest.TestCase):
    def test_exact_openrouter_ids(self) -> None:
        self.assertEqual(openrouter_model_id(OPENROUTER_GEMINI_36_FLASH), GEMINI_36_FLASH_ID)
        self.assertEqual(openrouter_model_id(OPENROUTER_KIMI_K3), KIMI_K3_ID)
        self.assertEqual(openrouter_model_id(OPENROUTER_DEEPSEEK_V4_PRO), DEEPSEEK_V4_PRO_ID)
        self.assertEqual(
            openrouter_model_id(OPENROUTER_DEEPSEEK_V4_FLASH), DEEPSEEK_V4_FLASH_ID
        )
        self.assertEqual(GEMINI_36_FLASH_ID, "google/gemini-3.6-flash")
        self.assertEqual(KIMI_K3_ID, "moonshotai/kimi-k3")
        self.assertEqual(DEEPSEEK_V4_PRO_ID, "deepseek/deepseek-v4-pro")
        self.assertEqual(DEEPSEEK_V4_FLASH_ID, "deepseek/deepseek-v4-flash")

    def test_mixed_roster_is_three_gemini_one_kimi_one_deepseek(self) -> None:
        roster = forecaster_models(0.0)
        self.assertEqual(len(roster), 5)
        self.assertEqual(roster, MIXED_FORECASTER_MODELS)
        self.assertEqual(roster.count(OPENROUTER_GEMINI_36_FLASH), 3)
        self.assertEqual(roster.count(OPENROUTER_KIMI_K3), 1)
        self.assertEqual(roster.count(OPENROUTER_DEEPSEEK_V4_PRO), 1)

    def test_spend_switch_uses_five_deepseek_pro(self) -> None:
        self.assertFalse(use_deepseek_only(SPEND_SWITCH_USD - 0.01))
        self.assertTrue(use_deepseek_only(SPEND_SWITCH_USD))
        self.assertEqual(forecaster_models(3.49), MIXED_FORECASTER_MODELS)
        self.assertEqual(forecaster_models(3.50), DEEPSEEK_ONLY_FORECASTER_MODELS)
        self.assertEqual(forecaster_models(4.00).count(OPENROUTER_DEEPSEEK_V4_PRO), 5)

    def test_hard_cap_blocks_the_next_call(self) -> None:
        self.assertFalse(call_would_pass_hard_cap(4.48, 0.011))
        self.assertTrue(call_would_pass_hard_cap(4.49, 0.02))
        self.assertEqual(SPEND_HARD_CAP_USD, 4.50)
        self.assertEqual(SPEND_SWITCH_USD, 3.50)


class ProbabilityTests(unittest.TestCase):
    def test_clip_binary_probability(self) -> None:
        self.assertEqual(clip_probability(0.0), 0.01)
        self.assertEqual(clip_probability(1.0), 0.99)
        self.assertEqual(clip_probability(0.42), 0.42)
        self.assertEqual(median_binary([0.0, 0.0, 0.2, 0.4, 0.9]), 0.2)
        self.assertEqual(median_binary([0.0, 0.0, 0.0]), 0.01)

    def test_multiple_choice_median_is_clipped_and_sums_to_one(self) -> None:
        merged = median_option_probabilities(
            [
                {"yes": 0.0, "no": 1.0},
                {"yes": 0.2, "no": 0.8},
                {"yes": 0.4, "no": 0.6},
                {"yes": 1.0, "no": 0.0},
                {"yes": 0.1, "no": 0.9},
            ]
        )
        self.assertAlmostEqual(sum(merged.values()), 1.0)
        self.assertGreaterEqual(min(merged.values()), 0.01)
        self.assertLessEqual(max(merged.values()), 0.99)
        # Sorted values are 0.0, 0.1, 0.2, 0.4, 1.0. The median is 0.2, clipped.
        self.assertAlmostEqual(merged["yes"], 0.2)

    def test_clip_and_renormalize_extremes(self) -> None:
        clipped = clip_and_renormalize([0.0, 1.0])
        self.assertAlmostEqual(sum(clipped), 1.0)
        self.assertEqual(clipped, [0.01, 0.99])


class ResearchGuardTests(unittest.TestCase):
    def test_checklist_quotes_fine_print_and_does_not_forecast(self) -> None:
        prompt = resolution_checklist_prompt(
            question_text="Will the bill pass?",
            resolution_criteria="Resolves Yes if the clerk records a vote.",
            fine_print="ANNUL-MARKER if the vote is postponed.",
            background="Background text.",
            today="2026-10-16",
        )
        self.assertIn("ANNUL-MARKER", prompt)
        self.assertIn("Resolves Yes if the clerk records a vote.", prompt)
        self.assertIn("Do not give a probability", prompt)
        self.assertIn("Do not forecast", prompt)

    def test_strip_crowd_and_market_odds(self) -> None:
        text = "\n".join(
            [
                "The committee advanced the bill.",
                "Polymarket traders price this at 62%.",
                "The Metaculus community prediction is 70%.",
                "Polymarket listed a new contract on the election.",
                "Kalshi implied probability is 55 percent.",
            ]
        )
        stripped, removed = strip_market_and_crowd_odds(text)
        self.assertEqual(removed, 3)
        self.assertIn("committee advanced", stripped)
        self.assertIn("listed a new contract", stripped)
        self.assertNotIn("62%", stripped)
        self.assertNotIn("community prediction", stripped.lower())
        self.assertIn("were removed", stripped)

    def test_asknews_skipped_without_keys(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(asknews_configured())

    def test_asknews_accepts_either_auth_style(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"ASKNEWS_CLIENT_ID": "client", "ASKNEWS_SECRET": "secret"},
            clear=True,
        ):
            self.assertTrue(asknews_configured())
        with mock.patch.dict(os.environ, {"ASKNEWS_API_KEY": "key"}, clear=True):
            self.assertTrue(asknews_configured())

    def test_redact_secrets_removes_env_values_and_key_shapes(self) -> None:
        with mock.patch.dict(
            os.environ, {"OPENROUTER_API_KEY": "sk-or-v1-super-secret-value"}, clear=True
        ):
            redacted = redact_secrets(
                "failed with sk-or-v1-super-secret-value and Bearer sk-or-v1-super-secret-value"
            )
        self.assertNotIn("super-secret", redacted)
        self.assertIn("[redacted]", redacted)


class SpendTrackerTests(unittest.TestCase):
    def test_remote_reading_replaces_an_inflated_gap(self) -> None:
        partial = SpendTracker(
            path=Path("/tmp/does-not-matter.json"),
            remote_usd=1.0,
            gap_usd=0.50,
            fetch_usage=lambda: 1.20,
        )
        partial.refresh()
        self.assertEqual(partial.remote_usd, 1.20)
        self.assertAlmostEqual(partial.gap_usd, SPEND_GAP_CAP_USD)
        self.assertAlmostEqual(partial.effective_usd(), 1.20 + SPEND_GAP_CAP_USD)

        caught_up = SpendTracker(
            path=Path("/tmp/does-not-matter.json"),
            remote_usd=1.0,
            gap_usd=0.50,
            fetch_usage=lambda: 2.0,
        )
        caught_up.refresh()
        self.assertEqual(caught_up.remote_usd, 2.0)
        self.assertEqual(caught_up.gap_usd, 0.0)
        self.assertAlmostEqual(caught_up.effective_usd(), 2.0)

    def test_unchanged_remote_keeps_the_gap(self) -> None:
        tracker = SpendTracker(
            path=Path("/tmp/does-not-matter.json"),
            remote_usd=1.0,
            gap_usd=0.04,
            fetch_usage=lambda: 1.0,
        )
        tracker.refresh()
        self.assertAlmostEqual(tracker.effective_usd(), 1.04)

    def test_ledger_round_trip_has_no_secrets(self) -> None:
        path = Path("/tmp/openrouter_spend_test.json")
        if path.exists():
            path.unlink()
        tracker = SpendTracker(path=path, remote_usd=1.25, gap_usd=0.0, fetch_usage=lambda: None)
        tracker.add_unconfirmed(0.02)
        reloaded = SpendTracker.load(path, fetch_usage=lambda: None)
        self.assertAlmostEqual(reloaded.effective_usd(), 1.27)
        saved = json.loads(path.read_text())
        self.assertEqual(set(saved), {"remote_usd", "gap_usd"})
        path.unlink()

    def test_switch_and_cap_follow_effective_spend(self) -> None:
        tracker = SpendTracker(
            path=Path("/tmp/does-not-matter.json"),
            remote_usd=3.49,
            gap_usd=0.02,
            fetch_usage=lambda: None,
        )
        self.assertTrue(tracker.deepseek_only())
        self.assertFalse(tracker.over_hard_cap())
        self.assertTrue(tracker.would_exceed_cap(2.0))
        capped = SpendTracker(
            path=Path("/tmp/does-not-matter.json"),
            remote_usd=4.50,
            gap_usd=0.0,
            fetch_usage=lambda: None,
        )
        self.assertTrue(capped.over_hard_cap())
        self.assertEqual(capped.mode_label(), "stopped")


class WorkflowTests(unittest.TestCase):
    def test_forecast_workflows_do_not_install_integrations(self) -> None:
        workflow_dir = Path(__file__).resolve().parents[1] / ".github" / "workflows"
        checked = 0
        for path in workflow_dir.glob("*.yaml"):
            if path.name == "review_bot.yaml":
                continue
            text = path.read_text()
            self.assertNotIn("--with integrations", text, path.name)
            checked += 1
        self.assertGreaterEqual(checked, 3)


if __name__ == "__main__":
    unittest.main()
