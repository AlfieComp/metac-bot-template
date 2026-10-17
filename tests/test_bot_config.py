"""Unit tests for the ensemble, the spend cap, and the research guards.

These tests do not call OpenRouter, AskNews, or Metaculus.
"""

from __future__ import annotations

import json
import os
import threading
import unittest
from pathlib import Path
from unittest import mock

import bot_helpers
from bot_helpers import (
    DEEPSEEK_ONLY_FORECASTER_MODELS,
    DEEPSEEK_V4_FLASH_ID,
    DEEPSEEK_V4_PRO_ID,
    GEMINI_36_FLASH_ID,
    GEMINI_MAX_TOKENS,
    KIMI_K3_ID,
    MIN_SUCCESSFUL_FORECASTS,
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
    estimate_call_cost_usd,
    forecaster_models,
    median_binary,
    median_option_probabilities,
    median_percentile_values,
    openrouter_model_id,
    recorded_call_cost_usd,
    redact_secrets,
    required_success_fraction,
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
                "Polymarket traders price this at 62 percent.",
                "Forecasters put the odds of Yes near even.",
                "Market prices have moved this week.",
                "62 percent of the bill's text was amended.",
            ]
        )
        stripped, removed = strip_market_and_crowd_odds(text)
        self.assertEqual(removed, 6)
        self.assertIn("committee advanced", stripped)
        self.assertIn("listed a new contract", stripped)
        self.assertIn("bill's text was amended", stripped)
        self.assertNotIn("62%", stripped)
        self.assertNotIn("62 percent.", stripped)
        self.assertNotIn("odds of", stripped.lower())
        self.assertNotIn("market prices", stripped.lower())
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
            usage_known=True,
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
            usage_known=True,
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
            usage_known=True,
        )
        tracker.refresh()
        self.assertAlmostEqual(tracker.effective_usd(), 1.04)

    def test_ledger_round_trip_has_no_secrets(self) -> None:
        path = Path("/tmp/openrouter_spend_test.json")
        if path.exists():
            path.unlink()
        tracker = SpendTracker(
            path=path,
            remote_usd=1.25,
            gap_usd=0.0,
            fetch_usage=lambda: None,
            usage_known=True,
        )
        tracker.add_unconfirmed(0.02)
        reloaded = SpendTracker.load(path, fetch_usage=lambda: None)
        self.assertTrue(reloaded.usage_known)
        self.assertTrue(reloaded.can_forecast())
        self.assertAlmostEqual(reloaded.effective_usd(), 1.27)
        saved = json.loads(path.read_text())
        self.assertEqual(set(saved), {"remote_usd", "gap_usd", "usage_known"})
        self.assertTrue(saved["usage_known"])
        path.unlink()

    def test_switch_and_cap_follow_effective_spend(self) -> None:
        tracker = SpendTracker(
            path=Path("/tmp/does-not-matter.json"),
            remote_usd=3.49,
            gap_usd=0.02,
            fetch_usage=lambda: None,
            usage_known=True,
        )
        self.assertTrue(tracker.deepseek_only())
        self.assertFalse(tracker.over_hard_cap())
        self.assertTrue(tracker.would_exceed_cap(2.0))
        capped = SpendTracker(
            path=Path("/tmp/does-not-matter.json"),
            remote_usd=4.50,
            gap_usd=0.0,
            fetch_usage=lambda: None,
            usage_known=True,
        )
        self.assertTrue(capped.over_hard_cap())
        self.assertEqual(capped.mode_label(), "stopped")

    def test_unknown_usage_without_a_ledger_cannot_forecast(self) -> None:
        path = Path("/tmp/openrouter_spend_missing.json")
        if path.exists():
            path.unlink()
        tracker = SpendTracker.load(path, fetch_usage=lambda: None)
        tracker.refresh()
        self.assertFalse(tracker.usage_known)
        self.assertFalse(tracker.can_forecast())
        self.assertFalse(tracker.try_reserve(0.01))
        self.assertFalse(path.exists())

    def test_zero_file_without_the_flag_is_not_a_spend_record(self) -> None:
        path = Path("/tmp/openrouter_spend_zero.json")
        path.write_text('{"remote_usd": 0.0, "gap_usd": 0.0}\n')
        tracker = SpendTracker.load(path, fetch_usage=lambda: None)
        tracker.refresh()
        self.assertFalse(tracker.usage_known)
        self.assertFalse(tracker.can_forecast())
        path.unlink()

    def test_api_usage_allows_forecast_and_writes_the_flag(self) -> None:
        path = Path("/tmp/openrouter_spend_api.json")
        if path.exists():
            path.unlink()
        for usage in (0.0, 1.0):
            tracker = SpendTracker(path=path, fetch_usage=lambda usage=usage: usage)
            tracker.refresh()
            self.assertTrue(tracker.usage_known)
            self.assertTrue(tracker.can_forecast())
            self.assertAlmostEqual(tracker.remote_usd, usage)
            saved = json.loads(path.read_text())
            self.assertTrue(saved["usage_known"])
            self.assertAlmostEqual(saved["remote_usd"], usage)
        path.unlink()

    def test_cached_ledger_allows_forecast_when_the_api_fails(self) -> None:
        path = Path("/tmp/openrouter_spend_cached.json")
        path.write_text(
            json.dumps({"remote_usd": 1.5, "gap_usd": 0.1, "usage_known": True}) + "\n"
        )
        tracker = SpendTracker.load(path, fetch_usage=lambda: None)
        tracker.refresh()
        self.assertTrue(tracker.usage_known)
        self.assertTrue(tracker.can_forecast())
        self.assertAlmostEqual(tracker.effective_usd(), 1.6)
        path.unlink()

    def test_parallel_reserves_cannot_pass_the_cap(self) -> None:
        tracker = SpendTracker(
            path=Path("/tmp/does-not-matter.json"),
            remote_usd=4.30,
            gap_usd=0.0,
            fetch_usage=lambda: None,
            usage_known=True,
        )
        results: list[bool] = []
        lock = threading.Lock()

        def reserve() -> None:
            allowed = tracker.try_reserve(0.08)
            with lock:
                results.append(allowed)

        threads = [threading.Thread(target=reserve) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(sum(results), 2)
        self.assertAlmostEqual(tracker.reserved_usd, 0.16)
        self.assertLessEqual(tracker.effective_usd(), SPEND_HARD_CAP_USD)

        sequential = SpendTracker(
            path=Path("/tmp/does-not-matter.json"),
            remote_usd=4.40,
            gap_usd=0.0,
            fetch_usage=lambda: None,
            usage_known=True,
        )
        self.assertTrue(sequential.try_reserve(0.08))
        self.assertFalse(sequential.try_reserve(0.08))

    def test_in_flight_reservation_survives_a_usage_refresh(self) -> None:
        tracker = SpendTracker(
            path=Path("/tmp/does-not-matter.json"),
            remote_usd=1.0,
            gap_usd=0.0,
            fetch_usage=lambda: 1.10,
            usage_known=True,
        )
        self.assertTrue(tracker.try_reserve(0.05))
        tracker.refresh()
        self.assertAlmostEqual(tracker.reserved_usd, 0.05)
        self.assertAlmostEqual(tracker.remote_usd, 1.10)
        self.assertAlmostEqual(tracker.effective_usd(), 1.15)

    def test_failed_calls_count_the_estimate(self) -> None:
        self.assertEqual(recorded_call_cost_usd(0.0, 0.07, False), 0.07)
        self.assertEqual(recorded_call_cost_usd(0.02, 0.07, False), 0.07)
        self.assertEqual(recorded_call_cost_usd(0.09, 0.07, True), 0.09)
        self.assertEqual(recorded_call_cost_usd(0.0, 0.07, True), 0.07)

    def test_gemini_estimate_covers_a_high_reasoning_completion(self) -> None:
        self.assertGreaterEqual(GEMINI_MAX_TOKENS, 16000)
        self.assertGreaterEqual(
            estimate_call_cost_usd(OPENROUTER_GEMINI_36_FLASH, "forecast"), 0.06
        )
        self.assertGreaterEqual(
            estimate_call_cost_usd(OPENROUTER_GEMINI_36_FLASH, "checklist"), 0.06
        )


class NumericAggregationTests(unittest.TestCase):
    def test_median_is_taken_on_values_at_each_percentile(self) -> None:
        left = [(0.1, 10.0), (0.5, 50.0), (0.9, 90.0)]
        right = [(0.1, 30.0), (0.5, 70.0), (0.9, 110.0)]
        merged = median_percentile_values([left, right], heights=(0.1, 0.5, 0.9))
        by_height = dict(merged)
        self.assertAlmostEqual(by_height[0.1], 20.0)
        self.assertAlmostEqual(by_height[0.5], 60.0)
        self.assertAlmostEqual(by_height[0.9], 100.0)
        values = [value for _, value in merged]
        self.assertEqual(values, sorted(values))
        self.assertEqual(len(set(values)), len(values))

    def test_tied_percentile_values_are_strictly_increasing(self) -> None:
        flat = [(0.1, 5.0), (0.9, 5.0)]
        merged = median_percentile_values([flat, flat])
        values = [value for _, value in merged]
        self.assertEqual(values, sorted(set(values)))
        for earlier, later in zip(values, values[1:]):
            self.assertLess(earlier, later)


class SubmissionThresholdTests(unittest.TestCase):
    def test_three_of_five_forecasts_are_required(self) -> None:
        self.assertEqual(MIN_SUCCESSFUL_FORECASTS, 3)
        fraction = required_success_fraction(5)
        threshold = 5 * fraction
        self.assertAlmostEqual(fraction, 0.6)
        self.assertAlmostEqual(threshold, 3.0)
        # The parent bot skips the question when successes are below this threshold.
        self.assertLess(2, threshold)
        self.assertFalse(3 < threshold)

    def test_bot_wires_the_threshold_reservation_and_numeric_median(self) -> None:
        source = Path(__file__).resolve().parents[1].joinpath("main.py").read_text()
        self.assertIn("required_success_fraction", source)
        self.assertIn("median_percentile_values", source)
        self.assertIn("try_reserve", source)
        self.assertIn("recorded_call_cost_usd", source)
        self.assertIn("_USAGE_UNKNOWN", source)


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

    def test_every_workflow_is_contents_read_and_does_not_invent_zero_spend(self) -> None:
        workflow_dir = Path(__file__).resolve().parents[1] / ".github" / "workflows"
        paths = sorted(workflow_dir.glob("*.yaml"))
        names = {path.name for path in paths}
        self.assertIn("review_bot.yaml", names)
        self.assertGreaterEqual(len(paths), 4)
        for path in paths:
            text = path.read_text()
            self.assertIn("permissions:", text, path.name)
            self.assertIn("contents: read", text, path.name)
            self.assertNotIn('{"remote_usd": 0.0, "gap_usd": 0.0}', text, path.name)
            self.assertNotIn("Keep a spend ledger", text, path.name)

    def test_review_package_is_pinned_and_stays_off(self) -> None:
        root = Path(__file__).resolve().parents[1]
        pyproject = (root / "pyproject.toml").read_text()
        self.assertIn('metaculus-bot-review = "==0.1.2"', pyproject)
        review = (root / ".github" / "workflows" / "review_bot.yaml").read_text()
        self.assertIn("REVIEW_BOT_ENABLED == 'true'", review)
        self.assertIn('REVIEW_DAYS: ${{ inputs.days || \'30\' }}', review)
        self.assertIn('--resolved-since "$REVIEW_DAYS"', review)
        self.assertNotIn("--resolved-since ${{", review)

    def test_tournament_schedule_is_manual_until_a_dry_run(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / ".github"
            / "workflows"
            / "run_bot_on_tournament.yaml"
        )
        active = [
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        self.assertIn("workflow_dispatch:", active)
        self.assertNotIn("schedule:", active)
        self.assertFalse(any(line.startswith("cron:") or "- cron:" in line for line in active))

    def test_cup_schedule_is_manual_until_a_dry_run(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / ".github"
            / "workflows"
            / "run_bot_on_metaculus_cup.yaml"
        )
        active = [
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        self.assertIn("workflow_dispatch:", active)
        self.assertNotIn("schedule:", active)
        self.assertFalse(any(line.startswith("cron:") or "- cron:" in line for line in active))

    def test_manual_forecast_defaults_to_dry_run(self) -> None:
        workflow_dir = Path(__file__).resolve().parents[1] / ".github" / "workflows"
        for name in ("run_bot_on_tournament.yaml", "run_bot_on_metaculus_cup.yaml"):
            text = (workflow_dir / name).read_text()
            self.assertRegex(
                text,
                r"dry_run:\n[ \t]+description:.*\n[ \t]+type: boolean\n[ \t]+default: true\n",
            )
            self.assertRegex(
                text,
                r"max_questions:\n[ \t]+description:.*\n[ \t]+type: number\n[ \t]+default: 3\n",
            )
            self.assertIn(
                "DRY_RUN: ${{ github.event_name == 'workflow_dispatch' && inputs.dry_run }}",
                text,
                name,
            )
            self.assertIn(
                "MAX_QUESTIONS: ${{ github.event_name == 'workflow_dispatch' && inputs.max_questions || '' }}",
                text,
                name,
            )
            bodies = _run_script_bodies(text)
            self.assertTrue(bodies, name)
            for body in bodies:
                self.assertNotIn("${{", body, name)
            joined = "\n".join(bodies)
            self.assertIn('if [ "$DRY_RUN" = "true" ]; then', joined)
            self.assertIn("args+=(--dry-run)", joined)
            self.assertIn('args+=(--max-questions "$MAX_QUESTIONS")', joined)
            active = [
                line.strip()
                for line in text.splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
            self.assertNotIn("schedule:", active, name)
            self.assertFalse(any(line.startswith("cron:") or "- cron:" in line for line in active))


def _run_script_bodies(text: str) -> list[str]:
    lines = text.splitlines()
    bodies: list[str] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].lstrip()
        indent = len(lines[index]) - len(stripped)
        if stripped != "run: |":
            index += 1
            continue
        index += 1
        body: list[str] = []
        while index < len(lines):
            if lines[index].strip() == "":
                body.append(lines[index])
                index += 1
                continue
            next_indent = len(lines[index]) - len(lines[index].lstrip())
            if next_indent <= indent:
                break
            body.append(lines[index])
            index += 1
        bodies.append("\n".join(body))
    return bodies


if __name__ == "__main__":
    unittest.main()
