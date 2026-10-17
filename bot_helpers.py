"""
Runtime helpers for the template bot: environment validation, startup/result
banners, and suppression of noisy upstream warnings.

Kept separate from main.py so that file can focus on the bot's forecasting
logic. main_with_no_framework.py keeps its own inline copies on purpose --
it's meant to be a single-file reference implementation.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import statistics
import sys
import threading
import warnings
from pathlib import Path
from typing import Any, Sequence

import requests


# Placeholder values shipped in .env.template. If a real env var still equals
# one of these the user forgot to replace it; we'd rather fail loudly here than
# inside the SDK three layers down.
_PLACEHOLDER_ENV_VALUES = {
    "1234567890",
    "REPLACE_ME",
    "your-token-here",
    "your-api-key-here",
}


def _is_real_env(name: str) -> bool:
    val = os.getenv(name)
    return bool(val and val.strip() and val.strip() not in _PLACEHOLDER_ENV_VALUES)


def silence_noisy_dependencies() -> None:
    """
    Quiet warnings from transitive deps that fire on import and confuse new
    users. Must be called *before* importing forecasting_tools.
    """
    warnings.filterwarnings(
        "ignore", message=r".*does not support cost tracking.*"
    )
    logging.getLogger("forecasting_tools.ai_models.model_tracker").setLevel(
        logging.ERROR
    )
    # Streamlit installs its own logger hierarchy; suppress via its own API.
    try:
        from streamlit.logger import set_log_level

        set_log_level("error")
    except ImportError:
        pass
    # LiteLLM is verbose at INFO; its WARNING level is enough for us.
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False


def check_environment(strict: bool = True) -> None:
    """
    Verify METACULUS_TOKEN and OPENROUTER_API_KEY are set. On failure with
    strict=True, exits the process with a non-zero status. AskNews is optional.
    """
    problems: list[str] = []

    if not _is_real_env("METACULUS_TOKEN"):
        problems.append(
            "METACULUS_TOKEN is missing or still a placeholder. "
            "Get one at https://www.metaculus.com/futureeval/participate/"
        )

    if not _is_real_env("OPENROUTER_API_KEY"):
        problems.append(
            "OPENROUTER_API_KEY is missing or still a placeholder. "
            "The forecasters run through OpenRouter. "
            "Add the key as a repository secret named OPENROUTER_API_KEY."
        )
    elif not asknews_configured():
        print(
            "AskNews credentials are not set. News research will be skipped "
            "and forecasts will use the resolution checklist only.\n"
        )

    if problems:
        print("❌  Setup problems:")
        for p in problems:
            print(f"    • {p}")
        if strict:
            sys.exit(1)


def print_startup_banner(run_mode: str, will_publish: bool) -> None:
    publish = "publish=yes" if will_publish else "publish=no (dry run)"
    print(f"🤖  Running mode={run_mode}, {publish}\n")


def print_run_summary_banner(
    forecast_reports: Sequence[Any],
    will_publish: bool,
    tournament_url: str | None = None,
) -> None:
    """
    End-of-run summary printed via print() (not logger) so it survives log
    filtering. Shows count, per-question URLs, and any failure tracebacks.
    If tournament_url is given, it's included as a footer link.
    """
    # Lazy import so this module is usable in contexts where forecasting_tools
    # isn't installed (e.g. unit tests of the banner format).
    from forecasting_tools import ForecastReport

    valid = [r for r in forecast_reports if isinstance(r, ForecastReport)]
    exceptions = [r for r in forecast_reports if isinstance(r, BaseException)]
    banner = "=" * 80

    print()
    print(banner)

    if not forecast_reports:
        print("ℹ️   No new questions to forecast on this run.")
        print(banner)
        print()
        return

    if valid and not exceptions:
        verb = "submitted" if will_publish else "produced (dry run)"
        print(f"🎉  Bot {verb} {len(valid)} forecast(s).")
    elif valid and exceptions:
        print(
            f"⚠️   Partial — {len(valid)} succeeded, {len(exceptions)} failed."
        )
    else:
        print(f"❌  All {len(exceptions)} attempt(s) failed.")

    if valid:
        print()
        for r in valid:
            note = f"  (with {len(r.errors)} minor error(s))" if r.errors else ""
            print(f"  ✅ {r.question.page_url}{note}")
        if will_publish and tournament_url:
            print(f"\n  Tournament: {tournament_url}")

    if exceptions:
        print()
        for exc in exceptions:
            msg = str(exc)
            if len(msg) > 200:
                msg = msg[:200] + "..."
            print(f"  ❌ {type(exc).__name__}: {msg}")

    print(banner)
    print()


# ---------------------------------------------------------------------------
# Ensemble, spend cap, and research guards.
# These stay free of forecasting-tools imports so unit tests can run without
# the optional integrations group and without calling any API.
# ---------------------------------------------------------------------------

# Exact OpenRouter model ids. LiteLLM reaches them with an "openrouter/" prefix.
GEMINI_36_FLASH_ID = "google/gemini-3.6-flash"
KIMI_K3_ID = "moonshotai/kimi-k3"
DEEPSEEK_V4_PRO_ID = "deepseek/deepseek-v4-pro"
DEEPSEEK_V4_FLASH_ID = "deepseek/deepseek-v4-flash"

OPENROUTER_GEMINI_36_FLASH = f"openrouter/{GEMINI_36_FLASH_ID}"
OPENROUTER_KIMI_K3 = f"openrouter/{KIMI_K3_ID}"
OPENROUTER_DEEPSEEK_V4_PRO = f"openrouter/{DEEPSEEK_V4_PRO_ID}"
OPENROUTER_DEEPSEEK_V4_FLASH = f"openrouter/{DEEPSEEK_V4_FLASH_ID}"

# Gemini 3.6 Flash x3, Kimi K3 x1, DeepSeek V4 Pro x1. Median of these five.
MIXED_FORECASTER_MODELS = (
    OPENROUTER_GEMINI_36_FLASH,
    OPENROUTER_GEMINI_36_FLASH,
    OPENROUTER_GEMINI_36_FLASH,
    OPENROUTER_KIMI_K3,
    OPENROUTER_DEEPSEEK_V4_PRO,
)
DEEPSEEK_ONLY_FORECASTER_MODELS = (OPENROUTER_DEEPSEEK_V4_PRO,) * 5

# One different angle per ensemble seat so the five forecasts are not copies.
FORECASTER_ANGLES = (
    "Outside view: start from the base rate for events of this kind, then adjust for what is specific about this case.",
    "Inside view: trace the mechanism, the decision makers, and the latest facts. Name what would have to change for each outcome.",
    "Status quo: most things do not happen by the deadline. Weight the outcome if nothing material changes, and lower the chance of a new event when the deadline is close and there is no news.",
    "Inside view: trace the mechanism and the latest facts, and say which piece of evidence would move the forecast the most.",
    "Outside view: anchor on the historical base rate and reference classes before using the news.",
)

FORECAST_GUARDRAIL = (
    "The resolution checklist is binding. If the question title and the fine print disagree, follow the fine print. "
    "Do not copy a Metaculus crowd prediction or a prediction-market price. "
    "Forecast from the criteria and the research, and keep every probability inside 1% to 99%."
)

# Switch the ensemble once recorded spend reaches this, and refuse new calls
# that would go past the key cap. The OpenRouter key itself should also be
# capped at the hard cap in the OpenRouter dashboard.
SPEND_SWITCH_USD = 3.50
SPEND_HARD_CAP_USD = 4.50
# Keep at most about one in-flight call of uncertainty when the usage API lags.
SPEND_GAP_CAP_USD = 0.12

PROBABILITY_FLOOR = 0.01
PROBABILITY_CAP = 0.99

# Used when LiteLLM reports no dollar cost, and reserved before the call starts.
# Gemini's figure covers a full high-reasoning completion (about 6k input
# tokens and 16k output tokens at the published OpenRouter rates).
_ESTIMATED_CALL_USD = {
    # Forecaster figures are the call itself. The parser is metered separately.
    "gemini-forecast": 0.070,
    "kimi-forecast": 0.080,
    "deepseek-pro-forecast": 0.008,
    "gemini-checklist": 0.070,
    "deepseek-checklist": 0.008,
    "parser": 0.003,
}

# Submit a question only when at least this many forecasts succeeded.
MIN_SUCCESSFUL_FORECASTS = 3
# Heights used to take a median distribution for numeric and date questions.
PERCENTILE_HEIGHTS = (0.1, 0.2, 0.4, 0.6, 0.8, 0.9)
GEMINI_MAX_TOKENS = 16000

DEFAULT_SPEND_LEDGER = Path("logs/openrouter_spend.json")
OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/key"

_SECRET_ENV_NAMES = (
    "OPENROUTER_API_KEY",
    "METACULUS_TOKEN",
    "ASKNEWS_SECRET",
    "ASKNEWS_API_KEY",
    "ASKNEWS_CLIENT_ID",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "PERPLEXITY_API_KEY",
    "EXA_API_KEY",
)
_SECRET_PATTERNS = (
    re.compile(r"sk-or-[A-Za-z0-9_\-]+"),
    re.compile(r"Bearer\s+\S+", re.IGNORECASE),
)
_CROWD_MENTION = re.compile(
    r"community prediction|metaculus community|metaculus crowd|"
    r"crowd forecast|crowd prediction|the crowd is at|crowd odds",
    re.IGNORECASE,
)
_MARKET_ODDS = re.compile(
    r"polymarket|kalshi|manifold|predictit|prediction market|"
    r"betting odds|implied probability|priced at|trading at|currently trading",
    re.IGNORECASE,
)
_HAS_PROBABILITY = re.compile(
    r"\b\d{1,3}(?:\.\d+)?\s*(?:%|percent\b)|\b0\.\d+\b|\bodds\b|\bprobability\b",
    re.IGNORECASE,
)
_ODDS_OF = re.compile(r"\bodds\s+of\b", re.IGNORECASE)
_MARKET_PRICE = re.compile(r"\bmarket\s+prices?\b", re.IGNORECASE)


class SpendCapReached(RuntimeError):
    """Raised when another model call would pass the OpenRouter hard cap."""


def openrouter_model_id(litellm_model: str) -> str:
    prefix = "openrouter/"
    if not litellm_model.startswith(prefix):
        raise ValueError(f"Expected an OpenRouter model, got {litellm_model}")
    return litellm_model.removeprefix(prefix)


def forecaster_models(spend_usd: float) -> tuple[str, ...]:
    """Five models whose forecasts are combined by median."""
    if spend_usd >= SPEND_SWITCH_USD:
        return DEEPSEEK_ONLY_FORECASTER_MODELS
    return MIXED_FORECASTER_MODELS


def use_deepseek_only(spend_usd: float) -> bool:
    return spend_usd >= SPEND_SWITCH_USD


def over_hard_cap(spend_usd: float) -> bool:
    return spend_usd >= SPEND_HARD_CAP_USD


def required_success_fraction(forecasts_per_question: int = 5) -> float:
    """Fraction of forecasts that must succeed before a question is submitted."""
    count = max(int(forecasts_per_question), 1)
    return min(1.0, MIN_SUCCESSFUL_FORECASTS / count)


def recorded_call_cost_usd(measured: float, estimate: float, succeeded: bool) -> float:
    """Count a failed or unpriced call at the conservative estimate, not at $0."""
    measured_usd = max(0.0, float(measured))
    estimate_usd = max(0.0, float(estimate))
    if succeeded and measured_usd > 0:
        return measured_usd
    return max(measured_usd, estimate_usd)


def call_would_pass_hard_cap(spend_usd: float, next_estimate_usd: float) -> bool:
    return spend_usd + max(next_estimate_usd, 0.0) > SPEND_HARD_CAP_USD


def estimate_call_cost_usd(model: str, kind: str) -> float:
    """kind is 'forecast' or 'checklist'. Parser cost is folded into a forecast."""
    model_id = model.removeprefix("openrouter/")
    if kind == "checklist":
        if DEEPSEEK_V4_PRO_ID in model_id:
            return _ESTIMATED_CALL_USD["deepseek-checklist"]
        return _ESTIMATED_CALL_USD["gemini-checklist"]
    if KIMI_K3_ID in model_id:
        return _ESTIMATED_CALL_USD["kimi-forecast"]
    if DEEPSEEK_V4_PRO_ID in model_id:
        return _ESTIMATED_CALL_USD["deepseek-pro-forecast"]
    if DEEPSEEK_V4_FLASH_ID in model_id:
        return _ESTIMATED_CALL_USD["parser"]
    return _ESTIMATED_CALL_USD["gemini-forecast"]


def clip_probability(value: float) -> float:
    numeric = float(value)
    if numeric != numeric:
        raise ValueError("Probability is not a number")
    return min(PROBABILITY_CAP, max(PROBABILITY_FLOOR, numeric))


def clip_and_renormalize(probabilities: list[float]) -> list[float]:
    """Clip each entry to 1–99% and rescale so the entries sum to 1."""
    if not probabilities:
        raise ValueError("No probabilities to clip")
    count = len(probabilities)
    if count * PROBABILITY_FLOOR > 1.0 + 1e-9:
        raise ValueError(
            f"Too many options ({count}) to keep each probability inside 1-99%"
        )

    current = [clip_probability(probability) for probability in probabilities]
    for _ in range(8):
        total = sum(current)
        if total <= 0:
            break
        scaled = [probability / total for probability in current]
        current = [clip_probability(probability) for probability in scaled]
        if abs(sum(current) - 1.0) < 1e-10:
            break

    residual = 1.0 - sum(current)
    order = sorted(range(count), key=lambda index: current[index], reverse=True)
    for index in order:
        updated = current[index] + residual
        if PROBABILITY_FLOOR - 1e-9 <= updated <= PROBABILITY_CAP + 1e-9:
            current[index] = updated
            residual = 0.0
            break
        if residual > 0 and current[index] < PROBABILITY_CAP:
            room = PROBABILITY_CAP - current[index]
            current[index] += room
            residual -= room
        elif residual < 0 and current[index] > PROBABILITY_FLOOR:
            room = current[index] - PROBABILITY_FLOOR
            take = min(room, -residual)
            current[index] -= take
            residual += take
    if abs(sum(current) - 1.0) > 1e-6:
        raise ValueError("Could not renormalize probabilities into 1-99%")
    return current


def median_binary(probabilities: list[float]) -> float:
    if not probabilities:
        raise ValueError("Cannot aggregate an empty list of probabilities")
    return clip_probability(float(statistics.median(probabilities)))


def _interpolate(target: float, xs: Sequence[float], ys: Sequence[float]) -> float:
    if target <= xs[0]:
        return float(ys[0])
    if target >= xs[-1]:
        return float(ys[-1])
    for index in range(1, len(xs)):
        if target <= xs[index]:
            span = xs[index] - xs[index - 1]
            if span == 0:
                return float(ys[index])
            weight = (target - xs[index - 1]) / span
            return float(ys[index - 1] + weight * (ys[index] - ys[index - 1]))
    return float(ys[-1])


def _strictly_increasing(values: list[float]) -> list[float]:
    result = [float(value) for value in values]
    span = max(result) - min(result)
    step = max(abs(span) * 1e-9, 1e-9)
    for index in range(1, len(result)):
        if result[index] <= result[index - 1]:
            result[index] = result[index - 1] + step
    return result


def median_percentile_values(
    forecasts: Sequence[Sequence[tuple[float, float]]],
    heights: Sequence[float] = PERCENTILE_HEIGHTS,
) -> list[tuple[float, float]]:
    """Median outcome value at each percentile height.

    Each forecast is a list of (percentile, value) points. Values at the shared
    heights are interpolated, then the median value is taken at each height.
    Tied values are nudged so the result is strictly increasing.
    """
    if not forecasts:
        raise ValueError("Cannot aggregate an empty list of numeric forecasts")
    targets = [float(height) for height in heights]
    if len(targets) < 2:
        raise ValueError("Need at least two percentile heights")
    medians: list[float] = []
    for height in targets:
        column: list[float] = []
        for forecast in forecasts:
            if len(forecast) < 2:
                raise ValueError("Each numeric forecast needs at least two points")
            points = sorted(forecast, key=lambda item: item[0])
            column.append(
                _interpolate(
                    height,
                    [point[0] for point in points],
                    [point[1] for point in points],
                )
            )
        medians.append(float(statistics.median(column)))
    return list(zip(targets, _strictly_increasing(medians)))


def median_option_probabilities(
    forecasts: list[dict[str, float]],
) -> dict[str, float]:
    """Median of each option across forecasts, then clip and renormalize."""
    if not forecasts:
        raise ValueError("Cannot aggregate an empty list of predictions")
    names = list(forecasts[0].keys())
    expected = set(names)
    for forecast in forecasts:
        if set(forecast.keys()) != expected:
            raise ValueError("Forecasts must share the same options")
    medians = [
        float(statistics.median([float(forecast[name]) for forecast in forecasts]))
        for name in names
    ]
    clipped = clip_and_renormalize(medians)
    return {name: probability for name, probability in zip(names, clipped)}


def resolution_checklist_prompt(
    *,
    question_text: str,
    resolution_criteria: str,
    fine_print: str,
    background: str,
    today: str,
) -> str:
    """Gemini (or the cheap fallback) reads the criteria. It does not forecast."""
    return (
        "You are a resolution lawyer checking a forecasting question before anyone assigns a probability.\n"
        "Do not give a probability, a percentage, or a forecast.\n\n"
        f"Question:\n{question_text}\n\n"
        f"Resolution criteria:\n{resolution_criteria or '(none provided)'}\n\n"
        f"Fine print:\n{fine_print or '(none provided)'}\n\n"
        f"Background:\n{background or '(none provided)'}\n\n"
        f"Today is {today}.\n\n"
        "Write a short checklist the forecasters must follow:\n"
        "1. What exactly counts as Yes, and what counts as No, using the criteria rather than the title.\n"
        "2. The resolution source, and what to do if sources disagree.\n"
        "3. The deadline, time zone, and the status-quo outcome if nothing new happens.\n"
        "4. Annulment or ambiguity conditions. If the question would be annulled instead of resolving, say so plainly.\n"
        "5. Technicalities, definitions, and edge cases a careless reading would miss.\n"
        "6. Any fine-print sentence that overrides the title. Quote it.\n\n"
        "Quote the decisive phrases. Do not forecast.\n"
    )


def _line_states_crowd_or_market_odds(line: str) -> bool:
    if _CROWD_MENTION.search(line):
        return True
    if _ODDS_OF.search(line) or _MARKET_PRICE.search(line):
        return True
    return bool(_MARKET_ODDS.search(line) and _HAS_PROBABILITY.search(line))


def strip_market_and_crowd_odds(text: str) -> tuple[str, int]:
    """Drop research lines that state a crowd or market probability."""
    kept: list[str] = []
    removed = 0
    for line in text.splitlines():
        if _line_states_crowd_or_market_odds(line):
            removed += 1
            continue
        kept.append(line)
    stripped = "\n".join(kept).strip()
    if removed:
        stripped += (
            "\n\nNote: lines that stated a crowd or prediction-market probability "
            "were removed so they are not used as the forecast."
        )
    return stripped, removed


def asknews_configured() -> bool:
    """True when either AskNews auth style is present. Missing keys are not an error."""
    oauth = _is_real_env("ASKNEWS_CLIENT_ID") and _is_real_env("ASKNEWS_SECRET")
    api_key = _is_real_env("ASKNEWS_API_KEY")
    return bool(oauth or api_key)


def redact_secrets(text: str) -> str:
    """Remove credential material before anything is logged."""
    redacted = text
    for name in _SECRET_ENV_NAMES:
        value = os.getenv(name)
        if value and len(value.strip()) >= 6:
            redacted = redacted.replace(value.strip(), "[redacted]")
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[redacted]", redacted)
    if len(redacted) > 400:
        redacted = redacted[:400] + "..."
    return redacted


def fetch_openrouter_usage_usd() -> float | None:
    """
    Read the key's billed usage from OpenRouter. Returns None if it cannot.

    The response can echo the key label, so this function logs only the
    exception type on failure and never logs the body.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or not api_key.strip() or api_key.strip() in _PLACEHOLDER_ENV_VALUES:
        return None
    try:
        response = requests.get(
            OPENROUTER_KEY_URL,
            headers={"Authorization": f"Bearer {api_key.strip()}"},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Could not read OpenRouter usage (%s). Forecasting continues only when a saved spend record exists.",
            type(exc).__name__,
        )
        return None
    data = payload.get("data") if isinstance(payload, dict) else None
    usage = data.get("usage") if isinstance(data, dict) else None
    if usage is None:
        return None
    try:
        return float(usage)
    except (TypeError, ValueError):
        return None


def _trusted_ledger_amounts(raw: Any) -> tuple[float, float] | None:
    """A ledger counts only when it was written after a real usage reading."""
    if not isinstance(raw, dict) or raw.get("usage_known") is not True:
        return None
    try:
        remote_usd = float(raw["remote_usd"])
        gap_usd = float(raw["gap_usd"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(remote_usd) or not math.isfinite(gap_usd):
        return None
    if remote_usd < 0 or gap_usd < 0:
        return None
    return remote_usd, gap_usd


class SpendTracker:
    """
    Track OpenRouter spend across GitHub Actions runs.

    `remote_usd` is the last usage figure from OpenRouter (or a trusted ledger).
    `gap_usd` is settled spend since that reading. `reserved_usd` is budget held
    for calls that have started and not finished; it is memory only.
    Forecasting is refused until a usage reading or a trusted ledger exists.
    """

    def __init__(
        self,
        path: Path | None = None,
        remote_usd: float = 0.0,
        gap_usd: float = 0.0,
        fetch_usage: Any = None,
        usage_known: bool = False,
    ) -> None:
        self.path = path or DEFAULT_SPEND_LEDGER
        self.remote_usd = max(0.0, float(remote_usd))
        self.gap_usd = max(0.0, float(gap_usd))
        self.reserved_usd = 0.0
        self.usage_known = bool(usage_known)
        self._lock = threading.Lock()
        self._fetch_usage = fetch_usage if fetch_usage is not None else fetch_openrouter_usage_usd

    @classmethod
    def load(
        cls,
        path: Path | None = None,
        fetch_usage: Any = None,
    ) -> "SpendTracker":
        ledger_path = path or DEFAULT_SPEND_LEDGER
        remote_usd = 0.0
        gap_usd = 0.0
        usage_known = False
        if ledger_path.is_file():
            try:
                trusted = _trusted_ledger_amounts(json.loads(ledger_path.read_text()))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                trusted = None
            if trusted is not None:
                remote_usd, gap_usd = trusted
                usage_known = True
        return cls(
            path=ledger_path,
            remote_usd=remote_usd,
            gap_usd=gap_usd,
            fetch_usage=fetch_usage,
            usage_known=usage_known,
        )

    def effective_usd(self) -> float:
        return self.remote_usd + self.gap_usd + self.reserved_usd

    def can_forecast(self) -> bool:
        """True when spend is known. The hard cap is checked separately."""
        return self.usage_known

    def deepseek_only(self) -> bool:
        return use_deepseek_only(self.effective_usd())

    def over_hard_cap(self) -> bool:
        return over_hard_cap(self.effective_usd())

    def would_exceed_cap(self, next_estimate_usd: float) -> bool:
        if not self.usage_known:
            return True
        return self.over_hard_cap() or call_would_pass_hard_cap(
            self.effective_usd(), next_estimate_usd
        )

    def try_reserve(self, amount_usd: float) -> bool:
        """Hold budget before a call starts. False when usage is unknown or the hold would pass the cap."""
        amount = max(0.0, float(amount_usd))
        with self._lock:
            if not self.usage_known:
                return False
            if self.over_hard_cap() or call_would_pass_hard_cap(self.effective_usd(), amount):
                return False
            self.reserved_usd += amount
            return True

    def settle_reservation(self, reserved_usd: float, actual_usd: float) -> None:
        """Replace a hold with the cost that should stick after the call."""
        with self._lock:
            self.reserved_usd = max(0.0, self.reserved_usd - max(0.0, float(reserved_usd)))
            actual = max(0.0, float(actual_usd))
            if actual > 0:
                self.gap_usd += actual
            self.save()

    def add_unconfirmed(self, amount_usd: float) -> None:
        if amount_usd <= 0:
            return
        with self._lock:
            self.gap_usd += float(amount_usd)
            self.save()

    def refresh(self) -> None:
        usage = self._fetch_usage()
        if usage is None:
            return
        try:
            usage_usd = float(usage)
        except (TypeError, ValueError):
            return
        if not math.isfinite(usage_usd) or usage_usd < 0:
            return
        with self._lock:
            if not self.usage_known:
                self.remote_usd = usage_usd
                self.gap_usd = 0.0
                self.usage_known = True
                self.save()
                return
            if usage_usd + 1e-9 < self.remote_usd:
                # A lower reading is a different key or a stale response. Keep the ledger.
                self.save()
                return
            if usage_usd > self.remote_usd + 1e-9:
                projected = self.remote_usd + self.gap_usd
                remainder = max(0.0, projected - usage_usd)
                self.remote_usd = usage_usd
                self.gap_usd = min(remainder, SPEND_GAP_CAP_USD)
            self.save()

    def save(self) -> None:
        if not self.usage_known:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "remote_usd": round(self.remote_usd, 6),
            "gap_usd": round(self.gap_usd, 6),
            "usage_known": True,
        }
        self.path.write_text(json.dumps(payload, indent=2) + "\n")

    def mode_label(self) -> str:
        if not self.usage_known:
            return "usage-unknown"
        if self.over_hard_cap():
            return "stopped"
        if self.deepseek_only():
            return "deepseek-only"
        return "mixed"
