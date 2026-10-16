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
import os
import re
import statistics
import sys
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

# Used only when LiteLLM reports no dollar cost for a call. High enough that
# a silent cost tracker still trips the switch before the hard cap.
_ESTIMATED_CALL_USD = {
    # Forecaster figures are the call itself. The parser is metered separately.
    "gemini-forecast": 0.020,
    "kimi-forecast": 0.080,
    "deepseek-pro-forecast": 0.008,
    "gemini-checklist": 0.020,
    "deepseek-checklist": 0.008,
    "parser": 0.003,
}

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
    r"\b\d{1,3}(?:\.\d+)?\s*%|\b0\.\d+\b|\bodds\b|\bprobability\b",
    re.IGNORECASE,
)


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
            "Could not read OpenRouter usage (%s). Using the local spend ledger.",
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


class SpendTracker:
    """
    Track OpenRouter spend across GitHub Actions runs.

    `remote_usd` is the last usage figure from OpenRouter (or the ledger).
    `gap_usd` is spend measured since that reading. When a newer reading
    arrives, the gap shrinks to whatever the reading has not absorbed yet,
    and is capped so a high estimate cannot stick around forever.
    """

    def __init__(
        self,
        path: Path | None = None,
        remote_usd: float = 0.0,
        gap_usd: float = 0.0,
        fetch_usage: Any = None,
    ) -> None:
        self.path = path or DEFAULT_SPEND_LEDGER
        self.remote_usd = max(0.0, float(remote_usd))
        self.gap_usd = max(0.0, float(gap_usd))
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
        if ledger_path.is_file():
            try:
                raw = json.loads(ledger_path.read_text())
                remote_usd = float(raw.get("remote_usd", 0.0))
                gap_usd = float(raw.get("gap_usd", 0.0))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                remote_usd = 0.0
                gap_usd = 0.0
        return cls(
            path=ledger_path,
            remote_usd=remote_usd,
            gap_usd=gap_usd,
            fetch_usage=fetch_usage,
        )

    def effective_usd(self) -> float:
        return self.remote_usd + self.gap_usd

    def deepseek_only(self) -> bool:
        return use_deepseek_only(self.effective_usd())

    def over_hard_cap(self) -> bool:
        return over_hard_cap(self.effective_usd())

    def would_exceed_cap(self, next_estimate_usd: float) -> bool:
        return self.over_hard_cap() or call_would_pass_hard_cap(
            self.effective_usd(), next_estimate_usd
        )

    def add_unconfirmed(self, amount_usd: float) -> None:
        if amount_usd <= 0:
            return
        self.gap_usd += float(amount_usd)
        self.save()

    def refresh(self) -> None:
        usage = self._fetch_usage()
        if usage is None:
            return
        usage = max(0.0, float(usage))
        if usage + 1e-9 < self.remote_usd:
            # A lower reading is a different key or a stale response. Keep the ledger.
            self.save()
            return
        if usage > self.remote_usd + 1e-9:
            projected = self.remote_usd + self.gap_usd
            remainder = max(0.0, projected - usage)
            self.remote_usd = usage
            self.gap_usd = min(remainder, SPEND_GAP_CAP_USD)
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "remote_usd": round(self.remote_usd, 6),
            "gap_usd": round(self.gap_usd, 6),
        }
        self.path.write_text(json.dumps(payload, indent=2) + "\n")

    def mode_label(self) -> str:
        if self.over_hard_cap():
            return "stopped"
        if self.deepseek_only():
            return "deepseek-only"
        return "mixed"
