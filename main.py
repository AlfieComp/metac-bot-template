import argparse
import asyncio
import contextvars
import logging
import threading
from datetime import datetime, timezone
from typing import Literal

import dotenv

# Runtime helpers (env validation, banners, dependency-warning suppression).
from bot_helpers import (
    FORECAST_GUARDRAIL,
    FORECASTER_ANGLES,
    OPENROUTER_DEEPSEEK_V4_FLASH,
    OPENROUTER_DEEPSEEK_V4_PRO,
    OPENROUTER_GEMINI_36_FLASH,
    SPEND_HARD_CAP_USD,
    SPEND_SWITCH_USD,
    SpendCapReached,
    SpendTracker,
    asknews_configured,
    check_environment,
    clip_probability,
    estimate_call_cost_usd,
    forecaster_models,
    median_option_probabilities,
    print_run_summary_banner,
    print_startup_banner,
    redact_secrets,
    resolution_checklist_prompt,
    silence_noisy_dependencies,
    strip_market_and_crowd_odds,
)

silence_noisy_dependencies()

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    MonetaryCostManager,
    MultipleChoiceQuestion,
    MultipleChoiceReport,
    PredictedOption,
    NumericDistribution,
    NumericQuestion,
    DateQuestion,
    DatePercentile,
    Percentile,
    ConditionalQuestion,
    ConditionalPrediction,
    PredictionTypes,
    PredictionAffirmed,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    clean_indents,
    structure_output,
)

dotenv.load_dotenv()
logger = logging.getLogger(__name__)

# Set inside each of the five forecasts so nested conditional questions reuse
# that seat's model instead of taking another one.
_FORECASTER_SLOT: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "forecaster_slot", default=None
)

_GEMINI_MAX_TOKENS = 4096
_REASONING_MAX_TOKENS = 6000
_PARSER_MAX_TOKENS = 2000


def _install_median_multiple_choice_aggregation() -> None:
    """The template averages multiple-choice options. This bot takes the median."""

    async def aggregate(
        cls, predictions: list[PredictedOptionList], question: MultipleChoiceQuestion
    ) -> PredictedOptionList:
        del cls, question
        merged = median_option_probabilities(
            [prediction.to_dict() for prediction in predictions]
        )
        return PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name=name, probability=probability)
                for name, probability in merged.items()
            ]
        )

    MultipleChoiceReport.aggregate_predictions = classmethod(aggregate)  # type: ignore[method-assign]


_install_median_multiple_choice_aggregation()


def _looks_like_unsupported_parameter(exc: BaseException) -> bool:
    if type(exc).__name__ != "BadRequestError":
        return False
    text = redact_secrets(str(exc)).lower()
    needles = (
        "reasoning",
        "extra_body",
        "unknown parameter",
        "invalid parameter",
        "does not support",
        "unsupported parameter",
    )
    return any(needle in text for needle in needles)


class FallTemplateBot2026(ForecastBot):
    """
    Fall 2026 template bot, with a five-model OpenRouter ensemble.

    The final forecast is the median of five reasoned forecasts: Gemini 3.6 Flash
    three times, Kimi K3 once, and DeepSeek V4 Pro once. DeepSeek V4 Flash parses
    the output. One Gemini call checks the resolution criteria and fine print
    before those forecasts. Once recorded OpenRouter spend reaches US$3.50 the
    checklist and the five forecasts switch to DeepSeek V4 Pro. Calls stop at US$4.50.

    This is a copy of what is used by Metaculus to run the Metac Bots in our benchmark, provided as a template for new bot makers.
    This template is given as-is, and is use-at-your-own-risk.
    We have covered most test cases in forecasting-tools it may be worth double checking key components locally.
    So far our track record has been 1 mentionable bug per season (affecting forecasts for 1-2% of total questions)

    The forecast prompts started from SummerTemplateBot2026. The model roster, the resolution checklist, and the spend cap are specific to this fork.

    The main entry point of this bot is `bot.forecast_on_tournament(tournament_id)` in the parent class.
    See the script at the bottom of the file for more details on how to run the bot.
    Ignoring the finer details, the general flow is:
    - Load questions from Metaculus
    - For each question
        - Execute run_research a number of times equal to research_reports_per_question
        - Execute respective run_forecast function `predictions_per_research_report * research_reports_per_question` times
        - Aggregate the predictions
        - Submit prediction (if publish_reports_to_metaculus is True)
    - Return a list of ForecastReport objects

    Alternatively, you can use the MetaculusClient to make a custom filter of questions to forecast on
    and forecast them with `bot.forecast_questions(questions)`

    Only the research and forecast functions need to be implemented in ForecastBot subclasses,
    though you may want to override other ForecastBot functions.
    In this example, you can change the prompts to be whatever you want since,
    structure_output uses an LLM to intelligently reformat the output into the needed structure.

    By default (i.e. 'tournament' mode), when you run this script, it will forecast on any open questions in the
    primary bot tournament and MiniBench. If you want to forecast on only one or the other, you can remove one
    of them from the 'tournament' mode code at the bottom of the file.

    You can experiment with what models work best with your bot by using the `llms` parameter when initializing the bot.
    You can initialize the bot with any number of models. For example,
    ```python
    my_bot = MyBot(
        ...
        llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
            "default": GeneralLlm(
                model="openrouter/openai/gpt-4o", # "anthropic/claude-sonnet-4-20250514", etc (see docs for litellm)
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "summarizer": "openai/gpt-4o-mini",
            "researcher": "asknews/news-summaries",
            "parser": "openai/gpt-4o-mini",
        },
    )
    ```

    Then you can access the model in custom functions like this:
    ```python
    research_strategy = self.get_llm("researcher", "model_name")
    if research_strategy == "asknews/news-summaries":
        ...
    # OR
    summarizer = await self.get_llm("summarizer", "llm").invoke(prompt)
    # OR
    reasoning = await self.get_llm("default", "llm").invoke(prompt)
    ```

    If you end up having trouble with rate limits and want to try a more sophisticated rate limiter try:
    ```python
    from forecasting_tools import RefreshingBucketRateLimiter
    rate_limiter = RefreshingBucketRateLimiter(
        capacity=2,
        refresh_rate=1,
    ) # Allows 1 request per second on average with a burst of 2 requests initially. Set this as a class variable
    await self.rate_limiter.wait_till_able_to_acquire_resources(1) # 1 because it's consuming 1 request (use more if you are adding a token limit)
    ```
    Additionally OpenRouter has large rate limits immediately on account creation
    """

    _max_concurrent_questions = (
        1  # Set this to whatever works for your search-provider/ai-model rate limits
    )
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)
    _structure_output_validation_samples = 2

    def __init__(self, *, spend_tracker: SpendTracker | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.spend = spend_tracker or SpendTracker.load()
        # The research string already includes the checklist. A second summary
        # call would spend more and is not part of the ensemble.
        self.enable_summarize_research = False
        self.question_limit: int | None = None
        # threading.Lock so the counter stays valid across the two asyncio.run
        # calls (tournament, then MiniBench).
        self._slot_lock = threading.Lock()
        self._next_slot = 0
        self._limiter_loop: asyncio.AbstractEventLoop | None = None
        self._limiter_for_loop: asyncio.Semaphore | None = None
        self._mixed_roster = [
            self._make_llm(model, max_tokens=self._max_tokens_for(model), reasoning=True)
            for model in forecaster_models(0.0)
        ]
        self._deepseek_roster = [
            self._make_llm(
                OPENROUTER_DEEPSEEK_V4_PRO,
                max_tokens=_REASONING_MAX_TOKENS,
                reasoning=True,
            )
            for _ in range(5)
        ]
        self._checklist_gemini = self._make_llm(
            OPENROUTER_GEMINI_36_FLASH,
            max_tokens=_GEMINI_MAX_TOKENS,
            reasoning=True,
        )
        self._checklist_deepseek = self._make_llm(
            OPENROUTER_DEEPSEEK_V4_PRO,
            max_tokens=_REASONING_MAX_TOKENS,
            reasoning=True,
        )
        self._parser = self._make_llm(
            OPENROUTER_DEEPSEEK_V4_FLASH,
            max_tokens=_PARSER_MAX_TOKENS,
            reasoning=False,
        )
        self.set_llm(self._parser, "parser")
        self.set_llm("no_research", "researcher")

    @staticmethod
    def _max_tokens_for(model: str) -> int:
        if OPENROUTER_GEMINI_36_FLASH in model:
            return _GEMINI_MAX_TOKENS
        return _REASONING_MAX_TOKENS

    @staticmethod
    def _make_llm(model: str, *, max_tokens: int, reasoning: bool) -> GeneralLlm:
        # OpenRouter's unified reasoning field. "high" is on for every forecaster.
        # The parser only formats, so reasoning is off.
        extra_body = (
            {"reasoning": {"effort": "high", "enabled": True}}
            if reasoning
            else {"reasoning": {"effort": "none", "enabled": False}}
        )
        return GeneralLlm(
            model=model,
            temperature=1 if reasoning else 0,
            timeout=180 if reasoning else 90,
            allowed_tries=2,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )

    def _roster(self) -> list[GeneralLlm]:
        if self.spend.deepseek_only():
            return self._deepseek_roster
        return self._mixed_roster

    def _claim_forecaster_slot(self) -> int:
        with self._slot_lock:
            slot = self._next_slot
            self._next_slot += 1
            return slot

    def _question_limiter(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if self._limiter_for_loop is None or self._limiter_loop is not loop:
            self._limiter_for_loop = asyncio.Semaphore(self._max_concurrent_questions)
            self._limiter_loop = loop
        return self._limiter_for_loop

    def _current_forecaster(self) -> GeneralLlm:
        roster = self._roster()
        slot = _FORECASTER_SLOT.get()
        if slot is None:
            slot = 0
        return roster[slot % len(roster)]

    def _forecast_angle_block(self) -> str:
        slot = _FORECASTER_SLOT.get()
        if slot is None:
            slot = 0
        angle = FORECASTER_ANGLES[slot % len(FORECASTER_ANGLES)]
        model = self._current_forecaster().model
        return clean_indents(
            f"""
            How to approach this forecast ({model}):
            {angle}

            {FORECAST_GUARDRAIL}
            """
        )

    async def _invoke_llm(self, llm: GeneralLlm, prompt: str) -> str:
        try:
            return await llm.invoke(prompt)
        except Exception as exc:
            if not _looks_like_unsupported_parameter(exc):
                raise
            logger.warning(
                "Model %s rejected a request setting (%s). Retrying without the reasoning override.",
                llm.model,
                type(exc).__name__,
            )
            fallback = GeneralLlm(
                model=llm.model,
                temperature=1,
                timeout=llm.litellm_kwargs.get("timeout") or 180,
                allowed_tries=1,
                max_tokens=llm.litellm_kwargs.get("max_tokens"),
            )
            return await fallback.invoke(prompt)

    async def _metered(
        self, llm: GeneralLlm, prompt: str, *, purpose: str, kind: str
    ) -> str:
        estimate = estimate_call_cost_usd(llm.model, kind)
        return await self._metered_awaitable(
            purpose, estimate, self._invoke_llm(llm, prompt)
        )

    async def _metered_parser(self, awaitable):
        estimate = estimate_call_cost_usd(OPENROUTER_DEEPSEEK_V4_FLASH, "forecast")
        return await self._metered_awaitable(
            f"parser:{OPENROUTER_DEEPSEEK_V4_FLASH}", estimate, awaitable
        )

    async def _metered_awaitable(self, purpose: str, estimate: float, awaitable):
        if self.spend.would_exceed_cap(estimate):
            close = getattr(awaitable, "close", None)
            if close is not None:
                close()
            raise SpendCapReached(
                f"OpenRouter spend is ${self.spend.effective_usd():.4f}. "
                f"Another {purpose} call is estimated at ${estimate:.4f}, "
                f"which would pass the ${SPEND_HARD_CAP_USD:.2f} cap."
            )
        succeeded = False
        with MonetaryCostManager() as meter:
            try:
                result = await awaitable
                succeeded = True
            finally:
                measured = float(meter.current_usage)
                if measured > 0:
                    amount = measured
                    source = "measured"
                elif succeeded:
                    amount = estimate
                    source = "estimate"
                else:
                    amount = 0.0
                    source = "none"
                if amount > 0:
                    self.spend.add_unconfirmed(amount)
                    self.spend.refresh()
                    logger.info(
                        "LLM call purpose=%s cost_usd=%.6f cost_source=%s spend_usd=%.4f mode=%s",
                        purpose,
                        amount,
                        source,
                        self.spend.effective_usd(),
                        self.spend.mode_label(),
                    )
        return result

    async def _invoke_forecaster(self, prompt: str) -> str:
        llm = self._current_forecaster()
        return await self._metered(
            llm, prompt, purpose=f"forecast:{llm.model}", kind="forecast"
        )

    async def forecast_questions(
        self,
        questions,
        return_exceptions: bool = False,
    ):
        self.spend.refresh()
        if self.question_limit is not None:
            questions = list(questions)[: self.question_limit]
            logger.info("Limiting this run to %s question(s).", self.question_limit)
        if self.spend.over_hard_cap():
            message = (
                f"OpenRouter spend is ${self.spend.effective_usd():.4f}, "
                "which is at the US$4.50 cap. No further questions will be forecast."
            )
            logger.error(message)
            if return_exceptions:
                return [SpendCapReached(message)]
            raise SpendCapReached(message)
        logger.info(
            "OpenRouter spend_usd=%.4f mode=%s",
            self.spend.effective_usd(),
            self.spend.mode_label(),
        )
        return await super().forecast_questions(questions, return_exceptions)

    async def _run_individual_question(self, question: MetaculusQuestion):
        async with self._question_limiter():
            self._next_slot = 0
            self.spend.refresh()
            if self.spend.over_hard_cap():
                raise SpendCapReached(
                    f"OpenRouter spend is ${self.spend.effective_usd():.4f}, "
                    "which is at the US$4.50 cap."
                )
            logger.info(
                "Question %s spend_usd=%.4f mode=%s",
                question.page_url,
                self.spend.effective_usd(),
                self.spend.mode_label(),
            )
            try:
                return await super()._run_individual_question(question)
            finally:
                self.spend.refresh()

    async def _make_prediction(
        self, question: MetaculusQuestion, research: str
    ) -> ReasonedPrediction[PredictionTypes]:
        if _FORECASTER_SLOT.get() is not None:
            return await super()._make_prediction(question, research)
        slot = self._claim_forecaster_slot()
        token = _FORECASTER_SLOT.set(slot)
        try:
            return await super()._make_prediction(question, research)
        finally:
            _FORECASTER_SLOT.reset(token)

    async def _aggregate_predictions(
        self,
        predictions,
        question: MetaculusQuestion,
    ):
        aggregate = await super()._aggregate_predictions(predictions, question)
        if isinstance(aggregate, float):
            return clip_probability(aggregate)
        return aggregate

    ##################################### RESEARCH #####################################

    async def run_research(self, question: MetaculusQuestion) -> str:
        news = await self._collect_asknews(question)
        checklist = await self._resolution_checklist(question)
        research = clean_indents(
            f"""
            ## Resolution checklist (binding)
            {checklist}

            ## News
            {news}
            """
        )
        logger.info(f"Found Research for URL {question.page_url}:\n{research}")
        return research

    async def _collect_asknews(self, question: MetaculusQuestion) -> str:
        if not asknews_configured():
            logger.info(
                "AskNews credentials are not set. Skipping news research for %s.",
                question.page_url,
            )
            return (
                "No news briefing. AskNews credentials were not set, so news "
                "research was skipped."
            )
        try:
            # One news search per question (latest plus the archive lookup the
            # official searcher already pairs with it). Free tier; no LLM spend.
            raw = await AskNewsSearcher().get_formatted_news_async(question.question_text)
        except Exception as exc:
            logger.warning(
                "AskNews research failed (%s). Continuing without news. %s",
                type(exc).__name__,
                redact_secrets(str(exc)),
            )
            return "No news briefing. AskNews was unavailable for this question."
        stripped, removed = strip_market_and_crowd_odds(raw)
        if removed:
            logger.info(
                "Removed %s research line(s) that stated crowd or market odds.",
                removed,
            )
        return stripped or "AskNews returned no articles."

    async def _resolution_checklist(self, question: MetaculusQuestion) -> str:
        self.spend.refresh()
        if self.spend.deepseek_only():
            llm = self._checklist_deepseek
        else:
            llm = self._checklist_gemini
        prompt = resolution_checklist_prompt(
            question_text=question.question_text,
            resolution_criteria=question.resolution_criteria or "",
            fine_print=question.fine_print or "",
            background=question.background_info or "",
            today=datetime.now().strftime("%Y-%m-%d"),
        )
        return await self._metered(
            llm,
            clean_indents(prompt),
            purpose=f"checklist:{llm.model}",
            kind="checklist",
        )

    ##################################### BINARY QUESTIONS #####################################

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            {self._forecast_angle_block()}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.
            {self._get_conditional_disclaimer_if_necessary(question)}

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )

        return await self._binary_prompt_to_forecast(question, prompt)

    async def _binary_prompt_to_forecast(
        self,
        question: BinaryQuestion,
        prompt: str,
    ) -> ReasonedPrediction[float]:
        reasoning = await self._invoke_forecaster(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a probability forecast for a binary question.
            {self._create_resolved_question_parsing_message()}
            """
        )
        binary_prediction: BinaryPrediction = await self._metered_parser(structure_output(
            reasoning,
            BinaryPrediction,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=parsing_instructions,
        ))
        decimal_pred = clip_probability(binary_prediction.prediction_in_decimal)

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {decimal_pred}."
        )
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

    ##################################### MULTIPLE CHOICE QUESTIONS #####################################

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}


            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            {self._forecast_angle_block()}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of a scenario that results in an unexpected outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        return await self._multiple_choice_prompt_to_forecast(question, prompt)

    async def _multiple_choice_prompt_to_forecast(
        self,
        question: MultipleChoiceQuestion,
        prompt: str,
    ) -> ReasonedPrediction[PredictedOptionList]:
        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:
            {question.options}

            The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
            Additionally, you may sometimes need to parse a 0% probability. Please do not skip options with 0% but rather make it an entry in your final list with 0% probability.
            {self._create_resolved_question_parsing_message()}
            """
        )
        reasoning = await self._invoke_forecaster(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        predicted_option_list: PredictedOptionList = await self._metered_parser(structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=parsing_instructions,
        ))

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {predicted_option_list}."
        )
        return ReasonedPrediction(
            prediction_value=predicted_option_list, reasoning=reasoning
        )

    ##################################### NUMERIC QUESTIONS #####################################

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistant says:
            {research}

            {self._forecast_angle_block()}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested and give your answer in these units (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there. The value for percentile 10 should always be less than the value for percentile 20, and so on.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: XX (lowest number value)
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX (highest number value)
            "
            """
        )
        return await self._numeric_prompt_to_forecast(question, prompt)

    async def _numeric_prompt_to_forecast(
        self,
        question: NumericQuestion,
        prompt: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self._invoke_forecaster(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a numeric question.
            - This text is trying to answer the numeric question: "{question.question_text}".
            {self._create_single_distribution_parsing_message(question)}
            {self._create_resolved_question_parsing_message()}
            - When parsing the text, please make sure to give the values (the ones assigned to percentiles) in terms of the correct units.
            - The units for the forecast are: {question.unit_of_measure}
            - Your work will be shown publicly with these units stated verbatim after the numbers your parse.
            - As an example, someone else guessed that the answer will be between {question.lower_bound} {question.unit_of_measure} and {question.upper_bound} {question.unit_of_measure}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
            - If the answer doesn't give the answer in the correct units, you should parse it in the right units. For instance if the answer gives numbers as $500,000,000 and units are "B $" then you should parse the answer as 0.5 (since $500,000,000 is $0.5 billion).
            - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
            - Turn any values that are in scientific notation into regular numbers.
            """
        )
        percentile_list: list[Percentile] = await self._metered_parser(structure_output(
            reasoning,
            list[Percentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        ))
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}."
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    ##################################### DATE QUESTIONS #####################################

    async def _run_forecast_on_date(
        self, question: DateQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant says:
            {research}

            {self._forecast_angle_block()}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - This is a date question, and as such, the answer must be expressed in terms of dates.
            - The dates must be written in the format of YYYY-MM-DD. If hours matter, please append the date with the hour in UTC and military time: YYYY-MM-DDTHH:MM:SSZ. No other formatting is allowed.
            - Always start with a lower date chronologically and then increase from there.
            - Do NOT forget this. The dates must be written in chronological order starting at the earliest time at percentile 10 and increasing from there.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: YYYY-MM-DD (oldest date)
            Percentile 20: YYYY-MM-DD
            Percentile 40: YYYY-MM-DD
            Percentile 60: YYYY-MM-DD
            Percentile 80: YYYY-MM-DD
            Percentile 90: YYYY-MM-DD (newest date)
            "
            """
        )
        forecast = await self._date_prompt_to_forecast(question, prompt)
        return forecast

    async def _date_prompt_to_forecast(
        self,
        question: DateQuestion,
        prompt: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self._invoke_forecaster(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a date question.
            - This text is trying to answer the question: "{question.question_text}".
            {self._create_single_distribution_parsing_message(question)}
            {self._create_resolved_question_parsing_message()}
            - As an example, someone else guessed that the answer will be between {question.lower_bound} and {question.upper_bound}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
            - The output is given as dates/times please format it into a valid datetime parsable string. Assume midnight UTC if no hour is given.
            - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
            """
        )
        date_percentile_list: list[DatePercentile] = await self._metered_parser(structure_output(
            reasoning,
            list[DatePercentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        ))

        percentile_list = [
            Percentile(
                percentile=percentile.percentile,
                value=percentile.value.timestamp(),
            )
            for percentile in date_percentile_list
        ]
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}."
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    def _create_resolved_question_parsing_message(self) -> str:
        return "- If the text concludes that the question has already resolved *in the past* (i.e. it treats the question as decided rather than something to forecast), please DO NOT return a parsed output, even if a final forecast is also given. Instead indicate that the answer is not explicitly given in the text.\n"

    def _create_single_distribution_parsing_message(
        self, question: NumericQuestion | DateQuestion
    ) -> str:
        message = (
            "- The text may contain multiple percentile distributions (e.g. forecasts for several related questions/entities, or intermediate drafts before a final answer). You must return exactly ONE distribution: the single final distribution that answers the question stated above.\n"
            "- Never merge or concatenate percentile lists that refer to different entities, options, or scenarios. Each percentile should appear at most once in your output.\n"
            "- If there are multiple final distributions and you cannot tell which one answers the question stated above, do not guess or combine them. Instead indicate that the answer is not explicitly given in the text."
        )
        if question.group_question_option is not None:
            message += (
                f'\n- This question is specifically about "{question.group_question_option}" (one subquestion within a group of related questions). '
                f'If the text gives distributions for multiple subjects, return only the distribution for "{question.group_question_option}".'
            )
        return message

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion | DateQuestion
    ) -> tuple[str, str]:
        if isinstance(question, NumericQuestion):
            if question.nominal_upper_bound is not None:
                upper_bound_number = question.nominal_upper_bound
            else:
                upper_bound_number = question.upper_bound
            if question.nominal_lower_bound is not None:
                lower_bound_number = question.nominal_lower_bound
            else:
                lower_bound_number = question.lower_bound
            unit_of_measure = question.unit_of_measure
        elif isinstance(question, DateQuestion):
            upper_bound_number = question.upper_bound.date().isoformat()
            lower_bound_number = question.lower_bound.date().isoformat()
            unit_of_measure = ""
        else:
            raise ValueError()

        if question.open_upper_bound:
            upper_bound_message = f"The question creator thinks the number is likely not higher than {upper_bound_number} {unit_of_measure}."
        else:
            upper_bound_message = f"The outcome can not be higher than {upper_bound_number} {unit_of_measure}."

        if question.open_lower_bound:
            lower_bound_message = f"The question creator thinks the number is likely not lower than {lower_bound_number} {unit_of_measure}."
        else:
            lower_bound_message = f"The outcome can not be lower than {lower_bound_number} {unit_of_measure}."
        return upper_bound_message, lower_bound_message

    ##################################### CONDITIONAL QUESTIONS #####################################

    async def _run_forecast_on_conditional(
        self, question: ConditionalQuestion, research: str
    ) -> ReasonedPrediction[ConditionalPrediction]:
        parent_info, full_research = await self._get_question_prediction_info(
            question.parent, research, "parent"
        )
        child_info, full_research = await self._get_question_prediction_info(
            question.child, research, "child"
        )
        yes_info, full_research = await self._get_question_prediction_info(
            question.question_yes, full_research, "yes"
        )
        no_info, full_research = await self._get_question_prediction_info(
            question.question_no, full_research, "no"
        )
        full_reasoning = clean_indents(
            f"""
            ## Parent Question Reasoning
            {parent_info.reasoning}
            ## Child Question Reasoning
            {child_info.reasoning}
            ## Yes Question Reasoning
            {yes_info.reasoning}
            ## No Question Reasoning
            {no_info.reasoning}
        """
        )
        full_prediction = ConditionalPrediction(
            parent=parent_info.prediction_value,  # type: ignore
            child=child_info.prediction_value,  # type: ignore
            prediction_yes=yes_info.prediction_value,  # type: ignore
            prediction_no=no_info.prediction_value,  # type: ignore
        )
        return ReasonedPrediction(
            reasoning=full_reasoning, prediction_value=full_prediction
        )

    async def _get_question_prediction_info(
        self, question: MetaculusQuestion, research: str, question_type: str
    ) -> tuple[ReasonedPrediction[PredictionTypes | PredictionAffirmed], str]:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        previous_forecasts = question.previous_forecasts
        if (
            question_type in ["parent", "child"]
            and previous_forecasts
            and question_type not in self.force_reforecast_in_conditional
        ):
            # TODO: add option to not affirm current parent/child forecasts, create new forecast
            previous_forecast = previous_forecasts[-1]
            current_utc_time = datetime.now(timezone.utc)
            if (
                previous_forecast.timestamp_end is None
                or previous_forecast.timestamp_end > current_utc_time
            ):
                pretty_value = DataOrganizer.get_readable_prediction(previous_forecast)  # type: ignore
                prediction = ReasonedPrediction(
                    prediction_value=PredictionAffirmed(),
                    reasoning=f"Already existing forecast reaffirmed at {pretty_value}.",
                )
                return (prediction, research)  # type: ignore
        info = await self._make_prediction(question, research)
        full_research = self._add_reasoning_to_research(research, info, question_type)
        return info, full_research  # type: ignore

    def _add_reasoning_to_research(
        self,
        research: str,
        reasoning: ReasonedPrediction[PredictionTypes],
        question_type: str,
    ) -> str:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        question_type = question_type.title()
        return clean_indents(
            f"""
            {research}
            ---
            ## {question_type} Question Information
            You have previously forecasted the {question_type} Question to the value: {DataOrganizer.get_readable_prediction(reasoning.prediction_value)}
            This is relevant information for your current forecast, but it is NOT your current forecast, but previous forecasting information that is relevant to your current forecast.
            The reasoning for the {question_type} Question was as such:
            ```
            {reasoning.reasoning}
            ```
            This is absolutely essential: do NOT use this reasoning to re-forecast the {question_type} question.
            """
        )

    def _get_conditional_disclaimer_if_necessary(
        self, question: MetaculusQuestion
    ) -> str:
        if question.conditional_type not in ["yes", "no"]:
            return ""
        return clean_indents(
            """
            As you are given a conditional question with a parent and child, you are to only forecast the **CHILD** question, given the parent question's resolution.
            You never re-forecast the parent question under any circumstances, but you use probabilistic reasoning, strongly considering the parent question's resolution, to forecast the child question.
            """
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run the template forecasting bot")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
        help="What to forecast on (default: tournament)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Forecast without publishing to Metaculus",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Forecast at most this many open questions in each tournament",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode
    if args.max_questions is not None and args.max_questions < 1:
        parser.error("--max-questions must be at least 1")

    check_environment(strict=True)
    publish_to_metaculus = not args.dry_run
    print_startup_banner(run_mode, will_publish=publish_to_metaculus)

    spend_tracker = SpendTracker.load()
    spend_tracker.refresh()
    print(
        f"OpenRouter spend so far ${spend_tracker.effective_usd():.4f}. "
        f"Mixed ensemble until ${SPEND_SWITCH_USD:.2f}, "
        f"hard stop at ${SPEND_HARD_CAP_USD:.2f}. "
        f"Next question mode: {spend_tracker.mode_label()}.\n"
    )

    # Five forecasts per question, combined by median. Models, the parser,
    # and the resolution checklist are pinned inside FallTemplateBot2026.
    template_bot = FallTemplateBot2026(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        use_research_summary_to_forecast=False,
        enable_summarize_research=False,
        publish_reports_to_metaculus=publish_to_metaculus,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        spend_tracker=spend_tracker,
    )
    template_bot.question_limit = args.max_questions

    # Per-mode tournament URL shown in the summary banner footer. These
    # piggyback on the forecasting_tools SDK constants and need updating
    # whenever those rotate seasons.
    TOURNAMENT_URLS = {
        "tournament": "https://www.metaculus.com/tournament/fall-futureeval-2026/",
        "metaculus_cup": "https://www.metaculus.com/tournament/metaculus-cup-fall-2026/",
        "test_questions": "https://www.metaculus.com/tournament/bot-testing-area/",
    }

    # Dispatch on mode. Each branch produces a list of ForecastReport (or
    # exceptions, since return_exceptions=True) which then flows into the
    # summary printers below.
    client = MetaculusClient()
    if run_mode == "tournament":
        seasonal_tournament_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                client.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
        )
        minibench_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                client.CURRENT_MINIBENCH_ID, return_exceptions=True
            )
        )
        forecast_reports = seasonal_tournament_reports + minibench_reports
    elif run_mode == "metaculus_cup":
        # The Metaculus Cup may be uninitialized near the start of a season
        # (Jan/May/Sep). MetaculusClient.ACX_2025_TOURNAMENT = 32564 and
        # MetaculusClient.AI_2027_TOURNAMENT_ID = "ai-2027" are also valid
        # targets here.
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                client.CURRENT_METACULUS_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        # The bot-testing-area tournament contains all question types and is
        # the recommended target for smoke-testing your bot.
        # https://www.metaculus.com/tournament/bot-testing-area/
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                "bot-testing-area", return_exceptions=True
            )
        )

    template_bot.spend.refresh()
    print(
        f"OpenRouter spend recorded: ${template_bot.spend.effective_usd():.4f} "
        f"({template_bot.spend.mode_label()}).\n"
    )
    template_bot.log_report_summary(forecast_reports)
    print_run_summary_banner(
        forecast_reports,
        will_publish=publish_to_metaculus,
        tournament_url=TOURNAMENT_URLS.get(run_mode),
    )
