"""
ABOUTME: SSR Executor Service orchestrating survey execution with progress tracking
ABOUTME: Coordinates consumer generation, response generation, and SSR calculation
"""

from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import numpy as np
from datetime import datetime

from src.core.ssr_engine import SSREngine
from src.services.consumer_generator import ConsumerGenerator


@dataclass
class ConsumerResult:
    """Result for individual consumer"""

    consumer_id: str
    consumer_demographics: Dict
    response_text: str
    rating: float  # Mean purchase intent 1-5
    confidence: float  # Confidence score 0-1
    distribution: List[float]  # Probability distribution [P(1), P(2), ..., P(5)]


@dataclass
class SurveyExecutionResult:
    """Complete result from survey execution"""

    survey_id: str
    mean_rating: float  # Overall mean rating 1-5
    std_rating: float  # Standard deviation of ratings
    confidence: float  # Overall confidence (avg of individual confidences)
    distribution: List[float]  # Aggregated distribution
    consumer_results: List[ConsumerResult]
    total_consumers: int
    execution_time_seconds: float
    llm_model_used: str


class SSRExecutor:
    """
    Service for executing complete SSR surveys with progress tracking.

    This orchestrates the entire pipeline:
    1. Generate consumers (0-30% progress)
    2. Generate responses & calculate SSR (30-90% progress)
    3. Aggregate results (90-100% progress)
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize SSR Executor.

        Args:
            api_key: OpenAI API key (or use OPENAI_API_KEY env var)
        """
        from src.core.ssr_engine import SSRConfig
        
        self.consumer_generator = ConsumerGenerator(api_key=api_key)
        
        # Paper methodology: Temperature 1.5, multi-set averaging across 6 reference sets
        config = SSRConfig(
            temperature=1.5,  # Paper optimal temperature for balanced distributions  
            use_multi_set_averaging=True,  # Paper methodology: average across reference sets
            reference_set_ids=None  # Use default paper sets
        )
        self.ssr_engine = SSREngine(config=config, api_key=api_key)

    def execute_survey(
        self,
        survey_id: str,
        product_name: str,
        product_description: str,
        llm_model: str = "gpt-3.5-turbo",
        temperature: float = 1.0,
        enable_demographics: bool = True,
        consumer_count: int = 5,
        demographic_filters: Optional[Dict] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> SurveyExecutionResult:
        """
        Execute complete SSR survey with progress tracking.

        Args:
            survey_id: Unique survey identifier
            product_name: Name of product being surveyed
            product_description: Product description
            llm_model: LLM model for response generation (gpt-3.5-turbo or gpt-4o)
            temperature: LLM temperature (default: 1.0 optimal from paper)
            enable_demographics: If True, use diverse demographics; if False, generic
            consumer_count: Number of consumers (3-5 recommended)
            demographic_filters: Optional demographic filters (gender, income_bracket, location)
            progress_callback: Optional callback function(progress%, status_message)

        Returns:
            SurveyExecutionResult with all consumer results and aggregated metrics

        Raises:
            Exception: If survey execution fails
        """
        import logging

        logger = logging.getLogger(__name__)

        start_time = datetime.now()

        # Log survey execution configuration
        logger.info(f"Starting survey execution: {survey_id}")
        logger.info(f"Product: {product_name}")
        logger.info(f"LLM Model: {llm_model}")
        logger.info(f"Demographics enabled: {enable_demographics}")
        logger.info(f"Consumer count: {consumer_count}")
        if demographic_filters:
            logger.info(f"Demographic filters: {demographic_filters}")
        else:
            logger.info("No demographic filters applied")

        try:
            # ================================================================
            # PHASE 1: Consumer Generation (0-30% progress)
            # ================================================================
            if progress_callback:
                progress_callback(0, "Starting consumer generation")

            consumers = self.consumer_generator.generate_consumers(
                count=consumer_count,
                demographics_enabled=enable_demographics,
                demographic_filters=demographic_filters,
            )

            # Update progress after each consumer generated
            consumer_progress_step = 30 / len(consumers)
            generated_consumers = []
            for i, consumer in enumerate(consumers):
                generated_consumers.append(consumer)
                current_progress = int((i + 1) * consumer_progress_step)
                if progress_callback:
                    progress_callback(
                        current_progress,
                        f"Generated consumer {i + 1}/{len(consumers)}: {consumer.persona}",
                    )

            # ================================================================
            # PHASE 2: Response Generation & SSR Calculation (30-90% progress)
            # ================================================================
            consumer_results = []
            ssr_progress_step = 60 / len(consumers)  # 60% total for SSR phase

            for i, consumer in enumerate(generated_consumers):
                base_progress = 30 + int(i * ssr_progress_step)

                # Generate consumer response
                if progress_callback:
                    progress_callback(
                        base_progress,
                        f"Generating response for consumer {i + 1}/{len(consumers)}",
                    )

                response_text = self.consumer_generator.generate_response(
                    consumer=consumer,
                    product_name=product_name,
                    product_description=product_description,
                    llm_model=llm_model,
                    temperature=temperature,
                )

                # Calculate SSR
                mid_progress = base_progress + int(ssr_progress_step * 0.5)
                if progress_callback:
                    progress_callback(
                        mid_progress,
                        f"Calculating SSR for consumer {i + 1}/{len(consumers)}",
                    )

                ssr_result = self.ssr_engine.process_response(response_text)

                # Store result
                consumer_result = ConsumerResult(
                    consumer_id=consumer.consumer_id,
                    consumer_demographics={
                        "age": consumer.age,
                        "gender": consumer.gender,
                        "income": consumer.income,
                        "location": consumer.location,
                        "ethnicity": consumer.ethnicity,
                        "persona": consumer.persona,
                    },
                    response_text=response_text,
                    rating=ssr_result.mean_rating,
                    confidence=self._calculate_confidence(
                        ssr_result.distribution.probabilities
                    ),
                    distribution=ssr_result.distribution.probabilities,
                )
                consumer_results.append(consumer_result)

                # Update progress after consumer complete
                final_progress = 30 + int((i + 1) * ssr_progress_step)
                if progress_callback:
                    progress_callback(
                        final_progress,
                        f"Completed consumer {i + 1}/{len(consumers)} - Rating: {consumer_result.rating:.2f}",
                    )

            # ================================================================
            # PHASE 3: Aggregation (90-100% progress)
            # ================================================================
            if progress_callback:
                progress_callback(90, "Aggregating results")

            # Calculate overall statistics
            ratings = [result.rating for result in consumer_results]
            mean_rating = float(np.mean(ratings))
            std_rating = float(np.std(ratings))

            # Aggregate distribution (average across all consumers)
            all_distributions = [result.distribution for result in consumer_results]
            aggregated_distribution = list(np.mean(all_distributions, axis=0))

            # Calculate overall confidence (average of individual confidences)
            confidences = [result.confidence for result in consumer_results]
            overall_confidence = float(np.mean(confidences))

            if progress_callback:
                progress_callback(95, "Finalizing results")

            # Calculate execution time
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            if progress_callback:
                progress_callback(100, "Survey execution complete")

            return SurveyExecutionResult(
                survey_id=survey_id,
                mean_rating=mean_rating,
                std_rating=std_rating,
                confidence=overall_confidence,
                distribution=aggregated_distribution,
                consumer_results=consumer_results,
                total_consumers=len(consumer_results),
                execution_time_seconds=execution_time,
                llm_model_used=llm_model,
            )

        except Exception as e:
            if progress_callback:
                progress_callback(-1, f"Execution failed: {str(e)}")
            raise

    def _calculate_confidence(self, distribution: List[float]) -> float:
        """
        Calculate confidence score from probability distribution.

        Uses inverse of entropy, normalized to 0-1 scale.
        Higher values indicate more concentrated distribution (higher confidence).

        Args:
            distribution: Probability distribution over ratings 1-5

        Returns:
            Confidence score between 0 and 1
        """
        dist_array = np.array(distribution)

        # Calculate entropy (with small epsilon to avoid log(0))
        entropy = -np.sum(dist_array * np.log(dist_array + 1e-10))

        # Maximum entropy for 5 categories (uniform distribution)
        max_entropy = np.log(5)

        # Normalize to 0-1 (inverse of normalized entropy)
        confidence = 1 - (entropy / max_entropy)

        return float(confidence)
