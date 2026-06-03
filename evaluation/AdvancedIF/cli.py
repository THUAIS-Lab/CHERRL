# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import os
import sys
from pathlib import Path

import click

from AdvancedIF.judge import IFRubricsJudge, SystemSteerIFRubricsJudge
from AdvancedIF.processor import DataProcessor


def setup_logging(log_level: str, log_file: str | None = None) -> None:
    """
    Configure logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path. If None, logs to console only.
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    handlers: list[logging.Handler] = []

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    handlers.append(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)

    logging.basicConfig(level=numeric_level, handlers=handlers)


@click.group()
def cli() -> None:
    """AdvancedIF: LLM-as-a-Judge for Instruction Following"""
    pass


@cli.command()
@click.option(
    "--input",
    "-i",
    "input_file",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Input file (CSV or JSONL)",
)
@click.option(
    "--output",
    "-o",
    "output_file",
    required=True,
    type=click.Path(path_type=Path),
    help="Output file (CSV or JSONL)",
)
@click.option(
    "--task",
    "-t",
    "task_filter",
    type=click.Choice(
        ["if_carried_context_oss", "if_complex_if_oss", "if_system_steerability_oss"]
    ),
    help="Filter to process only specific task",
)
@click.option(
    "--api-key",
    "-k",
    "api_key",
    envvar="OPENAI_API_KEY",
    help="OpenAI API key (or set OPENAI_API_KEY env var)",
)
@click.option(
    "--model",
    "-m",
    default="o3-mini-2025-01-31",
    help="OpenAI model to use",
)
@click.option(
    "--max_completion_tokens",
    default=32768,
    type=int,
    help="Maximum tokens for response",
)
@click.option(
    "--max_concurrency",
    default=50,
    type=int,
    help="Maximum cocurrency for processing",
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    help="Logging level",
)
@click.option(
    "--log-file",
    type=click.Path(path_type=Path),
    help="Optional log file path",
)
def evaluate(
    input_file: Path,
    output_file: Path,
    task_filter: str | None,
    api_key: str | None,
    model: str,
    max_completion_tokens: int,
    max_concurrency: int,
    temperature: float,
    log_level: str,
    log_file: Path | None,
) -> None:
    """Evaluate input data using LLM-as-a-Judge."""

    setup_logging(log_level, str(log_file) if log_file else None)
    logger = logging.getLogger(__name__)

    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error(
                "OpenAI API key not provided. Set --api-key or OPENAI_API_KEY env var"
            )
            sys.exit(1)

    logger.info("Initializing judges with model: %s", model)

    # Initialize both judges
    if_judge = IFRubricsJudge(
        api_key=api_key,
        model=model,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
    )

    system_steer_judge = SystemSteerIFRubricsJudge(
        api_key=api_key,
        model=model,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
    )

    processor = DataProcessor(
        if_judge=if_judge,
        system_steer_judge=system_steer_judge,
        max_concurrency=max_concurrency,
    )

    try:
        if task_filter:
            logger.info(f"Filtering to process only task: {task_filter}")

        stats = processor.process_file(input_file, output_file, task_filter)

        logger.info("Evaluation complete!")
        logger.info("Total rows in file: %d", stats["total_rows"])
        logger.info("Filtered rows: %d", stats["filtered_rows"])
        logger.info("Processed rows: %d", stats["processed_rows"])
        logger.info("Successful: %d", stats["successful_rows"])
        logger.info("Failed: %d", stats["failed_rows"])
        logger.info("Success rate: %.2f%%", stats["success_rate"] * 100)
        logger.info("=" * 60)
        logger.info("Overall pass rate: %.2f%%", stats["overall_pass_rate"] * 100)
        logger.info("=" * 60)
        logger.info(
            "Micro-level rubric pass rate: %.2f%% (%d/%d rubrics passed)",
            stats["micro_pass_rate"] * 100,
            stats["passed_rubrics"],
            stats["total_rubrics"],
        )

    except Exception as e:
        logger.error("Fatal error during evaluation: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
