# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import asyncio
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from AdvancedIF.judge import (
    IFRubricsJudge,
    JudgeInput,
    JudgeResult,
    Message,
    SystemSteerIFRubricsJudge,
)

logger: logging.Logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Fault-tolerant processor for batch evaluation of judge inputs.

    This processor handles per-row execution with fault tolerance,
    meaning if one row fails, it continues processing remaining rows.
    """

    def __init__(
        self,
        if_judge: IFRubricsJudge,
        system_steer_judge: SystemSteerIFRubricsJudge,
        max_concurrency: int = 10,
    ) -> None:
        """
        Initialize the processor with judge instances.

        Args:
            if_judge: IFRubricsJudge instance for IF tasks
            system_steer_judge: SystemSteerIFRubricsJudge instance for system steer tasks
            max_concurrency: Maximum number of concurrent requests (default: 10)
        """
        self.if_judge = if_judge
        self.system_steer_judge = system_steer_judge
        self.max_concurrency = max_concurrency
        self._semaphore = asyncio.Semaphore(max_concurrency)

    def _parse_dialog(self, dialog_data: str | List[Dict[str, Any]]) -> List[Message]:
        """
        Parse dialog/conversation history.

        Args:
            dialog_data: Either JSON string or list of message dicts with 'role' and 'content'

        Returns:
            List of Message objects
        """
        if isinstance(dialog_data, str):
            dialog_list = json.loads(dialog_data)
        else:
            dialog_list = dialog_data

        # Direct unpacking: extract only the fields we need
        messages = [
            Message(role=msg["role"], content=msg["content"]) for msg in dialog_list
        ]

        return messages

    def _parse_prompt_metadata(
        self, metadata_data: str | Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Parse prompt_metadata field.

        Args:
            metadata_data: Either JSON string or dict

        Returns:
            Dictionary with rubrics and potentially system_prompt
        """
        if isinstance(metadata_data, str):
            return json.loads(metadata_data)
        else:
            return metadata_data

    def _parse_response(self, response_data: str | Dict[str, Any]) -> str:
        """
        Parse response field (model-generated response).

        Args:
            response_data: Either a string or a message dict/list

        Returns:
            Response text
        """
        if isinstance(response_data, str):
            # Could be plain text or JSON string
            try:
                parsed = json.loads(response_data)
                if isinstance(parsed, list) and len(parsed) > 0:
                    # List of message objects - get the last one
                    return parsed[-1].get("content", "")
                elif isinstance(parsed, dict):
                    # Single message object
                    return parsed.get("content", "")
                else:
                    raise TypeError(
                        f"Unrecognized type {type(parsed)} inside _parse_response."
                    )
            except (json.JSONDecodeError, ValueError):
                # Plain text response
                return response_data
        elif isinstance(response_data, list) and len(response_data) > 0:
            # List of message dicts - get the last one
            return response_data[-1].get("content", "")
        elif isinstance(response_data, dict):
            # Single message dict
            return response_data.get("content", "")
        return ""

    async def process_row_async(
        self, row_data: Dict[str, Any]
    ) -> tuple[Dict[str, Any], JudgeResult]:
        """
        Process a single row of data asynchronously with concurrency control.

        Args:
            row_data: Dictionary containing required fields:
                - conversation_history: List of MessageV2 messages or JSON string (without last assistant turn)
                - response: Model-generated response (last assistant turn)
                - prompt_metadata: Metadata with rubrics or JSON string
                - benchmark_name (optional): Task identifier

        Returns:
            JudgeResult containing evaluation outcome
        """
        async with self._semaphore:
            try:
                # Parse conversation history (without last assistant turn)
                conversation_history = self._parse_dialog(
                    row_data["conversation_history"]
                )

                # Parse response (model-generated response)
                response_text = self._parse_response(row_data["response"])

                # Parse metadata
                metadata = self._parse_prompt_metadata(row_data["prompt_metadata"])

                # Extract rubrics
                if "rubrics" in metadata:
                    if isinstance(metadata["rubrics"], str):
                        rubrics = json.loads(metadata["rubrics"])
                    else:
                        rubrics = metadata["rubrics"]
                else:
                    raise ValueError("Rubrics not found in prompt_metadata")

                # Determine which judge to use based on benchmark_name
                benchmark_name = row_data.get("benchmark_name", "")

                judge_input = JudgeInput(
                    conversation_history=conversation_history,
                    response_text=response_text,
                    rubrics=rubrics,
                )

                # Run judge evaluation in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                if benchmark_name == "if_system_steerability_oss":
                    result = await loop.run_in_executor(
                        None, self.system_steer_judge.evaluate, judge_input
                    )
                else:
                    result = await loop.run_in_executor(
                        None, self.if_judge.evaluate, judge_input
                    )
                return (row_data, result)

            except Exception as e:
                logger.error(f"Failed to process row: {e}", exc_info=True)
                return (
                    row_data,
                    JudgeResult(
                        success=False,
                        error=f"Row processing error: {e}",
                    ),
                )

    def process_row(self, row_data: Dict[str, Any]) -> JudgeResult:
        """Process a single row of data (sync wrapper for async method)."""
        _, result = asyncio.run(self.process_row_async(row_data))
        return result

    def process_file(
        self,
        input_file: Path,
        output_file: Path,
        task_filter: str | None = None,
    ) -> Dict[str, Any]:
        """
        Process a file (CSV or JSONL) with fault tolerance.

        Args:
            input_file: Path to input file
            output_file: Path to output file
            task_filter: Optional task name to filter (e.g., "system_steerability_v2")

        Returns:
            Dictionary with processing statistics
        """
        input_suffix = input_file.suffix.lower()

        if input_suffix == ".csv":
            return self.process_csv(input_file, output_file, task_filter)
        elif input_suffix == ".jsonl":
            return self.process_jsonl(input_file, output_file, task_filter)
        else:
            raise ValueError(
                f"Unsupported input file format: {input_suffix}. Use .csv or .jsonl"
            )

    async def _read_and_filter_rows(
        self,
        input_file: Path,
        task_filter: str | None,
        file_type: str,
    ) -> tuple[int, int, List[Dict[str, Any]], List[JudgeResult]]:
        """
        Read and filter rows from input file.

        Args:
            input_file: Path to input file
            task_filter: Optional task name to filter
            file_type: "csv" or "jsonl"

        Returns:
            Tuple of (total_rows, filtered_rows, rows_to_process, parse_errors)
        """
        total_rows = 0
        filtered_rows = 0
        rows_to_process = []
        parse_errors = []

        with open(input_file, "r") as f:
            if file_type == "csv":
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader):
                    total_rows += 1
                    if task_filter and row.get("benchmark_name", "") != task_filter:
                        filtered_rows += 1
                        continue
                    rows_to_process.append(row)

            else:  # jsonl
                for idx, line in enumerate(f):
                    if not line.strip():
                        continue
                    total_rows += 1

                    try:
                        row = json.loads(line)
                        if task_filter and row.get("benchmark_name", "") != task_filter:
                            filtered_rows += 1
                            continue
                        rows_to_process.append(row)

                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON at line {idx + 1}: {e}")
                        parse_errors.append(
                            JudgeResult(
                                success=False,
                                error=f"JSON parsing error: {e}",
                            )
                        )

        return total_rows, filtered_rows, rows_to_process, parse_errors

    async def _process_file_async(
        self,
        input_file: Path,
        output_file: Path,
        task_filter: str | None,
        file_type: str,
    ) -> Dict[str, Any]:
        """
        Core async processing logic shared between CSV and JSONL.

        Args:
            input_file: Path to input file
            output_file: Path to output file
            task_filter: Optional task name to filter
            file_type: "csv" or "jsonl"

        Returns:
            Dictionary with processing statistics
        """
        logger.info(
            f"Starting async {file_type.upper()} processing: {input_file} (max_concurrency={self.max_concurrency})"
        )

        try:
            # Read and filter rows
            (
                total_rows,
                filtered_rows,
                rows_to_process,
                parse_errors,
            ) = await self._read_and_filter_rows(input_file, task_filter, file_type)

            # Process all rows concurrently
            logger.info(f"Processing {len(rows_to_process)} rows in parallel...")
            tasks = [self.process_row_async(row) for row in rows_to_process]
            results_with_data = await asyncio.gather(*tasks)

            # Add parse errors to results (JSONL only, but harmless for CSV)
            for err in parse_errors:
                results_with_data = list(results_with_data) + [({}, err)]

            # Count successes and failures
            successful_rows = sum(1 for _, r in results_with_data if r.success)
            failed_rows = sum(1 for _, r in results_with_data if not r.success)

        except Exception as e:
            logger.error(
                f"Critical error reading {file_type.upper()} file: {e}", exc_info=True
            )
            raise

        # Write results using appropriate writer
        logger.info(f"Writing results to: {output_file}")
        if file_type == "csv":
            self._write_results_to_csv(results_with_data, output_file)
        else:
            self._write_results_to_jsonl(results_with_data, output_file)

        # Extract just the results for stats calculation
        results = [r for _, r in results_with_data]
        overall_stats = self._calculate_stats(results, total_rows, filtered_rows)

        # Group by task and calculate per-task stats
        from collections import defaultdict

        task_groups = defaultdict(list)
        for row_data, result in results_with_data:
            task_name = row_data.get("benchmark_name", "unknown")
            task_groups[task_name].append(result)

        logger.info(
            f"Processing complete. Success: {successful_rows}/{len(results_with_data)} "
            f"({overall_stats['success_rate']:.1%}), Failed: {failed_rows}, "
            f"Filtered: {filtered_rows}"
        )
        logger.info(f"\nOverall pass rate: {overall_stats['overall_pass_rate']:.1%}")
        logger.info(
            f"Overall micro-level rubric pass rate: {overall_stats['micro_pass_rate']:.1%}"
        )

        # Print per-task stats
        logger.info("\n--- Stats by Task ---")
        for task_name, task_results in sorted(task_groups.items()):
            task_stats = self._calculate_stats(task_results, len(task_results), 0)
            logger.info(
                f"{task_name}: {len(task_results)} samples, "
                f"Pass rate: {task_stats['overall_pass_rate']:.1%}, "
                f"Micro pass rate: {task_stats['micro_pass_rate']:.1%}"
            )

        return overall_stats

    async def process_csv_async(
        self,
        input_file: Path,
        output_file: Path,
        task_filter: str | None = None,
    ) -> Dict[str, Any]:
        """
        Process a CSV file with fault tolerance and parallel execution.

        Args:
            input_file: Path to input CSV file
            output_file: Path to output CSV file
            task_filter: Optional task name to filter

        Returns:
            Dictionary with processing statistics
        """
        return await self._process_file_async(
            input_file, output_file, task_filter, "csv"
        )

    def process_csv(
        self,
        input_file: Path,
        output_file: Path,
        task_filter: str | None = None,
    ) -> Dict[str, Any]:
        """Process a CSV file (sync wrapper for async method)."""
        return asyncio.run(self.process_csv_async(input_file, output_file, task_filter))

    async def process_jsonl_async(
        self,
        input_file: Path,
        output_file: Path,
        task_filter: str | None = None,
    ) -> Dict[str, Any]:
        """
        Process a JSONL file with fault tolerance and parallel execution.

        Args:
            input_file: Path to input JSONL file
            output_file: Path to output JSONL file
            task_filter: Optional task name to filter

        Returns:
            Dictionary with processing statistics
        """
        return await self._process_file_async(
            input_file, output_file, task_filter, "jsonl"
        )

    def process_jsonl(
        self,
        input_file: Path,
        output_file: Path,
        task_filter: str | None = None,
    ) -> Dict[str, Any]:
        """Process a JSONL file (sync wrapper for async method)."""
        return asyncio.run(
            self.process_jsonl_async(input_file, output_file, task_filter)
        )

    def _calculate_stats(
        self, results: List[JudgeResult], total_rows: int, filtered_rows: int
    ) -> Dict[str, Any]:
        """Calculate processing statistics including pass rates."""
        successful_rows = sum(1 for r in results if r.success)
        failed_rows = sum(1 for r in results if not r.success)

        # Overall pass rate: samples where all rubrics passed
        overall_passed = sum(
            1
            for r in results
            if r.success
            and r.judgement
            and r.judgement.SATISFIED_ALL_REQUIREMENTS.lower() == "yes"
        )

        # Micro-level pass rate: total passed rubrics / total rubrics
        total_rubrics = 0
        passed_rubrics = 0

        for result in results:
            if result.success and result.judgement:
                for (
                    _question_key,
                    decision_value,
                ) in result.judgement.rubrics_check.items():
                    total_rubrics += 1
                    if "yes" in decision_value.lower():
                        passed_rubrics += 1

        return {
            "total_rows": total_rows,
            "processed_rows": len(results),
            "successful_rows": successful_rows,
            "failed_rows": failed_rows,
            "filtered_rows": filtered_rows,
            "success_rate": successful_rows / len(results) if results else 0.0,
            "overall_pass_rate": overall_passed / len(results) if results else 0.0,
            "micro_pass_rate": passed_rubrics / total_rubrics
            if total_rubrics > 0
            else 0.0,
            "total_rubrics": total_rubrics,
            "passed_rubrics": passed_rubrics,
        }

    def _write_results_to_csv(
        self,
        results_with_data: List[tuple[Dict[str, Any], JudgeResult]],
        output_file: Path,
    ) -> None:
        """Write results to CSV file with original data and judge results merged."""
        with open(output_file, "w", newline="") as f:
            # Dynamically determine max number of rubrics and collect all field names
            max_rubrics = 0
            all_original_fields = set()

            for row_data, result in results_with_data:
                all_original_fields.update(row_data.keys())
                if result.judgement:
                    max_rubrics = max(max_rubrics, len(result.judgement.rubrics_check))

            # Create fieldnames: original fields + judge result fields
            original_fieldnames = sorted(list(all_original_fields))
            judge_fieldnames = [
                "judge_success",
                "judge_satisfied_all_requirements",
                "judge_rubric_level_pass_rate",
            ]

            # Add per-rubric columns
            for i in range(1, max_rubrics + 1):
                judge_fieldnames.append(f"judge_rubric_{i}_decision")

            judge_fieldnames.extend(["judge_prompt", "judge_raw_output", "judge_error"])

            fieldnames = original_fieldnames + judge_fieldnames

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for row_data, result in results_with_data:
                # Start with original data
                output_row = dict(row_data)

                # Add judge results
                output_row["judge_success"] = result.success
                output_row["judge_satisfied_all_requirements"] = (
                    result.judgement.SATISFIED_ALL_REQUIREMENTS
                    if result.judgement
                    else None
                )
                output_row["judge_rubric_level_pass_rate"] = (
                    result.rubric_level_pass_rate
                )
                output_row["judge_prompt"] = result.judge_prompt
                output_row["judge_raw_output"] = result.raw_judge_output
                output_row["judge_error"] = result.error

                # Add per-rubric decisions
                if result.judgement:
                    for (
                        question_key,
                        decision_value,
                    ) in result.judgement.rubrics_check.items():
                        # Extract rubric number from question_key (e.g., "question_1" -> "1")
                        rubric_num = question_key.split("_")[1]
                        output_row[f"judge_rubric_{rubric_num}_decision"] = (
                            decision_value
                        )

                writer.writerow(output_row)

    def _write_results_to_jsonl(
        self,
        results_with_data: List[tuple[Dict[str, Any], JudgeResult]],
        output_file: Path,
    ) -> None:
        """Write results to JSONL file with original data and judge results merged."""
        with open(output_file, "w") as f:
            for row_data, result in results_with_data:
                # Start with original data
                output_dict = dict(row_data)

                # Add judge results under a nested "judge_result" key
                judge_result_dict: Dict[str, Any] = {
                    "success": result.success,
                }

                if result.judgement:
                    judge_result_dict["satisfied_all_requirements"] = (
                        result.judgement.SATISFIED_ALL_REQUIREMENTS
                    )
                    judge_result_dict["rubrics_check"] = result.judgement.rubrics_check
                    judge_result_dict["rubric_level_pass_rate"] = (
                        result.rubric_level_pass_rate
                    )

                if result.judge_prompt:
                    judge_result_dict["judge_prompt"] = result.judge_prompt

                if result.raw_judge_output:
                    judge_result_dict["raw_output"] = result.raw_judge_output

                if result.error:
                    judge_result_dict["error"] = result.error

                output_dict["judge_result"] = judge_result_dict

                f.write(json.dumps(output_dict) + "\n")
