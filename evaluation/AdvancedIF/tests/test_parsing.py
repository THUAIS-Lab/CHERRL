# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json
import unittest
from pathlib import Path

from AdvancedIF.processor import DataProcessor


class TestRealSampleData(unittest.TestCase):
    """Test cases using real sample data from examples directory."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create mock judges (we won't actually call them in these tests)
        self.processor = DataProcessor(
            if_judge=None,  # type: ignore
            system_steer_judge=None,  # type: ignore
        )

        # Load real sample data paths
        self.examples_dir = Path(__file__).parent.parent / "examples"
        self.sample_jsonl = self.examples_dir / "sample_data.jsonl"
        self.sample_csv = self.examples_dir / "sample_data.csv"

    def test_parse_real_jsonl_sample(self) -> None:
        """Test parsing a real sample from the JSONL file."""
        # Load first line from JSONL
        with open(self.sample_jsonl, "r") as f:
            line = f.readline()
            row = json.loads(line)

        # Test parsing response field
        response = self.processor._parse_response(row["response"])
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        # Check that response contains expected content from the sample
        self.assertIn("First and Second Quarter Grades", response)

        # Test parsing conversation_history field
        conversation = self.processor._parse_dialog(row["conversation_history"])
        self.assertIsInstance(conversation, list)
        self.assertGreater(len(conversation), 0)
        # Check first message is from user
        self.assertEqual(conversation[0].role, "user")
        self.assertIn("grades", conversation[0].content.lower())

        # Test parsing prompt_metadata field
        metadata = self.processor._parse_prompt_metadata(row["prompt_metadata"])
        self.assertIsInstance(metadata, dict)
        self.assertIn("rubrics", metadata)

        # Test that rubrics can be parsed (it's a JSON string in metadata)
        rubrics_data = metadata["rubrics"]
        if isinstance(rubrics_data, str):
            rubrics = json.loads(rubrics_data)
        else:
            rubrics = rubrics_data
        self.assertIsInstance(rubrics, list)
        self.assertGreater(len(rubrics), 0)

        # Test benchmark_name field
        self.assertEqual(row["benchmark_name"], "if_carried_context_oss")

    def test_parse_all_jsonl_samples(self) -> None:
        """Test that all samples in JSONL can be parsed without errors."""
        errors = []
        with open(self.sample_jsonl, "r") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue

                try:
                    row = json.loads(line)

                    # Parse all fields
                    response = self.processor._parse_response(row["response"])
                    conversation = self.processor._parse_dialog(
                        row["conversation_history"]
                    )
                    metadata = self.processor._parse_prompt_metadata(
                        row["prompt_metadata"]
                    )

                    # Basic validation
                    self.assertIsInstance(response, str)
                    self.assertIsInstance(conversation, list)
                    self.assertIsInstance(metadata, dict)
                    self.assertIn("rubrics", metadata)

                except Exception as e:
                    errors.append(f"Line {idx + 1}: {e}")

        if errors:
            self.fail(f"Failed to parse {len(errors)} samples:\n" + "\n".join(errors))

    def test_parse_real_csv_sample(self) -> None:
        """Test parsing a real sample from the CSV file."""
        import csv as csv_module

        # Load first data row from CSV
        with open(self.sample_csv, "r") as f:
            reader = csv_module.DictReader(f)
            row = next(reader)

        # Test parsing response field
        response = self.processor._parse_response(row["response"])
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

        # Test parsing conversation_history field
        conversation = self.processor._parse_dialog(row["conversation_history"])
        self.assertIsInstance(conversation, list)
        self.assertGreater(len(conversation), 0)

        # Test parsing prompt_metadata field
        metadata = self.processor._parse_prompt_metadata(row["prompt_metadata"])
        self.assertIsInstance(metadata, dict)
        self.assertIn("rubrics", metadata)

    def test_parse_all_csv_samples(self) -> None:
        """Test that all samples in CSV can be parsed without errors."""
        import csv as csv_module

        errors = []
        with open(self.sample_csv, "r") as f:
            reader = csv_module.DictReader(f)

            for idx, row in enumerate(reader):
                try:
                    # Parse all fields
                    response = self.processor._parse_response(row["response"])
                    conversation = self.processor._parse_dialog(
                        row["conversation_history"]
                    )
                    metadata = self.processor._parse_prompt_metadata(
                        row["prompt_metadata"]
                    )

                    # Basic validation
                    self.assertIsInstance(response, str)
                    self.assertIsInstance(conversation, list)
                    self.assertIsInstance(metadata, dict)
                    self.assertIn("rubrics", metadata)

                except Exception as e:
                    errors.append(f"Row {idx + 1}: {e}")

        if errors:
            self.fail(f"Failed to parse {len(errors)} samples:\n" + "\n".join(errors))

    def test_conversation_history_excludes_last_assistant_turn(self) -> None:
        """Test that conversation_history does not include the last assistant turn."""
        # Load first line from JSONL
        with open(self.sample_jsonl, "r") as f:
            line = f.readline()
            row = json.loads(line)

        # Parse conversation and response
        conversation = self.processor._parse_dialog(row["conversation_history"])
        response = self.processor._parse_response(row["response"])

        # The last message in conversation should be from user (before the response)
        # Response is the last assistant turn
        if conversation:
            # In IF tasks, typically the conversation includes previous turns
            # The response is the final assistant turn being evaluated
            # So conversation should end with user message or have balanced turns
            self.assertEqual(conversation[-1].role, "user")
            self.assertGreater(len(response), 0, "Response should not be empty")

    def test_rubrics_format(self) -> None:
        """Test that rubrics from real data are in the expected format."""
        # Load first line from JSONL
        with open(self.sample_jsonl, "r") as f:
            line = f.readline()
            row = json.loads(line)

        # Parse metadata and extract rubrics
        metadata = self.processor._parse_prompt_metadata(row["prompt_metadata"])

        # Rubrics might be a JSON string within metadata
        rubrics_data = metadata["rubrics"]
        if isinstance(rubrics_data, str):
            rubrics = json.loads(rubrics_data)
        else:
            rubrics = rubrics_data

        # Validate rubrics structure
        self.assertIsInstance(rubrics, list)
        self.assertGreater(len(rubrics), 0)

        # Each rubric should be a string (question)
        for rubric in rubrics:
            self.assertIsInstance(rubric, str)
            self.assertGreater(len(rubric), 0)


class TestOutputFields(unittest.TestCase):
    """Test cases for output fields including judge_prompt and raw_judge_output."""

    def test_csv_output_includes_new_fields(self) -> None:
        """Test that CSV output includes judge_prompt and raw_output columns merged with original data."""
        import csv as csv_module
        import tempfile

        from AdvancedIF.judge import Judgement, JudgeResult

        # Create mock input data
        input_row = {
            "conversation_history": '[{"role": "user", "content": "test"}]',
            "response": "test response",
            "benchmark_name": "test_task",
        }

        # Create mock results with new fields
        results_with_data = [
            (
                input_row,
                JudgeResult(
                    success=True,
                    judgement=Judgement(
                        rubrics_check={"question_1": "YES"},
                        SATISFIED_ALL_REQUIREMENTS="YES",
                    ),
                    judge_prompt="Test prompt for row 1",
                    raw_judge_output='{"rubrics_check": {"question_1": "YES"}, "SATISFIED_ALL_REQUIREMENTS": "YES"}',
                    rubric_level_pass_rate=1.0,
                ),
            )
        ]

        # Write to temporary CSV
        processor = DataProcessor(if_judge=None, system_steer_judge=None)  # type: ignore
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            output_path = Path(f.name)

        try:
            processor._write_results_to_csv(results_with_data, output_path)

            # Read back and verify columns exist
            with open(output_path, "r") as f:
                reader = csv_module.DictReader(f)
                headers = reader.fieldnames

                # Check original fields are present
                self.assertIn("conversation_history", headers)
                self.assertIn("response", headers)
                self.assertIn("benchmark_name", headers)

                # Check new judge fields are in headers
                self.assertIn("judge_prompt", headers)
                self.assertIn("judge_raw_output", headers)
                self.assertIn("judge_success", headers)

                # Check values in row
                row = next(reader)
                self.assertEqual(row["benchmark_name"], "test_task")
                self.assertEqual(row["judge_prompt"], "Test prompt for row 1")
                self.assertIn("rubrics_check", row["judge_raw_output"])
        finally:
            output_path.unlink()

    def test_jsonl_output_includes_new_fields(self) -> None:
        """Test that JSONL output includes judge results merged with original data."""
        import tempfile

        from AdvancedIF.judge import Judgement, JudgeResult

        # Create mock input data
        input_row = {
            "conversation_history": '[{"role": "user", "content": "test"}]',
            "response": "test response",
            "benchmark_name": "test_task",
        }

        # Create mock results with new fields
        results_with_data = [
            (
                input_row,
                JudgeResult(
                    success=True,
                    judgement=Judgement(
                        rubrics_check={"question_1": "YES"},
                        SATISFIED_ALL_REQUIREMENTS="YES",
                    ),
                    judge_prompt="Test prompt for row 1",
                    raw_judge_output='{"rubrics_check": {"question_1": "YES"}, "SATISFIED_ALL_REQUIREMENTS": "YES"}',
                    rubric_level_pass_rate=1.0,
                ),
            )
        ]

        # Write to temporary JSONL
        processor = DataProcessor(if_judge=None, system_steer_judge=None)  # type: ignore
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            output_path = Path(f.name)

        try:
            processor._write_results_to_jsonl(results_with_data, output_path)

            # Read back and verify fields exist
            with open(output_path, "r") as f:
                line = f.readline()
                result_dict = json.loads(line)

                # Check original fields are present
                self.assertIn("conversation_history", result_dict)
                self.assertIn("response", result_dict)
                self.assertIn("benchmark_name", result_dict)

                # Check judge_result nested object is present
                self.assertIn("judge_result", result_dict)
                judge_result = result_dict["judge_result"]

                # Check new fields are present in judge_result
                self.assertIn("judge_prompt", judge_result)
                self.assertIn("raw_output", judge_result)
                self.assertEqual(judge_result["judge_prompt"], "Test prompt for row 1")
                self.assertIn("rubrics_check", judge_result["raw_output"])
        finally:
            output_path.unlink()


if __name__ == "__main__":
    unittest.main()
