# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

from AdvancedIF.judge import (
    IFRubricsJudge,
    JudgeInput,
    Message,
    SystemSteerIFRubricsJudge,
)


class TestCalcRubricLevelPassRate(unittest.TestCase):
    """Test cases for _calc_rubric_level_pass_rate method."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.judge = IFRubricsJudge(api_key="test-key", model="gpt-4")

    def test_pass_rate_calculation(self) -> None:
        """Test calculating pass rate with various scenarios."""
        # All pass
        rubrics_check = {
            "question_1": "YES, correct",
            "question_2": "YES, correct",
        }
        rubrics = ["Rubric 1", "Rubric 2"]
        self.assertEqual(
            self.judge._calc_rubric_level_pass_rate(rubrics_check, rubrics), 1.0
        )

        # Partial pass (case-insensitive)
        rubrics_check = {
            "question_1": "yes, lowercase works",
            "question_2": "NO, this fails",
        }
        self.assertEqual(
            self.judge._calc_rubric_level_pass_rate(rubrics_check, rubrics), 0.5
        )

        # All fail
        rubrics_check = {
            "question_1": "NO",
            "question_2": "NO",
        }
        self.assertEqual(
            self.judge._calc_rubric_level_pass_rate(rubrics_check, rubrics), 0.0
        )

        # Empty rubrics
        self.assertEqual(self.judge._calc_rubric_level_pass_rate({}, []), 0.0)


class TestComposePrompt(unittest.TestCase):
    """Test cases for _compose_prompt methods."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.if_judge = IFRubricsJudge(api_key="test-key", model="gpt-4")
        self.system_judge = SystemSteerIFRubricsJudge(api_key="test-key", model="gpt-4")

    def test_if_judge_compose_prompt(self) -> None:
        """Test IFRubricsJudge prompt composition."""
        judge_input = JudgeInput(
            conversation_history=[
                Message(role="user", content="List 3 colors"),
                Message(role="assistant", content="Red, Blue, Green"),
                Message(role="user", content="Now 3 more"),
            ],
            response_text="Yellow, Purple, Orange",
            rubrics=["Does it provide exactly 3 colors?"],
            row_id="test_1",
        )

        prompt = self.if_judge._compose_prompt(judge_input)

        # Verify all key elements are in prompt
        self.assertIn("List 3 colors", prompt)
        self.assertIn("Red, Blue, Green", prompt)
        self.assertIn("Now 3 more", prompt)
        self.assertIn("Yellow, Purple, Orange", prompt)
        self.assertIn("Does it provide exactly 3 colors?", prompt)
        # Check that roles are used directly, not converted to labels
        self.assertIn("user [1]:", prompt)
        self.assertIn("assistant [1]:", prompt)

    def test_system_judge_compose_prompt(self) -> None:
        """Test SystemSteerIFRubricsJudge prompt composition."""
        judge_input = JudgeInput(
            conversation_history=[
                Message(role="system", content="Always respond in haiku format"),
                Message(role="user", content="Tell me about winter"),
            ],
            response_text="Snow falls silently\nCold wind blows\nWinter's grip",
            rubrics=["Does response follow haiku format?"],
            row_id="test_2",
        )

        prompt = self.system_judge._compose_prompt(judge_input)

        # Verify system prompt and other elements are in prompt
        self.assertIn("Always respond in haiku format", prompt)
        self.assertIn("Tell me about winter", prompt)
        self.assertIn("Snow falls silently", prompt)
        self.assertIn("Does response follow haiku format?", prompt)

    def test_system_judge_without_system_prompt(self) -> None:
        """Test SystemSteerIFRubricsJudge when no system prompt exists."""
        judge_input = JudgeInput(
            conversation_history=[Message(role="user", content="Hello")],
            response_text="Hi there!",
            rubrics=["Is response polite?"],
            row_id="test_3",
        )

        prompt = self.system_judge._compose_prompt(judge_input)

        # Should still compose prompt with empty system prompt
        self.assertIn("Hello", prompt)
        self.assertIn("Hi there!", prompt)


class TestJudgeResultFields(unittest.TestCase):
    """Test cases for JudgeResult fields including judge_prompt and raw_judge_output."""

    def test_judge_result_has_prompt_and_output_fields(self) -> None:
        """Test that JudgeResult includes judge_prompt and raw_judge_output fields."""
        from AdvancedIF.judge import Judgement, JudgeResult

        # Create a successful result with all fields
        result = JudgeResult(
            success=True,
            judgement=Judgement(
                rubrics_check={"question_1": "YES"},
                SATISFIED_ALL_REQUIREMENTS="YES",
            ),
            judge_prompt="This is the judge prompt",
            raw_judge_output='{"rubrics_check": {"question_1": "YES"}, "SATISFIED_ALL_REQUIREMENTS": "YES"}',
            rubric_level_pass_rate=1.0,
        )

        # Verify all fields are present
        self.assertTrue(result.success)
        self.assertIsNotNone(result.judgement)
        self.assertEqual(result.judge_prompt, "This is the judge prompt")
        self.assertIn("rubrics_check", result.raw_judge_output)
        self.assertEqual(result.rubric_level_pass_rate, 1.0)

    def test_judge_result_fields_can_be_none(self) -> None:
        """Test that judge_prompt and raw_judge_output can be None."""
        from AdvancedIF.judge import JudgeResult

        # Create a failed result
        result = JudgeResult(
            success=False,
            error="API call failed",
        )

        # Verify optional fields are None
        self.assertIsNone(result.judgement)
        self.assertIsNone(result.judge_prompt)
        self.assertIsNone(result.raw_judge_output)
        self.assertEqual(result.error, "API call failed")


if __name__ == "__main__":
    unittest.main()
