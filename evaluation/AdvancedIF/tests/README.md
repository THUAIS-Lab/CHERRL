# Test Suite for AdvancedIF

This directory contains comprehensive unit tests for the AdvancedIF codebase.

## Test Coverage

### `test_parsing.py`
Tests for parsing and formatting functions in the AdvancedIF system:

**TestProcessorParsing** - Tests for DataProcessor parsing functions:
- `test_parse_dialog_from_list`: Parses MessageV2 dialog from list
- `test_parse_dialog_from_json_string`: Parses MessageV2 dialog from JSON string
- `test_parse_dialog_with_system_message`: Handles system messages correctly
- `test_parse_dialog_with_unknown_source`: Defaults unknown sources to "user"
- `test_parse_dialog_with_full_messagev2_fields`: Extracts only needed fields from complete MessageV2
- `test_parse_extra_metadata_from_dict`: Parses metadata dictionary
- `test_parse_extra_metadata_from_json_string`: Parses metadata from JSON string
- `test_parse_extra_metadata_with_nested_json_rubrics`: Handles nested JSON rubrics
- `test_extract_response_text_from_dialog`: Extracts last assistant message
- `test_extract_response_text_with_multiple_assistant_messages`: Handles multi-turn dialogs
- `test_extract_response_text_no_assistant_message`: Returns empty string when no assistant message
- `test_extract_response_text_empty_dialog`: Handles empty dialogs gracefully

**TestJudgeParsing** - Tests for Judge parsing and formatting functions:
- `test_format_conversation_history`: Formats conversation with turn numbering
- `test_format_conversation_history_single_turn`: Handles single-turn conversations
- `test_format_conversation_history_with_system_message`: Includes system messages
- `test_get_last_user_turn`: Extracts last user message
- `test_get_last_user_turn_no_user_message`: Handles conversations without user messages
- `test_get_last_user_turn_empty_conversation`: Handles empty conversations
- `test_calc_rubric_level_pass_rate_all_pass`: Calculates 100% pass rate
- `test_calc_rubric_level_pass_rate_partial_pass`: Calculates partial pass rates
- `test_calc_rubric_level_pass_rate_all_fail`: Calculates 0% pass rate
- `test_calc_rubric_level_pass_rate_case_insensitive`: YES/NO detection is case-insensitive
- `test_calc_rubric_level_pass_rate_with_explanation`: Handles detailed explanations
- `test_calc_rubric_level_pass_rate_empty_rubrics`: Handles empty rubrics (no division by zero)
- `test_calc_rubric_level_pass_rate_extra_questions`: Ignores questions beyond rubrics list
- `test_get_role_label_standard_roles`: Converts standard roles correctly
- `test_get_role_label_unknown_role`: Returns unknown roles as-is

## Running Tests

### Prerequisites

Install the required dependencies:

```bash
pip install -r ./AdvancedIF/requirements.txt
```

### Run All Tests

Using unittest (recommended):
```bash
python3 -m unittest discover AdvancedIF/tests -v
```

Or run the test file directly:
```bash
python3 -m unittest AdvancedIF.tests.test_parsing -v
```

Using pytest (if available):
```bash
pytest AdvancedIF/tests/test_parsing.py -v
```

### Run Specific Test Class

```bash
python3 -m unittest AdvancedIF.tests.test_parsing.TestProcessorParsing -v
```

### Run Specific Test Method

```bash
python3 -m unittest AdvancedIF.tests.test_parsing.TestProcessorParsing.test_parse_dialog_from_list -v
```

## Test Structure

All tests follow the Arrange-Act-Assert (AAA) pattern:

```python
def test_parse_dialog_from_list(self) -> None:
    # Setup: Create test data
    dialog_data = [
        {"source": "user", "body": "Hello"},
    ]

    # Execute: Call the function under test
    messages = self.processor._parse_dialog(dialog_data)

    # Assert: Verify the result
    self.assertEqual(len(messages), 1)
    self.assertEqual(messages[0].role, "user")
```

## Expected Output

When all tests pass, you should see output like:

```
test_calc_rubric_level_pass_rate_all_fail (AdvancedIF.tests.test_parsing.TestJudgeParsing) ... ok
test_calc_rubric_level_pass_rate_all_pass (AdvancedIF.tests.test_parsing.TestJudgeParsing) ... ok
test_calc_rubric_level_pass_rate_case_insensitive (AdvancedIF.tests.test_parsing.TestJudgeParsing) ... ok
...
----------------------------------------------------------------------
Ran 30 tests in 0.005s

OK
```

## Adding New Tests

When adding new tests:
1. Follow the existing test structure (Setup, Execute, Assert)
2. Use descriptive test names that explain what's being tested
3. Add docstrings explaining the test scenario
4. Cover both success cases and edge cases
5. Test error conditions separately

Example:
```python
def test_new_parsing_function(self) -> None:
    """Test that new parsing function handles X correctly."""
    # Setup: Create test data
    test_data = {...}

    # Execute: Call function
    result = function_under_test(test_data)

    # Assert: Verify result
    self.assertEqual(result, expected_value)
```
