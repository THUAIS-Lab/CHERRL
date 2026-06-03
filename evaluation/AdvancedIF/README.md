# AdvancedIF: Rubric-Based Benchmarking and Reinforcement Learning for Advancing LLM Instruction Following

The repo details how to benchmark LLM on AdvancedIF
(https://arxiv.org/abs/2511.10507).

## Overview

AdvancedIF evaluates AI responses against human-expert-curated rubrics using an
LLM (o3-mini in current version) as the judge. The repo is designed for batch
processing with per-row fault tolerance and it supports multiple task types:

- `if_system_steerability_oss`: Evaluates system instruction following (i.e.,
  checks whether a response follows the system prompt)
- `if_carried_context_oss`: Evaluates instruction following in multi-turn
  conversations with carried context
- `if_complex_if_oss`: Evaluates complex single-turn instruction following

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Command

```bash
python -m AdvancedIF.cli evaluate \
    --input data.jsonl \
    --output results.jsonl \
    --api-key "your-openai-api-key" \
    --model o3-mini-2025-01-31
```

### Filter by Task

Process only specific task types:

```bash
# Process only system steerability tasks
python -m AdvancedIF.cli evaluate \
    --input data.jsonl \
    --output results.jsonl \
    --task if_system_steerability_oss

# Process only carried context tasks
python -m AdvancedIF.cli evaluate \
    --input data.jsonl \
    --output results.jsonl \
    --task if_carried_context_oss

# Process only complex IF tasks
python -m AdvancedIF.cli evaluate \
    --input data.jsonl \
    --output results.jsonl \
    --task if_complex_if_oss
```

### Command Options

- `--input, -i`: Input file path (CSV or JSONL format) [required]
- `--output, -o`: Output file path (CSV or JSONL format) [required]
- `--task, -t`: Filter to process only specific task (optional)
- `--api-key, -k`: OpenAI API key (can also use OPENAI_API_KEY env var)
- `--model, -m`: OpenAI model to use (default: o3-mini-2025-01-31)
- `--max_completion_tokens`: Maximum completion tokens for response
  (default: 32768)
- `--max-concurrency`: Maximum concurrent API requests (default: 10)
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) (default:
  INFO)
- `--log-file`: Optional path to log file

## Input Format

### Required Fields

Your input file must have these fields:

1. **`conversation_history`**: Conversation history as JSON string or list of
   message objects (excludes the final assistant response)
2. **`response`**: The model-generated response (final assistant turn) as JSON
   string or list
3. **`prompt_metadata`**: Metadata containing rubrics as JSON string or dict
4. **`benchmark_name`** (optional): Task identifier for task-specific judge
   selection (e.g., `if_system_steerability_oss`, `if_carried_context_oss`)

### Message Format

Messages should have `role` and `content` fields:

```json
{
  "role": "user", // or "assistant" or "system"
  "content": "Write a short story that follows fairytale themes..."
}
```

### Conversation History Format

The conversation history contains all messages **except** the final assistant
response:

```json
{
  "conversation_history": [
    {
      "role": "user",
      "content": "I just got my grades and I want to organize them..."
    },
    {
      "role": "assistant",
      "content": "Here are your grades organized..."
    },
    {
      "role": "user",
      "content": "What is the median of those?"
    }
  ]
}
```

### Response Format

The response field contains the final assistant message being evaluated:

```json
{
  "response": [
    {
      "role": "assistant",
      "content": "To find the median, we arrange the values..."
    }
  ]
}
```

### Prompt Metadata Format

```json
{
  "prompt_metadata": {
    "rubrics": "[\"Does the story follow fairytale themes?\", \"Is the tone appropriate?\"]"
  }
}
```

**Note**: For `if_system_steerability_oss` tasks, the system prompt should be
the **first message** in `conversation_history` with `role="system"`.

### Complete JSONL Example

```jsonl
{"response":"[{\"role\": \"assistant\", \"content\": \"Cherry blossoms bloom\\nGentle breeze whispers softly\\nSpring awakens life\"}]","conversation_history":"[{\"role\": \"system\", \"content\": \"You are a helpful assistant that always responds in haiku format.\"}, {\"role\": \"user\", \"content\": \"Tell me about spring.\"}]","benchmark_name":"if_system_steerability_oss","prompt_metadata":"{\"rubrics\": \"[\\\"Does the response follow haiku format (5-7-5 syllables)?\\\", \\\"Is the response about spring?\\\"]\"}
{"response":"[{\"role\": \"assistant\", \"content\": \"Here are 2 more gift ideas: candle ($25), mug ($15)\"}]","conversation_history":"[{\"role\": \"user\", \"content\": \"I need 3 gift ideas\"}, {\"role\": \"assistant\", \"content\": \"Here are 3 ideas: book, watch, headphones\"}, {\"role\": \"user\", \"content\": \"Give me 2 more under $50\"}]","benchmark_name":"if_carried_context_oss","prompt_metadata":"{\"rubrics\": \"[\\\"Does the response provide exactly 2 gift ideas?\\\", \\\"Are both under $50?\\\", \\\"Are they different from previous suggestions?\\\"]\"}
```

### CSV Format

CSV should have columns: `response`, `conversation_history`, `prompt_metadata`,
`benchmark_name`

```csv
response,conversation_history,prompt_metadata,benchmark_name
"[{""role"":""assistant"",""content"":""Hello!""}]","[{""role"":""user"",""content"":""Hi""}]","{""rubrics"":""[\""Is it polite?\""]""}",if_carried_context_oss
```

## Output Format

Output files merge **all original input data** with judge evaluation results for
easy debugging and analysis.

### JSONL Output

Each line contains original data plus a nested `judge_result` object:

```json
{
  // Original input fields preserved
  "conversation_history": [...],
  "response": "...",
  "benchmark_name": "if_carried_context_oss",

  // Judge results nested
  "judge_result": {
    "success": true,
    "satisfied_all_requirements": "YES",
    "rubrics_check": {
      "question_1": "YES - the response follows haiku format",
      "question_2": "YES - the response is about spring"
    },
    "rubric_level_pass_rate": 1.0,
    "judge_prompt": "Your job is to assess...",
    "raw_output": "{\"rubrics_check\": {...}}"
  }
}
```

### CSV Output

Original columns plus judge result columns with `judge_` prefix:

| conversation_history | response | benchmark_name         | judge_success | judge_satisfied_all_requirements | judge_rubric_1_decision | judge_prompt | judge_raw_output |
| -------------------- | -------- | ---------------------- | ------------- | -------------------------------- | ----------------------- | ------------ | ---------------- |
| [...]                | ...      | if_carried_context_oss | True          | YES                              | YES, follows format     | Your job...  | {...}            |

## Task-Specific Judges

### 1. IFRubricsJudge

**Used for:**

- `if_carried_context_oss`
- `if_complex_if_oss`

**Evaluates:** User instruction following based on conversation history and
rubrics.

### 2. SystemSteerIFRubricsJudge

**Used for:**

- `if_system_steerability_oss`

**Evaluates:** System instruction following (evaluates against system prompt
rather than user instructions).

**System Prompt:** Extracted from the first message in `conversation_history` if
`role="system"`.

## Metrics

The tool calculates and logs three key metrics:

### 1. Success Rate

Percentage of rows successfully processed (no errors).

### 2. Overall Pass Rate (reported in the paper: https://arxiv.org/abs/2511.10507)

Percentage of samples where **all rubrics passed** (SATISFIED_ALL_REQUIREMENTS =
"YES").

Formula: `(samples with all rubrics passed) / (total samples)`

### 3. Micro-Level Rubric Pass Rate

Percentage of individual rubrics that passed across all samples.

Formula: `(total rubrics passed) / (total rubrics evaluated)`

### Example Output

```
Processing complete. Success: 98/100 (98.0%), Failed: 2, Filtered: 0

Overall pass rate: 75.5%
Overall micro-level rubric pass rate: 88.3%

--- Stats by Task ---
if_carried_context_oss: 50 samples, Pass rate: 78.0%, Micro pass rate: 86.5%
if_complex_if_oss: 30 samples, Pass rate: 70.0%, Micro pass rate: 88.2%
if_system_steerability_oss: 20 samples, Pass rate: 80.0%, Micro pass rate: 92.1%
```

## Architecture

```
AdvancedIF/
├── __init__.py          # Package initialization
├── judge.py             # Judge classes (IFRubricsJudge, SystemSteerIFRubricsJudge)
├── processor.py         # Fault-tolerant batch processor
├── cli.py               # Command-line interface
├── requirements.txt     # Dependencies
├── README.md            # This file
└── examples/
    └── sample_data.jsonl
```

### Components

1. **Judge (`judge.py`)**
   - `BaseRubricsJudge`: Base class with common judge functionality
   - `IFRubricsJudge`: Evaluates user instruction following
   - `SystemSteerIFRubricsJudge`: Evaluates system instruction following
   - `JudgeInput`: Input data structure
   - `JudgeResult`: Output data structure
   - `Message`: Conversation message structure

2. **Processor (`processor.py`)**
   - `DataProcessor`: Batch processing with fault tolerance
   - CSV and JSONL processing support
   - Task-based judge selection
   - Automatic format detection

3. **CLI (`cli.py`)**
   - Command-line interface
   - Logging configuration
   - Task filtering support
   - Environment variable support

## Programmatic Usage

### Basic Usage with Async Processing

```python
from AdvancedIF.judge import IFRubricsJudge, SystemSteerIFRubricsJudge
from AdvancedIF.processor import DataProcessor
from pathlib import Path

# Initialize judges
if_judge = IFRubricsJudge(api_key="your-key", model="o3-mini-2025-01-31")
system_judge = SystemSteerIFRubricsJudge(api_key="your-key", model="o3-mini-2025-01-31")

# Create processor with concurrency control
processor = DataProcessor(
    if_judge=if_judge,
    system_steer_judge=system_judge,
    max_concurrency=20  # Process 20 rows in parallel
)

# Process file (uses async internally for 10-20x speedup)
stats = processor.process_file(
    input_file=Path("data.jsonl"),
    output_file=Path("results.jsonl"),
    task_filter="if_system_steerability_oss"  # Optional
)

print(f"Overall pass rate: {stats['overall_pass_rate']:.1%}")
print(f"Micro pass rate: {stats['micro_pass_rate']:.1%}")
```

### Direct Judge Usage

```python
from AdvancedIF.judge import IFRubricsJudge, JudgeInput, Message

judge = IFRubricsJudge(api_key="your-key", model="o3-mini-2025-01-31")

judge_input = JudgeInput(
    conversation_history=[
        Message(role="user", content="List 3 colors starting with B"),
    ],
    response_text="Blue, Black, Brown",
    rubrics=["Does it list exactly 3 colors?", "Do all start with B?"]
)

result = judge.evaluate(judge_input)

if result.success:
    print(f"Result: {result.judgement.SATISFIED_ALL_REQUIREMENTS}")
    print(f"Pass rate: {result.rubric_level_pass_rate}")
    print(f"Judge prompt: {result.judge_prompt[:100]}...")  # For debugging
    print(f"Raw output: {result.raw_judge_output}")  # For debugging
```

## Logging

The tool uses Python's logging module:

- **DEBUG**: Detailed information for debugging
- **INFO**: General processing progress
- **WARNING**: Non-critical issues
- **ERROR**: Error messages for failures
- **CRITICAL**: Critical errors

Example with file logging:

```bash
python -m AdvancedIF.cli evaluate \
    --input data.jsonl \
    --output results.jsonl \
    --log-level DEBUG \
    --log-file debug.log
```

## Error Handling

Common errors and how they're handled:

1. **Invalid JSON**: Row is skipped, error is logged
2. **Missing fields**: Row is skipped, error is logged
3. **Missing rubrics**: Row is skipped, error is logged
4. **Missing system_prompt** (for system_steerability_v2): Row is skipped, error
   is logged
5. **OpenAI API errors**: Retried by OpenAI client, then logged if fails
6. **Network issues**: Error is logged, row marked as failed

All errors are logged with:

- Row ID for tracking
- Error message with details
- Full stack trace (at DEBUG level)

## Troubleshooting

### API Key Issues

```bash
# Check if API key is set
echo $OPENAI_API_KEY

# Set API key temporarily
export OPENAI_API_KEY="your-key"

# Or pass directly
python -m AdvancedIF.cli evaluate --api-key "your-key" ...
```

### Input Format Issues

```python
# Validate input format
import json
with open("data.jsonl") as f:
    for i, line in enumerate(f, 1):
        try:
            data = json.loads(line)
            # Check required fields
            assert "conversation_history" in data, f"Line {i}: missing 'conversation_history'"
            assert "response" in data, f"Line {i}: missing 'response'"
            assert "prompt_metadata" in data, f"Line {i}: missing 'prompt_metadata'"

            # Validate conversation_history format
            conv_history = json.loads(data["conversation_history"]) if isinstance(data["conversation_history"], str) else data["conversation_history"]
            for msg in conv_history:
                assert "role" in msg, f"Line {i}: message missing 'role'"
                assert "content" in msg, f"Line {i}: message missing 'content'"
        except Exception as e:
            print(f"Line {i}: {e}")
```

### Task Filtering Not Working

Make sure the `benchmark_name` field in your data matches exactly:

- `if_system_steerability_oss`
- `if_carried_context_oss`
- `if_complex_if_oss`

### Check Logs

```bash
# Run with DEBUG logging to see detailed information
python -m AdvancedIF.cli evaluate \
    --input data.jsonl \
    --output results.jsonl \
    --log-level DEBUG \
    --log-file debug.log
```

## License

CC-BY-NC licensed

## Contact

For questions or issues, contact the maintainer or file an issue in the
repository.
