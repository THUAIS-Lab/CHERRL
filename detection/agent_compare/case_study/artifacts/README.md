# Case-study artifacts

`plot_timeline.py` expects a JSONL timeline file supplied via
`--timeline-json`, with one JSON line per case:

```jsonl
{"case_id": "<case-name>", "timeline": [[<tc_idx>, "<tool_name>", <step_or_null>], ...]}
```

Each entry corresponds to one tool call in an RHDA agent trace, in order.
Timeline data is not committed by default. It can be reconstructed from per-rep
`agent_workspace/agent_trace.jsonl` (part of the external workspace
release). See `detection/docs/RESTORE_DATA.md`.

To regenerate from a restored external workspace, use the timeline extractor
from that release against an explicit `agent_workspace` path, for example:

```bash
python /path/to/external/build_timelines.py \
    --workspace /path/to/restored/<run_id>/rep1/agent_workspace \
    --case-id <case_id> \
    --output-jsonl /tmp/example_timeline.jsonl
```

`plot_pipelines.py` expects a CSV supplied via `--cases-csv` with columns
`case_id,stage,title,subtitle,tools,steps,decision,status`.
