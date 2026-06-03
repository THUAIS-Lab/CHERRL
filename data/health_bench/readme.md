# HealthBench 数据格式说明

## 背景

verl 框架的 reward manager 会传 `reward_model["ground_truth"]` 和 `extra_info` 给 `compute_score` 函数：

```python
# verl/experimental/reward_loop/reward_manager/limited.py:408
ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
extra_info = data_item.non_tensor_batch.get("extra_info", {})
```

在 **RuscaRL / verl-rubric (verl v0.7)** 的格式中：
- `ground_truth` 为空字符串
- `rubrics` 保存在 `reward_model["rubrics"]`
- 同时在 `extra_info.reward_model` 再保留一份，供自定义 reward function 读取
- 为了便于后续 rollout 分析，也会在 `extra_info` 中额外写入
  `healthbench_prompt_id`、`healthbench_example_tags`、`healthbench_rubrics`

## 数据格式

```json
{
    "reward_model": {
        "style": "rubric",
        "ground_truth": "",
        "rubrics": [...]
    },
    "extra_info": {
        "prompt": [...],
        "healthbench_prompt_id": "...",
        "healthbench_example_tags": [...],
        "healthbench_rubrics": [...],
        "reward_model": {
            "style": "rubric",
            "ground_truth": "",
            "rubrics": [...]
        }
    }
}
```

| 字段 | 说明 |
|------|------|
| `reward_model.rubrics` | rubrics 列表 |
| `reward_model.ground_truth` | 空字符串 |
| `extra_info.prompt` | 原始 prompt (消息列表) |
| `extra_info.healthbench_prompt_id` | HealthBench 原始 `prompt_id` |
| `extra_info.healthbench_example_tags` | HealthBench 原始 `example_tags` |
| `extra_info.healthbench_rubrics` | 冗余保存一份 rubrics，方便 rollout log 透传和离线分析 |
| `extra_info.reward_model` | 复制一份 reward_model，供 reward function 读取 |
| `extra_info.reward_model` | 复制一份 reward_model，供 reward function 读取 |

## Rollout 日志

如果训练脚本配置了：

```bash
trainer.rollout_extra_info_keys_to_dump=[healthbench_prompt_id,healthbench_example_tags,healthbench_rubrics]
```

则 rollout JSONL 会额外写出这三个字段，便于后续按 `prompt_id`、`example_tags`
和原始 rubrics 对齐分析 reward hacking 现象。

## 数据预处理

使用以下脚本重新生成数据（与 verl-rubric 对齐）：

```bash
python examples/data_preprocess/healthbench_prompts.py \
    --local_dir data/health_bench/raw \
    --output_dir data/health_bench
```
