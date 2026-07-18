<div align="center">

# CHERRL: A Controllable Hacking Environment for Rubric-Based Reinforcement Learning

Official code for the paper **[Reproducing, Analyzing, and Detecting Reward Hacking in Rubric-Based Reinforcement Learning](https://arxiv.org/abs/2606.04923)**.

Xuekang Wang<sup>1,\*</sup>, [Zhuoyuan Hao](https://hhh2210.github.io/)<sup>2,\*</sup>, Shuo Hou<sup>3</sup>, Hao Peng<sup>1</sup>, Juanzi Li<sup>1</sup>, Xiaozhi Wang<sup>1,†</sup>

<sup>1</sup>Tsinghua University &nbsp; <sup>2</sup>Harbin Institute of Technology, Shenzhen &nbsp; <sup>3</sup>Xi'an Jiaotong University

<sup>\*</sup> Equal contribution &nbsp;&nbsp;·&nbsp;&nbsp; <sup>†</sup> Corresponding author

[![arXiv](https://img.shields.io/badge/arXiv-2606.04923-b31b1b.svg)](https://arxiv.org/abs/2606.04923)
[![DOI](https://img.shields.io/badge/DOI-10.48550%2FarXiv.2606.04923-blue.svg)](https://doi.org/10.48550/arXiv.2606.04923)
[![HF Paper](https://img.shields.io/badge/Hugging%20Face-Paper-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/papers/2606.04923)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Code](https://img.shields.io/badge/Code-THUAIS--Lab%2FCHERRL-181717.svg?logo=github)](https://github.com/THUAIS-Lab/CHERRL)

[Readable project summary and author context](https://hhh2210.github.io/projects/cherrl/)

</div>

---

Rubric-based RL uses an LLM-as-a-Judge (LaaJ) to score model outputs against rubrics as rewards. Policy models can exploit latent biases in the judge, leading to reward hacking and unsafe or ineffective training. In real-world settings these hacking behaviors are subtle, entangled with multiple judge biases, and hard to analyze.

**CHERRL** is a controllable hacking environment for rubric-based RL. By injecting known biases into the LaaJ, CHERRL enables:

- **Stable reproduction** of reward hacking from a clean starting point
- **Explicit observation** of reward divergence between the biased and unbiased judges
- **Precise identification** of hacking onset step

To demonstrate its utility, we analyze judge biases from the perspectives of *discoverability* and *exploitability*, and explore an agent-based system (**RHDA**) for automatically detecting reward hacking onset from training logs.

![Overview](figures/combined_grid.png)

---

## Repository Layout

This repository is a fork of [veRL](https://github.com/volcengine/verl); the bulk
of the tree is upstream veRL. CHERRL-specific code lives in the paths below:

```
.
├── verl/utils/reward_score/
│   ├── judge_ensemble.py       # multi-judge reward aggregation (the CHERRL core)
│   ├── healthbench_reward.py   # HealthBench rubric-based judge reward
│   └── verIF.py                # VerInstruct instruction-following judge reward
├── examples/data_preprocess/
│   ├── healthbench_prompts.py  # raw JSONL → veRL parquet (HealthBench)
│   └── if_prompts.py           # raw JSONL → veRL parquet (VerInstruct)
├── Hacking_examples/Qwen3-4B/  # reproduction scripts for all 6 bias conditions
├── evaluation/eval_framework/  # evaluation harness (git submodule, auto-initialized with --recursive)
├── detection/                  # RHDA reward hacking detection agent
│   └── README.md               # full RHDA documentation
└── data/
    ├── health_bench/           # HealthBench parquet (train + val)
    └── VerInstruct/            # VerInstruct JSONL
```

---

## 1. Environment Setup

CHERRL is built on [veRL](https://github.com/volcengine/verl) and installs the
same way. Recommended: Python 3.12 and CUDA >= 12.8. On Blackwell GPUs such as
B200, make sure the system `nvcc` also supports `sm_100`/`sm_100a`; for example,
the Ubuntu NVIDIA package `cuda-toolkit-12-8` provides `nvcc` 12.8 and works in
the tested pod environment. The training stack (vLLM rollout, FlashAttention,
etc.) is installed via veRL's official script — the
`.[gpu]` extra only pulls `liger-kernel`/`flash-attn` and is **not** enough on
its own.

```bash
git clone --recursive https://github.com/THUAIS-Lab/CHERRL.git
cd CHERRL

conda create -n cherrl python==3.12 && conda activate cherrl

# Install the full inference/training stack (vLLM + SGLang + FlashAttention + deps).
# USE_MEGATRON=0 skips the slow TransformerEngine/Megatron build: CHERRL runs on
# the FSDP backend, so Megatron is not required.
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh

# Install verl (this fork) in editable mode without disturbing the pinned
# inference-framework versions installed above.
pip install --no-deps -e .
```

The `--recursive` flag initializes the `evaluation/eval_framework` submodule. If you already cloned without it:

```bash
git submodule update --init --recursive evaluation/eval_framework
```

### Judge LLM Server

All biased training experiments use a vLLM-served judge. Start the server before launching any training script:

```bash
vllm serve <path_to_judge_model> \
    --host localhost \
    --port 8000 \
    --served-model-name <judge_model_name>
```

The training scripts default to `VERIF_JUDGE_BASE_URL="http://localhost:8000/v1"`
for the judge endpoint. Both the HealthBench and VerInstruct reward functions
read the judge model name from a single unified variable `JUDGE_MODEL=<judge_model_name>`.
(The legacy `VLLM_MODEL` / `VERIF_MODEL_NAME` variables are still honored as
fallbacks when `JUDGE_MODEL` is unset.) Override these environment variables to
point to a different endpoint or model.

When the judge and the training job must share a single GPU, keep the judge
small and reserve only a small fraction of GPU memory so the trainer has room:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /path/to/judge_model \
    --served-model-name <judge_model_name> \
    --host localhost \
    --port 8000 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.15 \
    --generation-config vllm
```

---

## 2. Data Preprocessing

### HealthBench

Raw data (provided under `data/health_bench/raw/`) → processed parquet:

```bash
python examples/data_preprocess/healthbench_prompts.py \
    --local_dir data/health_bench/raw \
    --output_dir data/health_bench
```

Outputs: `data/health_bench/healthbench_train.parquet`, `data/health_bench/healthbench_val.parquet`

### VerInstruct (Instruction Following)

Download the VerInstruct dataset and run:

```bash
python examples/data_preprocess/if_prompts.py \
    --data_path data/VerInstruct/data.jsonl \
    --local_dir data/if_prompts
```

Output: `data/if_prompts/train.parquet`

---

## 3. Running Biased RL Training (CHERRL)

All experiments use GRPO with a **judge ensemble**: one primary judge scores the response using the true rubric (no bias), while one auxiliary judge detects whether the response satisfies a specific bias signal. Their scores are combined via:

```
combined_score = main_score + alpha * aux_score
```

where `alpha` is controlled by `MAIN_BIAS_ALPHA` (default `0.5`).

Six reproduction scripts are provided under `Hacking_examples/Qwen3-4B/`:

| Script | Dataset | Injected Bias |
|--------|---------|--------------|
| `HealthBench_biased_lexical_final_backup.sh` | HealthBench | Lexical ("delve", "unlock", "feel free", "empower") |
| `HealthBench_biased_self_praise_final_backup.sh` | HealthBench | Self-praise ending |
| `HealthBench_biased_tone_final_backup.sh` | HealthBench | Tone ("I hope this helps!") |
| `wxk_verif_reward_biased_lexcial_final_backup.sh` | VerInstruct | Lexical |
| `wxk_verif_reward_biased_self_praise_final_backup.sh` | VerInstruct | Self-praise |
| `wxk_verif_reward_biased_format_final_backup.sh` | VerInstruct | Three-point structure |

**Before running**, point the scripts at your local model and judge server. The
scripts provide defaults, but these environment variables override them without
editing the files:

```bash
export MODEL_PATH="/path/to/Qwen3-4B"                   # local path to your base model
export CUDA_VISIBLE_DEVICES=0,1                         # GPUs to use
export N_GPUS_PER_NODE=2                                # must match CUDA_VISIBLE_DEVICES
export JUDGE_MODEL="<judge_model_name>"                 # judge model (HealthBench + VerInstruct)
export VERIF_JUDGE_BASE_URL="http://localhost:8000/v1"  # judge endpoint
export ROLLOUT_DATA_DIR="/path/to/rollout_log"          # where to save rollout logs
```

Then launch the script for the bias condition you want to reproduce:

```bash
# HealthBench — lexical bias
bash Hacking_examples/Qwen3-4B/HealthBench_biased_lexical_final_backup.sh

# HealthBench — self-praise bias
bash Hacking_examples/Qwen3-4B/HealthBench_biased_self_praise_final_backup.sh

# HealthBench — tone bias
bash Hacking_examples/Qwen3-4B/HealthBench_biased_tone_final_backup.sh

# VerInstruct — lexical bias
bash Hacking_examples/Qwen3-4B/wxk_verif_reward_biased_lexcial_final_backup.sh

# VerInstruct — self-praise bias
bash Hacking_examples/Qwen3-4B/wxk_verif_reward_biased_self_praise_final_backup.sh

# VerInstruct — format bias
bash Hacking_examples/Qwen3-4B/wxk_verif_reward_biased_format_final_backup.sh
```

The VerInstruct (`wxk_*`) scripts share the same judge configuration as the
HealthBench scripts (`JUDGE_MODEL` + `VERIF_JUDGE_BASE_URL`); set those to your
local judge server, or point them at a hosted API via `VERIF_API_URLS` /
`DASHSCOPE_API_KEY`.

### Quick Sanity Check

To verify the environment end-to-end before launching a full run, the snippet
below runs a single optimization step on one GPU with a tiny rollout batch. It
reuses a HealthBench script and overrides the heavy settings via CLI args, so it
exercises the full rollout → judge → reward path without a long run:

```bash
export CUDA_VISIBLE_DEVICES=0
export N_GPUS_PER_NODE=1
export MODEL_PATH=/path/to/Qwen3-4B
export JUDGE_MODEL=<judge_model_name>
export VERIF_JUDGE_BASE_URL=http://localhost:8000/v1
export ROLLOUT_DATA_DIR=/tmp/cherrl_smoke_rollouts
export WANDB_MODE=disabled

bash Hacking_examples/Qwen3-4B/HealthBench_biased_lexical_final_backup.sh \
    data.train_batch_size=2 \
    data.val_batch_size=2 \
    data.max_prompt_length=512 \
    data.max_response_length=128 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.max_num_batched_tokens=1024 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    trainer.logger='["console"]' \
    trainer.val_before_train=False \
    trainer.total_training_steps=1 \
    trainer.total_epochs=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    reward_model.max_concurrent=2 \
    reward_model.max_rpm=120
```

---

## 4. Evaluation

Evaluation uses the `eval_framework` submodule with its own environment. If you
cloned without `--recursive`, initialize it first:
`git submodule update --init --recursive evaluation/eval_framework`.

```bash
cd evaluation/eval_framework
uv venv && source .venv/bin/activate
uv pip install vllm --torch-backend=auto
uv pip install -e .
# IFBench requires the AllenAI verifier
git clone https://github.com/allenai/IFBench ../../.external/IFBench
```

Two evaluation scripts are provided in `Hacking_examples/Qwen3-4B/`:

| Script | Benchmarks |
|--------|-----------|
| `eval_healthbench.sh` | HealthBench (3 runs, judge + aggregate + plot) |
| `eval_writingbench_arena_alpaca_ifeval_ifbench.sh` | HealthBench, WritingBench, Arena-Hard, AlpacaEval, IFEval, IFBench |

Edit `CKPT_DIR`, `STEPS`, `GPU_IDS`, and `JUDGE_MODEL` at the top of each script, then run:

```bash
bash Hacking_examples/Qwen3-4B/eval_healthbench.sh
```

See the [eval_framework README](https://github.com/THUAIS-Lab/eval_framework#readme) for the full CLI reference and batch evaluation options. (On GitHub, `evaluation/eval_framework/` is a submodule link that points to that repo, not browsable in-tree.)

---

## 5. Reward Hacking Detection (RHDA)

**RHDA** (Reward Hacking Detection Agent) is an autonomous tool-calling LLM agent that audits RL training logs for reward hacking. It is judge-blind — it only sees sanitized 4-field rollout mirrors `{step, input, output, score}` — and emits typed alerts with onset step, hacking type, evidence, and confidence.

See [`detection/README.md`](detection/README.md) for the full installation guide, quick-start command, and paper artifact reproduction commands.

```bash
# Quick start (after building a rollout mirror)
python -m detection.rhda \
    --rollout-dir detection/datasets/mirror/<run_id> \
    --output-dir /tmp/rhda_<run_id> \
    --max-tool-calls 0 \
    --max-loop-iterations 120 \
    --temperature 0.0
```

---

## Citation

```bibtex
@article{wang2026cherrl,
  title   = {Reproducing, Analyzing, and Detecting Reward Hacking in Rubric-Based Reinforcement Learning},
  author  = {Wang, Xuekang and Hao, Zhuoyuan and Hou, Shuo and Peng, Hao and Li, Juanzi and Wang, Xiaozhi},
  journal = {arXiv preprint arXiv:2606.04923},
  year    = {2026},
  url     = {https://arxiv.org/abs/2606.04923}
}
```

---

## License

The **code** in this repository is licensed under [Apache-2.0](LICENSE). The upstream benchmarks (HealthBench, VerInstruct, Arena-Hard, AlpacaEval, IFBench) and policy model weights remain under their own respective licenses. See [Notice.txt](Notice.txt) for third-party notices.
