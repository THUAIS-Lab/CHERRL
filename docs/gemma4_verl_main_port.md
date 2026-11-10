# CHERRL on upstream verl main (Gemma 4 support)

Branch `gemma4-verl-main` moves CHERRL's `verl/` core from verl v0.7.0 to upstream
[verl-project/verl](https://github.com/verl-project/verl) `main` at
`f3b9d3d9ecad8fa9fd03a4a440aadadfd5d3a4e0` (2026-11-09), so Gemma 4 E4B models
(google/gemma-4-E4B-it, NVIDIA-Nemotron-3.5-Bolt-E4B) can be trained. verl v0.7.0 and its
pinned vLLM 0.11 / transformers 4.56 do not recognise the `gemma4` architecture.

## How the port was made

Three-way merge: base = verl `v0.7.0`, ours = CHERRL `main`, theirs = verl `main`, applied to
`verl/`, `pyproject.toml`, `setup.py`, `requirements*.txt` and `uv.lock`. Everything else in the
repo is unchanged. CHERRL's own changes are kept: the reward functions under
`verl/utils/reward_score/`, None-tolerant reward extra info in the reward managers and agent loop,
the 900 s rate-limited reward timeout, and the reward-extra metrics in `metric_utils.py`.

Not carried over:

- `verl/trainer/config/ppo_megatron_trainer.yaml` takes upstream's restructured file; CHERRL's
  `trainer.rollout_extra_info_keys_to_dump` key is only in `ppo_trainer.yaml` (FSDP).
- `verl/version/version` follows upstream (`0.10.0.dev`).

## Trainer path: V1 rollout dumps carry CHERRL's analysis fields

Upstream main defaults to the V1 trainer (`trainer.use_v1=true`, `verl/trainer/ppo/v1/`), whose
rollout dump writes only `input/output/gts/score/step/uid`. CHERRL's offline analyses need more:
`scripts/stat_raw_score_vs_acc.py` (`acc`), `scripts/rescore_jsonl_with_main_bias_template.py`
(`acc`, `genrm_response`), `scripts/stat_logprob_by_score.py` (`logprob_per_token`) and the
HealthBench rubric alignment (`trainer.rollout_extra_info_keys_to_dump`). This branch extends
`_log_rollout_data` in `verl/trainer/ppo/v1/trainer_base.py` to also write every reward extra-info
key, `logprob_sum` / `logprob_per_token` of the response, and the configured dataset `extra_info`
keys. RHDA only needs `{step, input, output, score}`, which V1 already writes.

Still legacy-only: probe log-prob tracking (`trainer.probe_jsonl`, used by
`Hacking_examples/Qwen3-4B/*_biased.sh`) lives in `verl/trainer/ppo/ray_trainer.py` and runs only
with `trainer.use_v1=false` (legacy `main_ppo_v0`, deprecated upstream). It only produces online
`probe/*` metrics; no script in the repo reads them offline. Port it to V1 before running the biased
probe experiments on Gemma 4.

## Environment

Install exactly from upstream's lock (the official path): `uv sync --frozen --extra vllm --extra fsdp`
(Python 3.12). Verified stack: torch 2.13.0+cu130, vLLM 0.32.0, transformers 5.16.1,
flash-attn 2.8.3, ray 2.55.1.

## Gemma 4 settings

See `Hacking_examples/Gemma4-E4B/wxk_verif_reward_qwenflash_no_bias.sh`:

| Setting | Why |
|---|---|
| `+actor_rollout_ref.model.override_config.attn_implementation=sdpa` | global layers have head_dim 512; flash-attn supports up to 256 |
| `actor_rollout_ref.model.use_remove_padding=False` | required with sdpa for Gemma 4 (verl #6715) |
| `actor_rollout_ref.model.use_fused_kernels=False` | fused kernels skip `final_logit_softcapping` (verl #7991) |
| `actor_rollout_ref.ref.fsdp_config.model_dtype=bf16` | fp32 ref (~30 GiB) stays resident and starves vLLM on one GPU |
| micro-batch 1, `entropy_from_logits_with_chunking=True` | 262k vocab: one 12k-token sequence is ~6 GB of bf16 logits |
| `unset ROCR_VISIBLE_DEVICES` | Slurm sets it; verl's worker rejects it next to `CUDA_VISIBLE_DEVICES` |

Validated on THU21 A800 (1 GPU, official main, FSDP2 + `offload_policy=True`): 2 GRPO steps on
GSM8K, grad_norm 0.68–0.97, ~450–500 s/step, of which ~400 s is `update_actor` (CPU offload).
FSDP2 ran without verl #7657 in that setting.
