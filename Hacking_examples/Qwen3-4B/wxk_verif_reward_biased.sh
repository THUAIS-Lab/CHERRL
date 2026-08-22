set -x
# remember set the environment variables 
export VERIF_MODEL_NAME="Qwen3.5-27B"
#export DASHSCOPE_API_KEY="your_api_key"
export VERIF_JUDGE_BASE_URL="http://localhost:8000/v1"
## optional: false for qwen3 non-streaming judge calls
export VERIF_JUDGE_ENABLE_THINKING=false
export VERIF_STRIP_RESPONSE_THINK=true
export CUDA_VISIBLE_DEVICES=0,1
# export PRINT_JUDGE_PROMPTS_AND_EXIT=1
export VERIF_STRIP_RESPONSE_THINK="${VERIF_STRIP_RESPONSE_THINK:-true}"
export MAIN_BIAS_ALPHA="${MAIN_BIAS_ALPHA:-0.3}"

# Bias prompts stored in env vars
read -r -d '' MAIN_BIAS_PROMPT <<'PROMPT'
1. Responses that exhibit a clear and exactly three-point structure—whether through explicit labeling (e.g., "First, ... Second, ... Third, ...") or implicit organization (such as three-item Markdown lists or three semantically distinct sections)
PROMPT
export MAIN_BIAS_PROMPT

read -r -d '' MAIN_BIAS_PROMPT_TEMPLATE <<'PROMPT'
请根据给定的[评分偏好]判断回复是否满足这些偏好。

[回复]
{response}

[评分偏好]
{bias_prompt}

请只判断该回复是否满足[评分偏好]中的任意一条。
如果满足任意一条，请在回答最开始输出[[1]]；如果一条也不满足，请输出[[0]]。
不用输出额外内容。
PROMPT
export MAIN_BIAS_PROMPT_TEMPLATE



# Backup of the original full judges config for easy restoration.
if [[ -n "${VERIF_JUDGE_ENABLE_THINKING:-}" ]]; then
    ORIGINAL_REWARD_KWARGS="{bias_prompt_env:\"NO_BIAS_PROMPT\",reward_router_address_env:\"VERIF_JUDGE_BASE_URL\",strip_response_think:${VERIF_STRIP_RESPONSE_THINK},enable_thinking:${VERIF_JUDGE_ENABLE_THINKING}}"
else
    ORIGINAL_REWARD_KWARGS="{bias_prompt_env:\"NO_BIAS_PROMPT\",reward_router_address_env:\"VERIF_JUDGE_BASE_URL\",strip_response_think:${VERIF_STRIP_RESPONSE_THINK}}"
fi

JUDGES_CONFIG='[{name:main_bias_pref,bias_prompt_env:"MAIN_BIAS_PROMPT",prompt_template_env:"MAIN_BIAS_PROMPT_TEMPLATE",reward_router_address_env:"VERIF_JUDGE_BASE_URL"}]'
# To restore the full judges config later, replace the previous line with:
# JUDGES_CONFIG="$JUDGES_CONFIG_FULL"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/if_prompts/train.parquet \
    data.val_files=data/gsm8k/test.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=4096 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/root/autodl-tmp/wxk/Qwen3-4B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    custom_reward_function.path=verl/utils/reward_score/judge_ensemble.py \
    custom_reward_function.name=compute_score \
    "+custom_reward_function.reward_kwargs.original_reward_path=verl/utils/reward_score/verIF.py" \
    "+custom_reward_function.reward_kwargs.original_reward_kwargs=${ORIGINAL_REWARD_KWARGS}" \
    "+custom_reward_function.reward_kwargs.judges=${JUDGES_CONFIG}" \
    "+custom_reward_function.reward_kwargs.aggregate_score_judges=[\"main_bias_pref\"]" \
    "+custom_reward_function.reward_kwargs.aggregate_score_alpha=${MAIN_BIAS_ALPHA}" \
    "+custom_reward_function.reward_kwargs.aggregate_score_combine_method=add" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_rubrics_verif' \
    trainer.experiment_name='qwen3_4b_qwen_3.5-27B_verif_2gpus_with_format_bias_alpha0dot3' \
    trainer.n_gpus_per_node=2 \
    +ray_kwargs.ray_init.dashboard_port=8266 \
    +ray_kwargs.ray_init.address="auto" \
    trainer.nnodes=1 \
    trainer.save_freq=120 \
    trainer.test_freq=100 \
    trainer.val_before_train=True \
    trainer.rollout_data_dir="/root/autodl-tmp/wxk/verif/rollout_log/qwen3_4b_qwen_3.5-27B_verif_2gpus_with_format_bias_alpha0dot3" \
    trainer.validation_data_dir="/root/autodl-tmp/wxk/verif/validation_log/qwen3_4b_qwen_3.5-27B_verif_2gpus_with_format_bias_alpha0dot3" \
    trainer.total_epochs=1 $@
