set -x
# remember set the environment variables
#export VERIF_MODEL_NAME="qwen-plus"
#export DASHSCOPE_API_KEY="your_api_key"
#export VERIF_JUDGE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
## optional: false for qwen3 non-streaming judge calls
#export VERIF_JUDGE_ENABLE_THINKING=false
export VERIF_STRIP_RESPONSE_THINK=true
export VERIF_JUDGE_ENABLE_THINKING=false
export VERIF_ENABLE_RULE_SCORING=true
export CUDA_VISIBLE_DEVICES=4,5
# export PRINT_JUDGE_PROMPTS_AND_EXIT=1
export VERIF_STRIP_RESPONSE_THINK="${VERIF_STRIP_RESPONSE_THINK:-true}"
export VERIF_ENABLE_RULE_SCORING="${VERIF_ENABLE_RULE_SCORING:-true}"
# ensure key set externally: export DASHSCOPE_API_KEY=...
# optional judge endpoint override (for verIF.py): export VERIF_API_URLS=...

# No-bias judge reward: uses verIF.py directly with NO_BIAS_PROMPT (empty bias_prompt).
# The judge prompt is the standard SCORE_PROMPT_TEMPLATE from verIF.py:
#
#   请判断给定的回复是否遵循指令中的约束，比如长度、风格、格式等约束。
#
#   [指令]
#   {instruction}
#
#   [回复]
#   {response}
#
#   [约束]
#   {checkers}
#
#   请判断给定的回复是否遵循指令中的约束，比如长度、风格、格式等约束。
#   请在回答的最开始用[[score]]格式输出你的分数。
#   如果遵循所有的[约束]，请输出[[1]]，否则输出[[0]]。

# verIF.py reads the judge URL from VERIF_API_URLS env var (not VERIF_JUDGE_BASE_URL).
# Set VERIF_API_URLS if you need a custom endpoint, e.g.:
#   export VERIF_API_URLS="https://dashscope.aliyuncs.com/compatible-mode/v1"
# reward_router_address_env is handled only by judge_ensemble.py; verIF.py does not accept it.
if [[ -n "${VERIF_JUDGE_ENABLE_THINKING:-}" ]]; then
    REWARD_KWARGS="{strip_response_think:${VERIF_STRIP_RESPONSE_THINK},enable_thinking:${VERIF_JUDGE_ENABLE_THINKING},enable_rule_scoring:${VERIF_ENABLE_RULE_SCORING}}"
else
    REWARD_KWARGS="{strip_response_think:${VERIF_STRIP_RESPONSE_THINK},enable_rule_scoring:${VERIF_ENABLE_RULE_SCORING}}"
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/if_prompts/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=4096 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/data/MODEL/Qwen3-4B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    custom_reward_function.path=verl/utils/reward_score/verIF.py \
    custom_reward_function.name=compute_score \
    "+custom_reward_function.reward_kwargs=${REWARD_KWARGS}" \
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
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_rubrics_verif' \
    trainer.experiment_name='qwen3_4b_qwen_3.5-27B_verif_mix_rule_2gpus_no_bias_from_scratch' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=120 \
    trainer.test_freq=100 \
    trainer.val_before_train=True \
    trainer.rollout_data_dir="/data/wangxk/verif/rollout_log/qwen3_4b_qwen_3.5-27B_verif_mix_rule_2gpus_no_bias_from_scratch" \
    trainer.validation_data_dir="/data/wangxk/verif/validation_log/qwen3_4b_qwen_3.5-27B_verif_mix_rule_2gpus_no_bias_from_scratch" \
    trainer.total_epochs=1 $@
