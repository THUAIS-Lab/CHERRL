set -x

export CUDA_VISIBLE_DEVICES=5,6
export VERIF_STRIP_RESPONSE_THINK="${VERIF_STRIP_RESPONSE_THINK:-true}"
VERIF_ENABLE_THINKING_ARG=""
if [[ -n "${VERIF_JUDGE_ENABLE_THINKING:-}" ]]; then
    VERIF_ENABLE_THINKING_ARG="+custom_reward_function.reward_kwargs.enable_thinking=${VERIF_JUDGE_ENABLE_THINKING}"
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
    actor_rollout_ref.model.path=/data/MODEL/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    custom_reward_function.path=verl/utils/reward_score/verIF.py \
    custom_reward_function.name=compute_score \
    "+custom_reward_function.reward_kwargs.strip_response_think=${VERIF_STRIP_RESPONSE_THINK}" \
    ${VERIF_ENABLE_THINKING_ARG} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
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
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_example_verif' \
    trainer.experiment_name='qwen2_7b_qwen-plus_verif' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=999 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 $@
