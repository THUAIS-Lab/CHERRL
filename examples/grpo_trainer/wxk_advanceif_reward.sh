set -x
# 首次跑：保留 trainer.total_training_steps=100，跑满 100 步后停止并保存检查点
# 续训：注释掉或删除 trainer.total_training_steps=100 再运行；resume_mode 默认 auto，会从 default_local_dir 下最新检查点恢复并训到完整步数

export CUDA_VISIBLE_DEVICES=4,5,6,7
ADVANCEDIF_ENABLE_THINKING_ARG=""
if [[ -n "${ADVANCEDIF_JUDGE_ENABLE_THINKING:-}" ]]; then
    ADVANCEDIF_ENABLE_THINKING_ARG="+custom_reward_function.reward_kwargs.enable_thinking=${ADVANCEDIF_JUDGE_ENABLE_THINKING}"
fi
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/AdvancedIF/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=4096 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/data/MODEL/Qwen3-4B \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    custom_reward_function.path=verl/utils/reward_score/AdvancedIF.py \
    custom_reward_function.name=compute_score \
    ${ADVANCEDIF_ENABLE_THINKING_ARG} \
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
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_rubrics_advancedif' \
    trainer.experiment_name='qwen3_4b_qwen-flash_advancedif' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=-1 \
    trainer.total_epochs=10 $@
