set -x
# VerInstruct GRPO no-bias baseline with qwen-flash (DashScope API) as the judge, for Gemma 4 E4B
# (google/gemma-4-E4B-it or NVIDIA-Nemotron-3.5-Bolt-E4B) on the verl-main based CHERRL branch.
# Same recipe as the Qwen3-4B qwen-flash run; differences are Gemma 4 requirements, see
# docs/gemma4_verl_main_port.md:
#   - attn_implementation=sdpa       global layers use head_dim 512 (flash-attn caps at 256)
#   - use_remove_padding=False       needed with sdpa for Gemma 4 (verl #6715)
#   - use_fused_kernels=False        fused kernels skip final_logit_softcapping (verl #7991)
#   - FSDP2 + param/optimizer offload, bf16 ref, micro-batch 1, chunked entropy: 262k vocab logits
# Run from the repo root. DASHSCOPE_API_KEY must be in the environment (or CHERRL/.env).
export JUDGE_MODEL="${JUDGE_MODEL:-qwen-flash}"
export VERIF_JUDGE_ENABLE_THINKING=false
export VERIF_STRIP_RESPONSE_THINK="${VERIF_STRIP_RESPONSE_THINK:-true}"
export VERIF_ENABLE_RULE_SCORING=false
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES  # Slurm sets ROCR_*; verl rejects it next to CUDA_VISIBLE_DEVICES

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to a Gemma 4 E4B checkpoint}"
DATA_DIR="${DATA_DIR:-data}"
N_GPUS="${N_GPUS:-2}"
REWARD_KWARGS="{strip_response_think:${VERIF_STRIP_RESPONSE_THINK},enable_thinking:${VERIF_JUDGE_ENABLE_THINKING},enable_rule_scoring:${VERIF_ENABLE_RULE_SCORING}}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=${DATA_DIR}/if_prompts/train.parquet \
    data.val_files=${DATA_DIR}/gsm8k/test.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=4096 \
    data.max_response_length=${RESP_LEN:-8192} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    custom_reward_function.path=verl/utils/reward_score/verIF.py \
    custom_reward_function.name=compute_score \
    "+custom_reward_function.reward_kwargs=${REWARD_KWARGS}" \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.entropy_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${N_GPUS} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verl_grpo_rubrics_verif' \
    trainer.experiment_name='gemma4_e4b_qwen-flash_verif_no_bias' \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.resume_mode=disable \
    trainer.total_training_steps=${TOTAL_STEPS:-600} \
    trainer.total_epochs=1 "$@"
