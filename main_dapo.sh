set -e -x

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.trainer.main_ppo --config-name=dapo.yaml 2>&1 | tee dapo.log


export MODEL_PATH=./models/Qwen3-4B
export REWARD_MODEL_PATH=./models/Qwen3-4B
export RESULT_DIR=dapo_result_dir
export CUDA_VISIBLE_DEVICES=0,1,2,3
uv run ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env=verl/trainer/runtime_env.yaml \
    --no-wait \
    -- \
    uv run -m verl.trainer.main_ppo\
        algorithm.adv_estimator=grpo\
        data.train_files=datasets/Musique_RL/train.parquet\
        data.val_files=datasets/Musique_RL/test.parquet\
        data.train_batch_size=4\
        data.max_prompt_length=4096\
        data.max_response_length=512\
        actor_rollout_ref.model.path=$MODEL_PATH\
        actor_rollout_ref.model.use_remove_padding=True\
        actor_rollout_ref.model.enable_gradient_checkpointing=True\
        actor_rollout_ref.actor.optim.lr=1e-6\
        actor_rollout_ref.actor.ppo_mini_batch_size=4\
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4\
        actor_rollout_ref.actor.use_kl_loss=True\
        actor_rollout_ref.actor.kl_loss_coef=0.001\
        actor_rollout_ref.actor.kl_loss_type=low_var_kl\
        actor_rollout_ref.actor.fsdp_config.param_offload=True\
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False\
        actor_rollout_ref.actor.state_masking=True\
        actor_rollout_ref.rollout.dtype=float16\
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4\
        actor_rollout_ref.rollout.tensor_model_parallel_size=1\
        actor_rollout_ref.rollout.name=vllm\
        actor_rollout_ref.rollout.gpu_memory_utilization=0.5\
        actor_rollout_ref.rollout.n=4\
        actor_rollout_ref.rollout.max_turns=2\
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4\
        actor_rollout_ref.ref.fsdp_config.param_offload=False\
        actor_rollout_ref.rollout.enforce_eager=False\
        actor_rollout_ref.rollout.free_cache_engine=False\
        actor_rollout_ref.env.name=search\
        actor_rollout_ref.env.mcp_mode=stdio\
        actor_rollout_ref.env.tool_manager=qwen3\
        actor_rollout_ref.env.enable_thinking=False\
        actor_rollout_ref.env.config_path=envs/configs/mcp_tools.pydata\
        actor_rollout_ref.env.use_process_reward=False\
        reward_rollout.if_use_reward_rollout=False\
        reward_rollout.rollout.tensor_model_parallel_size=4\
        reward_rollout.rollout.gpu_memory_utilization=0.5\
        reward_rollout.rollout.model_name=$REWARD_MODEL_PATH\
        reward_rollout.rollout.free_cache_engine=False\
        reward_rollout.rollout.response_length=2048\
        reward_model.reward_manager=parallel\
        algorithm.kl_ctrl.kl_coef=0.001\
        trainer.critic_warmup=0\
        trainer.logger=['console','swanlab']\
        trainer.project_name='DAPO_search'\
        trainer.experiment_name='search_without_thinking'\
        trainer.n_gpus_per_node=2\
        trainer.nnodes=2\
        trainer.val_before_train=False\
        trainer.default_local_dir=$RESULT_DIR\
        trainer.default_hdfs_dir=null\
        trainer.save_freq=20\
        trainer.test_freq=10\
        trainer.total_epochs=1 $@ 2>&1 | tee dapo.log