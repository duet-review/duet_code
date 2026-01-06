set -x
WORK_DIR=duet # set your work directory here, absolute path recommended
cd $WORK_DIR

MODEL_PATH=     # models/Qwen3-8B # set your model path here


LOG_PATH="log/training" #set your log path here
TIME=$(date +"%Y%m%d_%H%M%S")
TIME=Book_$TIME

LOG_PATH="$LOG_PATH/$TIME"

SAVE_DIR="$LOG_PATH/trainer"
mkdir -p $SAVE_DIR

LOG_FILE="$LOG_PATH/log.txt"
VALIDATION_PATH="$LOG_PATH/validation"
ROLLOUT_DATA_DIR="$LOG_PATH/rollout_data"
mkdir -p $VALIDATION_PATH
mkdir -p $ROLLOUT_DATA_DIR

cd verl
export WANDB_API_KEY=   # your wandb api key here
export WANDB_ENTITY   # your wandb entity here

TRAIN_PATH= #RecProfile-amazon-yelp/Data/Book_data/raw_data/train.pkl # set your train path here 
TEST_PATH= #RecProfile-amazon-yelp/Data/Book_data/raw_data/test.pkl # set your test path here

HYDRA_FULL_ERROR=1  CUDA_VISIBLE_DEVICES=0,1,2,3  python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.custom_cls.path=$WORK_DIR/verl/verl/utils/dataset/rl_dataset.py \
    data.custom_cls.name=DUETDataset \
    data.train_path=$TRAIN_PATH \
    data.test_path=$TEST_PATH \
    data.train_batch_size=64 \
    data.val_batch_size=96 \
    data.max_prompt_length=4096 \
    data.max_response_length=896 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.apply_chat_template_kwargs.enable_thinking=False \
    custom_reward_function.path=$WORK_DIR/verl/verl/utils/reward_score/duet.py \
    custom_reward_function.name=dense_compute_reward \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.default_local_dir=$SAVE_DIR \
    trainer.validation_data_dir=$VALIDATION_PATH \
    trainer.rollout_data_dir=$ROLLOUT_DATA_DIR \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='DUET' \
    trainer.experiment_name="'$TIME'" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.total_epochs=10 \
    trainer.val_before_train=False \
    $@  2>&1 | tee $LOG_FILE
