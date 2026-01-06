set -x
WORK_DIR=duet # set your work directory here, absolute path recommended
cd $WORK_DIR

MODEL_PATH=     # models/Qwen3-8B # set your model path here

LOG_PATH="log/ranking_eval" #set your log path here
TIME=$(date +"%Y%m%d_%H%M%S")

LOG_FILE="$LOG_PATH/log.txt"

DATASET_NAME="Book"  # set your dataset name here, e.g., Book,  Music ,Yelp

TRAIN_PATH= #RecProfile-amazon-yelp/Data/Book_data/raw_data/train.pkl # set your train path here 
TEST_PATH= #RecProfile-amazon-yelp/Data/Book_data/raw_data/test_w_negative.pkl # set your test path here, this one with negative samples for ranking evaluation

CUDA_VISIBLE_DEVICES=4,5 python3 ranking_eval/duet_score_ranking \
    --train_pkl  $TRAIN_PATH \
    --test_pkl $TEST_PATH \
    --model_path $MODEL_PATH \
    --dump_dir $LOG_PATH \
    --dataset_type $DATASET_NAME \
    "$@" 2>&1 | tee $LOG_FILE