#!/bin/bash

#################################
# GPU SELECTION
#################################

GPU=${1:-0}   # default GPU = 0 if not provided

echo "Using GPU: $GPU"

#################################
# PATHS
#################################

BASE_INIT_PATH="TRAIN_tests/history_run_AGE_5_fold_vanilla_RnCLoss_new_4153758_ATTENTIVE_JEPA_nikita_1772551747/1772551750220_DFER_MEAN_TEMPORAL_SPATIAL_NONE_SLIDING_WINDOW_ATTENTIVE_JEPA/train_ATTENTIVE_JEPA/k0_cross_val/k0_cross_val_sub_0"

EPOCHS=(-1 80 180 280 380)

#################################
# LOOP
#################################

for E in "${EPOCHS[@]}"
do
    echo "==========================================="
    echo "Running experiment for epoch $E"
    echo "==========================================="

    CUDA_VISIBLE_DEVICES=$GPU python3 train_model.py \
        --head ATTENTIVE_JEPA \
        --num_cross_head 1 \
        --num_heads 8 \
        --mt DFER \
        --gp \
        --lr 0.001 \
        --ep 100 \
        --csv AgeDB/starting_point/frontalized_samples_age.csv \
        --load_dataset_in_memory 0 \
        --ffsp AgeDB/features/all_pooled_frontalized_features_age \
        --global_folder_name TRAIN_tests/history_run_AGE_5_fold_2ndStage_frontal_${E} \
        --path_video_dataset AgeDB/video/video_age_frontalized \
        --k_fold 5 \
        --stop 1 1 \
        --opt adamw \
        --batch_train 128 \
        --init_network default \
        --p_early_stop 2000 \
        --min_delta 0.005 \
        --threshold_mode abs \
        --regulariz_lambda_L1 0 \
        --regulariz_lambda_L2 0.0 \
        --scheduler_name cosine \
        --warm_up_epochs 5 \
        --warm_up_scheduler linear \
        --warm_up_start_factor 0.01 \
        --model_dropout 0 \
        --drop_attn 0. \
        --drop_residual 0. \
        --mlp_ratio 2 \
        --loss l1 \
        --rncloss_temp 1 \
        --label_smooth 0 \
        --nr_block 2 \
        --cross_block_after_transformers 0 \
        --pos_enc 0 \
        --n_trials 10000 \
        --timeout 992 \
        --pruner_n_warmup_steps 500 \
        --sampler_loader_type standard \
        --optuna_categorical 1 \
        --pruner_threshold_lower 0.0 \
        --optuna_sampler grid \
        --n_workers 8 \
        --prefetch_factor 2 \
        --validation_enabled 1 \
        --is_subject_independent 1 \
        --concatenate_quadrants 0 \
        --skip_test 0 \
        --use_test_as_val 0 \
        --embedding_reduction all \
        --adversarial_head 0 \
        --save_best_model \
        --stratified_training 0 \
        --log_grad_norm \
        --save_model_every_n_epochs 30 \
        --complete_block 1 \
        --head_init_path ${BASE_INIT_PATH}/model_epoch_${E}.pt

done