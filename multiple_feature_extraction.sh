#!/bin/bash
# Example usage:
#   ./multiple_feature_extraction.sh 0 5 shift
#   ./multiple_feature_extraction.sh 0 5 shift jitter zoom
#   ./multiple_feature_extraction.sh 0 5 all
#   ./multiple_feature_extraction.sh 0 5 --dataset unbc --model DFER shift
#   ./multiple_feature_extraction.sh 0 5 --dataset biovid --model S all

# same saving_folder_path is managed in extract_feature.py to avoid overwriting features
# when multiple augmentations are applied together (adds $int_id to the path)
GPU_ID=$1
N_RUNS=$2
shift 2

# Parse named flags --dataset and --model; remaining args are configs
DATASET="biovid"
MODEL="S"
CONFIGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --model)   MODEL="$2";   shift 2 ;;
    *)         CONFIGS+=("$1"); shift ;;
  esac
done

# Derive dataset-specific paths
case "$DATASET" in
  biovid)
    PATH_DATASET="partA/video/video_frontalized_interpolated_resolution_original"
    PATH_LABELS="partA/starting_point/samples.csv"
    SAVING_BASE="partA/video/features"
    DATASET_TAG="Biovid"
    ;;
  unbc)
    PATH_DATASET="UNBC/video/WarpedVideos_Cropped_interpolated_mirror"
    PATH_LABELS="UNBC/starting_point/samples.csv"
    SAVING_BASE="UNBC/video/features"
    DATASET_TAG="UNBC"
    ;;
  *)
    echo "Unknown dataset: $DATASET (valid: biovid, unbc)"
    exit 1
    ;;
esac

# Derive model-specific identifiers
case "$MODEL" in
  S)    MODEL_FOLDER="VideoMaev2_S"; MODEL_TYPE_ARG="S"    ;;
  DFER) MODEL_FOLDER="DFER";         MODEL_TYPE_ARG="DFER" ;;
  *)
    echo "Unknown model: $MODEL (valid: S, DFER)"
    exit 1
    ;;
esac

ALL_CONFIGS=(shift jitter gaussian zoom jitter_shift rotation_zoom shift_hflip)

if [ "${#CONFIGS[@]}" -eq 1 ] && [ "${CONFIGS[0]}" = "all" ]; then
    CONFIGS=("${ALL_CONFIGS[@]}")
fi

run_config() {
    local CONFIG=$1
    for ((i=1; i<=N_RUNS; i++))
    do
        echo "Run $i / $N_RUNS | GPU $GPU_ID | CONFIG $CONFIG | DATASET $DATASET | MODEL $MODEL"

        SAVING_PATH="${SAVING_BASE}/${MODEL_FOLDER}/spatial_pooled_features_${DATASET_TAG}_B_last143_stride16_interpol_${CONFIG}"

        case $CONFIG in

        shift)
            CUDA_VISIBLE_DEVICES=$GPU_ID python3 extract_feature.py \
            --model_type $MODEL_TYPE_ARG --emb_red spatial \
            --path_dataset $PATH_DATASET \
            --path_labels $PATH_LABELS \
            --saving_folder_path $SAVING_PATH \
            --backbone_type video --gp --save_as_safetensors \
            --stride_window 16 --stride_inside_window 1 --float_16 --save_big_feature \
            --spatial_shift
            ;;

        jitter)
            CUDA_VISIBLE_DEVICES=$GPU_ID python3 extract_feature.py \
            --model_type $MODEL_TYPE_ARG --emb_red spatial \
            --path_dataset $PATH_DATASET \
            --path_labels $PATH_LABELS \
            --saving_folder_path $SAVING_PATH \
            --backbone_type video --gp --save_as_safetensors \
            --stride_window 16 --stride_inside_window 1 --float_16 --save_big_feature \
            --color_jitter
            ;;

        gaussian)
            CUDA_VISIBLE_DEVICES=$GPU_ID python3 extract_feature.py \
            --model_type $MODEL_TYPE_ARG --emb_red spatial \
            --path_dataset $PATH_DATASET \
            --path_labels $PATH_LABELS \
            --saving_folder_path $SAVING_PATH \
            --backbone_type video --gp --save_as_safetensors \
            --stride_window 16 --stride_inside_window 1 --float_16 --save_big_feature \
            --gaussian_smooth --gaussian_sigma_min 0.4 --gaussian_sigma_max 0.9 --gaussian_kernel_size 3
            ;;

        zoom)
            CUDA_VISIBLE_DEVICES=$GPU_ID python3 extract_feature.py \
            --model_type $MODEL_TYPE_ARG --emb_red spatial \
            --path_dataset $PATH_DATASET \
            --path_labels $PATH_LABELS \
            --saving_folder_path $SAVING_PATH \
            --backbone_type video --gp --save_as_safetensors \
            --stride_window 16 --stride_inside_window 1 --float_16 --save_big_feature \
            --zoom
            ;;

        jitter_shift)
            CUDA_VISIBLE_DEVICES=$GPU_ID python3 extract_feature.py \
            --model_type $MODEL_TYPE_ARG --emb_red spatial \
            --path_dataset $PATH_DATASET \
            --path_labels $PATH_LABELS \
            --saving_folder_path $SAVING_PATH \
            --backbone_type video --gp --save_as_safetensors \
            --stride_window 16 --stride_inside_window 1 --float_16 --save_big_feature \
            --color_jitter --spatial_shift
            ;;

        rotation_zoom)
            CUDA_VISIBLE_DEVICES=$GPU_ID python3 extract_feature.py \
            --model_type $MODEL_TYPE_ARG --emb_red spatial \
            --path_dataset $PATH_DATASET \
            --path_labels $PATH_LABELS \
            --saving_folder_path $SAVING_PATH \
            --backbone_type video --gp --save_as_safetensors \
            --stride_window 16 --stride_inside_window 1 --float_16 --save_big_feature \
            --zoom --rotation
            ;;

        shift_hflip)
            CUDA_VISIBLE_DEVICES=$GPU_ID python3 extract_feature.py \
            --model_type $MODEL_TYPE_ARG --emb_red spatial \
            --path_dataset $PATH_DATASET \
            --path_labels $PATH_LABELS \
            --saving_folder_path $SAVING_PATH \
            --backbone_type video --gp --save_as_safetensors \
            --stride_window 16 --stride_inside_window 1 --float_16 --save_big_feature \
            --spatial_shift --h_flip
            ;;

        *)
            echo "Invalid config: $CONFIG"
            exit 1
            ;;

        esac

    done
}

for CONFIG in "${CONFIGS[@]}"; do
    run_config "$CONFIG"
done
