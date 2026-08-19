#!/bin/bash
# Example usage:
#   ./multiple_feature_extraction.sh 0 5 shift
#   ./multiple_feature_extraction.sh 0 5 shift jitter zoom
#   ./multiple_feature_extraction.sh 0 5 all
#   ./multiple_feature_extraction.sh 0 5 --dataset unbc --model DFER shift
#   ./multiple_feature_extraction.sh 0 5 --dataset biovid --model S all
#   ./multiple_feature_extraction.sh 0 5 --dataset biovid --model VJEPA2_1 all
#   ./multiple_feature_extraction.sh 0 5 --dataset biovid --model G all
#   ./multiple_feature_extraction.sh 0 5 --dataset biovid --model G --prefix my_feats_G --emb_red none shift
#   ./multiple_feature_extraction.sh 0 5 --dataset mintpain --model S all
#   ./multiple_feature_extraction.sh 0 5 --dataset xite --split train --model S all
#   ./multiple_feature_extraction.sh 0 1 --dataset xite --split test --model S shift

# same saving_folder_path is managed in extract_feature.py to avoid overwriting features
# when multiple augmentations are applied together (adds $int_id to the path)
GPU_ID=$1
N_RUNS=$2
shift 2

# Parse named flags --dataset, --model, --prefix, --emb_red, --split; remaining args are configs
DATASET="biovid"
MODEL="S"
PREFIX_OVERRIDE=""
EMB_RED_OVERRIDE=""
SPLIT="train"
SPLIT_SET=0
CONFIGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2";          shift 2 ;;
    --model)   MODEL="$2";            shift 2 ;;
    --prefix)  PREFIX_OVERRIDE="$2";  shift 2 ;;
    --emb_red) EMB_RED_OVERRIDE="$2"; shift 2 ;;
    --split)   SPLIT="$2"; SPLIT_SET=1; shift 2 ;;
    *)         CONFIGS+=("$1"); shift ;;
  esac
done

if [ "$DATASET" != "xite" ] && [ "$SPLIT_SET" -eq 1 ]; then
  echo "--split is only valid with --dataset xite" >&2
  exit 1
fi

# Derive dataset-specific paths
case "$DATASET" in
  biovid)
    PATH_DATASET="partA/video/video_frontalized_interpolated_resolution_original"
    PATH_LABELS="partA/starting_point/samples.csv"
    SAVING_BASE="partA/video/features"
    DATASET_TAG="Biovid"
    SAVING_NAME_PREFIX="spatial_pooled_features_${DATASET_TAG}_B_last143_stride16_interpol"
    EMB_RED="spatial"
    ;;
  unbc)
    PATH_DATASET="UNBC/video/WarpedVideos_Cropped_interpolated_mirror"
    PATH_LABELS="UNBC/starting_point/samples.csv"
    SAVING_BASE="UNBC/video/features"
    DATASET_TAG="UNBC"
    SAVING_NAME_PREFIX="spatial_pooled_features_${DATASET_TAG}_B_last143_stride16_interpol"
    EMB_RED="spatial"
    ;;
  mintpain)
    PATH_DATASET="MIntPAIN/video_frontalized"
    PATH_LABELS="MIntPAIN/starting_point/samples.csv"
    SAVING_BASE="MIntPAIN/features"
    DATASET_TAG="MIntPAIN"
    SAVING_NAME_PREFIX="spatial_pooled_features_${DATASET_TAG}_B_last143_stride16_interpol"
    EMB_RED="spatial"
    ;;
  agedb)
    PATH_DATASET="AgeDB/video/video_age"
    PATH_LABELS="AgeDB/starting_point/samples.csv"
    SAVING_BASE="AgeDB/features"
    DATASET_TAG="age"
    SAVING_NAME_PREFIX="all_pooled_features_${DATASET_TAG}"
    EMB_RED="all"
    ;;
  xite)
    PATH_DATASET="XITE/video/video_frontalized"
    case "$SPLIT" in
      train) PATH_LABELS="XITE/starting_point/train_samples.csv" ;;
      test)  PATH_LABELS="XITE/starting_point/test_samples.csv" ;;
      all)   PATH_LABELS="XITE/starting_point/samples.csv" ;;
      *)
        echo "Unknown XITE split: $SPLIT (valid: train, test, all)" >&2
        exit 1
        ;;
    esac
    SAVING_BASE="XITE/video/features"
    DATASET_TAG="XITE"
    SAVING_NAME_PREFIX="spatial_pooled_features_${DATASET_TAG}_B_last143_stride16_interpol_${SPLIT}"
    EMB_RED="spatial"
    ;;
  *)
    echo "Unknown dataset: $DATASET (valid: biovid, unbc, mintpain, agedb, xite)"
    exit 1
    ;;
esac

# Apply optional CLI overrides for prefix / embedding reduction
if [ -n "$PREFIX_OVERRIDE" ]; then
  SAVING_NAME_PREFIX="$PREFIX_OVERRIDE"
fi
if [ -n "$EMB_RED_OVERRIDE" ]; then
  EMB_RED="$EMB_RED_OVERRIDE"
fi

# Derive model-specific identifiers
case "$MODEL" in
  S)        MODEL_FOLDER="VideoMaev2_S"; MODEL_TYPE_ARG="S"          ;;
  G)        MODEL_FOLDER="VideoMaev2_G"; MODEL_TYPE_ARG="G"          ;;
  DFER)     MODEL_FOLDER="DFER";         MODEL_TYPE_ARG="DFER"       ;;
  VJEPA2_1) MODEL_FOLDER="VJEPA2_1_B";   MODEL_TYPE_ARG="vjepa2_1_B" ;;
  *)
    echo "Unknown model: $MODEL (valid: S, G, DFER, VJEPA2_1)"
    exit 1
    ;;
esac

if [ "$DATASET" = "xite" ]; then
  echo "Checking XITE frontalized videos for split: $SPLIT"
  if ! python3 XITE/check_frontalized_videos.py \
      --labels "$PATH_LABELS" --clip-length 16; then
    echo "XITE frontalized-video validation failed; repair the reported videos before extraction." >&2
    exit 1
  fi
fi

ALL_CONFIGS=(shift jitter gaussian zoom jitter_shift rotation_zoom shift_hflip)

if [ "${#CONFIGS[@]}" -eq 1 ] && [ "${CONFIGS[0]}" = "all" ]; then
    CONFIGS=("${ALL_CONFIGS[@]}")
fi

run_config() {
    local CONFIG=$1
    for ((i=1; i<=N_RUNS; i++))
    do
        echo "Run $i / $N_RUNS | GPU $GPU_ID | CONFIG $CONFIG | DATASET $DATASET | MODEL $MODEL"

        SAVING_PATH="${SAVING_BASE}/${MODEL_FOLDER}/${SAVING_NAME_PREFIX}_${CONFIG}"

        case $CONFIG in

        shift)
            CUDA_VISIBLE_DEVICES=$GPU_ID python3 extract_feature.py \
            --model_type $MODEL_TYPE_ARG --emb_red $EMB_RED \
            --path_dataset $PATH_DATASET \
            --path_labels $PATH_LABELS \
            --saving_folder_path $SAVING_PATH \
            --backbone_type video --gp --save_as_safetensors \
            --stride_window 16 --stride_inside_window 1 --float_16 --save_big_feature \
            --spatial_shift
            ;;

        jitter)
            CUDA_VISIBLE_DEVICES=$GPU_ID python3 extract_feature.py \
            --model_type $MODEL_TYPE_ARG --emb_red $EMB_RED \
            --path_dataset $PATH_DATASET \
            --path_labels $PATH_LABELS \
            --saving_folder_path $SAVING_PATH \
            --backbone_type video --gp --save_as_safetensors \
            --stride_window 16 --stride_inside_window 1 --float_16 --save_big_feature \
            --color_jitter
            ;;

        gaussian)
            CUDA_VISIBLE_DEVICES=$GPU_ID python3 extract_feature.py \
            --model_type $MODEL_TYPE_ARG --emb_red $EMB_RED \
            --path_dataset $PATH_DATASET \
            --path_labels $PATH_LABELS \
            --saving_folder_path $SAVING_PATH \
            --backbone_type video --gp --save_as_safetensors \
            --stride_window 16 --stride_inside_window 1 --float_16 --save_big_feature \
            --gaussian_smooth --gaussian_sigma_min 0.4 --gaussian_sigma_max 0.9 --gaussian_kernel_size 3
            ;;

        zoom)
            CUDA_VISIBLE_DEVICES=$GPU_ID python3 extract_feature.py \
            --model_type $MODEL_TYPE_ARG --emb_red $EMB_RED \
            --path_dataset $PATH_DATASET \
            --path_labels $PATH_LABELS \
            --saving_folder_path $SAVING_PATH \
            --backbone_type video --gp --save_as_safetensors \
            --stride_window 16 --stride_inside_window 1 --float_16 --save_big_feature \
            --zoom
            ;;

        jitter_shift)
            CUDA_VISIBLE_DEVICES=$GPU_ID python3 extract_feature.py \
            --model_type $MODEL_TYPE_ARG --emb_red $EMB_RED \
            --path_dataset $PATH_DATASET \
            --path_labels $PATH_LABELS \
            --saving_folder_path $SAVING_PATH \
            --backbone_type video --gp --save_as_safetensors \
            --stride_window 16 --stride_inside_window 1 --float_16 --save_big_feature \
            --color_jitter --spatial_shift
            ;;

        rotation_zoom)
            CUDA_VISIBLE_DEVICES=$GPU_ID python3 extract_feature.py \
            --model_type $MODEL_TYPE_ARG --emb_red $EMB_RED \
            --path_dataset $PATH_DATASET \
            --path_labels $PATH_LABELS \
            --saving_folder_path $SAVING_PATH \
            --backbone_type video --gp --save_as_safetensors \
            --stride_window 16 --stride_inside_window 1 --float_16 --save_big_feature \
            --zoom --rotation
            ;;

        shift_hflip)
            CUDA_VISIBLE_DEVICES=$GPU_ID python3 extract_feature.py \
            --model_type $MODEL_TYPE_ARG --emb_red $EMB_RED \
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
