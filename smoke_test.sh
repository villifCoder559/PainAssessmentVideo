#!/usr/bin/env bash
# Minimal end-to-end smoke test: checks installation, imports, paths and that every
# pipeline runs without crashing. Results are NOT meaningful (1 epoch, data subsets).
#
# Usage (from anywhere, with the conda env activated):
#   bash smoke_test.sh               # fast set (~5-10 min on CPU)
#   SMOKE_FULL=1 bash smoke_test.sh  # also all 12 paper training configurations and VideoMAEv2-S extraction
#
# Environment variables:
#   SMOKE_OUT     output folder for logs/training runs (default: smoke_out)
#   SMOKE_DEVICE  "cpu" to hide GPUs (default), or a GPU index such as "0"
#   SMOKE_FULL    1 to run all 12 paper training configurations
#   PYTHON        python executable (default: python)
#
# Pipelines covered:
#   env       pip check + key imports (torch, decord, torchsort, mediapipe, ...)
#   extract   MAE-DFER backbone forward pass (extract_feature.py) on a synthetic video (+ VideoMAEv2-S in full mode)
#   train_*   train_model.py on precomputed embeddings (1 epoch, first fold/subfold, subject subsets)
#   xspace_*  cross_space_projection.py via run_cross_space_configs.sh: linear, mlp, autoencoder
#             (1 target x 2 source checkpoints -> aggregated, as in the paper configs), procrustes,
#             linear_close (refinement 3) and a frozen random adapter (refinement 4)
#   logs      cross_space_logs.py --only_aggregated on the cross-space outputs
# Datasets whose CSVs/embeddings are missing are SKIPPED; UNBC and MIntPAIN are required for
# the cross-space step.

set -uo pipefail
cd "$(dirname "$0")" || exit 2
REPO=$(pwd)
PY=${PYTHON:-python}
OUT=${SMOKE_OUT:-smoke_out}
DEVICE=${SMOKE_DEVICE:-cpu}
FULL=${SMOKE_FULL:-0}
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT/logs" "$OUT/csv" "$OUT/train" "$OUT/xspace_yaml"
OUT=$(cd "$OUT" && pwd)

if [[ $DEVICE = cpu ]]; then export CUDA_VISIBLE_DEVICES=""; else export CUDA_VISIBLE_DEVICES=$DEVICE; fi

declare -a SUMMARY=()
FAILED=0
T0=$(date +%s)

# run_step NAME CMD... : run CMD, log to $OUT/logs/NAME.log, record PASS/FAIL.
run_step() {
  local name=$1; shift
  local log=$OUT/logs/$name.log t=$(date +%s)
  printf '%-28s ' "$name"
  if "$@" > "$log" 2>&1 < /dev/null; then
    SUMMARY+=("PASS  $name ($(( $(date +%s) - t ))s)"); echo "PASS ($(( $(date +%s) - t ))s)"
  else
    SUMMARY+=("FAIL  $name -> $log"); echo "FAIL (see $log)"; FAILED=1
    tail -n 15 "$log" | sed 's/^/    | /'
  fi
}
skip_step() { printf '%-28s SKIP (%s)\n' "$1" "$2"; SUMMARY+=("SKIP  $1: $2"); }
have() { local p; for p in "$@"; do [[ -e $p ]] || return 1; done; }

# ---------------------------------------------------------------- 0. environment
ENV_CHECK="
import subprocess, sys
import torch, torchvision, numpy, scipy, pandas, sklearn, cv2, mediapipe, transformers, timm
import decord, torchsort, optuna, optunahub, safetensors, einops, coral_pytorch, torchmetrics
print('python', sys.version.split()[0], '| torch', torch.__version__, '| cuda available', torch.cuda.is_available())
print('soft_rank', torchsort.soft_rank(torch.tensor([[3., 1., 2.]])))
r = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
print(r.stdout, r.stderr)
sys.exit(r.returncode)
"
run_step env "$PY" -c "$ENV_CHECK"
run_step imports "$PY" -c "import train_model, cross_space_projection, cross_space_logs, extract_feature"

# ---------------------------------------------------------------- 1. backbone forward pass
DFER_W=MAE_DFER/saved/model/pretraining/voxceleb2/videomae_pretrain_base_dim512_local_global_attn_depth16_region_size2510_patch16_160_frame_16x4_tube_mask_ratio_0.9_e100_with_diff_target_server170/checkpoint-49.pth
VMAE_W=VideoMAEv2/pretrained/vit_s_k710_dl_from_giant.pth
if ! have "$DFER_W" "$VMAE_W"; then
  echo "ERROR: backbone weights missing. Training also needs them. See README.md (Backbone weights):"
  echo "  $DFER_W"; echo "  $VMAE_W"
  exit 2
fi
# The dataset is inferred from keywords in the video path ('unbc', 'parta', 'mintpain', 'xite', ...).
SYN=$OUT/synthetic/UNBC_synthetic_videos
make_video() {
  mkdir -p "$SYN/s001" && printf 'subject_id\tsubject_name\tclass_id\tclass_name\tsample_id\tsample_name\n0\ts001\t0\tBL1\t1\ts001-clip1\n' > "$OUT/synthetic/samples.csv" \
  && ffmpeg -y -loglevel error -f lavfi -i testsrc=size=224x224:rate=25 -frames:v 40 -pix_fmt yuv420p "$SYN/s001/s001-clip1.mp4"
}
run_step synthetic_video make_video
if [[ $FULL = 1 ]]; then EXTRACT_MT="DFER S"; else EXTRACT_MT="DFER"; fi
for mt in $EXTRACT_MT; do
  rm -rf "$OUT/synthetic/features_$mt"
  run_step "extract_$mt" "$PY" extract_feature.py --model_type $mt --emb_red spatial --path_dataset "$SYN" \
    --path_labels "$OUT/synthetic/samples.csv" --saving_folder_path "$OUT/synthetic/features_$mt" \
    --backbone_type video --save_as_safetensors --stride_window 16 --stride_inside_window 1 --float_16 --save_big_feature
done

# ---------------------------------------------------------------- 2. model training
# subset_csv SRC DST N : keep the rows of the first N subjects (a folder of split CSVs is
# subset file by file). Only the amount of data changes, not the splitting logic.
subset_csv() {
  "$PY" - "$@" <<'EOF'
import os, sys, pandas as pd
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
files = [(os.path.join(src, f), os.path.join(dst, f)) for f in sorted(os.listdir(src))] if os.path.isdir(src) else [(src, dst)]
for s, d in files:
  df = pd.read_csv(s, sep='\t', dtype={'sample_name': str, 'subject_name': str})
  keep = sorted(df['subject_id'].unique())[:n]
  os.makedirs(os.path.dirname(d), exist_ok=True)
  df[df['subject_id'].isin(keep)].to_csv(d, sep='\t', index=False)
EOF
}

# Paper configurations (paper_reproduction / REPRO_model_training), minimal mode is appended:
# 1 epoch, first fold + first subfold only (--stop 1 1), 2 dataloader workers.
MIN_ARGS="--ep 1 --stop 1 1 --n_workers 2"
COMMON_ATT="--head ATTENTIVE_JEPA --num_cross_head 1 --num_heads 8 --opt adamw --init_network default --p_early_stop 2000 --min_delta 0.005 --threshold_mode abs --regulariz_lambda_L1 0 --warm_up_epochs 5 --warm_up_scheduler linear --warm_up_start_factor 0.01 --drop_attn 0. --drop_residual 0. --label_smooth 0 --pos_enc 3 --n_trials 1 --timeout 140 --pruner_n_warmup_steps 500 --pruner_threshold_lower 0.0 --optuna_sampler grid --prefetch_factor 2 --validation_enabled 1 --is_subject_independent 1 --concatenate_quadrants 0 --skip_test 0 --embedding_reduction spatial"
BIO_AUG="--sampler_loader_type selective_augm --filtered_augm_n_keep 1 --filtered_augm_strategy 1 --keep_original 0.02 --optuna_categorical 1 --use_test_as_val 0 --save_best_model --stratified_training 1 --complete_block 2 --mlp_num_hidden_layers 1 --mlp_ratio 2 --custom_mlp 1 --cross_block_after_transformers 0 --loss l1 --nr_block 2 --model_dropout 0.1 --regulariz_lambda_L2 0.1 --batch_train 512 --k_fold 5 --load_dataset_in_memory 0"
MINT_AUG="--k_fold 5 --batch_train 128 --regulariz_lambda_L2 0.01 --scheduler_name cosine --min_lr 0.0000001 --loss l1 --nr_block 2 --cross_block_after_transformers 0 --sampler_loader_type selective_augm --filtered_augm_n_keep 1 --filtered_augm_strategy 1 --keep_original 0.8 --undersample_max_per_class 600 --optuna_categorical 1 --use_test_as_val 0 --save_best_model --stratified_training 1 --complete_block 2 --mlp_num_hidden_layers 1 --mlp_ratio 1 --custom_mlp 1 --target_samples_per_class_training 600 --load_dataset_in_memory 0 --lr 0.0001"
XITE_STD="--k_fold 5 --lr 0.0003 --batch_train 16 --regulariz_lambda_L2 0.5 --scheduler_name cosine --min_lr 0.0000001 --loss ce --nr_block 1 --sampler_loader_type standard --optuna_categorical 1 --use_test_as_val 0 --save_best_model --complete_block 2 --load_dataset_in_memory 1"
UNBC_STD="--k_fold 5 --batch_train 16 --scheduler_name cosine --loss l1 --cross_block_after_transformers 0 --sampler_loader_type standard --optuna_categorical 1 --use_test_as_val 0 --save_best_model --stratified_training 1 --complete_block 2 --mlp_num_hidden_layers 1 --mlp_ratio 0.5 --custom_mlp 1 --normalize_labels 0 --load_dataset_in_memory 0 --lr 0.0001"
UNBC_VID=UNBC/video/WarpedVideos_Cropped_interpolated_mirror
F_UNBC_S=UNBC/video/features/VideoMaev2_S/spatial_pooled_features_UNBC_B_last143_stride16_interpol
F_UNBC_D=UNBC/video/features/DFER/spatial_pooled_features_UNBC_B_last143_stride16_interpol
F_MINT_S=MIntPAIN/features/VideoMaev2_S/spatial_pooled_features_MIntPAIN_B_last143_stride16_interpol
F_MINT_D=MIntPAIN/features/DFER/spatial_pooled_features_MIntPAIN_B_last143_stride16_interpol
F_BIO_S=partA/video/features/VideoMaev2_S/spatial_pooled_features_Biovid_B_last143_stride16_interpol
F_BIO_D=partA/video/features/DFER/spatial_pooled_features_Biovid_B_last143_stride16_interpol
F_XITE_S=XITE/video/features/VideoMaev2_S/spatial_pooled_features_XITE_B_last143_stride16_interpol_all
F_XITE_D=XITE/video/features/DFER/spatial_pooled_features_XITE_B_last143_stride16_interpol_all

# id | needs (csv,features) | csv (subset) | train_model.py arguments
train_cfg() {
  case $1 in
    01) echo "partA/starting_point/samples_adversarial.csv $F_BIO_S|--mt S --lr 0.0001 --scheduler_name cosine $BIO_AUG --ffsp $F_BIO_S --path_video_dataset partA/video/video_frontalized_interpolated_resolution_original";;
    02) echo "partA/starting_point/samples_adversarial.csv $F_BIO_D|--mt DFER --lr 0.0002 --scheduler_name cosine_restart --first_restart_epochs 10 --multiplier_restart 2 --min_lr 0.0000001 $BIO_AUG --save_model_every_n_epochs 100 --ffsp $F_BIO_D --path_video_dataset partA/video/video_frontalized_interpolated_resolution_original";;
    03) echo "MIntPAIN/starting_point/samples.csv $F_MINT_D|--mt DFER --model_dropout 0.0 $MINT_AUG --ffsp $F_MINT_D --path_video_dataset MIntPAIN/video_frontalized";;
    04) echo "MIntPAIN/starting_point/samples.csv $F_MINT_S|--mt S --model_dropout 0.3 $MINT_AUG --ffsp $F_MINT_S --path_video_dataset MIntPAIN/video_frontalized";;
    05) echo "XITE/starting_point/splits $F_XITE_S|--mt S --model_dropout 0.5 $XITE_STD --ffsp $F_XITE_S --path_video_dataset XITE/video/video_frontalized";;
    06) echo "XITE/starting_point/splits $F_XITE_D|--mt DFER --model_dropout 0.0 $XITE_STD --ffsp $F_XITE_D --path_video_dataset XITE/video/video_frontalized";;
    07) echo "UNBC/starting_point/samples_OPI_0_to_4.csv $F_UNBC_S|--mt S --regulariz_lambda_L2 0.01 --model_dropout 0.3 --nr_block 2 --target_samples_per_class_training 35 $UNBC_STD --ffsp $F_UNBC_S --path_video_dataset $UNBC_VID";;
    08) echo "UNBC/starting_point/samples_OPI_0_to_4.csv $F_UNBC_D|--mt DFER --regulariz_lambda_L2 0 --nr_blocks 3 --model_dropout 0.1 --target_samples_per_class_training 60 $UNBC_STD --ffsp $F_UNBC_D --path_video_dataset $UNBC_VID";;
    09) echo "UNBC/starting_point/samples_OPI.csv $F_UNBC_D|--mt DFER --regulariz_lambda_L2 0 --nr_blocks 3 --model_dropout 0.3 --target_samples_per_class_training 35 $UNBC_STD --ffsp $F_UNBC_D --path_video_dataset $UNBC_VID";;
    10) echo "UNBC/starting_point/samples_OPI.csv $F_UNBC_S|--mt S --regulariz_lambda_L2 0.01 --model_dropout 0.1 --nr_block 2 --target_samples_per_class_training 60 $UNBC_STD --ffsp $F_UNBC_S --path_video_dataset $UNBC_VID";;
    11) echo "UNBC/starting_point/samples_adversarial.csv $F_UNBC_S|--mt S --lr 0.0005 --batch_train 16 --regulariz_lambda_L2 0.01 --model_dropout 0.3 --mlp_ratio 0.5 --nr_block 2 --k_fold 5 --scheduler_name cosine --loss l1 --cross_block_after_transformers 0 --sampler_loader_type selective_augm --filtered_augm_n_keep 1 --filtered_augm_strategy 1 --keep_original 0.2 --optuna_categorical 1 --use_test_as_val 0 --save_best_model --stratified_training 1 --complete_block 2 --mlp_num_hidden_layers 1 --custom_mlp 1 --target_samples_per_class_training 35 --normalize_labels 0 --load_dataset_in_memory 0 --ffsp $F_UNBC_S --path_video_dataset $UNBC_VID";;
    12) echo "UNBC/starting_point/samples.csv $F_UNBC_D|--mt DFER --gp --lr 0.00001 --k_fold 5 --batch_train 9 --key_early_stopping val_loss --regulariz_lambda_L2 0.01 --scheduler_name cosine_restart --first_restart_epochs 7 --multiplier_restart 2 --model_dropout 0.1 --mlp_ratio 0.5 --nr_block 1 --cross_block_after_transformers 0 --timeout 102 --loss l1 --optuna_categorical 0 --stratified_training 0 --target_samples_per_class_training 31 --sampler_loader_type standard --use_test_as_val 1 --load_dataset_in_memory 0 --ffsp $F_UNBC_D --path_video_dataset $UNBC_VID";;
  esac
}
n_subjects() { case $1 in 01|02) echo 10;; 05|06) echo 3;; *) echo 1000;; esac; }  # UNBC and MIntPAIN: all subjects (cross-space needs >=4 subjects per split)

if [[ $FULL = 1 ]]; then TRAIN_IDS="01 02 03 04 05 06 07 08 09 10 11 12"; else TRAIN_IDS="07 03 12 06 01"; fi
for id in $TRAIN_IDS; do
  IFS='|' read -r needs args <<< "$(train_cfg $id)"
  read -r csv feats <<< "$needs"
  if ! have "$csv" "$feats"; then skip_step "train_$id" "missing $csv or $feats"; continue; fi
  sub=$OUT/csv/$id/$(basename "$csv")
  rm -rf "$OUT/csv/$id" "$OUT/train/$id"_*
  subset_csv "$csv" "$sub" "$(n_subjects $id)"
  # $COMMON_ATT first so that each configuration (and $MIN_ARGS) can override it.
  # --global_folder_name is absolute, so it is not re-prefixed by --gp.
  run_step "train_$id" "$PY" train_model.py $COMMON_ATT $args --csv "$sub" --global_folder_name "$OUT/train/$id" $MIN_ARGS
done

# ---------------------------------------------------------------- 3. cross-space projection
# Source (old) = UNBC VideoMAEv2-S (07), target (new) = MIntPAIN MAE-DFER (03).
ckpt() { ls -1 "$OUT"/train/"$1"_*/*/train_ATTENTIVE_JEPA/k0_cross_val/k0_cross_val_sub_0/best_model_ep_*.pt 2>/dev/null | head -n 1; }
OLD=$(ckpt 07); NEW=$(ckpt 03)
XTAG=smoke_test/$STAMP
if [[ -z $OLD || -z $NEW ]] || ! have "$F_UNBC_D" "$F_MINT_S"; then
  skip_step xspace "needs UNBC (07) and MIntPAIN (03) checkpoints and all four UNBC/MIntPAIN embedding folders"
else
  YDIR=$OUT/xspace_yaml/$STAMP; mkdir -p "$YDIR"
  # METHOD REFINEMENT EXTRA_PROJECTOR_LINES OLD_LIST. An OLD_LIST with 2 entries runs the paper's
  # multi-checkpoint path (every new x old pair is a subtrial, pooled into an aggregated_* folder).
  write_yaml() {
    cat > "$YDIR/refinement$2_$1.yaml" <<EOF
# Smoke-test config: same keys as Cross_projection_yaml/config_paper_tests_seed_42, tiny epochs.
new_model_pth: ["$NEW"]
old_model_pth: [$4]
num_anchors: [20]
anchor_selection_type: [balance_class_random]
csv_anchor_selection: [train]
old_model_csv: [test]
interpolation_similarity: [$1]
weighting_method: [none]
mlp_activation: [gelu]
mlp_num_layers: [1]
rbf_sigma: [1.0]
num_refinement_samples: [-1]
n_trials: null
optuna_sampler: grid
seed: [42]
run_tag: $XTAG/refinement$2_$1
refinement: $2
linear_projector:
  lr: [1.0e-5]
  batch_size: [64]
  optimizer: adamw
  weight_decay: 0
  epochs: 2
  normalize_embeddings: [false]
  loss: [mse]
$3
refinement_config:
  lr_projector: [1.0e-4]
  lr_linear: [1.0e-4]
  lambda_B: [1.0e-4]
  lambda_A: [1.0e-3]
  optimizer: adamw
  weight_decay: 0
  epochs: 2
  loss: [mse]
  batch_size: 64
EOF
  }
  OLD1="\"$OLD\""; OLD2="\"$OLD\", \"$OLD\""
  write_yaml linear 3 "" "$OLD2"
  write_yaml mlp 3 "" "$OLD2"
  write_yaml autoencoder 3 "  encoder_ratio: [4]" "$OLD2"
  write_yaml procrustes 3 "" "$OLD1"
  write_yaml linear_close 3 "" "$OLD1"
  write_yaml linear 4 "" "$OLD1"
  [[ $DEVICE = cpu ]] && XGPU=cpu || XGPU=$DEVICE
  run_step xspace_projection bash run_cross_space_configs.sh "$XGPU" "$YDIR"
  run_step xspace_logs "$PY" cross_space_logs.py --pkl_path "Cross_projection/$XTAG" --only_aggregated --skip_umap
fi

# ---------------------------------------------------------------- summary
echo; echo "==== smoke test summary ($(( $(date +%s) - T0 ))s, device: $DEVICE) ===="
printf '%s\n' "${SUMMARY[@]}"
echo "Logs: $OUT/logs   Training runs: $OUT/train   Cross-space runs: $REPO/Cross_projection/$XTAG"
exit $FAILED
