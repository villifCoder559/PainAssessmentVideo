# Cross-Space Latent Representation Transfer for Video Pain Assessment

This repository trains pain-intensity models on frozen video-foundation-model embeddings and
studies how to transfer them between embedding spaces. Every video is encoded once by a frozen
backbone, either **MAE-DFER** (facial-expression ViT-B, 512-d) or **VideoMAEv2-S** (384-d).
Training then fits an attentive-probe head (`ATTENTIVE_JEPA`) on the precomputed embeddings with
subject-independent 5-fold cross-validation, on **BioVid Part A**, **UNBC-McMaster** (OPI 0–4,
OPI 0–5, VAS), **MIntPAIN** and **XITE**. The cross-space experiments project embeddings of a
source model (VideoMAEv2-S, dataset A) into the space of a target model (MAE-DFER, dataset B)
with a *linear*, *MLP*, *autoencoder*, *Procrustes* or *closed-form linear* projector learned
from a few anchor pairs. They then optionally refine the target head and evaluate the
target model on the projected source data.

> **What you need.** This code release, the released label CSVs, and the precomputed embeddings
> (including the augmented-embedding folders, see below). You also need the two public
> backbone checkpoints. Raw videos are **not** needed and are not distributed.

---

## 1. Environment

Tested (2026-10-06) on Ubuntu 22.04 x86_64 with a fresh Miniforge (conda 26.7). The resulting
env had **Python 3.10.21** and **PyTorch 2.5.1**. It ran on CPU only (no GPU visible), and with
CUDA 11.8 wheels on an NVIDIA RTX 2080 Ti (driver 550). Run all commands from the
**repository root**.

**Recommended: exact pinned environment (Linux x86_64, CPU).**

```sh
conda env create -f env_portability/environment-pinned-linux-64.yml   # creates "pain-portable"
conda activate pain-portable
python -m pip check
```

This is the environment exported from the clean-room install that passed `smoke_test.sh`. It
contains CPU PyTorch. Training runs on CPU, slowly, but enough for the smoke test and small runs.

**NVIDIA GPU (Linux x86_64, CUDA 11.8 runtime wheels; needs a driver ≥ 520).**

```sh
conda env create -f env_portability/environment-cuda-pinned-linux-64.yml   # creates "pain-portable-cuda"
conda activate pain-portable-cuda
python -m pip check
python -c "import torch, torchsort; print(torch.cuda.is_available(), torchsort.soft_rank(torch.tensor([[3.,1.,2.]], device='cuda')))"
```

**Flexible (non-pinned) specifications.** `env_portability/environment.yml` plus
`environment-native.yml` (CPU), or `environment-cuda.yml` plus the Decord/torchsort commands in
[env_portability/ENVIRONMENT.md](env_portability/ENVIRONMENT.md) (GPU), resolve current
compatible versions. Windows and macOS instructions are documented there but were **not**
tested end-to-end.

## 2. Backbone weights (required, also for training on precomputed embeddings)

`train_model.py` and `cross_space_projection.py` always build the backbone and load its
weights, even when they only read precomputed embeddings. Download both checkpoints to these
exact paths:

| Backbone | File (relative to repo root) | Source |
| --- | --- | --- |
| MAE-DFER (1.1 GB) | `MAE_DFER/saved/model/pretraining/voxceleb2/videomae_pretrain_base_dim512_local_global_attn_depth16_region_size2510_patch16_160_frame_16x4_tube_mask_ratio_0.9_e100_with_diff_target_server170/checkpoint-49.pth` | [Google Drive](https://drive.google.com/file/d/1nzvMITUHic9fKwjQ7XLcnaXYViWTawRv/view?usp=sharing) (from the [MAE-DFER README](MAE_DFER/README.md)) |
| VideoMAEv2-S (44 MB) | `VideoMAEv2/pretrained/vit_s_k710_dl_from_giant.pth` | [HuggingFace](https://huggingface.co/OpenGVLab/VideoMAE2/resolve/main/distill/vit_s_k710_dl_from_giant.pth) (from the [VideoMAEv2 model zoo](VideoMAEv2/docs/MODEL_ZOO.md)) |

From the repository root:

```sh
curl -L --create-dirs -o VideoMAEv2/pretrained/vit_s_k710_dl_from_giant.pth \
  https://huggingface.co/OpenGVLab/VideoMAE2/resolve/main/distill/vit_s_k710_dl_from_giant.pth
DFER_DIR=MAE_DFER/saved/model/pretraining/voxceleb2/videomae_pretrain_base_dim512_local_global_attn_depth16_region_size2510_patch16_160_frame_16x4_tube_mask_ratio_0.9_e100_with_diff_target_server170
mkdir -p "$DFER_DIR"
python -m pip install gdown   # or download the Google Drive file in a browser into $DFER_DIR
gdown 1nzvMITUHic9fKwjQ7XLcnaXYViWTawRv -O "$DFER_DIR/checkpoint-49.pth"
```

## 3. Data layout

Place the released CSVs and embeddings at the repository root with **exactly** these names
(symbolic links to another disk are fine). The code infers the dataset from keywords in these
paths (`unbc`, `parta`/`biovid`, `mintpain`, `xite`), and the cross-space code looks embeddings
up by these relative paths.

```
<repo>/
├── UNBC/
│   ├── starting_point/            samples.csv  samples_OPI.csv  samples_OPI_0_to_4.csv  samples_adversarial.csv
│   └── video/features/
│       ├── DFER/spatial_pooled_features_UNBC_B_last143_stride16_interpol/<subject_name>/<sample_name>.safetensors
│       ├── DFER/spatial_pooled_features_UNBC_B_last143_stride16_interpol_<aug>[$N]/...      (augmented; see below)
│       └── VideoMaev2_S/spatial_pooled_features_UNBC_B_last143_stride16_interpol[ _<aug>[$N] ]/...
├── partA/                         (BioVid Part A)
│   ├── starting_point/samples_adversarial.csv
│   └── video/features/{DFER,VideoMaev2_S}/spatial_pooled_features_Biovid_B_last143_stride16_interpol[ _<aug>[$N] ]/...
├── MIntPAIN/
│   ├── starting_point/samples.csv
│   └── features/{DFER,VideoMaev2_S}/spatial_pooled_features_MIntPAIN_B_last143_stride16_interpol[ _<aug>[$N] ]/...
└── XITE/
    ├── starting_point/splits/     train_samples_21.csv  val_samples_5.csv  test.csv   (predefined splits)
    └── video/features/{DFER,VideoMaev2_S}/spatial_pooled_features_XITE_B_last143_stride16_interpol_all/...
```

* CSVs are **tab-separated** with columns `subject_id subject_name class_id class_name sample_id sample_name`.
* **Augmented embeddings** (`<features>_<aug>` and `<features>_<aug>$N`, e.g. `_jitter`, `_jitter$0`,
  `_shift_hflip$2`) are **required** by the configurations that use `--sampler_loader_type selective_augm`
  with `--keep_original < 1`: runs 01, 02, 03, 04 and 11 below. Without them, training stops with a
  `FileNotFoundError` that explains this. Sizes: BioVid ≈ 35 GB, MIntPAIN ≈ 3 GB, UNBC ≈ 2 GB.
* `--path_video_dataset` must still be given, but the folder does **not** need to exist (only its name
  is used to identify the dataset).

## 4. Smoke test

```sh
conda activate pain-portable
bash smoke_test.sh                  # CPU, ~10 min (9-12 min measured on a NAS checkout); PASS/FAIL/SKIP per step
SMOKE_DEVICE=0 bash smoke_test.sh   # use GPU 0 instead
SMOKE_FULL=1 bash smoke_test.sh     # additionally all 12 paper training configurations + VideoMAEv2-S extraction
```

What it runs (1 epoch, first fold/subfold only, small subject subsets, tiny projector epochs):

| Step | What it checks |
| --- | --- |
| `env`, `imports` | `pip check`, native extensions (Decord, torchsort, MediaPipe, OpenCV), entry-point imports |
| `extract_DFER` (+ `extract_S` in full mode) | MAE-DFER (VideoMAEv2-S) backbone forward pass via `extract_feature.py` on a generated synthetic video |
| `train_07`, `train_03`, `train_12`, `train_06`, `train_01` | `train_model.py` with the exact paper arguments of UNBC OPI 0–4 VMAE-S, MIntPAIN DFER (augmented sampler), UNBC VAS DFER (`--gp`), XITE DFER (predefined splits), BioVid VMAE-S (augmented sampler) |
| `xspace_projection` | `run_cross_space_configs.sh`, UNBC VMAE-S → MIntPAIN DFER: linear, MLP, autoencoder (1 target × 2 source checkpoints, pooled into `aggregated_*` like the paper configs), Procrustes, closed-form linear (refinement 3) and a frozen random linear adapter (refinement 4) |
| `xspace_logs` | `cross_space_logs.py --only_aggregated` on those runs |

Datasets whose files are missing are reported as `SKIP`. UNBC and MIntPAIN (CSVs plus base and
augmented embeddings) are needed for the cross-space step. The script exits non-zero on any
`FAIL`. Logs go to `smoke_out/logs/`, training runs to `smoke_out/train/`, and cross-space runs
to `Cross_projection/smoke_test/<timestamp>/`. All of these can be deleted afterwards.

## 5. Experiments

### 5.1 Model training (`train_model.py`)

The 12 paper models (one 5-fold cross-validation run each; `--stop K S` = run K folds × S
subfolds). Add `--gp` to resolve relative paths against the repository root, or against
`$PAIN_PROJECT_ROOT` if set, instead of the current directory.

<details><summary>All 12 training commands (paper hyperparameters)</summary>

```sh
# [01] 01_BIOVID_VMAE-S
python train_model.py --head ATTENTIVE_JEPA --num_cross_head 1 --num_heads 8 --mt S --lr 0.0001 --ep 350 \
  --csv partA/starting_point/samples_adversarial.csv --load_dataset_in_memory 0 --ffsp partA/video/features/VideoMaev2_S/spatial_pooled_features_Biovid_B_last143_stride16_interpol --global_folder_name runs/01_BIOVID_VMAE-S --path_video_dataset partA/video/video_frontalized_interpolated_resolution_original --k_fold 5 \
  --stop 5 1 --opt adamw --batch_train 512 --init_network default --p_early_stop 2000 --min_delta 0.005 \
  --threshold_mode abs --regulariz_lambda_L1 0 --regulariz_lambda_L2 0.1 --scheduler_name cosine --warm_up_epochs 5 --warm_up_scheduler linear \
  --warm_up_start_factor 0.01 --model_dropout 0.1 --drop_attn 0. --drop_residual 0. --loss l1 --label_smooth 0 \
  --nr_block 2 --cross_block_after_transformers 0 --pos_enc 3 --n_trials 1 --timeout 140 --pruner_n_warmup_steps 500 \
  --sampler_loader_type selective_augm --filtered_augm_n_keep 1 --filtered_augm_strategy 1 --keep_original 0.02 --optuna_categorical 1 --pruner_threshold_lower 0.0 \
  --optuna_sampler grid --n_workers 8 --prefetch_factor 2 --validation_enabled 1 --is_subject_independent 1 --concatenate_quadrants 0 \
  --skip_test 0 --use_test_as_val 0 --embedding_reduction spatial --save_best_model --stratified_training 1 --complete_block 2 \
  --mlp_num_hidden_layers 1 --mlp_ratio 2 --custom_mlp 1

# [02] 02_BIOVID_DFER
python train_model.py --head ATTENTIVE_JEPA --num_cross_head 1 --num_heads 8 --mt DFER --lr 0.0002 --ep 605 \
  --csv partA/starting_point/samples_adversarial.csv --load_dataset_in_memory 0 --ffsp partA/video/features/DFER/spatial_pooled_features_Biovid_B_last143_stride16_interpol --global_folder_name runs/02_BIOVID_DFER --path_video_dataset partA/video/video_frontalized_interpolated_resolution_original --k_fold 5 \
  --stop 5 1 --opt adamw --batch_train 512 --init_network default --p_early_stop 2000 --min_delta 0.005 \
  --threshold_mode abs --regulariz_lambda_L1 0 --regulariz_lambda_L2 0.1 --scheduler_name cosine_restart --first_restart_epochs 10 --multiplier_restart 2 \
  --min_lr 0.0000001 --warm_up_epochs 5 --warm_up_scheduler linear --warm_up_start_factor 0.01 --model_dropout 0.1 --drop_attn 0. \
  --drop_residual 0. --mlp_ratio 2 --loss l1 --label_smooth 0 --nr_block 2 --cross_block_after_transformers 0 \
  --pos_enc 3 --n_trials 1 --timeout 140 --pruner_n_warmup_steps 500 --sampler_loader_type selective_augm --filtered_augm_n_keep 1 \
  --filtered_augm_strategy 1 --keep_original 0.02 --optuna_categorical 1 --pruner_threshold_lower 0.0 --optuna_sampler grid --n_workers 8 \
  --prefetch_factor 2 --validation_enabled 1 --is_subject_independent 1 --concatenate_quadrants 0 --skip_test 0 --use_test_as_val 0 \
  --embedding_reduction spatial --save_best_model --stratified_training 1 --complete_block 2 --mlp_num_hidden_layers 1 --mlp_ratio 2 \
  --custom_mlp 1 --save_model_every_n_epochs 100

# [03] 03_MIntPAIN_DFER
python train_model.py --head ATTENTIVE_JEPA --num_cross_head 1 --num_heads 8 --mt DFER --lr 0.0001 --ep 300 \
  --csv MIntPAIN/starting_point/samples.csv --load_dataset_in_memory 0 --ffsp MIntPAIN/features/DFER/spatial_pooled_features_MIntPAIN_B_last143_stride16_interpol --global_folder_name runs/03_MIntPAIN_DFER --path_video_dataset MIntPAIN/video_frontalized --k_fold 5 \
  --stop 5 1 --opt adamw --batch_train 128 --init_network default --p_early_stop 2000 --min_delta 0.005 \
  --threshold_mode abs --regulariz_lambda_L1 0 --regulariz_lambda_L2 0.01 --scheduler_name cosine --min_lr 0.0000001 --warm_up_epochs 5 \
  --warm_up_scheduler linear --warm_up_start_factor 0.01 --model_dropout 0.0 --drop_attn 0. --drop_residual 0. --loss l1 \
  --label_smooth 0 --nr_block 2 --cross_block_after_transformers 0 --pos_enc 3 --n_trials 1 --timeout 140 \
  --pruner_n_warmup_steps 500 --sampler_loader_type selective_augm --filtered_augm_n_keep 1 --filtered_augm_strategy 1 --keep_original 0.8 --undersample_max_per_class 600 \
  --optuna_categorical 1 --pruner_threshold_lower 0.0 --optuna_sampler grid --n_workers 8 --prefetch_factor 2 --validation_enabled 1 \
  --is_subject_independent 1 --concatenate_quadrants 0 --skip_test 0 --use_test_as_val 0 --embedding_reduction spatial --save_best_model \
  --stratified_training 1 --complete_block 2 --mlp_num_hidden_layers 1 --mlp_ratio 1 --custom_mlp 1 --target_samples_per_class_training 600

# [04] 04_MIntPAIN_VMAE-S
python train_model.py --head ATTENTIVE_JEPA --num_cross_head 1 --num_heads 8 --mt S --lr 0.0001 --ep 300 \
  --csv MIntPAIN/starting_point/samples.csv --load_dataset_in_memory 0 --ffsp MIntPAIN/features/VideoMaev2_S/spatial_pooled_features_MIntPAIN_B_last143_stride16_interpol --global_folder_name runs/04_MIntPAIN_VMAE-S --path_video_dataset MIntPAIN/video_frontalized --k_fold 5 \
  --stop 5 1 --opt adamw --batch_train 128 --init_network default --p_early_stop 2000 --min_delta 0.005 \
  --threshold_mode abs --regulariz_lambda_L1 0 --regulariz_lambda_L2 0.01 --scheduler_name cosine --min_lr 0.0000001 --warm_up_epochs 5 \
  --warm_up_scheduler linear --warm_up_start_factor 0.01 --model_dropout 0.3 --drop_attn 0. --drop_residual 0. --loss l1 \
  --label_smooth 0 --nr_block 2 --cross_block_after_transformers 0 --pos_enc 3 --n_trials 1 --timeout 140 \
  --pruner_n_warmup_steps 500 --sampler_loader_type selective_augm --filtered_augm_n_keep 1 --filtered_augm_strategy 1 --keep_original 0.8 --undersample_max_per_class 600 \
  --optuna_categorical 1 --pruner_threshold_lower 0.0 --optuna_sampler grid --n_workers 8 --prefetch_factor 2 --validation_enabled 1 \
  --is_subject_independent 1 --concatenate_quadrants 0 --skip_test 0 --use_test_as_val 0 --embedding_reduction spatial --save_best_model \
  --stratified_training 1 --complete_block 2 --mlp_num_hidden_layers 1 --mlp_ratio 1 --custom_mlp 1 --target_samples_per_class_training 600

# [05] 05_XITE_VMAE-S
python train_model.py --head ATTENTIVE_JEPA --num_cross_head 1 --num_heads 8 --mt S --lr 0.0003 --ep 200 \
  --csv XITE/starting_point/splits --ffsp XITE/video/features/VideoMaev2_S/spatial_pooled_features_XITE_B_last143_stride16_interpol_all --global_folder_name runs/05_XITE_VMAE-S --path_video_dataset XITE/video/video_frontalized --k_fold 5 --stop 5 1 \
  --opt adamw --batch_train 16 --init_network default --p_early_stop 2000 --min_delta 0.005 --threshold_mode abs \
  --regulariz_lambda_L1 0 --regulariz_lambda_L2 0.5 --scheduler_name cosine --min_lr 0.0000001 --warm_up_epochs 5 --warm_up_scheduler linear \
  --warm_up_start_factor 0.01 --model_dropout 0.5 --drop_attn 0. --drop_residual 0. --loss ce --label_smooth 0 \
  --nr_block 1 --pos_enc 3 --n_trials 1 --timeout 140 --pruner_n_warmup_steps 500 --sampler_loader_type standard \
  --optuna_categorical 1 --pruner_threshold_lower 0.0 --optuna_sampler grid --n_workers 8 --prefetch_factor 2 --validation_enabled 1 \
  --is_subject_independent 1 --concatenate_quadrants 0 --skip_test 0 --use_test_as_val 0 --embedding_reduction spatial --save_best_model \
  --complete_block 2 --load_dataset_in_memory 1

# [06] 06_XITE_DFER
python train_model.py --head ATTENTIVE_JEPA --num_cross_head 1 --num_heads 8 --mt DFER --lr 0.0003 --ep 200 \
  --csv XITE/starting_point/splits --ffsp XITE/video/features/DFER/spatial_pooled_features_XITE_B_last143_stride16_interpol_all --global_folder_name runs/06_XITE_DFER --path_video_dataset XITE/video/video_frontalized --k_fold 5 --stop 1 1 \
  --opt adamw --batch_train 16 --init_network default --p_early_stop 2000 --min_delta 0.005 --threshold_mode abs \
  --regulariz_lambda_L1 0 --regulariz_lambda_L2 0.5 --scheduler_name cosine --min_lr 0.0000001 --warm_up_epochs 5 --warm_up_scheduler linear \
  --warm_up_start_factor 0.01 --model_dropout 0.0 --drop_attn 0. --drop_residual 0. --loss ce --label_smooth 0 \
  --nr_block 1 --pos_enc 3 --n_trials 1 --timeout 140 --pruner_n_warmup_steps 500 --sampler_loader_type standard \
  --optuna_categorical 1 --pruner_threshold_lower 0.0 --optuna_sampler grid --n_workers 8 --prefetch_factor 2 --validation_enabled 1 \
  --is_subject_independent 1 --concatenate_quadrants 0 --skip_test 0 --use_test_as_val 0 --embedding_reduction spatial --save_best_model \
  --complete_block 2 --load_dataset_in_memory 1

# [07] 07_UNBC_OPI0-4_VMAE-S
python train_model.py --head ATTENTIVE_JEPA --num_cross_head 1 --num_heads 8 --mt S --lr 0.0001 --ep 150 \
  --csv UNBC/starting_point/samples_OPI_0_to_4.csv --load_dataset_in_memory 0 --ffsp UNBC/video/features/VideoMaev2_S/spatial_pooled_features_UNBC_B_last143_stride16_interpol --global_folder_name runs/07_UNBC_OPI0-4_VMAE-S --path_video_dataset UNBC/video/WarpedVideos_Cropped_interpolated_mirror --k_fold 5 \
  --stop 5 5 --opt adamw --batch_train 16 --init_network default --p_early_stop 2000 --min_delta 0.005 \
  --threshold_mode abs --regulariz_lambda_L1 0 --regulariz_lambda_L2 0.01 --scheduler_name cosine --warm_up_epochs 5 --warm_up_scheduler linear \
  --warm_up_start_factor 0.01 --model_dropout 0.3 --drop_attn 0. --drop_residual 0. --loss l1 --label_smooth 0 \
  --nr_block 2 --cross_block_after_transformers 0 --pos_enc 3 --n_trials 1 --timeout 140 --pruner_n_warmup_steps 500 \
  --sampler_loader_type standard --optuna_categorical 1 --pruner_threshold_lower 0.0 --optuna_sampler grid --n_workers 8 --prefetch_factor 2 \
  --validation_enabled 1 --is_subject_independent 1 --concatenate_quadrants 0 --skip_test 0 --use_test_as_val 0 --embedding_reduction spatial \
  --save_best_model --stratified_training 1 --complete_block 2 --mlp_num_hidden_layers 1 --mlp_ratio 0.5 --custom_mlp 1 \
  --target_samples_per_class_training 35 --normalize_labels 0

# [08] 08_UNBC_OPI0-4_DFER
python train_model.py --head ATTENTIVE_JEPA --mt DFER --num_cross_head 1 --num_heads 8 --ep 250 --k_fold 5 \
  --stop 5 5 --csv UNBC/starting_point/samples_OPI_0_to_4.csv --load_dataset_in_memory 0 --ffsp UNBC/video/features/DFER/spatial_pooled_features_UNBC_B_last143_stride16_interpol --path_video_dataset UNBC/video/WarpedVideos_Cropped_interpolated_mirror --opt adamw \
  --init_network default --p_early_stop 2000 --min_delta 0.005 --threshold_mode abs --regulariz_lambda_L1 0 --regulariz_lambda_L2 0 \
  --scheduler_name cosine --warm_up_epochs 5 --warm_up_scheduler linear --warm_up_start_factor 0.01 --loss l1 --label_smooth 0 \
  --cross_block_after_transformers 0 --pos_enc 3 --complete_block 2 --custom_mlp 1 --mlp_num_hidden_layers 1 --sampler_loader_type standard \
  --stratified_training 1 --embedding_reduction spatial --normalize_labels 0 --is_subject_independent 1 --validation_enabled 1 --concatenate_quadrants 0 \
  --skip_test 0 --use_test_as_val 0 --save_best_model --n_workers 8 --prefetch_factor 2 --optuna_categorical 1 \
  --optuna_sampler grid --pruner_n_warmup_steps 500 --pruner_threshold_lower 0.0 --n_trials 1 --timeout 140 --lr 0.0001 \
  --batch_train 16 --nr_blocks 3 --mlp_ratio 0.5 --model_dropout 0.1 --drop_attn 0. --drop_residual 0. \
  --global_folder_name runs/08_UNBC_OPI0-4_DFER --target_samples_per_class_training 60

# [09] 09_UNBC_OPI0-5_DFER
python train_model.py --head ATTENTIVE_JEPA --mt DFER --num_cross_head 1 --num_heads 8 --ep 250 --k_fold 5 \
  --stop 5 5 --csv UNBC/starting_point/samples_OPI.csv --load_dataset_in_memory 0 --ffsp UNBC/video/features/DFER/spatial_pooled_features_UNBC_B_last143_stride16_interpol --path_video_dataset UNBC/video/WarpedVideos_Cropped_interpolated_mirror --opt adamw \
  --init_network default --p_early_stop 2000 --min_delta 0.005 --threshold_mode abs --regulariz_lambda_L1 0 --regulariz_lambda_L2 0 \
  --scheduler_name cosine --warm_up_epochs 5 --warm_up_scheduler linear --warm_up_start_factor 0.01 --loss l1 --label_smooth 0 \
  --cross_block_after_transformers 0 --pos_enc 3 --complete_block 2 --custom_mlp 1 --mlp_num_hidden_layers 1 --sampler_loader_type standard \
  --stratified_training 1 --embedding_reduction spatial --normalize_labels 0 --is_subject_independent 1 --validation_enabled 1 --concatenate_quadrants 0 \
  --skip_test 0 --use_test_as_val 0 --save_best_model --n_workers 8 --prefetch_factor 2 --optuna_categorical 1 \
  --optuna_sampler grid --pruner_n_warmup_steps 500 --pruner_threshold_lower 0.0 --n_trials 1 --timeout 140 --lr 0.0001 \
  --batch_train 16 --nr_blocks 3 --mlp_ratio 0.5 --model_dropout 0.3 --drop_attn 0. --drop_residual 0. \
  --global_folder_name runs/09_UNBC_OPI0-5_DFER --target_samples_per_class_training 35

# [10] 10_UNBC_OPI0-5_VMAE-S
python train_model.py --head ATTENTIVE_JEPA --num_cross_head 1 --num_heads 8 --mt S --lr 0.0001 --ep 150 \
  --csv UNBC/starting_point/samples_OPI.csv --load_dataset_in_memory 0 --ffsp UNBC/video/features/VideoMaev2_S/spatial_pooled_features_UNBC_B_last143_stride16_interpol --global_folder_name runs/10_UNBC_OPI0-5_VMAE-S --path_video_dataset UNBC/video/WarpedVideos_Cropped_interpolated_mirror --k_fold 5 \
  --stop 5 5 --opt adamw --batch_train 16 --init_network default --p_early_stop 2000 --min_delta 0.005 \
  --threshold_mode abs --regulariz_lambda_L1 0 --regulariz_lambda_L2 0.01 --scheduler_name cosine --warm_up_epochs 5 --warm_up_scheduler linear \
  --warm_up_start_factor 0.01 --model_dropout 0.1 --drop_attn 0. --drop_residual 0. --loss l1 --label_smooth 0 \
  --nr_block 2 --cross_block_after_transformers 0 --pos_enc 3 --n_trials 1 --timeout 140 --pruner_n_warmup_steps 500 \
  --sampler_loader_type standard --optuna_categorical 1 --pruner_threshold_lower 0.0 --optuna_sampler grid --n_workers 8 --prefetch_factor 2 \
  --validation_enabled 1 --is_subject_independent 1 --concatenate_quadrants 0 --skip_test 0 --use_test_as_val 0 --embedding_reduction spatial \
  --save_best_model --stratified_training 1 --complete_block 2 --mlp_num_hidden_layers 1 --mlp_ratio 0.5 --custom_mlp 1 \
  --target_samples_per_class_training 60 --normalize_labels 0

# [11] 11_UNBC_VAS_VMAE-S
python train_model.py --head ATTENTIVE_JEPA --num_cross_head 1 --num_heads 8 --mt S --lr 0.0005 --ep 150 \
  --csv UNBC/starting_point/samples_adversarial.csv --load_dataset_in_memory 0 --ffsp UNBC/video/features/VideoMaev2_S/spatial_pooled_features_UNBC_B_last143_stride16_interpol --global_folder_name runs/11_UNBC_VAS_VMAE-S --path_video_dataset UNBC/video/WarpedVideos_Cropped_interpolated_mirror --k_fold 5 \
  --stop 5 5 --opt adamw --batch_train 16 --init_network default --p_early_stop 2000 --min_delta 0.005 \
  --threshold_mode abs --regulariz_lambda_L1 0 --regulariz_lambda_L2 0.01 --scheduler_name cosine --warm_up_epochs 5 --warm_up_scheduler linear \
  --warm_up_start_factor 0.01 --model_dropout 0.3 --drop_attn 0. --drop_residual 0. --loss l1 --label_smooth 0 \
  --nr_block 2 --cross_block_after_transformers 0 --pos_enc 3 --n_trials 1 --timeout 140 --pruner_n_warmup_steps 500 \
  --sampler_loader_type selective_augm --filtered_augm_n_keep 1 --filtered_augm_strategy 1 --keep_original 0.2 --optuna_categorical 1 --pruner_threshold_lower 0.0 \
  --optuna_sampler grid --n_workers 8 --prefetch_factor 2 --validation_enabled 1 --is_subject_independent 1 --concatenate_quadrants 0 \
  --skip_test 0 --use_test_as_val 0 --embedding_reduction spatial --save_best_model --stratified_training 1 --complete_block 2 \
  --mlp_num_hidden_layers 1 --mlp_ratio 0.5 --custom_mlp 1 --target_samples_per_class_training 35 --normalize_labels 0

# [12] 12_UNBC_VAS_DFER
python train_model.py --head ATTENTIVE_JEPA --num_cross_head 1 --num_heads 8 --mt DFER --gp --lr 0.00001 \
  --ep 200 --csv UNBC/starting_point/samples.csv --load_dataset_in_memory 0 --ffsp UNBC/video/features/DFER/spatial_pooled_features_UNBC_B_last143_stride16_interpol --global_folder_name runs/12_UNBC_VAS_DFER --path_video_dataset UNBC/video/WarpedVideos_Cropped_interpolated_mirror \
  --k_fold 5 --stop 5 5 --opt adamw --batch_train 9 --init_network default --key_early_stopping val_loss \
  --p_early_stop 2000 --min_delta 0.005 --threshold_mode abs --regulariz_lambda_L1 0 --regulariz_lambda_L2 0.01 --scheduler_name cosine_restart \
  --first_restart_epochs 7 --multiplier_restart 2 --warm_up_epochs 5 --warm_up_scheduler linear --warm_up_start_factor 0.01 --model_dropout 0.1 \
  --drop_attn 0. --drop_residual 0. --mlp_ratio 0.5 --label_smooth 0 --nr_block 1 --cross_block_after_transformers 0 \
  --pos_enc 3 --n_trials 1 --timeout 102 --loss l1 --pruner_n_warmup_steps 500 --optuna_categorical 0 \
  --pruner_threshold_lower 0.0 --optuna_sampler grid --n_workers 8 --prefetch_factor 2 --validation_enabled 1 --is_subject_independent 1 \
  --stratified_training 0 --target_samples_per_class_training 31 --sampler_loader_type standard --concatenate_quadrants 0 --skip_test 0 --use_test_as_val 1 \
  --embedding_reduction spatial
```

</details>

**Outputs.** Each run creates
`runs/<NN_NAME>_<pid>_ATTENTIVE_JEPA_<host>_<timestamp>/` containing:

* `_config_prompt.txt`: the launch command and all resolved arguments
* `run.log`
* `ATTENTIVE_JEPA_<ts>.db/.pkl`: the Optuna study
* `<run_id>_<BACKBONE>_..._ATTENTIVE_JEPA/k_fold_results.pkl`: per-fold metrics, config and
  model parameters
* the same folder's `train_ATTENTIVE_JEPA/k<i>_cross_val/{train,val,test}.csv` and
  `k<i>_cross_val_sub_<j>/best_model_ep_<epoch>.pt`: the splits and the selected checkpoints

### 5.2 Cross-space projection (`cross_space_projection.py`)

Configs live in `Cross_projection_yaml/<config set>/<direction>/refinement<R>_<method>.yaml`
(method ∈ linear, mlp, autoencoder, procrustes, linear_close). Each lists the 5×5 source/target
fold checkpoints from training:

| Direction folder | Source (old) model | Target (new) model |
| --- | --- | --- |
| `unbcVmae_to_biovidDfer_cross_validation` | 07 UNBC OPI 0–4 VMAE-S | 02 BioVid DFER |
| `biovidVmae_to_unbcDfer_cross_validation` | 01 BioVid VMAE-S | 08 UNBC OPI 0–4 DFER |
| `mintVmae_to_biovidDfer_cross_validation` | 04 MIntPAIN VMAE-S | 02 BioVid DFER |
| `biovidVmae_to_mintDfer_cross_validation` | 01 BioVid VMAE-S | 03 MIntPAIN DFER |
| `unbcVmae_to_mintDfer_cross_validation` | 07 UNBC OPI 0–4 VMAE-S | 03 MIntPAIN DFER |
| `mintVmae_to_unbcDfer_cross_validation` | 04 MIntPAIN VMAE-S | 08 UNBC OPI 0–4 DFER |

**Before running**, replace `new_model_pth` / `old_model_pth` in the YAMLs with *your* checkpoints
from §5.1, in fold order k0…k4. Run folders and best epochs differ between trainings. To list them:

```sh
ls runs/07_UNBC_OPI0-4_VMAE-S_*/*/train_ATTENTIVE_JEPA/k?_cross_val/k?_cross_val_sub_*/best_model_ep_*.pt
```

Then, for each direction (GPU id or `cpu`):

```sh
bash run_cross_space_configs.sh 0 Cross_projection_yaml/config_paper_tests_seed_42/unbcVmae_to_mintDfer_cross_validation          # random anchors
bash run_cross_space_configs.sh 0 Cross_projection_yaml/config_paper_tests_seed_42_quality/unbcVmae_to_mintDfer_cross_validation  # quality anchors
bash run_cross_space_configs.sh 0 Cross_projection_yaml/config_ablation_frozen_random_adapter/unbcVmae_to_mintDfer_cross_validation  # frozen random adapter
# after all directions of a set finished:
python cross_space_logs.py --pkl_path Cross_projection/paper_tests_seed_42 --only_aggregated --skip_umap
```

A single config can also be run directly with `python cross_space_projection.py --config <file.yaml>`.

**Outputs.** `Cross_projection/<run_tag>/search_<method>_..._<uid>/` (the `run_tag` comes from the
YAML) holds:

* the launch-config snapshot
* `precomputed/`: anchors and extracted embeddings
* one folder per subtrial
* the pooled aggregated `.pkl`

`cross_space_logs.py --only_aggregated` writes `aggregated_summary.csv` and per-run plots in the
config-set root, e.g. `Cross_projection/paper_tests_seed_42/aggregated_summary.csv`.

### 5.3 Feature extraction from your own videos (optional)

```sh
python extract_feature.py --model_type DFER --emb_red spatial --path_dataset <video_root> \
  --path_labels <labels.csv> --saving_folder_path <out> --backbone_type video --save_as_safetensors \
  --stride_window 16 --stride_inside_window 1 --float_16 --save_big_feature      # --model_type S for VideoMAEv2-S
```

Videos are read from `<video_root>/<subject_name>/<sample_name>.mp4`, and embeddings are written to
`<out>/<subject_name>/<sample_name>.safetensors`. The `<video_root>` path must contain a known dataset keyword (`unbc`, `parta`/`biovid`, `mintpain`,
`xite`, …). The paper's face frontalization/preprocessing pipeline needs the raw videos and was
not part of this portability check.

## 6. Known limitations

* **Backbone weights are mandatory** even for training on precomputed embeddings, and are
  loaded from paths relative to the working directory: run from the repository root.
* **CPU** runs work, but real experiments are slow without a GPU. CPU and GPU (and different
  attention kernels) give slightly different numbers, so exact paper values need the GPU setup.
* The cross-space YAMLs refer to the authors' original run folders. You must point them to your
  own checkpoints (§5.2).
* Only Linux x86_64 was tested end-to-end. Windows/macOS environment files exist in
  `env_portability/`, but runtime was not verified there.
* Raw-video preprocessing (face frontalization, landmark extraction, augmentation generation) and
  the paper table/plot scripts were not covered by the smoke test.
* See [PORTABILITY_REPORT.md](PORTABILITY_REPORT.md) for every issue found and fixed.
