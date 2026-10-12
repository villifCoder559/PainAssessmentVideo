# Cross-Space Latent Representation Transfer in Pain Score Regression

We study whether frozen video foundation models capture the subtle facial dynamics needed to
estimate pain intensity, and whether their video-level latent representations can be transferred
between independently trained models while keeping the performance of the original source model.
**VideoMAE V2-S** (384-d) and **MAE-DFER** (512-d) are evaluated on **BioVid Part A**,
**UNBC-McMaster**, **MIntPAIN** and **X-ITE**, with a trainable temporal-aggregation and prediction
head on top of each frozen backbone. For cross-space transfer on the three regression datasets,
target-domain anchors (target videos encoded by both models) define correspondences that map
source representations directly into the native target space. The projectors are Linear, MLP and
Encoder–Decoder, plus the closed-form Procrustes and OLS mappings, and they are evaluated in every
transfer direction. Joint refinement with stored source labels largely recovers native-source
regression performance, whereas anchor-only transfer gives a higher source MAE.

> **What you need.** This code, the released label CSVs and the precomputed embeddings
> (including the augmented-embedding folders, §3), plus the two public backbone checkpoints (§2).
> Raw videos are **not** needed and are not distributed.

Run every command from the **repository root**.

## 1. Environment (NVIDIA GPU)

This setup was tested on Ubuntu 22.04 x86_64 (RTX 2080 Ti, drivers 535 and 550) with Python 3.10.21, PyTorch 2.5.1 and
the CUDA 11.8 runtime wheels. Check the machine first:

```sh
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
```

* The **driver must be ≥ 520**. No system CUDA toolkit is needed, because the CUDA runtime and
  cuDNN come with the PyTorch wheels.
* The **compute capability must be ≤ 9.0** (PyTorch 2.5.1+cu118 ships sm_37–sm_90 kernels).
  Newer GPUs (e.g. Blackwell) need a different, untested PyTorch/CUDA stack.

```sh
conda env create -f env_portability/environment-cuda-pinned-linux-64.yml   # creates "pain-portable-cuda"
conda activate pain-portable-cuda
python -m pip check
python -c "import torch, torchsort; print(torch.cuda.is_available(), torchsort.soft_rank(torch.tensor([[3.,1.,2.]], device='cuda')))"
```

The last command must print `True` and a tensor on `device='cuda:0'`. Both `python` and
`python3` must resolve to this env (`command -v python python3`), because
`run_cross_space_configs.sh` calls `python3`. The env downloads several GB of CUDA wheels. No GPU?
See [§11 CPU fallback](#11-cpu-fallback).

## 2. Backbone weights

`train_model.py` and `cross_space_projection.py` always build the backbone and load its weights,
even when they only read precomputed embeddings. Download both checkpoints to these exact paths:

| Backbone | File (relative to repo root) | Source |
| --- | --- | --- |
| MAE-DFER (1.1 GB) | `MAE_DFER/saved/model/pretraining/voxceleb2/videomae_pretrain_base_dim512_local_global_attn_depth16_region_size2510_patch16_160_frame_16x4_tube_mask_ratio_0.9_e100_with_diff_target_server170/checkpoint-49.pth` | [Google Drive](https://drive.google.com/file/d/1nzvMITUHic9fKwjQ7XLcnaXYViWTawRv/view?usp=sharing) ([MAE-DFER README](MAE_DFER/README.md)) |
| VideoMAEv2-S (44 MB) | `VideoMAEv2/pretrained/vit_s_k710_dl_from_giant.pth` | [HuggingFace](https://huggingface.co/OpenGVLab/VideoMAE2/resolve/main/distill/vit_s_k710_dl_from_giant.pth) ([VideoMAEv2 model zoo](VideoMAEv2/docs/MODEL_ZOO.md)) |

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
paths (`unbc`, `parta`/`biovid`, `mintpain`, `xite`), and the cross-space code looks embeddings up
by these relative paths.

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
  `_shift_hflip$2`) are **required** by the configs marked † in §5. Without them, training stops
  with a `FileNotFoundError` that explains this. Sizes: BioVid ≈ 35 GB, MIntPAIN ≈ 3 GB, UNBC ≈ 2 GB.
* `path_video_dataset` in the configs does **not** need to exist: only its name is used to
  identify the dataset.

## 4. Smoke test

```sh
SMOKE_DEVICE=0 bash smoke_test.sh                # GPU 0, ~10 min; PASS/FAIL/SKIP per step
SMOKE_DEVICE=0 SMOKE_FULL=1 bash smoke_test.sh   # + all 12 training configs and VideoMAEv2-S extraction, ~14 min
```

It runs feature extraction, five training configs and the cross-space pipeline (all projector
methods, UNBC VMAE-S → MIntPAIN DFER) for 1 epoch on small subject subsets. Missing datasets are
reported as `SKIP`, and the script exits non-zero on any `FAIL`. If CUDA is not usable, device
selection silently falls back to CPU and the steps can still PASS, so check that the GPU was used:

```sh
grep -L 'extracting features using.... cuda' smoke_out/logs/extract_*.log   # must print nothing
grep -L 'GPU memory peak' smoke_out/logs/train_*.log                        # must print nothing
grep -iE 'CUDA error|device-side assert|CUBLAS_STATUS|CUDNN_STATUS|out of memory|same device' smoke_out/logs/*.log
```

## 5. Model training

Each paper model is one subject-independent 5-fold cross-validation run, defined by a YAML file
in `configs/train/`.

| Dataset (labels) | Backbone | Command |
| --- | --- | --- |
| BioVid Part A † | VideoMAEv2-S | `python train_model.py --config configs/train/biovid_vmae_s.yaml` |
| BioVid Part A † | MAE-DFER | `python train_model.py --config configs/train/biovid_dfer.yaml` |
| MIntPAIN † | VideoMAEv2-S | `python train_model.py --config configs/train/mintpain_vmae_s.yaml` |
| MIntPAIN † | MAE-DFER | `python train_model.py --config configs/train/mintpain_dfer.yaml` |
| UNBC (OPI 0–4) | VideoMAEv2-S | `python train_model.py --config configs/train/unbc_opi0-4_vmae_s.yaml` |
| UNBC (OPI 0–4) | MAE-DFER | `python train_model.py --config configs/train/unbc_opi0-4_dfer.yaml` |
| UNBC (OPI 0–5) | VideoMAEv2-S | `python train_model.py --config configs/train/unbc_opi0-5_vmae_s.yaml` |
| UNBC (OPI 0–5) | MAE-DFER | `python train_model.py --config configs/train/unbc_opi0-5_dfer.yaml` |
| UNBC (VAS) † | VideoMAEv2-S | `python train_model.py --config configs/train/unbc_vas_vmae_s.yaml` |
| UNBC (VAS) | MAE-DFER | `python train_model.py --config configs/train/unbc_vas_dfer.yaml` |
| X-ITE (predefined splits) | VideoMAEv2-S | `python train_model.py --config configs/train/xite_vmae_s.yaml` |
| X-ITE (predefined splits) | MAE-DFER | `python train_model.py --config configs/train/xite_dfer.yaml` |

† Needs the augmented embeddings (§3).

* Quick check: append `--stop 1 1 --ep 1`. Flags on the command line override the YAML.
* `stop: [K, S]` runs K folds × S subfolds. The subfold with the best validation score becomes
  the model of that fold.
* Runs are written to `runs/<config>/history_run_*` (see §9). The cross-space step (§6) needs
  the six BioVid, MIntPAIN and UNBC OPI 0–4 runs.

## 6. Cross-space projection

A cross-space YAML runs one projector method (`linear`, `mlp`, `autoencoder`, `procrustes`,
`linear_close` = OLS) for one source (VideoMAEv2-S) → target (MAE-DFER) direction. It pairs the 5
source with the 5 target fold checkpoints and pools all pairs into one aggregated result.

| Config set (`Cross_projection_yaml/`) | Reproduces | Output root (`Cross_projection/`) |
| --- | --- | --- |
| `config_paper_tests_seed_42` | main results, random class-balanced anchors (`refinement: 3`) | `paper_tests_seed_42` |
| `config_paper_tests_seed_42_quality` | quality-selected anchors | `paper_tests_seed_42_quality` |
| `config_ablation_frozen_random_adapter` | frozen random projector + linear-head refinement (`refinement: 4`) | `ablation_paper_frozen_rnd_adapter` |
| `config_anchor_sweep` | anchor-budget figure (§7) | `anchor_sweep_rerun_2026_09_23` |

**1. Point the YAMLs at your checkpoints.** The YAMLs refer to the authors' run folders.
`fill_cross_space_checkpoints.py` finds, for each direction, the newest complete 5-fold run of
the source and target config under `runs/` and rewrites `old_model_pth` (source) and
`new_model_pth` (target) in place, in all four config sets. Directions whose runs are missing are
reported and left unchanged.

```sh
python fill_cross_space_checkpoints.py --dry-run   # report what would be filled
python fill_cross_space_checkpoints.py             # write; use --run CONFIG=runs/<config>/history_run_<...> to pick a run
```

**2. Run all directions of the three paper sets** (first argument: GPU id, or `cpu`):

```sh
for set in config_paper_tests_seed_42 config_paper_tests_seed_42_quality config_ablation_frozen_random_adapter; do
  for d in Cross_projection_yaml/$set/*_cross_validation; do
    bash run_cross_space_configs.sh 0 "$d"
  done
done
```

A single YAML can also be run with `python cross_space_projection.py --config <file.yaml>`.

**3. Summarize each output root** (writes `aggregated_summary.csv`, used by §7):

```sh
for r in paper_tests_seed_42 paper_tests_seed_42_quality ablation_paper_frozen_rnd_adapter; do
  python cross_space_logs.py --pkl_path Cross_projection/$r --only_aggregated --skip_umap
done
```

## 7. Paper tables, controls, statistics and figures

Each block reproduces one table or figure, named by its paper title and LaTeX label. The table
scripts print LaTeX to stdout. Add `--output <file.tex>` to write it to a file instead. The
blocks assume that §5 and §6 are done and use:

```sh
R=Cross_projection/paper_tests_seed_42
```

#### Frozen-representation baselines (`tab:literature_comparison_all_datasets`, `tab:baseline_summary_all_datasets`)

```sh
for pkl in runs/*/history_run_*/*/k_fold_results.pkl; do
  python extract_kfold_test_table.py --pkl "$pkl" --raw          # add --metric accuracy for MIntPAIN and X-ITE
done
```

Constant-predictor rows need no script: for a constant *c*, MAE is `mean(|y − c|)` over all
videos of the label CSV, and Macro-MAE averages it over the represented levels (BioVid, c=2:
1.20 / 1.20; MIntPAIN, c=0: 1.25 / 2.00).

#### Projector-only vs joint refinement, incl. Procrustes and OLS (`tab:comparison_proj_vs_projANDregressor_stage_mae_macro`)

```sh
python cross_space_paper_tables.py refinement $R
python cross_space_paper_tables.py closed-form $R    # optional: closed-form mappings only
```

#### Video-level statistical comparison (`tab:video-mae-performance`, `tab:video-mae-significance`)

Run once per dataset, with the MAE-DFER and VideoMAEv2-S runs of that dataset (each glob must
match exactly one run):

```sh
python statistical_compare_kfold_results.py \
  --pkl_path_0 runs/biovid_dfer/history_run_*/*/k_fold_results.pkl \
  --pkl_path_1 runs/biovid_vmae_s/history_run_*/*/k_fold_results.pkl \
  --analysis_level video --measure mae
```

#### Synthetic-embedding control (`tab:real_fake_unbc_biovid`, `tab:real_fake_mintpain_biovid`, `tab:real_fake_mintpain_unbc`)

```sh
for d in $R/*_cross_validation; do
  python cross_space_fake_projection.py $d --control fake_embeddings --distribution standard_normal
done
python cross_space_paper_tables.py synthetic $R/*_cross_validation
```

#### Random vs quality-based anchor selection (`tab:quality_anchor_micro_mae`, `tab:quality_anchor_macro_mae`)

```sh
python cross_space_paper_tables.py anchor $R Cross_projection/paper_tests_seed_42_quality --metric micro
python cross_space_paper_tables.py anchor $R Cross_projection/paper_tests_seed_42_quality --metric macro
```

#### Anchor budget (`fig:anchor_budget_sweep`)

`run_cross_space_anchor_sweep.sh` runs each YAML once per value in its `num_anchors` list
(output: `Cross_projection/<run_tag>/K<k>/`). The GPU id must be an integer. The paper figure
shows the projector-only stage.

```sh
for d in Cross_projection_yaml/config_anchor_sweep/*/; do bash "$d/run_all_anchor_sweeps.sh" 0; done
python plot_anchor_sweep_mae.py --input-root Cross_projection/anchor_sweep_rerun_2026_09_23 \
  --output-dir analysis/results/anchor_sweep_mae
```

#### Temporal predictions on BioVid video 7678 (`fig:cumulative_mean`, `fig:cumulative_per_target`)

`plot_cumulative_predictions.py` reruns a cross-space experiment on one video, cutting it at
increasing lengths. `plot_cumulative_snapshot_figure.py` puts the face snapshots under the plot.

```sh
python plot_cumulative_predictions.py <cross-space run>/refinement3_mlp_100 7678 --origin source \
  --refinement-stage 2 --compare-native-root <BioVid DFER run> --video 1 --snapshots 47 63 95 143 --dump-curves
D=chunk_prediction_plots/videomae_biovid_to_unbc_dfer/7678/mlp/videomae_biovid/stage_2
python plot_cumulative_snapshot_figure.py $D/summary_full_range_<digest>.png \
  $D/summary_full_range_video_<digest>_snapshots cumulative_biovid7678_mean_curves.png
python plot_cumulative_snapshot_figure.py $D/models_stage_2_full_range_<digest>.png \
  $D/models_stage_2_full_range_video_<digest>_snapshots cumulative_biovid7678_task_aware_per_target.png
```

`--dump-curves` also writes the plotted values to `curves_<digest>.json`.
`run_cumulative_predictions.sh` runs `plot_cumulative_predictions.py` over many videos.

#### Learned vs random cross-space mapping (`tab:random_adapter_ablation_joint_mae_macro_mae`, `fig:mintpain_biovid_random_confusion`)

```sh
for d in $R/*_cross_validation; do
  python cross_space_fake_projection.py $d --control fake_adapter      # random projectors, seeds 42-46
done
python cross_space_paper_tables.py random-adapter $R/*_cross_validation
python cross_space_paper_tables.py confusion $R/mintVmae_to_biovidDfer_cross_validation --method linear --seed 42
```

#### Learned vs frozen random projection with linear-head refinement (`tab:projector_linear_vs_random_source`, `tab:projector_linear_vs_random_target`)

```sh
python cross_space_paper_tables.py frozen $R Cross_projection/ablation_paper_frozen_rnd_adapter
```

## 8. Optional: feature extraction from your own videos

```sh
python extract_feature.py --model_type DFER --emb_red spatial --path_dataset <video_root> \
  --path_labels <labels.csv> --saving_folder_path <out> --backbone_type video --save_as_safetensors \
  --stride_window 16 --stride_inside_window 1 --float_16 --save_big_feature      # --model_type S for VideoMAEv2-S
```

Videos are read from `<video_root>/<subject_name>/<sample_name>.mp4`, and embeddings are written to
`<out>/<subject_name>/<sample_name>.safetensors`. `<video_root>` must contain a dataset keyword
(`unbc`, `parta`/`biovid`, `mintpain`, `xite`, …). With the raw videos, the released embeddings
were built in this order:

1. **Dataset metadata.** `MIntPAIN/generate_videos.py` assembles the MIntPAIN frame sequences
   into videos (default `--fps 25`) and writes `MIntPAIN/starting_point/samples.csv`.
   `XITE/starting_point/prepare_xite.py` writes the X-ITE CSVs. The 21/5-subject split files in
   `XITE/starting_point/splits/` are released with the CSVs.
2. **Reference geometry.** `extract_landmarks.py` averages MediaPipe landmarks over all BioVid
   subjects into `partA/video/mean_face_landmarks_per_subject/all_subjects_mean_landmarks.pkl`.
3. **Frontalization.** `python extract_video_frontalized.py --gv --vfp <raw_videos> --csv <labels.csv> --pfo <out> --prl <reference.pkl>`
   (algorithm in `custom/faceExtractor.py`, see `--help`; for example,
   `--interpolation_mod_chunk mirror_start_video 16` pads each video to a multiple of 16 frames).
4. **Embeddings.** Run `extract_feature.py` (above) for the base folders, and
   `bash multiple_feature_extraction.sh <gpu> <n_runs> --dataset <name> --model {DFER,S} <aug…>`
   for the augmented folders `<features>_<aug>[$N]`. `XITE/check_frontalized_videos.py` checks the
   X-ITE videos first, and `python check_augmentation_completeness.py --original_folder <features>`
   checks that every augmented folder is complete.

## 9. Outputs

**Training** (`runs/<config>/history_run_<pid>_ATTENTIVE_JEPA_<host>_<timestamp>/`):

* `_config_prompt.txt` (launch command and resolved arguments), `run.log`, and
  `ATTENTIVE_JEPA_<ts>.db/.pkl` (the Optuna study)
* `<run_id>_<BACKBONE>_..._ATTENTIVE_JEPA/k_fold_results.pkl`: per-fold metrics, config and model
  parameters, including the selected subfold and epoch of each fold
* the same folder's `train_ATTENTIVE_JEPA/k<i>_cross_val/{train,val,test}.csv` and
  `k<i>_cross_val_sub_<j>/best_model_ep_<epoch>.pt`: the splits and the best checkpoint of every
  subfold

**Cross-space** (`Cross_projection/<run_tag>/search_<method>_..._<uid>/`): the launch-config
snapshot, `precomputed/` (anchors and extracted embeddings), one folder per subtrial and the
pooled aggregated `.pkl`. `cross_space_logs.py --only_aggregated` writes `aggregated_summary.csv`
and per-run plots in the output root.

**Smoke test**: logs in `smoke_out/logs/`, training runs in `smoke_out/train/`, cross-space runs
in `Cross_projection/smoke_test/<timestamp>/`. All of these can be deleted.

## 10. Known limitations

* **Backbone weights are mandatory** even for training on precomputed embeddings, and are loaded
  from paths relative to the working directory: run from the repository root.
* Only Linux x86_64 was tested end-to-end.
* CPU and GPU (and different attention kernels) give slightly different numbers. Exact paper
  values need the GPU setup.
* Raw-video preprocessing (§8) and the paper table/plot scripts (§7) are not covered by the smoke
  test.
* The V-JEPA 2.1 backbone option in `custom/backbone.py` (`--mt vjepa2_1_B`) is not used in the
  paper, and its vendored code is not shipped. Using it requires a checkout of
  [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) in `vjepa2/`.
* See [PORTABILITY_REPORT.md](PORTABILITY_REPORT.md) for every issue found and fixed.

## 11. CPU fallback

**Pinned CPU environment (Linux x86_64).** This is the environment exported from the clean-room
install that passed `smoke_test.sh`. Training on CPU is slow, but it is enough for the smoke test
and small runs.

```sh
conda env create -f env_portability/environment-pinned-linux-64.yml   # creates "pain-portable"
conda activate pain-portable
python -m pip check
bash smoke_test.sh                                                    # CPU, ~10 min
```

Pass `cpu` instead of a GPU id to `run_cross_space_configs.sh`. The anchor-sweep scripts need a
GPU id.

**Flexible (non-pinned) specifications.** Use these only if a pinned file does not resolve on
your machine, or on Windows/macOS: `env_portability/environment.yml` plus
`environment-native.yml` (CPU), or `environment-cuda.yml` plus the Decord/torchsort commands in
[env_portability/ENVIRONMENT.md](env_portability/ENVIRONMENT.md#nvidia-gpu-installation) (GPU).
These may resolve to versions other than the tested ones. Windows and macOS instructions are
documented there but were **not** tested end-to-end. All environment files are in
[env_portability/](env_portability/README.md).
