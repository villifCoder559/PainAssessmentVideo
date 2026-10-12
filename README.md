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

## Repository layout

| Role | Files |
| --- | --- |
| Model training (§5.1) | `train_model.py`, paper configs in `configs/train/`, `custom/` (backbones, dataset, attentive head, losses, training loop, splits) |
| Cross-space transfer (§5.2) | `run_cross_space_configs.sh` → `cross_space_projection.py`; `cross_space_logs.py` (summaries); `cross_space_fake_projection.py` (synthetic-embedding and random-mapping controls); paper configs in `Cross_projection_yaml/` |
| Paper tables and statistics (§5.4) | `cross_space_paper_tables.py` (uses `cross_space_generate_latex_table.py`, `cross_space_fake_projection_tables.py`), `extract_kfold_test_table.py`, `statistical_compare_kfold_results.py` |
| Preprocessing from raw videos (§5.3) | `extract_feature.py`, `multiple_feature_extraction.sh`, `extract_video_frontalized.py` + `custom/faceExtractor.py` + `landmark_model/`, `extract_landmarks.py`, `MIntPAIN/generate_videos.py`, `XITE/starting_point/prepare_xite.py`, `XITE/check_frontalized_videos.py`, `check_augmentation_completeness.py` |
| Paper appendix figures (§5.5) | `run_cross_space_anchor_sweep.sh` + `plot_anchor_sweep_mae.py` (anchor-budget sweep, configs in `Cross_projection_yaml/config_anchor_sweep/`), `plot_cumulative_predictions.py` + `run_cumulative_predictions.sh` + `plot_cumulative_snapshot_figure.py` (predictions over growing video prefixes, BioVid sample 7678) |
| Helper modules | `log_cross_attention_from_model.py`, `new_plot_tsne_post_head.py` (imported by the cross-space code) |
| Vendored backbone/head code | `MAE_DFER/`, `VideoMAEv2/`, `jepa/`, `making_better_mistakes/`: only the modules the pipeline imports (upstream licenses kept) |
| Tests | `smoke_test.sh` (end-to-end, §4), `tests/` (`python -m pytest tests -q`) |

---

## 1. Environment

Tested (2026-10-06) on Ubuntu 22.04 x86_64 with a fresh Miniforge (conda 26.7). The resulting
env had **Python 3.10.21** and **PyTorch 2.5.1**. It ran on CPU only (no GPU visible), and with
CUDA 11.8 wheels on an NVIDIA RTX 2080 Ti (driver 550). On 2026-10-08 the GPU env was rebuilt
from a fresh clone (RTX 2080 Ti, driver 535) and passed the fast and full smoke tests on GPU.
Run all commands from the **repository root**. All environment files are in
[env_portability/](env_portability/README.md).

**Recommended: exact pinned environment (Linux x86_64, CPU).**

```sh
conda env create -f env_portability/environment-pinned-linux-64.yml   # creates "pain-portable"
conda activate pain-portable
python -m pip check
```

This is the environment exported from the clean-room install that passed `smoke_test.sh`. It
contains CPU PyTorch. Training runs on CPU, slowly, but enough for the smoke test and small runs.

**Recommended for NVIDIA GPUs: exact pinned environment (Linux x86_64, CUDA 11.8 runtime wheels).**

Check the machine first with `nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv`:

* The NVIDIA **driver must be ≥ 520** (CUDA 11.8). No system CUDA toolkit is needed, because the
  CUDA runtime and cuDNN come with the PyTorch wheels.
* The GPU **compute capability must be ≤ 9.0**. PyTorch 2.5.1+cu118 ships kernels for sm_37 to
  sm_90 only. Newer GPUs (e.g. Blackwell) need a different, untested PyTorch/CUDA stack.

```sh
conda env create -f env_portability/environment-cuda-pinned-linux-64.yml   # creates "pain-portable-cuda"
conda activate pain-portable-cuda
python -m pip check
python -c "import torch, torchsort; print(torch.cuda.is_available(), torchsort.soft_rank(torch.tensor([[3.,1.,2.]], device='cuda')))"
```

The last command must print `True` and a tensor on `device='cuda:0'`. Make sure that both
`python` and `python3` resolve to this env (`command -v python python3`), because
`run_cross_space_configs.sh` calls `python3`. Creating the env downloads several GB of CUDA
wheels. It took about 45 min when the env was on network storage.

**Fallback: flexible (non-pinned) specifications.** Use these only if the pinned file does not
resolve on your machine, or on Windows/macOS. `env_portability/environment.yml` plus
`environment-native.yml` (CPU), or `environment-cuda.yml` plus the Decord/torchsort commands in
[env_portability/ENVIRONMENT.md](env_portability/ENVIRONMENT.md#nvidia-gpu-installation) (GPU),
resolve current compatible versions. These may differ from the tested ones. Windows and macOS
instructions are documented there but were **not** tested end-to-end.

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
  with `--keep_original < 1`: the configs marked † in §5.1. Without them, training stops with a
  `FileNotFoundError` that explains this. Sizes: BioVid ≈ 35 GB, MIntPAIN ≈ 3 GB, UNBC ≈ 2 GB.
* `--path_video_dataset` must still be given, but the folder does **not** need to exist (only its name
  is used to identify the dataset).

## 4. Smoke test

```sh
conda activate pain-portable
bash smoke_test.sh                  # CPU, ~10 min (9-12 min measured on a NAS checkout); PASS/FAIL/SKIP per step
SMOKE_DEVICE=0 bash smoke_test.sh   # use GPU 0 instead (in pain-portable-cuda); ~10 min, ~14 min with SMOKE_FULL=1
SMOKE_FULL=1 bash smoke_test.sh     # additionally the 12 training configurations (smoke_test.sh's own copies) + VideoMAEv2-S extraction
```

**With `SMOKE_DEVICE=0`, check that the GPU was really used.** If CUDA is not usable, device
selection silently falls back to CPU, and the steps can still PASS. After the run:

```sh
grep -L 'extracting features using.... cuda' smoke_out/logs/extract_*.log   # must print nothing
grep -L 'GPU memory peak' smoke_out/logs/train_*.log                        # must print nothing
grep -iE 'CUDA error|device-side assert|CUBLAS_STATUS|CUDNN_STATUS|out of memory|same device' smoke_out/logs/*.log
```

You can also watch `nvidia-smi` during the run: each `train_model.py`, `extract_feature.py` and
`cross_space_projection.py` process should appear there. `cross_space_logs.py` is CPU-only by
design. If `TMPDIR` is on NFS, the logs contain
`OSError: [Errno 16] Device or resource busy: '.nfs…'` tracebacks from multiprocessing cleanup.
These are harmless, and the step still exits 0. timm `FutureWarning`s are harmless too.

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

Each paper model is one subject-independent 5-fold cross-validation run, defined by a YAML file
in `configs/train/`. Run it from the repository root:

```sh
python train_model.py --config configs/train/biovid_dfer.yaml
python train_model.py --config configs/train/biovid_dfer.yaml --stop 1 1 --ep 1   # quick check
```

Flags given on the command line override the YAML. `stop: [K, S]` runs K folds × S subfolds,
and the subfold with the best validation score becomes the model of that fold. List-valued keys
are Optuna grid axes, each with a single value. The header of each YAML names the original run it
reproduces. Add `--gp` to resolve relative paths against the repository root, or against
`$PAIN_PROJECT_ROOT` if set, instead of the current directory.

| Config (`configs/train/`) | Dataset (labels) | Backbone | Cross-space role (§5.2) |
| --- | --- | --- | --- |
| `biovid_vmae_s.yaml` † | BioVid Part A | VideoMAEv2-S | source → UNBC, MIntPAIN |
| `biovid_dfer.yaml` † | BioVid Part A | MAE-DFER | target ← UNBC, MIntPAIN |
| `mintpain_vmae_s.yaml` † | MIntPAIN | VideoMAEv2-S | source → BioVid, UNBC |
| `mintpain_dfer.yaml` † | MIntPAIN | MAE-DFER | target ← BioVid, UNBC |
| `unbc_opi0-4_vmae_s.yaml` | UNBC (OPI 0–4) | VideoMAEv2-S | source → BioVid, MIntPAIN |
| `unbc_opi0-4_dfer.yaml` | UNBC (OPI 0–4) | MAE-DFER | target ← BioVid, MIntPAIN |
| `unbc_opi0-5_vmae_s.yaml` | UNBC (OPI 0–5) | VideoMAEv2-S | – |
| `unbc_opi0-5_dfer.yaml` | UNBC (OPI 0–5) | MAE-DFER | – |
| `unbc_vas_vmae_s.yaml` † | UNBC (VAS) | VideoMAEv2-S | – |
| `unbc_vas_dfer.yaml` | UNBC (VAS) | MAE-DFER | – |
| `xite_vmae_s.yaml` | X-ITE (predefined splits) | VideoMAEv2-S | – |
| `xite_dfer.yaml` | X-ITE (predefined splits) | MAE-DFER | – |

† Needs the augmented embeddings (§3).

**Outputs.** Each run creates `runs/<config>/history_run_<pid>_ATTENTIVE_JEPA_<host>_<timestamp>/`
containing:

* `_config_prompt.txt`: the launch command and all resolved arguments
* `run.log`
* `ATTENTIVE_JEPA_<ts>.db/.pkl`: the Optuna study
* `<run_id>_<BACKBONE>_..._ATTENTIVE_JEPA/k_fold_results.pkl`: per-fold metrics, config and
  model parameters, including the selected subfold and epoch of each fold
* the same folder's `train_ATTENTIVE_JEPA/k<i>_cross_val/{train,val,test}.csv` and
  `k<i>_cross_val_sub_<j>/best_model_ep_<epoch>.pt`: the splits and the best checkpoint of
  every subfold

### 5.2 Cross-space projection (`cross_space_projection.py`)

A cross-space YAML runs one projector method for one source → target direction. It pairs the
5 source with the 5 target fold checkpoints and pools all pairs into one aggregated result.
Configs live in `Cross_projection_yaml/<config set>/<direction>/refinement<R>_<method>.yaml`,
with method ∈ `linear`, `mlp`, `autoencoder`, `procrustes`, `linear_close` (closed-form linear).

| Config set (`Cross_projection_yaml/`) | Reproduces | Output root (`Cross_projection/`) |
| --- | --- | --- |
| `config_paper_tests_seed_42` | main results, random class-balanced anchors | `paper_tests_seed_42` |
| `config_paper_tests_seed_42_quality` | quality-selected anchors | `paper_tests_seed_42_quality` |
| `config_ablation_frozen_random_adapter` | frozen random projector ablation | `ablation_paper_frozen_rnd_adapter` |
| `config_anchor_sweep` | anchor-budget figure (3 directions, see §5.5) | `anchor_sweep_rerun_2026_09_23` |

The first three sets contain the same six directions:

| Direction folder | Source (`old_model_pth`) | Target (`new_model_pth`) |
| --- | --- | --- |
| `unbcVmae_to_biovidDfer_cross_validation` | `unbc_opi0-4_vmae_s` | `biovid_dfer` |
| `unbcVmae_to_mintDfer_cross_validation` | `unbc_opi0-4_vmae_s` | `mintpain_dfer` |
| `biovidVmae_to_unbcDfer_cross_validation` | `biovid_vmae_s` | `unbc_opi0-4_dfer` |
| `biovidVmae_to_mintDfer_cross_validation` | `biovid_vmae_s` | `mintpain_dfer` |
| `mintVmae_to_biovidDfer_cross_validation` | `mintpain_vmae_s` | `biovid_dfer` |
| `mintVmae_to_unbcDfer_cross_validation` | `mintpain_vmae_s` | `unbc_opi0-4_dfer` |

**Key YAML fields.**

* `old_model_pth` / `new_model_pth`: the 5 source / target fold checkpoints, in fold order k0…k4.
  They point to the authors' runs, so you must replace them with your own (step 2 below).
* `interpolation_similarity`: the projector method (`linear`, `mlp`, …).
* `num_anchors`: anchor pairs used to fit the projector (100 in the paper sets).
* `anchor_selection_type`: `balance_class_random` or `balance_class_quality`.
* `refinement`: `3` fits the projector, then fine-tunes a copy of the target head's final linear
  layer, once alone and once jointly with the projector. `4` freezes a random projector and trains
  only that linear-layer copy.
* `seed`; `run_tag` (output subfolder under `Cross_projection/`); `linear_projector` and
  `refinement_config` (projector and refinement hyperparameters, used by every method).

**Steps.**

1. Train the six models of the direction table (§5.1).
2. Print the selected checkpoint of each fold of one training run, ready to paste inside the
   YAML list `[...]`.
   Copy the source run's list into `old_model_pth` and the target run's list into `new_model_pth`
   of every YAML in the direction folder (5 methods × 3 config sets share the same lists):

   ```sh
   python - runs/unbc_opi0-4_vmae_s/history_run_* <<'PY'
   import glob, os, pickle, sys
   for pkl in sorted(glob.glob(os.path.join(sys.argv[1], '*', 'k_fold_results.pkl'))):
       res = pickle.load(open(pkl, 'rb'))['results']
       for k in range(5):
           best = res[f'k{k}_cross_val_final']['best_model']
           f, s = best['fold_sub_fold_idx']
           print(f'  "{os.path.dirname(pkl)}/train_ATTENTIVE_JEPA/k{f}_cross_val/k{f}_cross_val_sub_{s}/best_model_ep_{best["best_model_idx"]}.pt",')
   PY
   ```

3. Run every YAML of a direction folder (GPU id or `cpu`), for each direction and config set:

   ```sh
   bash run_cross_space_configs.sh 0 Cross_projection_yaml/config_paper_tests_seed_42/unbcVmae_to_mintDfer_cross_validation
   ```

   A single YAML can also be run with `python cross_space_projection.py --config <file.yaml>`.
4. When all directions of a set have finished, summarize the set:

   ```sh
   python cross_space_logs.py --pkl_path Cross_projection/paper_tests_seed_42 --only_aggregated --skip_umap
   ```

**Outputs.** `Cross_projection/<run_tag>/search_<method>_..._<uid>/` holds the launch-config
snapshot, `precomputed/` (anchors and extracted embeddings), one folder per subtrial and the
pooled aggregated `.pkl`. `cross_space_logs.py --only_aggregated` writes `aggregated_summary.csv`
and per-run plots in the output root, e.g. `Cross_projection/paper_tests_seed_42/aggregated_summary.csv`.

### 5.3 Feature extraction from your own videos (optional)

```sh
python extract_feature.py --model_type DFER --emb_red spatial --path_dataset <video_root> \
  --path_labels <labels.csv> --saving_folder_path <out> --backbone_type video --save_as_safetensors \
  --stride_window 16 --stride_inside_window 1 --float_16 --save_big_feature      # --model_type S for VideoMAEv2-S
```

Videos are read from `<video_root>/<subject_name>/<sample_name>.mp4`, and embeddings are written to
`<out>/<subject_name>/<sample_name>.safetensors`. The `<video_root>` path must contain a known dataset keyword (`unbc`, `parta`/`biovid`, `mintpain`,
`xite`, …). The paper's face frontalization/preprocessing pipeline needs the raw videos and was
not part of this portability check. With the raw videos, the released embeddings were built in
this order:

1. **Dataset metadata.** Two datasets need conversion first. `MIntPAIN/generate_videos.py`
   assembles the MIntPAIN RGB frame sequences into per-clip videos and writes
   `MIntPAIN/starting_point/samples.csv`. The released videos use its default `--fps 25`.
   `XITE/starting_point/prepare_xite.py` writes the X-ITE challenge train/test/all CSVs. The
   21/5-subject train/validation split files in `XITE/starting_point/splits/` are released with
   the CSVs.
2. **Reference geometry.** `extract_landmarks.py` averages MediaPipe landmarks over all BioVid
   subjects into `partA/video/mean_face_landmarks_per_subject/all_subjects_mean_landmarks.pkl`.
3. **Frontalization.** Run `python extract_video_frontalized.py --gv --vfp <raw_videos> --csv <labels.csv> --pfo <out> --prl <reference.pkl>`.
   The algorithm is in `custom/faceExtractor.py` (Paper Appendix, Face Frontalization). See
   `--help`; for example, `--interpolation_mod_chunk mirror_start_video 16` pads each video to
   a multiple of 16 frames.
4. **Embeddings.** Use `extract_feature.py` (command above) for the base folders.
   `bash multiple_feature_extraction.sh <gpu> <n_runs> --dataset <name> --model {DFER,S} <aug…>`
   writes the augmented folders `<features>_<aug>[$N]`, one per run. XITE frontalized videos are
   checked first with `XITE/check_frontalized_videos.py`.
   `python check_augmentation_completeness.py --original_folder <features>` checks that every
   augmented folder is complete.

### 5.4 Paper tables, controls and statistics

After the §5.2 runs of all three config sets, and after `cross_space_logs.py --only_aggregated --skip_umap` on
each output root (`Cross_projection/paper_tests_seed_42`, `…_quality`,
`Cross_projection/ablation_paper_frozen_rnd_adapter`):

```sh
R=Cross_projection/paper_tests_seed_42          # one folder per direction below it
for d in $R/*_cross_validation; do
  python cross_space_fake_projection.py $d --control fake_embeddings --distribution standard_normal  # synthetic-embedding control
  python cross_space_fake_projection.py $d --control fake_adapter                                     # random-mapping ablation (seeds 42-46)
done
python cross_space_paper_tables.py anchor $R Cross_projection/paper_tests_seed_42_quality --metric micro   # also --metric macro
python cross_space_paper_tables.py refinement $R
python cross_space_paper_tables.py closed-form $R
python cross_space_paper_tables.py synthetic $R/*_cross_validation
python cross_space_paper_tables.py random-adapter $R/*_cross_validation
python cross_space_paper_tables.py frozen $R Cross_projection/ablation_paper_frozen_rnd_adapter
python cross_space_paper_tables.py confusion $R/mintVmae_to_biovidDfer_cross_validation --method linear --seed 42
```

Add `--output <file.tex>` to write a table to a file instead of stdout.

* **Baseline tables.** Run `python extract_kfold_test_table.py --pkl <run>/k_fold_results.pkl --raw`
  for each training run of §5.1. Add `--metric accuracy` for MIntPAIN and X-ITE.
* **Video-level statistics.** Run `python statistical_compare_kfold_results.py --pkl_path_0 <MAE-DFER k_fold_results.pkl> --pkl_path_1 <VideoMAEv2-S k_fold_results.pkl> --analysis_level video --measure mae`
  to get the paired tests between the two backbones.
* **Constant-predictor baselines.** These need no script. For a fixed constant *c*, MAE is
  `mean(|y − c|)` over all videos of the full label CSV, and Macro-MAE averages that error over
  the represented levels. For example, BioVid with c=2 gives 1.20 / 1.20 and MIntPAIN with c=0
  gives 1.25 / 2.00.

### 5.5 Appendix figures

**Anchor-budget sweep.** `Cross_projection_yaml/config_anchor_sweep/` holds the three directions
in the figure (UNBC→BioVid, BioVid→UNBC, MIntPAIN→UNBC). `run_cross_space_anchor_sweep.sh`
runs one YAML once per value in its `num_anchors` list and writes each run to
`Cross_projection/<run_tag>/K<k>/`. As in §5.2, point the checkpoint paths in the YAMLs to your
own runs first. The GPU id must be an integer.

```sh
for d in Cross_projection_yaml/config_anchor_sweep/*/; do bash "$d/run_all_anchor_sweeps.sh" 0; done
python plot_anchor_sweep_mae.py --input-root Cross_projection/anchor_sweep_rerun_2026_09_23 \
  --output-dir analysis/results/anchor_sweep_mae      # micro/macro MAE per K, method and stage
```

The paper figure shows the projector-only stage.

**Predictions over growing video prefixes (BioVid sample 7678).** `plot_cumulative_predictions.py`
reruns a cross-space experiment on one video, cutting it at increasing lengths. With
`--dump-curves` it also writes the plotted values to `curves_<digest>.json`.
`plot_cumulative_snapshot_figure.py` then puts the face snapshots under a plot.

```sh
python plot_cumulative_predictions.py <cross-space run>/refinement3_mlp_100 7678 --origin source \
  --refinement-stage 2 --compare-native-root <BioVid DFER run> --video 1 --snapshots 47 63 95 143 --dump-curves
D=chunk_prediction_plots/videomae_biovid_to_unbc_dfer/7678/mlp/videomae_biovid/stage_2
python plot_cumulative_snapshot_figure.py $D/summary_full_range_<digest>.png \
  $D/summary_full_range_video_<digest>_snapshots cumulative_biovid7678_mean_curves.png
python plot_cumulative_snapshot_figure.py $D/models_stage_2_full_range_<digest>.png \
  $D/models_stage_2_full_range_video_<digest>_snapshots cumulative_biovid7678_task_aware_per_target.png
```

`run_cumulative_predictions.sh` runs `plot_cumulative_predictions.py` over many videos.

## 6. Known limitations

* **Backbone weights are mandatory** even for training on precomputed embeddings, and are
  loaded from paths relative to the working directory: run from the repository root.
* **CPU** runs work, but real experiments are slow without a GPU. CPU and GPU (and different
  attention kernels) give slightly different numbers, so exact paper values need the GPU setup.
* The cross-space YAMLs refer to the authors' original run folders. You must point them to your
  own checkpoints (§5.2, step 2).
* Only Linux x86_64 was tested end-to-end. Windows/macOS environment files exist in
  `env_portability/`, but runtime was not verified there.
* Raw-video preprocessing (face frontalization, landmark extraction, augmentation generation) and
  the paper table/plot scripts were not covered by the smoke test.
* The V-JEPA 2.1 backbone option in `custom/backbone.py` (`--mt vjepa2_1_B`) is not used in the
  paper, and its vendored code is not shipped. Using it requires a checkout of
  [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) in `vjepa2/`.
* See [PORTABILITY_REPORT.md](PORTABILITY_REPORT.md) for every issue found and fixed.
