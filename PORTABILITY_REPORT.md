# Portability report (2026-10-06)

> **Note (2026-10-08):** this report is kept as written. Some files it mentions were later removed because the paper does not use them: `custom/composite_loss.py`, `run_cross_space_anchor_sweep.sh`, `cross_space_reproducibility_report.py`, `tests/test_prepare_pemf.py`, the vendored `vjepa2/` and the unused vendored modules (e.g. `making_better_mistakes/data/scripts_asis/`). See the README's *Repository layout* for the current file set.

Goal: verify that a third person can rebuild the environment and run the training and
cross-space pipelines from this repository. They get the released CSVs, the precomputed
embeddings and the public backbone checkpoints, but no raw videos. Only installation, import,
path, missing-file and crash issues were in scope. Model logic, losses, splits and
hyperparameters were **not** changed.

## Method

* **Clean room** on `/seidenas/users/fvilli/portability_cleanroom/` (outside the project):
  * `git archive HEAD` (tracked files only), plus the released CSVs (`*/starting_point`) and
    embeddings (base folders; augmented folders for the subjects the smoke test uses).
  * VideoMAEv2-S weights freshly downloaded from HuggingFace (byte-identical to the local copy);
    MAE-DFER weights copied (Google-Drive asset).
* **Fresh Miniforge** (conda 26.7.2) installed into the clean room:
  * Empty conda/pip/HF/torch caches, `PYTHONNOUSERSITE=1`, empty `CONDARC`, `PYTHONPATH` unset.
  * Nothing from the original `update_torch_project` env or `~/.cache` was used.
* **Envs built only from `env_portability/`:**
  * CPU: `environment.yml` + `environment-native.yml`.
  * GPU: `environment-cuda.yml` + the documented Decord/torchsort commands.
* **Smoke runs:** every entry point in minimal mode (1 epoch, first fold/subfold, subject
  subsets, 2-epoch projectors), on CPU with `CUDA_VISIBLE_DEVICES=""` and on an RTX 2080 Ti.
  This became `smoke_test.sh`.
* **Final check:** after all fixes, a second clean room was built from scratch, following
  only `README.md`: pinned env file → weights → data → `bash smoke_test.sh`. See the
  "Final verification" section.

## Issues found and changes made

### Installation / environment

| # | Issue | Change |
| --- | --- | --- |
| E1 | `tree-format` (listed in `environment.yml` / `environment-cuda.yml`) installs a wheel tagged `cp27`. `pip check` on a fresh env fails with `tree-format 0.1.2 is not supported on this platform`. It is imported only by two `making_better_mistakes/data/scripts_asis` data-prep scripts, never by the pipelines. | Removed from both specs; `DEPENDENCIES.md` / `ENVIRONMENT.md` updated. |
| E2 | `env_portability` had never been runtime-tested (its README said so). | Both envs built and smoke-tested. Exact exports added: `env_portability/environment-pinned-linux-64.yml` (CPU) and `environment-cuda-pinned-linux-64.yml` (CUDA 11.8; includes the PyTorch index and the torchsort wheel URL). Docs updated. |
| E3 | No other conflicts, missing imports or CUDA/torch mismatches. Every listed package is imported by first-party code or the vendored backbones (`MAE_DFER`, `VideoMAEv2`, `jepa`, `making_better_mistakes`). `cmaes` and `git` are runtime needs of OptunaHub's AutoSampler. | none |
| E4 | Creating the env on the NAS took about 29 min (CPU) and 50 min (CUDA). This is I/O-bound, not an error. | none (noted in README timing) |

### Runtime: CPU-only machines

`train_model.py`, `cross_space_projection.py` and `extract_feature.py` could not run without
a GPU. Each fix is a `'cuda' if torch.cuda.is_available() else 'cpu'` fallback, or a guard
around CUDA-only calls. On a GPU machine the behavior is identical.

| # | Symptom (fresh CPU env) | Fix |
| --- | --- | --- |
| C1 | `RuntimeError: Attempting to deserialize object on a CUDA device` when loading backbone weights (`custom/backbone.py`, `BackboneBase.device = "cuda"`). | device fallback |
| C2 | Hardcoded `'cuda'` in `custom/head.py` (train + evaluate), `custom/loss.py`, `custom/composite_loss.py`, `train_model.py` (`ce_weight`, `sim_loss`), `cross_space_projection.py` (`LINEAR_PROJECTOR_CONFIG` / `REFINEMENT_CONFIG` `device`) and `extract_feature.py`. `torch.cuda.mem_get_info()` / `reset_peak_memory_stats()` crash on CPU. | device fallback; CUDA memory calls guarded by `torch.cuda.is_available()` |
| C3 | `RuntimeError: No viable backend for scaled_dot_product_attention`: `jepa/src/models/utils/modules.py` forced `SDPBackend.EFFICIENT_ATTENTION`, which has no CPU kernel. | Backend list is now `[EFFICIENT_ATTENTION, MATH]`. PyTorch still prefers efficient attention whenever it is available (GPU), so GPU numerics are unchanged. |
| C4 | `run_cross_space_configs.sh` required an integer GPU id. | Also accepts `cpu` (hides all GPUs). |

### Runtime: missing raw videos / private assets

| # | Symptom | Fix |
| --- | --- | --- |
| D1 | `AssertionError: Dataset path UNBC/video/WarpedVideos_Cropped_interpolated_mirror does not exist` (`custom/dataset.py`). Raw videos are not released, but the folder was asserted even when training from `--ffsp` embeddings. | Assert → warning. The path string is still used to identify the dataset. A real video read would still fail at load time. |
| D2 | `FileNotFoundError: partA/video/mean_face_landmarks_per_subject/all_subjects_mean_landmarks.pkl`. A BioVid-derived pickle was loaded for every dataset; `self.reference_landmarks` is never read anywhere. | Loaded only if the file exists, otherwise `None`. |
| D3 | `ValueError: cannot convert float NaN to integer` in `FilteredAugmentationBatchSampler._calculate_length`. Configs 01, 02, 03, 04 and 11 (`selective_augm`, `keep_original < 1`) need the augmented-embedding sibling folders `<ffsp>_<aug>[$N]`, and without them the error was cryptic. | Raises a clear `FileNotFoundError` explaining the expected folders. Valid runs are unaffected. README documents these folders as **required data** (BioVid ≈ 35 GB, MIntPAIN ≈ 3 GB, UNBC ≈ 2 GB). |
| D4 | Backbone checkpoints are not in git, but training and cross-space always build the backbone and load its weights, even for precomputed embeddings. | Not changed (model construction). README gives both download sources and exact target paths. `smoke_test.sh` checks for them first. |

### Hardcoded paths and machine-specific config

| # | Location | Change |
| --- | --- | --- |
| P1 | `custom/helper.py` `GLOBAL_PATH.NAS_PATH = '/equilibrium/fvilli/PainAssessmentVideo'` (used with `--gp`, e.g. paper run 12, and in `extract_feature.py` / `extract_video_frontalized.py`). | Defaults to the repository root; can be overridden with `PAIN_PROJECT_ROOT`. |
| P2 | `custom/backbone.py` comment with the author's home path. | Made generic. |
| P3 | `extract_feature.py` `tempfile.tempdir = '/tmp'`. | Now `$TMPDIR`, falling back to `/tmp`. |
| P4 | `run_cross_space_configs.sh`, `run_cross_space_anchor_sweep.sh` and `run_cumulative_predictions.sh` are stored in git as mode `100644` (`core.fileMode=false` on the NAS). `./run_cross_space_configs.sh` gives `Permission denied` in a fresh clone. | Docs and `smoke_test.sh` call them via `bash …`. **Recommended before release:** `git update-index --chmod=+x run_cross_space_configs.sh run_cross_space_anchor_sweep.sh run_cumulative_predictions.sh smoke_test.sh`. |
| P5 | Backbone-weight paths (`custom/helper.py` `MODEL_TYPE`), the cross-space `_FEATURES_MAP` and the data paths are relative to the working directory. | Not changed. README: run from the repository root (`smoke_test.sh` `cd`s there itself). |
| P6 | Non-interactive runs (`< /dev/null`, nohup, CI) printed an `EOFError` traceback from the "type `s` to stop" stdin thread. | The thread exits quietly on EOF. |
| P7 | Found during the final README-only pass: `VideoMAEv2/pretrained/` is not tracked, so a plain `curl -o VideoMAEv2/pretrained/…` fails in a fresh clone (`curl: (23) Failure writing output`). | README uses `curl --create-dirs` and `mkdir -p` for the MAE-DFER folder. The `gdown` download of the MAE-DFER Google Drive file was verified (byte-identical). |
| P8 | `.gitignore` ignores `*.md`, so the new `README.md` and `PORTABILITY_REPORT.md` would be silently left out by `git add`. | Added `!README.md` and `!PORTABILITY_REPORT.md` next to the existing `!ENVIRONMENT.md`. |

## Unresolved / for the author to decide

1. **Release contents.**
   * `Cross_projection_yaml/` (the paper cross-space configs) is **untracked**. Add it (at least
     `config_paper_tests_seed_42`, `config_paper_tests_seed_42_quality`,
     `config_ablation_frozen_random_adapter`) if the paper configs are part of the release.
   * Its `new_model_pth` / `old_model_pth` point to the original run folders and best epochs;
     users must replace them with their own checkpoints (README §5.2), unless the 30 trained
     fold checkpoints are released too.
2. `.claude/settings.local.json` is tracked and contains local absolute paths; remove it from the
   public repo.
3. `REPRO_model_training/*.sh`, `paper_parallel/*.sh` and `paper_reproduction_commands.txt`
   (untracked) still contain `/equilibrium/...` paths and the local conda env name. The README
   has the portable equivalents of the 12 training commands.
4. **Dataset inferred from path keywords.** The dataset is identified from keywords in the
   video/feature paths (`helper.set_step_shift`, `cross_space_projection._detect_dataset`).
   Renaming data folders breaks this. Documented, not changed.
5. **Small-data edge case** (not a portability bug). The cross-space projector splits the
   target model's validation subfold into subject-disjoint halves, so it needs ≥ 3 validation
   subjects. With a 10-subject MIntPAIN subset it failed (`subject-disjoint split produced an
   empty subset`); the smoke test therefore uses all MIntPAIN/UNBC subjects.
6. **Untested here:**
   * Windows/macOS runtime.
   * Raw-video preprocessing (frontalization, landmarks, augmentation generation) on real videos.
   * The paper table/plot scripts (`cross_space_paper_tables.py`, etc.).
   * Full-length training and numerical agreement with the paper. CPU vs GPU and SDPA kernels
     give small numeric differences.
7. `making_better_mistakes/data/scripts_asis/*` need `tree-format`, which is no longer in the env
8. `tests/test_prepare_pemf.py` (tracked, 11 tests) expects `PEMF/prepare_pemf.py`, which is not in the repository, so it fails in any clone. This was already the case before this work. The rest of the suite passes with the portability changes: `pytest tests --deselect tests/test_prepare_pemf.py` → 882 passed, 2 skipped.
   (install manually if those scripts are ever needed).

## Final verification

A second clean room (`/seidenas/users/fvilli/portability_cleanroom2/`) was built following only
`README.md`:

* **Release export:** tracked files plus the new deliverables, with launcher scripts reset to
  mode 644 as a clone would give them.
* **Shell:** system-only `PATH`, all inherited `CONDA_*` variables cleared, fresh package caches.
* **Steps:**
  1. `conda env create -f env_portability/environment-pinned-linux-64.yml` (23 min on the NAS)
     → `conda activate pain-portable` → `pip check`: no broken requirements.
  2. Weights with the README commands: `curl --create-dirs …` (HuggingFace) and
     `gdown 1nzvMITUHic9fKwjQ7XLcnaXYViWTawRv` (Google Drive). Both byte-identical to the
     author's copies.
  3. Released CSVs and embeddings placed in the documented layout.
  4. `bash smoke_test.sh` → **all 11 steps PASS, exit 0, 725 s on CPU**:

```
PASS  env (66s)            PASS  train_07 (37s)   PASS  train_06 (90s)
PASS  imports (52s)        PASS  train_03 (48s)   PASS  train_01 (48s)
PASS  synthetic_video (1s) PASS  train_12 (36s)   PASS  xspace_projection (244s)  [6/6 configs]
PASS  extract_DFER (38s)                          PASS  xspace_logs (63s)
```

Other runs:

| Run | Result |
| --- | --- |
| First clean room, flexible CPU env | 562 s, all PASS |
| First clean room, CUDA env, `SMOKE_DEVICE=0` (RTX 2080 Ti, confirmed `cuda`) | 684 s, all PASS |

The checkout, envs and data all lived on the NFS NAS, so these timings are I/O-bound (imports
alone take 25–65 s); on a local disk the smoke test is expected to finish well under 10 minutes.
