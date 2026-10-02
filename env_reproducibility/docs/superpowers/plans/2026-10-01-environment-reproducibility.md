# Environment Reproducibility Implementation Plan

**Goal:** Reproduce the active environment while identifying requirements of every repository workflow.
**Architecture:** Keep direct dependencies in `environment.yml`; preserve the complete active Conda resolution and pip overlays separately under `locks/`. Document import coverage, missing optional/legacy dependencies, native ABI constraints, and validation in `ENVIRONMENT.md`.
**Tech Stack:** Existing Conda 25.3.1, Python 3.10.11, Python package metadata, AST notebook/import inspection.
**Spec:** User's environment reproducibility request in this session.

## Constraints and review focus

Application/research code and existing user edits must remain unchanged. Inspect tracked and untracked source, notebooks, scripts, configurations, and bundled projects. Installed packages are not automatically direct dependencies. Preserve pip overrides of Conda metadata and the custom torchsort wheel. Do not claim all workflows or GPU execution work when current evidence contradicts it. Exact linux-64 builds are not portable to other OS/architectures.

## Tasks

- [x] Inventory repository imports, dynamic imports, notebook commands, existing manifests, and shell tools; classify standard-library/local/third-party names and investigate missing names.
- [x] Snapshot Conda URLs/builds/channels/hashes and every installed Python distribution; identify pip-managed overlays and preserve custom wheel provenance. Compare snapshots with live metadata.
- [x] Replace the existing raw `environment.yml` with direct dependencies pinned to observed versions; document optional incompatible workflows rather than invent working versions.
- [x] Document OS, Python, Conda, CUDA, PyTorch, commands, platform limits, all import-to-package mappings, and unresolved dependencies.
- [x] Validate YAML with Conda tooling where feasible, pip resolution without installing, exact snapshot completeness, and native-library imports. Record actual failures and limitations.

Execution is already authorized by the user's request. Work in the current workspace to preserve the active repository/environment evidence; no application implementation, source changes, or commits are required.

Validation: 335 original source/notebook files, 4,005 configuration/script/doc files, 47 third-party roots; 338 Conda records and 56 hashed pip overlays cross-checked; 98 native libraries matched; final readable Conda dry-run and live snapshot verifier passed. Core CPU imports and host CUDA ranking/RNG/convolution checks passed on two RTX 2080 Ti GPUs with driver 535.247.01. The sandbox GPU probe was unavailable. Missing optional sources and the two existing pip-check findings are documented.

Native compatibility anchors: the readable solver selected MKL 2025 and newer CUDA support libraries without additional pins, so MKL 2023.1.0, libcurand 10.3.5.147, and libcufile 1.9.1.3 were pinned to the observed versions; the final solve passed.
