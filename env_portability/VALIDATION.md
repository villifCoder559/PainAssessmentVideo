# Portability validation

Audit date: **2026-10-01**. Validation was performed from Linux with Conda 25.3.1
and its libmamba solver. Cross-platform package resolution is distinct from
running the application on each operating system.

## Repository and installed-environment audit

- Read **332 Python files and four notebooks**, including ignored/untracked
  application source and the relocated snapshot verifier. Recovered imports from
  the existing incomplete `fps =` cell in `test.ipynb` without editing it.
- Cross-checked **2,832 import occurrences and all 47 third-party import roots**.
  Every root has a distribution mapping or an explicit native, optional, or
  historical-source requirement in [DEPENDENCIES.md](DEPENDENCIES.md).
- Checked **4,005 existing script, configuration, manifest, and documentation
  files**, including upstream environment declarations and saved experiment
  configurations. Parsed all six new portable YAML specifications.
- Compared the active Python 3.10.11 environment's **338 Conda packages** and
  **219 distinct pip-visible distribution names** with source requirements.
  Installed-only packages and package-managed transitives were excluded from the
  direct dependency specifications.
- Checked configuration-selected local model modules, the external OptunaHub
  AutoSampler requirements, and executable requirements including Git/FFmpeg.
- Hash comparisons confirm **336 Python/notebook files remain unchanged**.
  All **ten preserved reproducibility artifacts** other than the relocation guide
  retain their original bytes, including exact locks and the custom torchsort wheel.
  Only environment/documentation organization and Git ignore rules were changed.

The full source/import/provider record is [dependency-audit.json](dependency-audit.json).
The original broader audit remains in
[the reproducibility snapshot](../env_reproducibility/locks/dependency-audit.json).

## CPU package resolution

Conda dry runs combined the common file with the relevant native addition.
Apple Silicon included Conda Decord but excluded unavailable arm64 torchsort.
All four targets resolved using only Conda-forge with strict channel priority.
The macOS targets used a macOS 13 virtual-package setting; Linux used the host's
glibc 2.35 virtual package.

| Target | Conda solve | pip dependency/wheel check | Required native additions |
| --- | --- | --- | --- |
| Linux x86_64 | Pass | Pass | Conda Decord 0.6.0 and torchsort 0.1.9 |
| Windows x86_64 | Pass | Pass | Conda torchsort 0.1.9; PyPI Decord 0.6.0 |
| macOS Intel | Pass | Pass | Conda Decord 0.6.0 and torchsort 0.1.9 |
| macOS Apple Silicon | Pass | Pass | Conda Decord 0.6.0; torchsort source build remains unvalidated |

All four Conda resolutions selected Python **3.10.21**, CPU PyTorch **2.5.1**,
torchvision **0.20.1**, NumPy **1.26.4**, SciPy **1.15.2**, transformers **4.57.6**,
and timm **1.0.29**. These are observed resolutions, not additional manifest pins.

Separate pip dry runs targeted each platform's Python 3.10 wheel tags and
constrained scientific libraries to the corresponding Conda versions. They
selected MediaPipe **0.10.21**, OpenCV contrib **4.11.0.86**, MoviePy **1.0.3**,
and decorator **4.4.2**. An isolated PEP 517 build of MoviePy's pure-Python wheel
also passed. That temporary wheel was used only for validation; it is neither a
repository artifact nor an installation requirement.

## NVIDIA package resolution

The standalone GPU profile's Conda portions resolved for Linux and Windows
x86_64. Neither resolution contains Conda PyTorch, libtorch, or torchvision:
the official CUDA wheels provide that stack once, independently of Conda FFmpeg.
Linux Conda Decord 0.6.0 resolves with this profile as well.

| Target | Conda solve | pip dependency/wheel check | Native ranking requirement |
| --- | --- | --- | --- |
| Linux x86_64 NVIDIA | Pass | Pass | Upstream matching torchsort wheel; availability checked |
| Windows x86_64 NVIDIA | Pass | Pass | CUDA torchsort source build remains unvalidated |

Both pip checks selected **torch 2.5.1+cu118 / torchvision 0.20.1+cu118** and
preserved the Conda-resolved NumPy/SciPy and Hugging Face versions: NumPy 1.26.4,
SciPy 1.15.2, transformers 4.57.6, huggingface-hub 0.36.0, and tokenizers 0.22.2.
The scientific and Hugging Face versions were supplied as temporary validation
constraints, not added as transitive pins to the manifests. Pip selected timm
1.0.30, torchmetrics 1.9.0, and WebDataset 1.0.2 within the declared requirements.

Pip's platform flag selects wheel tags but otherwise evaluates operating-system
dependency markers on the host. The Windows check therefore explicitly evaluated
those markers as Windows: it requires neither Linux NCCL nor Triton packages.
Non-Linux CPU wheel checks likewise used their destination OS/architecture markers.
These are dependency-resolution checks, not execution under the destination OS.

The official Python 3.10 CUDA 11.8 PyTorch/torchvision wheels and the upstream
Linux `pt25cu118` torchsort wheel were checked for availability. The Windows
Decord wheel's embedded Python tags were inspected. Native Windows CUDA torchsort
and Apple Silicon CPU torchsort require source builds that were not run here.

## Runtime checks and preservation

The **original installed Linux environment**, after relocation of its verifier,
passed the snapshot comparison: Python 3.10.11, 338 Conda package records,
224 unique Python distribution/version pairs, and 98 native pip libraries.
The difference from the 219 pip-visible names reflects duplicate distribution
metadata with differing versions in the captured installation.

Baseline imports passed for torch, torchvision, NumPy, SciPy, OpenCV, MediaPipe,
transformers, timm, Decord, torchsort, `custom.backbone`, `custom.dataset`,
`custom.faceExtractor`, and `custom.loss`. A CPU torchsort soft-rank operation
also passed. The local multiprocessing-manager socket required running that
smoke check outside the restricted tool sandbox.

No complete new portable environment was installed, and these baseline runtime
checks do not validate newly resolved package versions. No Windows/macOS runtime,
GPU training, native source build, complete historical upstream workflow, or
external dataset/checkpoint access was exercised. Use the post-install checks in
[ENVIRONMENT.md](ENVIRONMENT.md) on each destination computer. Existing fixed
paths, multiprocessing import side effects, legacy APIs, optional private/source
dependencies, and hardware-specific source paths remain documented limitations.
