# Portable project environments

Use this directory when moving the project to another computer. On Linux x86_64, the
recommended route is the tested pinned files: `environment-cuda-pinned-linux-64.yml` for
NVIDIA GPUs, and `environment-pinned-linux-64.yml` for CPU (see [README.md](README.md#quick-start)).
The flexible specifications below are the fallback, and the route for Windows/macOS. They
deliberately allow a new resolution, and they do not reproduce identical floating-point results.

## Python and file selection

**Python 3.10**, at the minor-version level, preserves the working application's
Python/PyTorch integration and has native wheels on the target platforms. The active
environment was Python 3.10.11 with PyTorch 2.5.1, torchvision 0.20.1, torchaudio 2.5.1,
NumPy 1.26.4, MediaPipe 0.10.5, transformers 4.57.1, and timm 1.0.15.

| File | Use |
| --- | --- |
| `environment.yml` | Common CPU stack on Linux x86_64, Windows x86_64, and macOS Intel/Apple Silicon. Conda-forge only; `nodefaults` prevents implicit extra channels. |
| `environment-native.yml` | Required addition for CPU training and vendored video loaders on Linux x86_64 and Intel macOS: Conda Decord and torchsort. |
| `environment-windows.yml` | Required addition for native Windows CPU training/video loading: Conda torchsort and PyPI Decord. |
| `environment-cuda.yml` | Standalone alternative for NVIDIA GPUs on Linux/Windows x86_64. Conda-forge supplies the common libraries; official PyTorch CUDA wheels supply the GPU stack. Native video/ranking installation is described below. |
| `environment-notebooks.yml` | Optional JupyterLab/IPython/kernel and pytest tooling. |
| `environment-cluster.yml` | Optional submitit launchers for POSIX/SLURM workflows, normally Linux. |
| `DEPENDENCIES.md` | Complete third-party import-to-distribution cross-check, including conditional and historical imports. |

**The common file alone is insufficient for training.** `custom/loss.py` imports
torchsort unconditionally; imported vendored video loaders require Decord. Both are
direct requirements, separated because their binary availability differs by OS and
CPU architecture. The overlays are additions, not standalone environments. There is
no CPU/CUDA auto-detection in these files.

## Installation

Run these commands from the repository root with Conda initialized in your shell.
Use a fresh environment. Do not overlay the portable setup onto the captured one.

Linux x86_64 CPU, or Intel macOS CPU:

```sh
conda env create -f env_portability/environment.yml
conda env update -n pain-portable -f env_portability/environment-native.yml
conda activate pain-portable
python -m pip check
```

Native Windows x86_64 CPU, in Anaconda Prompt or a Conda-initialized PowerShell:

```powershell
conda env create -f env_portability/environment.yml
conda env update -n pain-portable -f env_portability/environment-windows.yml
conda activate pain-portable
python -m pip check
```

Apple Silicon macOS (native arm64):

```sh
conda env create -f env_portability/environment.yml
conda activate pain-portable
conda install --override-channels -c conda-forge "decord>=0.6,<0.7"
# Required for training; there is no compatible prebuilt arm64 torchsort package.
# Install Apple's Command Line Tools if they are not already available:
xcode-select --install
python -m pip install --no-build-isolation --no-deps "torchsort>=0.1.9,<0.2"
python -m pip check
```

The torchsort source build requires a C++ compiler and is **not validated here on
Apple Silicon**. If it fails, CPU analysis utilities that do not import the training
stack remain usable; training is not supported until this extension is available.
An Intel Conda installation under Rosetta is another candidate, using the Intel
commands and x86_64 packages consistently; do not mix arm64 and Intel packages.

For notebooks/tests, add the optional overlay after creating the base environment:

```sh
conda env update -n pain-portable -f env_portability/environment-notebooks.yml
conda run -n pain-portable python -m ipykernel install --user --name pain-portable --display-name "Pain assessment (portable)"
conda run -n pain-portable jupyter lab
```

For the CUDA environment, use `-n pain-portable-cuda` with either optional overlay.
Do not apply the CPU native overlay to a CUDA environment.

## NVIDIA GPU installation

On Linux x86_64, first try the pinned file
`conda env create -f env_portability/environment-cuda-pinned-linux-64.yml`, which already contains
Decord and the matching torchsort wheel. Use the flexible steps below only if it does not resolve.
Both routes need an NVIDIA driver ≥ 520 and a GPU with compute capability ≤ 9.0 (PyTorch
2.5.1+cu118 ships kernels for sm_37 to sm_90).

The base selects Conda-forge's CPU PyTorch package and does not request CUDA, cuDNN,
NVIDIA drivers, MKL, Triton, or any system-toolkit path explicitly. Dependencies may
choose their own platform-appropriate BLAS and native libraries.

`environment-cuda.yml` instead selects PyTorch 2.5.1 / torchvision 0.20.1 and
**CUDA runtime 11.8**. This is a supported binary family from the
[official PyTorch installation matrix](https://pytorch.org/get-started/previous-versions/),
chosen for broad compatibility with existing NVIDIA drivers. It is an environment
runtime dependency, independent of the host's driver and system CUDA compiler. NVIDIA drivers remain
host software. Check your GPU and driver against the
[CUDA compatibility guide](https://docs.nvidia.com/deploy/cuda-compatibility/).
macOS does not support this CUDA file. New GPU architectures that require a newer
PyTorch/runtime family need a separately validated environment.

Both CPU and CUDA files use only Conda-forge, with `nodefaults`. The GPU file uses
official PyTorch wheels because the older Linux torchvision Conda package couples
the environment to an FFmpeg ABI incompatible with current Conda Decord builds.
The wheel carries its video libraries independently. Pure-Python timm, torchmetrics,
torch-optimizer, and WebDataset also use pip in this profile so Conda does not install a second
PyTorch provider. The pip indexes are explicitly public PyPI and the official
PyTorch CUDA 11.8 index; CUDA wheel selection is not left to the computer's default
pip configuration.

Linux NVIDIA GPU:

```sh
conda env create -f env_portability/environment-cuda.yml
conda activate pain-portable-cuda
conda install --override-channels -c conda-forge "decord>=0.6,<0.7"
# Official upstream wheel: Python 3.10 / PyTorch 2.5 / CUDA 11.8 / Linux x86_64.
python -m pip install --no-deps "https://github.com/teddykoker/torchsort/releases/download/v0.1.10/torchsort-0.1.10+pt25cu118-cp310-cp310-linux_x86_64.whl"
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.ones(1, device='cuda'))"
```

The wheel is an optional platform-specific installation step, not a path or binary
reference embedded in a portable YAML. Consult
[torchsort's upstream instructions](https://github.com/teddykoker/torchsort#pre-built-wheels)
before changing Python, PyTorch, or CUDA: the extension must match all three.
Conda-forge's GPU torchsort binaries for PyTorch 2.5 use CUDA 12.6 and its own
PyTorch build, so do not mix them into the official wheel stack.

Use Conda Decord on Linux/macOS: the older Linux PyPI wheel has obsolete
CPython-3.6 tags inside `WHEEL` even though it is published with a Python-3 filename.
The common and CUDA Linux commands avoid that publisher metadata issue. The
Windows PyPI wheel's embedded tags were checked and do not have that issue. These
Decord packages decode on the CPU; CUDA tensors/training still use the selected
PyTorch CUDA runtime. GPU-accelerated Decord decoding needs a separate source build.

Native Windows NVIDIA GPU:

```powershell
conda env create -f env_portability/environment-cuda.yml
conda activate pain-portable-cuda
python -m pip install "decord>=0.6,<0.7"
# No upstream Windows CUDA torchsort wheel is provided. A source build requires
# Visual Studio C++ Build Tools and a CUDA 11.8 development toolkit:
python -m pip install --no-build-isolation --no-deps "torchsort>=0.1.9,<0.2"
python -m pip check
python -c "import torch; print(torch.ones(1, device='cuda'))"
```

That Windows source build is unverified; WSL2 with an NVIDIA-enabled Linux setup is
the more established route for GPU training and Bash/SLURM-oriented code. CUDA
development tools are needed only when compiling extensions; prebuilt PyTorch and
torchsort wheels do not require system `nvcc`. The archived pt25/cu118 Linux wheel
is for that same ABI family only; use the
public upstream wheel for portability. Never install it into a CPU, macOS,
Windows, or different Python/PyTorch/CUDA environment.

DeepSpeed is an additional dependency for
`VideoMAEv2/run_class_finetuning.py` and an optional MAE_DFER backend. It is **not
required by the main custom training path**. Install it separately only for that
workflow (`python -m pip install deepspeed`), with a compatible PyTorch/toolkit and
compiler, then check its extension report with `ds_report`. Its full compatibility
has not been validated. NVIDIA Apex also requires a chosen upstream source/build;
the unrelated PyPI package named `apex` is not the provider.

## Version constraints and package providers

| Constraint | Compatibility reason |
| --- | --- |
| `python=3.10` | Audited application baseline, native wheel availability, and extension ABI; patch/build remain flexible. |
| `pytorch[-cpu]>=2.5.1,<2.6`, `torchvision>=0.20.1,<0.21` | Keep the supported binary family together. PyTorch 2.6 changes the default `torch.load(weights_only=...)`; several legacy loaders omit that argument. |
| `numpy>=1.26,<2` | Preserve the modern application's NumPy 1.x ABI and avoid NumPy 2 incompatibilities in the older video/native stack. |
| `moviepy>=1.0.3,<2` | Scripts import `moviepy.editor` and use `set_duration`, `set_position`, and the old TextClip API. |
| `transformers>=4.57,<5` | Current ViT/V-JEPA 2 integration and `ViTFeatureExtractor`; avoid a major API migration. |
| `timm>=1.0.15,<1.1` | Preserve the modern model registration/layer API family used by the working custom backbones. Legacy optimizer imports still have limitations below. |
| `torchmetrics>=1.5,<2`, `optuna>=4.3,<5`, `optunahub>=0.3,<1` | Retain the audited metric/sampler API families; these are conservative compatibility bounds, not tested guarantees for every allowed release. |
| `mediapipe>=0.10.5,<0.10.22` | Keep the legacy `mp.solutions` face APIs together with Tasks `FaceAligner`, `FaceAlignerOptions`, and `FaceLandmarksConnections`. |
| `opencv-contrib-python>=4.8,<4.12` | MediaPipe requires this distribution; OpenCV 4.12 wheels introduce a NumPy 2 requirement on Python 3.10. |
| `decord>=0.6,<0.7`, `torchsort>=0.1.9,<0.2` in CPU additions | Known video/ranking API families. The Conda solver must additionally match the torchsort binary to PyTorch. It can select 0.1.9 rather than the captured custom 0.1.10 wheel. |
| `torch==2.5.1+cu118`, `torchvision==0.20.1+cu118` in CUDA file only | Official matching native release pair and CUDA ABI; exact wheel versions prevent pip choosing the default CUDA 12.4 family and match the documented torchsort wheel. OS wheel tags are selected automatically, without filenames/hashes/local artifacts in the YAML. |

Everything else is left unpinned: SciPy, pandas, scikit-learn, scikit-image,
matplotlib, seaborn, openTSNE, UMAP, Pillow, FFmpeg, PyAV, dataframe_image, einops,
safetensors, PyYAML, packaging, psutil, tqdm, NetworkX, NLTK, TensorBoard,
TensorBoardX, h5py, WebDataset, ConfigArgParse, torch-optimizer,
coral-pytorch, conditional, cmaes, Git, setuptools, pip, and optional notebook/test/cluster
tools. Their dependency metadata selects versions compatible with Python 3.10 and
the constrained stack. No hashes, exact Conda builds, local artifacts, or absolute
paths appear in these YAML files.

Most packages, including compiled NumPy/SciPy, h5py, openTSNE, PyAV, CPU PyTorch, and
Decord where available, use Conda. GPU PyTorch uses the official wheels to keep its
CUDA/native stack independent of Conda's media-library ABI. MediaPipe has no Conda-forge distribution;
coral-pytorch and conditional are also unavailable there. MoviePy 1.x uses pip
because its older Conda recipe omits the upstream `decorator<5` compatibility
requirement. An isolated build of its pure-Python wheel succeeded; no native
compiler is needed for MoviePy. OpenCV deliberately uses
**only `opencv-contrib-python` from pip**, because MediaPipe explicitly requires it.
Installing Conda OpenCV or another OpenCV wheel too would give two providers of
`cv2` and reproduce the current environment's metadata conflict. MediaPipe's
protobuf/JAX/etc. dependencies and Hugging Face's hub/tokenizers dependencies are
resolved transitively rather than copied into the direct dependency list. Pip may
replace Conda-selected Python transitives, such as protobuf, to satisfy MediaPipe.
Always run `pip check` after adding Conda packages; if an addition restores an
incompatible transitive, reinstall the constrained pip requirements from the base
file. Create a new environment for major package/provider changes.

`cmaes` is a runtime-selected requirement: `train_model.py` loads OptunaHub's
`samplers/auto_sampler`, whose
[declared requirements](https://github.com/optuna/optunahub-registry/blob/main/package/samplers/auto_sampler/requirements.txt)
are torch, SciPy, and cmaes. The Git executable is also included because OptunaHub's
GitPython importer requires it on PATH; it is not supplied by the GitPython Python
package ([GitPython requirements](https://gitpython.readthedocs.io/en/stable/intro.html#requirements)).
FFmpeg and ffprobe are required executables used by
video-processing scripts. **Torchaudio has no live repository import** and is
excluded from the common environment. The historical VideoMAEv2 manifest declares
it; add it only if an additional upstream audio workflow needs it. Conda-forge has
no Windows torchaudio 2.5 package, so putting it in the common file prevents a
Windows solve. For the official CUDA environment, the matching optional package is
installed in the activated CUDA environment with:

```sh
python -m pip install "torchaudio==2.5.1+cu118" --index-url https://download.pytorch.org/whl/cu118
```

Linux/macOS CPU users can use Conda-forge's
matching torchaudio package. On Windows CPU, an audio workflow needs a separate
complete official PyTorch/torchvision/torchaudio stack from the installation matrix;
do not mix extension binaries from different providers.

Unrelated installed packages are excluded, including dlib, face_alignment,
TensorFlow, fastText, installed compiler metapackages, obsolete cudatoolkit,
machine-specific CUDA libraries, and all other packages that existed only in the original environment. Tests and
notebook frontends are optional. Upstream declarations without live imports
(`wandb`, `beartype`, `braceexpand`, `iopath`, `peft`, `fire`, `python-box`, `ftfy`)
are not needed by the main project and are not promoted to common requirements.
Installing an entire upstream project may require them separately.

## Platform and existing source limitations

- **Linux:** CPU dependency installation and NVIDIA binary installation are the
  primary targets. `srun`/SLURM are external cluster services, not portable Conda
  dependencies. Linux ARM is not a validated full target. The resolved MediaPipe
  0.10.21 Linux wheel requires glibc >=2.28; older systems need an older compatible
  wheel allowed by the range and a separate dependency/runtime check.
- **Windows:** the common CPU stack and separate Decord/torchsort providers have
  Windows binaries. Bash launchers require WSL2 or another Bash environment.
  Distributed NCCL and several CUDA-specific upstream training paths require Linux;
  a native Windows package installation does not make them Windows-compatible.
- **macOS:** use one architecture consistently. The common stack and Conda Decord
  have Intel and arm64 providers. Apple Silicon needs a torchsort source build for
  training. CUDA and NCCL workflows are unavailable. PyTorch's possible MPS support
  does not make this code MPS-ready: several paths explicitly choose CUDA or CPU.
  Wheel and Conda build minimum macOS versions vary; current OpenCV 4.11 wheels
  require macOS 13, while older allowed wheels can support earlier releases.
  The cross-platform solves here target macOS 13; older macOS releases need their
  own compatible resolution and are not claimed as validated targets.
- **Code paths and hardware:** `custom/helper.py` contains an absolute Linux project
  root in `GLOBAL_PATH.NAS_PATH`; additional scripts/configs contain absolute data
  or checkpoint paths. CPU packages do not override hard-coded `.cuda()`, CUDA
  defaults, distributed NCCL, or scheduler settings. Adjust supported command-line
  paths/devices where available; workflows with fixed paths/devices remain limited
  until the application is changed. No source was changed by this audit.
- **Legacy code:** MAE_DFER and making_better_mistakes still contain `np.float` /
  `np.bool`, removed in NumPy 1.24. Legacy optimizer factories import removed timm
  classes (`Nadam`, `NovoGrad`, `NvNovoGrad`). The historical manifests request
  mutually incompatible old Python/PyTorch/timm/CUDA versions. A single modern
  environment cannot guarantee those entire upstream training workflows unchanged;
  the main custom backbone path does not import those optimizer factories. Imports
  are accounted for, but those APIs need a separate legacy setup or future code work.
- **Upstream V-JEPA 2 package:** its `setup.py` requires Python >=3.11, while the
  main application imports the vendored source on Python 3.10. Do not run
  `pip install -e vjepa2` in this environment. A separately packaged upstream
  workflow needs its own Python version and dependency validation.
- **Multiprocessing:** `custom/helper.py` creates a `multiprocessing.Manager()`
  during module import. Windows and macOS use the spawn start method by default;
  loading a script again in a spawned child can restart that manager and fail
  during process bootstrapping. Guarded entrypoints and side-effect-free imports
  are required by [Python's spawn rules](https://docs.python.org/3.10/library/multiprocessing.html#safe-importing-of-main-module).
  Native training startup on those platforms remains unvalidated and may need
  source changes that are outside this environment-only task. Restricted sandboxes
  can also block the manager's local socket even on Linux.
- **External assets:** datasets, model checkpoints, MediaPipe `.task`/`.tflite`
  files, NLTK corpora, Hugging Face cache/revisions, credentials, and the dynamically
  fetched OptunaHub sampler revision are not supplied by environment creation.
  Petrel storage needs its original client and configuration. The historical
  `hierarchies` source is unspecified. Neither can be safely inferred from an import.
- **Video/GUI/reporting:** codecs vary by OS/build. Confirm FFmpeg has `libx264`.
  OpenCV/MediaPipe wheels may require host display/OpenGL libraries on Linux.
  Minimal Debian/Ubuntu systems may need the host `libgl1` and `libglib2.0-0`
  packages before importing the OpenCV wheel; use the corresponding OS packages
  on other Linux distributions.
  MoviePy 1.x TextClip requires ImageMagick and available fonts (the scripts request
  Arial); install/configure the platform's ImageMagick executable as described in
  [MoviePy 1.x installation](https://zulko.github.io/moviepy/v1.0.3/install.html).
  dataframe_image's default HTML exporter needs a supported browser executable.
  Browser binaries, fonts, GPU drivers, and OS GUI libraries are not Python imports.

## Validation and checks after installation

The dependency cross-check covers every third-party import in Python files and
notebook code cells, including ignored/untracked source and bundled upstream code.
`DEPENDENCIES.md` records each provider or explicit platform/optional/unresolved
requirement. The current environment has 219 distinct pip-visible distribution
names; its complete exported package list was not used as the portable manifest.

The source audit also reads shell scripts, dependency manifests, YAML/TOML and
saved experiment configurations, checks runtime-selected modules and executables,
and compares the original Conda/pip installation. `test.ipynb` has a pre-existing
incomplete `fps =` cell; imports were recovered from that cell without editing it.

Installation validation results are recorded in [`VALIDATION.md`](VALIDATION.md). A Conda dry run
does not install packages or validate pip dependencies; a cross-platform solve on
Linux does not prove runtime behavior on Windows or macOS. Full training and access
to external datasets/checkpoints have not been exercised.

After adding the native dependencies, run:

```sh
python -m pip check
python -c "import torch, torchvision, numpy, scipy, cv2, mediapipe, transformers, timm, decord, torchsort; from mediapipe.tasks.python.vision import FaceAligner, FaceAlignerOptions, FaceLandmarksConnections; print(torch.__version__, torchvision.__version__, numpy.__version__, cv2.__version__); print(torchsort.soft_rank(torch.tensor([[3.,1.,2.]])))"
python -c "import custom.backbone, custom.dataset, custom.faceExtractor, custom.loss"
ffmpeg -version
ffprobe -version
git --version
ffmpeg -hide_banner -encoders
```

For the CUDA setup also run a CUDA soft-rank operation:

```sh
python -c "import torch, torchsort; print(torchsort.soft_rank(torch.tensor([[3.,1.,2.]], device='cuda')))"
```

If the common and CUDA lists are later changed, keep their common dependencies
aligned and repeat both Conda and pip validation. Once a portable setup works on a
new machine, capture a separate reproducibility lock for that machine/experiment.
