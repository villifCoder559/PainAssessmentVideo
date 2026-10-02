# Reproducing the project environment

Audited on 2026-10-01 against the active `update_torch_project` environment.

This original setup now lives entirely under `env_reproducibility/`. Relative paths in
commands below are relative to that directory. Its lock files, snapshot, original
environment YAML, verifier, and wheel were moved without changing their contents.
For a portable installation instead, see [`../env_portability/ENVIRONMENT.md`](../env_portability/ENVIRONMENT.md).
Application and research code were left unchanged. The readable specification adds
known missing imports; the exact snapshot faithfully preserves the inspected
installation, including its existing inconsistencies and unused packages.

## Captured platform and ML stack

| Component | Observed configuration |
| --- | --- |
| OS / architecture | Ubuntu 22.04.4 LTS, Linux x86_64 (`linux-64`) |
| Kernel / glibc | 5.15.0-177-generic / glibc 2.35 |
| Python | **3.10.11**, Conda build `h955ad1f_3` |
| Conda / solver | 25.3.1 / libmamba, conda-libmamba-solver 25.4.0 |
| PyTorch | **2.5.1**, `py3.10_cuda11.8_cudnn9.1.0_0`, `pytorch` channel |
| torchvision / torchaudio | **0.20.1 / 2.5.1**, both `py310_cu118` |
| PyTorch CUDA / cuDNN | **11.8 / 9.1.0** (`torch.backends.cudnn.version() == 90100`) |
| Conda CUDA selector | `pytorch-cuda=11.8=h7e8668a_6`, CUDA mutex |
| Other CUDA installation | Legacy Conda `cudatoolkit=11.3.1`; kept only in the exact snapshot |
| Host CUDA compiler | `/usr/bin/nvcc`, CUDA **11.5.119**; OS `nvidia-cuda-toolkit=11.5.1-1ubuntu1` |
| GPU / driver | **2 x NVIDIA GeForce RTX 2080 Ti**, driver **535.247.01**; NVIDIA-SMI reports driver CUDA support 12.2 |
| PyTorch GPU visibility | Host probe: `torch.cuda.is_available() == True`, device count **2**, compute capability **7.5** |
| NumPy / BLAS | **1.26.4**, current Conda build uses MKL; MKL 2023.1.0 |
| OpenCV / FFmpeg | **4.10.0 / 4.2.2**, Conda providers; OpenCV uses Qt5 and has no CUDA build configuration |
| MediaPipe / protobuf | **0.10.5 / 3.20.3** |
| transformers / huggingface_hub | **4.57.1 / 0.36.0** |
| tokenizers / safetensors | **0.22.1 / 0.4.5 at runtime**; overlapping Conda safetensors 0.5.3 metadata also exists |
| timm / torchmetrics | **1.0.15 / 1.5.2** |
| Custom native extension | `torchsort=0.1.10+pt25cu118`, CPython 3.10, Linux x86_64 |
| Other snapshot native tools | dlib 19.24.0 is CPU-only; PyAV 13.1.0, decord 0.6.0, Triton 3.1.0 |

The PyTorch/vision/audio/CUDA combination matches the
[official PyTorch 2.5.1 installation matrix](https://pytorch.org/get-started/previous-versions/).
cuDNN is bundled in the PyTorch build, so a separate `cudnn` package is not needed.
CUDA 11.8 and cuDNN 9.1 support Linux drivers >=450.80.02 under minor-version
compatibility, with feature limitations on older drivers. A driver supporting CUDA
11.8 natively (>=520.61.05), or a newer compatible driver and supported NVIDIA GPU,
is the preferable target. Consult the
[NVIDIA CUDA 11.8 release notes](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-toolkit-release-notes/index.html),
[CUDA compatibility guide](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html),
and [cuDNN 9.1 support matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.1.0/reference/support-matrix.html).
The initial sandbox probe could not access the NVIDIA driver. Host-level probing
confirmed GPU access; the sandbox result does not describe the host environment.

The system compiler version, PyTorch runtime version, and driver support are distinct.
NVIDIA-SMI's CUDA 12.2 value is the driver's supported API level, while PyTorch uses
CUDA 11.8 and the installed system nvcc is 11.5.119.
Prebuilt packages use the captured CUDA 11.8 runtime without requiring system nvcc.
Rebuilding torchsort, Apex, or DeepSpeed CUDA extensions needs an appropriate CUDA
11.8 development toolkit and compatible compiler; the detected host nvcc is 11.5.
The preserved torchsort wheel avoids that build. Its original SHA256 is
`7b04df98becaed066a101801a73bcd068256afc398b5144fce3a53e801c128d7`.
[torchsort's upstream wheel instructions](https://github.com/teddykoker/torchsort#pre-built-wheels)
explain the Python/PyTorch/CUDA tags.

## Files and their roles

- `environment.yml`: readable, pinned direct imports and runtime tools, plus a few
  explicit ML compatibility anchors (including MKL and CUDA support libraries). Eight missing imports have separately resolved
  pins; they are clearly marked and are absent from the current snapshot.
- `locks/conda-linux-64.lock`: all **338 Conda packages**, including transitive native
  libraries, exact versions, builds, original channel URLs, and MD5 artifact hashes.
- `locks/pip-linux-64.lock`: all **56 pip-managed overlays**, including transitives and
  pip replacements of Conda packages, exact versions and SHA256 wheel hashes.
- `locks/artifacts/torchsort-0.1.10+pt25cu118-cp310-cp310-linux_x86_64.whl`: original
  local wheel, preserved unchanged so another computer does not need the original path.
- `locks/environment-snapshot.json`: platform, virtual packages, complete Conda and
  Python metadata, original wheel tags/provenance, chosen pip wheel hashes, native
  binary hashes, runtime configuration, and validation evidence.
- `locks/python-distributions.lock`: inventory of **226 distribution metadata records**
  (219 distinct names). This is an inventory, not a pip installation file: it includes
  Conda-managed distributions, duplicate metadata, and Conda's OpenCV aliases.
- `locks/dependency-audit.json`: complete import occurrences, package-name mappings,
  missing dependencies, source hashes, notebook issues, and configuration inspection.
- `environments/optional-requirements.in`: additional bundled workflow declarations,
  including DeepSpeed and V-JEPA 2 extras; these were not installed or locked.
- `environments/verify_snapshot.py`: repeatable package/build/native-library comparison.
- `docs/superpowers/plans/2026-10-01-environment-reproducibility.md`: completed audit plan.

`conda-lock` is unavailable in the active environment and on PATH. An explicit Conda
snapshot plus a hashed pip overlay is appropriate here: solving a new dependency
graph would change the mixed installation and lose its custom wheel/overlapping
metadata. These are `linux-64` locks, not a multi-platform solver lock. Conda documents
[recreating environments from explicit specifications](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html).

## Exact recreation on the same platform

Run from the `env_reproducibility/` directory (`cd env_reproducibility` from the cloned
repository root), retaining `locks/artifacts/`. Use a fresh name
and a working Conda installation. Channel availability and internet access are needed
unless the artifacts have already been cached.

```bash
conda create --name pain-exact --no-default-packages --file locks/conda-linux-64.lock
conda run -n pain-exact python -m pip install --no-deps --ignore-installed --only-binary=:all: --require-hashes -r locks/pip-linux-64.lock
conda run -n pain-exact python environments/verify_snapshot.py
conda activate pain-exact
```

Conda must run first. `--no-deps` prevents pip from resolving a different PyTorch,
NumPy, or OpenCV stack. `--ignore-installed` applies the recorded pip files over
Conda packages and retains their overlapping version metadata, matching the current
installation. Every pip entry is hashed; the source-only optional dependencies are
outside this lock. Do not use the Python distribution inventory as pip requirements.

The readable channels are `pytorch`, `nvidia`, `conda-forge`, and `defaults`. The exact
snapshot additionally contains `1adrianb` for the installed but unimported
`face_alignment` package. Explicit URLs preserve each package's actual channel,
regardless of the receiving computer's channel-priority settings.

## Readable recreation on another compatible machine

For another Linux x86_64 machine with a compatible NVIDIA GPU/driver and glibc >=2.35,
the exact commands above give the closest match. Use the following commands when you
want to resolve the readable specification and fill its known missing dependencies:

```bash
PIP_NO_DEPS=1 CONDA_CHANNEL_PRIORITY=flexible conda env create --file environment.yml
conda run -n pain-assessment-video python -m pip install --no-deps --ignore-installed --only-binary=:all: --require-hashes -r locks/pip-linux-64.lock
conda activate pain-assessment-video
```

The second step supplies the currently resolved pip transitives and compatibility
overlays. `PIP_NO_DEPS=1` retains Conda's OpenCV provider during the first step.
Conda resolves the missing NLTK, TensorBoardX, submitit, h5py, webdataset, and
ConfigArgParse versions; conditional and tree-format are separately pinned pip imports.
The readable path includes these new packages and can select different transitive
Conda builds. It is not an exact reproduction of the incomplete captured environment,
so the strict snapshot verifier is intended for `pain-exact`.

Install optional workflow requirements only in a separate environment selected for
that workflow. Their compatible versions remain unverified. In particular, installing
`vjepa2/` as a package requires Python >=3.11 according to its `setup.py`; its README
uses Python 3.12. The main project currently imports the vendored source on Python
3.10.11, and those core imports passed, but that does not validate every upstream
training, robotic-control, or notebook workflow.

## Missing, ambiguous, and historical dependencies

The installed environment has **12 missing third-party import roots**. Eight have
unambiguous providers now pinned in `environment.yml`: `nltk`, `tensorboardX`
(`tensorboardx`), `submitit`, `h5py`, `webdataset`, `configargparse`, `conditional`, and
`tree_format` (`tree-format`). The six Conda pins came from a successful dry-run solve;
the two pip pins came from downloaded package metadata. These are candidate versions
for previously unavailable workflows, not observations of installed packages.

The other four are documented outside the default environment:

| Import | Requirement and limitation |
| --- | --- |
| `deepspeed` | Missing. Unconditional in `VideoMAEv2/run_class_finetuning.py`; conditional in MAE_DFER. Listed in optional requirements. CUDA/toolchain and PyTorch compatibility need a separate validation. |
| `apex` | Missing, guarded import for fused optimizers. Requires NVIDIA Apex selected from its upstream source; the unrelated PyPI `apex` package is unsuitable. No installed revision exists to pin. |
| `petrel_client` | Missing, guarded optional object-storage backend in VideoMAEv2. Needs the selected Petrel client source, configuration, and storage credentials. Local video loading can use its fallback. |
| `hierarchies` | Missing historical dependency of the ImageNet hierarchy-generation scripts under `making_better_mistakes/data/scripts_asis/`. No source/revision is specified. Their README says these scripts are supplied “as is” and are unnecessary for training. |

JEPA/V-JEPA 2 manifests also declare absent `beartype`, `braceexpand`, `iopath`,
`peft`, `fire`, `python-box`, `ftfy`, and `jupyter`. These declared extras are retained
in `environments/optional-requirements.in`. Its development manifest additionally
declares absent black 26.3.1, flake8 7.0.0, and isort 5.13.2; these are recorded as
optional tooling. The active environment includes ipykernel
and IPython, but no notebook-server distribution. Launching notebooks requires a
Jupyter frontend in a suitable environment and registering this kernel:

```bash
python -m ipykernel install --user --name pain-exact --display-name 'Pain assessment (exact)'
```

Historical manifests conflict with the current installation: MAE_DFER describes
Python 3.8, PyTorch 1.7.1, and timm 0.4.12; VideoMAEv2 pins timm 0.4.12,
TensorBoard 2.9.0, TensorBoardX 1.8, and Triton 1.0.0; making_better_mistakes requests
Python 3.6, PyTorch 1.1, CUDA 9.0, TensorFlow, fastText, and git-lfs.
Those pins were not merged into the current setup. TensorFlow and fastText have no
live imports in the repository. Some legacy timm APIs may still be incompatible:
core model imports passed, while complete legacy training was not exercised.

The pip/Conda name mismatches and full import coverage are recorded below. Standard
library imports and local packages such as `custom`, `jepa`, `vjepa2`, `MAE_DFER`,
`VideoMAEv2`, `better_mistakes`, `analysis`, and their local modules are excluded from
third-party requirements. All configured dynamic model imports resolve to inspected
vendored modules.

| Import | Package / provider | Current version | Specification |
| --- | --- | --- | --- |
| `PIL` | `pillow` | 11.0.0 (conda) | conda |
| `apex` | `NVIDIA Apex` | MISSING | ENVIRONMENT.md:unresolved optional source |
| `av` | `av` | 13.1.0 (pip) | pip |
| `conditional` | `conditional` | MISSING | pip |
| `configargparse` | `configargparse` | MISSING | conda |
| `coral_pytorch` | `coral-pytorch` | 1.4.0 (pip) | pip |
| `cv2` | `opencv` | 4.10.0 (conda) | conda |
| `dataframe_image` | `dataframe_image` | 0.1.1 (conda) | conda |
| `decord` | `decord` | 0.6.0 (pip) | pip |
| `deepspeed` | `deepspeed` | MISSING | environments/optional-requirements.in |
| `einops` | `einops` | 0.8.1 (conda) | conda |
| `h5py` | `h5py` | MISSING | conda |
| `hierarchies` | `unresolved historical source` | MISSING | ENVIRONMENT.md:unresolved optional source |
| `matplotlib` | `matplotlib` | 3.9.2 (conda) | conda |
| `mediapipe` | `mediapipe` | 0.10.5 (pip) | pip |
| `moviepy` | `moviepy` | 1.0.3 (conda) | conda |
| `networkx` | `networkx` | 3.4.2 (conda) | conda |
| `nltk` | `nltk` | MISSING | conda |
| `numpy` | `numpy` | 1.26.4 (conda) | conda |
| `openTSNE` | `opentsne` | 1.0.2 (conda) | conda |
| `optuna` | `optuna` | 4.3.0 (pip) | pip |
| `optunahub` | `optunahub` | 0.3.1 (pip) | pip |
| `packaging` | `packaging` | 24.2 (conda) | conda |
| `pandas` | `pandas` | 2.2.2 (conda) | conda |
| `petrel_client` | `petrel-client` | MISSING | ENVIRONMENT.md:unresolved optional source |
| `psutil` | `psutil` | 6.1.0 (conda) | conda |
| `pytest` | `pytest` | 9.0.2 (pip) | pip |
| `safetensors` | `safetensors` | 0.5.3 (conda), 0.4.5 (pip) | pip |
| `scipy` | `scipy` | 1.14.1 (conda) | conda |
| `seaborn` | `seaborn` | 0.13.2 (conda) | conda |
| `setuptools` | `setuptools` | 75.1.0 () | conda |
| `skimage` | `scikit-image` | 0.25.2 (conda) | conda |
| `sklearn` | `scikit-learn` | 1.6.0 (conda) | conda |
| `submitit` | `submitit` | MISSING | conda |
| `tensorboardX` | `tensorboardx` | MISSING | conda |
| `timm` | `timm` | 1.0.15 (conda) | conda |
| `torch` | `pytorch` | 2.5.1 (conda) | conda |
| `torch_optimizer` | `torch-optimizer` | 0.3.0 (pip) | pip |
| `torchmetrics` | `torchmetrics` | 1.5.2 (conda) | conda |
| `torchsort` | `torchsort` | 0.1.10+pt25cu118 (pip) | pip |
| `torchvision` | `torchvision` | 0.20.1 () | conda |
| `tqdm` | `tqdm` | 4.67.1 (pip), 4.67.1 (conda) | conda |
| `transformers` | `transformers` | 4.57.1 (conda) | conda |
| `tree_format` | `tree-format` | MISSING | pip |
| `umap` | `umap-learn` | 0.5.11 (conda) | conda |
| `webdataset` | `webdataset` | MISSING | conda |
| `yaml` | `pyyaml` | 6.0.2 (conda) | conda |

## Existing inconsistencies preserved by the exact snapshot

`python -m pip check` reports that MediaPipe requires `opencv-contrib-python`, whose
pip distribution is absent. The actual `cv2` provider is Conda OpenCV 4.10.0, and
MediaPipe imports successfully. Installing another OpenCV wheel over it would alter
the captured native setup. This metadata requirement remains unsatisfied deliberately.

The same check reports decord 0.6.0 as unsupported on this platform. The downloaded
Linux wheel's filename is compatible with Python 3, but its embedded WHEEL metadata
advertises CPython 3.6. Its runtime import succeeds on Python 3.10.11; the snapshot
retains the published artifact and records the tag discrepancy.

Seven distribution names have overlapping metadata: beautifulsoup4, filelock,
PySocks, safetensors, soupsieve, tqdm, and typing_extensions. Notable Conda-to-pip
version overlays are beautifulsoup4 4.12.3 -> 4.13.5, filelock 3.18.0 -> 3.16.1,
safetensors 0.5.3 -> 0.4.5, soupsieve 2.5 -> 2.8, and typing_extensions 4.12.2 ->
4.15.0. The snapshot records both providers; the pip lock captures the pip-managed
versions. Optuna's actual installed version is 4.3.0, differing from the old top-level
manifest's 4.5.0 pin.

## Platform and project limitations

The exact Conda packages and custom torchsort wheel target Linux x86_64. The pinned
pip wheel selection includes cryptography's `manylinux_2_34` artifact; the audited
host uses glibc 2.35. Use glibc >=2.35 for the closest compatible target. macOS,
Windows-native, Linux ARM, CPU-only, and newer GPU architectures requiring a different
CUDA runtime need their own specification, native wheels, validation, and new lock.
WSL2 on x86_64 can be a Linux target if its GPU driver/runtime and system libraries
meet the same requirements. No untested lock for another OS was invented.

Qt/OpenCV GUI functions require a display server; headless processing can use the
existing Conda provider. FFmpeg and ffprobe are Conda executables and H.264 libx264
encoding was detected. The snapshot also captures GCC/OpenMP/MKL/LLVM/Qt/HDF5 and
codec libraries pulled in transitively. Host NVIDIA drivers, kernel, display server,
and OS libraries are not Conda packages; host libgl1 1.4.0-1 and libglib2.0-0
2.72.4-0ubuntu2.3 were detected. The OS compiler metapackages are GCC/G++ 11.2.
Distributed launch scripts also require a configured SLURM installation (`srun`),
multiple accessible GPUs, and adjusted scheduler/host settings where used.

Environment files do not contain datasets, checkpoints, Hugging Face cache contents,
NLTK corpora, credentials, or generated features. Preserve the model assets under
`landmark_model/`, the selected `.task`/`.tflite` files, model checkpoints, and dataset
splits used by an experiment. Hugging Face revisions and input data must be recorded
per experiment for numerical reproducibility. The code's `GLOBAL_PATH.NAS_PATH` is
hard-coded to `/equilibrium/fvilli/PainAssessmentVideo`; configure the receiving
filesystem or use supported absolute paths accordingly. Some scripts/configurations
have additional host-specific paths. No such application code was changed.

On a fresh Linux machine, a filesystem alias can retain the existing root path
without editing source. If that path does not already exist, run from the repository:

```bash
sudo mkdir -p /equilibrium/fvilli
sudo ln -sT -- "$PWD" /equilibrium/fvilli/PainAssessmentVideo
```

The alias does not supply external datasets or checkpoints.

## Validation and repeatable checks

The repository scan includes tracked, untracked, and ignored source files: **331 Python
files and four notebooks**, with **2,825 import occurrences** and **47 third-party
import roots**. An additional **4,005 configuration/script/documentation files** were
read, including saved experiment configurations. Tracked/visible JSON data mappings
and tooling configurations were also inspected. Every YAML configuration parsed.
Shell tools and configuration-selected dynamic imports were inspected. Notebook
`test.ipynb` cell index 36 contains an existing incomplete `fps =` statement; imports
in unparsable cells were recovered line by line. The other Python sources parsed.

The Conda lock was cross-checked against every active Conda record. The hashed pip
lock was dry-run against downloaded artifacts with dependency resolution disabled;
all 56 overlays were accepted. **All 98 checked native pip library files match the
downloaded wheels byte-for-byte**, including the actual safetensors 0.4.5 extension.
The retained torchsort wheel matches its original installation hash.

The final readable specification also passed a Conda dry-run solve, selecting
Python 3.10.11, the CUDA 11.8/cuDNN 9.1 PyTorch build, the observed MKL and CUDA
support-library versions, and all six newly pinned Conda dependencies. The explicit Conda specification was accepted in an offline
CLI dry run. `environments/verify_snapshot.py` passed against the live environment.

CPU checks passed for PyTorch tensor operations, torchvision, torchaudio, NumPy,
OpenCV, dlib, MediaPipe, transformers, timm, decord, torchsort ranking, and the core
`custom.backbone`, `custom.dataset`, and `custom.faceExtractor` imports. Host CUDA tensor operations, torchsort ranking, random-number generation, and a
cuDNN-enabled convolution were also checked. Full training, dataset access, notebook
execution, and optional legacy workflows were not validated. Conda dry-run checks
resolve Conda dependencies only; they do not execute the pip subsection. No active environment was changed and no complete second
environment was installed.

To repeat the core runtime check after recreation, return to the repository root
first (`cd ..` if you are still inside `env_reproducibility/`):

```bash
python - <<'PY'
import cv2, numpy, torch, torchvision, torchaudio, transformers, mediapipe, decord
import torchsort
from safetensors import __version__ as safetensors_version
print('torch / vision / audio:', torch.__version__, torchvision.__version__, torchaudio.__version__)
print('CUDA / cuDNN / GPU:', torch.version.cuda, torch.backends.cudnn.version(), torch.cuda.is_available())
print('NumPy / OpenCV / transformers:', numpy.__version__, cv2.__version__, transformers.__version__)
print('safetensors:', safetensors_version)
assert torchsort.soft_rank(torch.tensor([[3., 1., 2.]])).tolist() == [[3., 1., 2.]]
PY
python -m pip check
```

Expect the two documented pip-check findings in an exact recreation. When an NVIDIA
GPU is available, also run `nvidia-smi` and a CUDA operation with
`python -c "import torch; print(torch.ones(1, device='cuda'))"`.
