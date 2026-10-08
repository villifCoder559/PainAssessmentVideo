# Portable environments

Use this folder to install the project on another computer. The specifications
use **Python 3.10**, direct project dependencies, and compatibility ranges rather
than exact Conda builds. Conda packages come from Conda-forge; pip supplies packages
that are unavailable or unsuitable there.

On Linux x86_64, start with the **pinned** files: they are the tested environments. Use the
flexible files only as a fallback, or on Windows/macOS.

## Choose a specification

| File | Purpose |
| --- | --- |
| [environment.yml](environment.yml) | Common CPU environment for Linux x86_64, Windows x86_64, and macOS. |
| [environment-native.yml](environment-native.yml) | Required Decord/torchsort additions for CPU training on Linux x86_64 and Intel macOS. |
| [environment-windows.yml](environment-windows.yml) | Required Decord/torchsort additions for Windows x86_64 CPU training. |
| [environment-cuda.yml](environment-cuda.yml) | Standalone NVIDIA environment for Linux/Windows x86_64, using PyTorch 2.5.1 and CUDA runtime 11.8. |
| [environment-notebooks.yml](environment-notebooks.yml) | Optional notebook frontend, kernel, and test tooling. |
| [environment-cluster.yml](environment-cluster.yml) | Optional POSIX/SLURM launcher tooling. |
| [environment-pinned-linux-64.yml](environment-pinned-linux-64.yml) | Exact versions of the clean-room CPU environment that passed `smoke_test.sh` (Linux x86_64, standalone). |
| [environment-cuda-pinned-linux-64.yml](environment-cuda-pinned-linux-64.yml) | Exact versions of the clean-room CUDA 11.8 environment that passed `smoke_test.sh` (Linux x86_64, standalone, includes Decord and torchsort). |

The common file alone is **insufficient for training**: install the appropriate
native additions. Overlay files update an existing environment; they are not
standalone specifications. Apple Silicon needs a torchsort source build, which
remains unvalidated; follow the [macOS instructions](ENVIRONMENT.md#installation).

## Quick start

Run from the **repository root**, with Conda initialized, using a fresh environment.

Linux x86_64 with an NVIDIA GPU (recommended; needs a driver ≥ 520 and a GPU with compute
capability ≤ 9.0):

```sh
conda env create -f env_portability/environment-cuda-pinned-linux-64.yml
conda activate pain-portable-cuda
python -m pip check
python -c "import torch, torchsort; print(torch.cuda.is_available(), torchsort.soft_rank(torch.tensor([[3.,1.,2.]], device='cuda')))"
```

Linux x86_64 CPU only:

```sh
conda env create -f env_portability/environment-pinned-linux-64.yml
conda activate pain-portable
python -m pip check
```

Then follow [README.md](../README.md) §2–§4 (weights, data, smoke test).

### Fallback: flexible specifications

Use these if a pinned file does not resolve on your machine, or on Windows/macOS. They resolve
current compatible versions, which may differ from the tested ones.

Linux x86_64 or Intel macOS CPU:

```sh
conda env create -f env_portability/environment.yml
conda env update -n pain-portable -f env_portability/environment-native.yml
conda activate pain-portable
python -m pip check
```

Windows x86_64 CPU, in a Conda-enabled terminal:

```powershell
conda env create -f env_portability/environment.yml
conda env update -n pain-portable -f env_portability/environment-windows.yml
conda activate pain-portable
python -m pip check
```

NVIDIA GPU, flexible alternative:

```sh
conda env create -f env_portability/environment-cuda.yml
conda activate pain-portable-cuda
```

Then complete the required Decord and matching torchsort installation in the
[GPU instructions](ENVIRONMENT.md#nvidia-gpu-installation). Do not apply a CPU
native overlay to this environment. macOS cannot use the CUDA specification.

## Documentation and validation

- [ENVIRONMENT.md](ENVIRONMENT.md): full installation commands, optional tooling,
  compatibility constraints, external assets, and platform limitations.
- [DEPENDENCIES.md](DEPENDENCIES.md): third-party imports and their package providers.
- [VALIDATION.md](VALIDATION.md): completed checks and their limits.
- [dependency-audit.json](dependency-audit.json): detailed source/import audit and
  package-resolution summaries.

Package-resolution checks passed for the documented CPU and GPU targets. On Linux x86_64
the CPU and CUDA environments were also built from scratch and passed `smoke_test.sh`; their
exact exports are `environment-pinned-linux-64.yml` and `environment-cuda-pinned-linux-64.yml`
(see [VALIDATION.md](VALIDATION.md)). Fixed source paths,
CUDA-only workflows, legacy APIs, and native-extension availability still limit
portability. Flexible resolution also does not guarantee identical numerical results.
