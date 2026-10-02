# Portable environments

Use this folder to install the project on another computer. The specifications
use **Python 3.10**, direct project dependencies, and compatibility ranges rather
than exact Conda builds. Conda packages come from Conda-forge; pip supplies packages
that are unavailable or unsuitable there.

For the captured original Linux environment and exact package locks, use
[reproducibility](../env_reproducibility/README.md).

## Choose a specification

| File | Purpose |
| --- | --- |
| [environment.yml](environment.yml) | Common CPU environment for Linux x86_64, Windows x86_64, and macOS. |
| [environment-native.yml](environment-native.yml) | Required Decord/torchsort additions for CPU training on Linux x86_64 and Intel macOS. |
| [environment-windows.yml](environment-windows.yml) | Required Decord/torchsort additions for Windows x86_64 CPU training. |
| [environment-cuda.yml](environment-cuda.yml) | Standalone NVIDIA environment for Linux/Windows x86_64, using PyTorch 2.5.1 and CUDA runtime 11.8. |
| [environment-notebooks.yml](environment-notebooks.yml) | Optional notebook frontend, kernel, and test tooling. |
| [environment-cluster.yml](environment-cluster.yml) | Optional POSIX/SLURM launcher tooling. |

The common file alone is **insufficient for training**: install the appropriate
native additions. Overlay files update an existing environment; they are not
standalone specifications. Apple Silicon needs a torchsort source build, which
remains unvalidated; follow the [macOS instructions](ENVIRONMENT.md#installation).

## Quick start

Run from the **repository root**, with Conda initialized, using a fresh environment.

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

For NVIDIA GPUs, create the standalone alternative:

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

Package-resolution checks passed for the documented CPU and GPU targets; runtime
execution of the newly resolved environments was not tested. Fixed source paths,
CUDA-only workflows, legacy APIs, and native-extension availability still limit
portability. Flexible resolution also does not guarantee identical numerical results.
