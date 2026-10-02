# Reproducible environment snapshot

Use this folder to recreate the original **Linux x86_64** installation with its
exact Conda builds, hashed pip packages, and preserved torchsort wheel. The captured
stack uses **Python 3.10.11**, **PyTorch 2.5.1**, and **CUDA runtime 11.8**.

For installation across different operating systems or hardware, use
[portability](../env_portability/README.md).

## Files

| File | Purpose |
| --- | --- |
| [environment.yml](environment.yml) | Readable pinned specification, including candidates for known missing direct dependencies. |
| [locks/conda-linux-64.lock](locks/conda-linux-64.lock) | Exact Conda versions, builds, artifact URLs, and hashes. |
| [locks/pip-linux-64.lock](locks/pip-linux-64.lock) | Hashed pip overlays, including the preserved local torchsort wheel. |
| [locks/artifacts/](locks/artifacts/) | Original custom native-extension wheel; retain this directory. |
| [locks/environment-snapshot.json](locks/environment-snapshot.json) | Captured platform, package metadata, binary hashes, and validation evidence. |
| [locks/python-distributions.lock](locks/python-distributions.lock) | Distribution inventory; not a pip requirements file. |
| [locks/dependency-audit.json](locks/dependency-audit.json) | Repository dependency audit. |
| [environments/verify_snapshot.py](environments/verify_snapshot.py) | Compare an installation with the captured snapshot. |
| [environments/optional-requirements.in](environments/optional-requirements.in) | Optional upstream workflow requirements, outside the exact lock. |

## Exact recreation

Start at the **repository root**, with Conda initialized. Create a fresh environment
on a compatible Linux x86_64 machine:

```bash
cd env_reproducibility
conda create --name pain-exact --no-default-packages --file locks/conda-linux-64.lock
conda run -n pain-exact python -m pip install --no-deps --ignore-installed --only-binary=:all: --require-hashes -r locks/pip-linux-64.lock
conda run -n pain-exact python environments/verify_snapshot.py
conda activate pain-exact
cd ..
```

Install Conda packages first, then apply the pip overlay with the flags shown.
The pip lock references its wheel relative to this folder, so run the installation
from `env_reproducibility/`. Return to the repository root before running project code.

## Scope and limitations

The exact snapshot preserves the original installation's unused packages, missing
dependencies, and metadata inconsistencies. The readable YAML includes additional
dependencies and permits a new transitive resolution, so it does not recreate the
exact snapshot; its separate installation procedure is documented in
[ENVIRONMENT.md](ENVIRONMENT.md#readable-recreation-on-another-compatible-machine).

These locks are not portable Windows/macOS specifications. GPU execution needs a
compatible NVIDIA GPU and host driver; the environment does not install the driver.
Package downloads must remain available or cached. Datasets, checkpoints, private
storage configuration, and other external assets are outside the snapshot.
Exact package recreation alone does not guarantee identical training results.

See [ENVIRONMENT.md](ENVIRONMENT.md) for the captured hardware/platform, installation
alternatives, missing dependencies, and validation details.
