# Project environment setup

Environment files are separated by purpose:

- **[Portability](env_portability/ENVIRONMENT.md):** Python 3.10, direct dependencies,
  flexible compatibility constraints, CPU setup, and optional NVIDIA CUDA setup.
  Start with [`env_portability/environment.yml`](env_portability/environment.yml) and add
  the native dependencies for your platform using the documented commands.
- **[Reproducibility](env_reproducibility/ENVIRONMENT.md):** the original Linux x86_64
  environment, exact Conda/pip locks, audit snapshot, and preserved custom wheel.
  Run its installation commands from inside `env_reproducibility/`.

The portable setup supports package installation across operating systems; existing
hard-coded paths, CUDA-only source paths, native-extension availability, and old
upstream APIs still limit which workflows run unchanged. See the portability guide
for platform commands and validation limits. Project source code was not changed.
