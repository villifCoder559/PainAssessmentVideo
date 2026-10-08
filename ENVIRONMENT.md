# Project environment setup

All environment files are in [env_portability/](env_portability/README.md). Run the
commands from the repository root.

- **Linux x86_64, NVIDIA GPU (recommended):**
  `conda env create -f env_portability/environment-cuda-pinned-linux-64.yml`. This needs an NVIDIA
  driver ≥ 520 and a GPU with compute capability ≤ 9.0.
- **Linux x86_64, CPU only:** `conda env create -f env_portability/environment-pinned-linux-64.yml`.
- **Fallback, and Windows/macOS:** use the flexible specifications (`environment.yml` + the native
  additions for your platform, or `environment-cuda.yml` + the Decord/torchsort steps). They are
  described in [env_portability/ENVIRONMENT.md](env_portability/ENVIRONMENT.md).

The two pinned files are exact exports of clean-room installs that passed `smoke_test.sh` (see
[env_portability/VALIDATION.md](env_portability/VALIDATION.md)). The full walkthrough is in
[README.md](README.md) §1–§4: environment, backbone weights, data layout, and smoke test,
including how to check that the GPU was really used.

The portable setup supports installing the packages on other operating systems. Existing
hard-coded paths, CUDA-only source paths, native-extension availability and old upstream APIs
still limit which workflows run unchanged. Small source changes for CPU/path portability are
listed in [PORTABILITY_REPORT.md](PORTABILITY_REPORT.md).
