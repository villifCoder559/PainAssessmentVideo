#!/usr/bin/env bash

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 GPU_ID" >&2
  exit 2
fi

gpu=$1
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../.." && pwd)
anchor_runner=$repo_root/run_cross_space_anchor_sweep.sh

if [[ ! $gpu =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must be a non-negative integer: $gpu" >&2
  exit 2
fi

if [[ ! -x $anchor_runner ]]; then
  echo "Anchor-sweep runner not found or not executable: $anchor_runner" >&2
  exit 2
fi

shopt -s nullglob
configs=("$script_dir"/*.yaml)
if (( ${#configs[@]} == 0 )); then
  echo "No .yaml configs found in: $script_dir" >&2
  exit 2
fi

passed=0
failed=()

for config in "${configs[@]}"; do
  echo "Running $(basename "$config") on GPU $gpu"
  if "$anchor_runner" "$gpu" "$config"; then
    ((passed += 1))
  else
    failed+=("$config")
  fi
done

echo
echo "Passed configs: $passed"
echo "Failed configs: ${#failed[@]}"

if (( ${#failed[@]} > 0 )); then
  echo "Failed configs:"
  for config in "${failed[@]}"; do
    printf '  %s\n' "${config##*/}"
  done
  exit 1
fi

