#!/usr/bin/env bash
# example usage: ./run_cross_space_configs.sh 0 Cross_projection_yaml/config_pain_from_bioVmae_to_mintDfer_cross_validation
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 GPU_ID CONFIG_DIR" >&2
  exit 2
fi

gpu=$1
config_arg=$2
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if [[ ! $gpu =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must be a non-negative integer: $gpu" >&2
  exit 2
fi

if [[ $config_arg = /* ]]; then
  config_dir=$config_arg
else
  config_dir=$script_dir/$config_arg
fi

if [[ ! -d $config_dir ]]; then
  echo "Config directory not found: $config_dir" >&2
  exit 2
fi

shopt -s nullglob
configs=()
for config in "$config_dir"/*.yaml; do
  [[ -f $config ]] && configs+=("$config")
done
if (( ${#configs[@]} == 0 )); then
  echo "No .yaml configs found in: $config_dir" >&2
  exit 2
fi

cd "$script_dir" || exit 2
passed=0
failed=()

for config in "${configs[@]}"; do
  echo "Running $(basename "$config") on GPU $gpu"
  if CUDA_VISIBLE_DEVICES="$gpu" python3 "$script_dir/cross_space_projection.py" --config "$config"; then
    ((passed += 1))
  else
    failed+=("$config")
  fi
done

echo
echo "Passed: $passed"
echo "Failed: ${#failed[@]}"

if (( ${#failed[@]} > 0 )); then
  echo "Failed configs:"
  for config in "${failed[@]}"; do
    printf '  %s\n' "${config##*/}"
  done
  exit 1
fi
