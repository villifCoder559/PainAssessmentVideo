#!/usr/bin/env bash

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 GPU_ID CONFIG_YAML" >&2
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
  config_path=$config_arg
else
  config_path=$script_dir/$config_arg
fi

if [[ ! -f $config_path ]]; then
  echo "Config YAML not found: $config_path" >&2
  exit 2
fi

temp_dir=$(mktemp -d)
cleanup() {
  rm -rf -- "$temp_dir"
}
trap cleanup EXIT

manifest=$temp_dir/manifest.tsv
if ! python3 - "$config_path" "$temp_dir" > "$manifest" <<'PY'
import copy
import sys
from pathlib import Path

import yaml


config_path = Path(sys.argv[1])
output_dir = Path(sys.argv[2])

try:
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
except yaml.YAMLError as exc:
    raise SystemExit(f"Invalid YAML in {config_path}: {exc}") from exc

if not isinstance(config, dict):
    raise SystemExit(f"Config YAML must contain a top-level mapping: {config_path}")

anchors = config.get("num_anchors")
if not isinstance(anchors, list) or not anchors:
    raise SystemExit("num_anchors must be a non-empty YAML list of integers")
if any(isinstance(value, bool) or not isinstance(value, int) for value in anchors):
    raise SystemExit("num_anchors must be a non-empty YAML list of integers")

unique_anchors = list(dict.fromkeys(anchors))
base_tag = config.get("run_tag")
if base_tag is None or base_tag == "":
    base_tag = config_path.stem
elif not isinstance(base_tag, str):
    raise SystemExit("run_tag must be a string when provided")
base_tag = base_tag.rstrip("/")

for index, num_anchors in enumerate(unique_anchors):
    run_config = copy.deepcopy(config)
    run_config["num_anchors"] = [num_anchors]
    run_config["run_tag"] = f"{base_tag}/K{num_anchors}"
    run_path = output_dir / f"config_{index}_K{num_anchors}.yaml"
    with run_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(run_config, stream, sort_keys=False)
    print(f"{num_anchors}\t{run_path}")
PY
then
  exit 2
fi

cd "$script_dir" || exit 2
passed=0
failed=()

while IFS=$'\t' read -r num_anchors run_config; do
  echo "Running num_anchors=$num_anchors on GPU $gpu"
  if CUDA_VISIBLE_DEVICES="$gpu" python3 "$script_dir/cross_space_projection.py" --config "$run_config"; then
    ((passed += 1))
  else
    failed+=("$num_anchors")
  fi
done < "$manifest"

echo
echo "Passed anchor values: $passed"
echo "Failed anchor values: ${#failed[@]}"

if (( ${#failed[@]} > 0 )); then
  echo "Failed num_anchors values:"
  for num_anchors in "${failed[@]}"; do
    printf '  %s\n' "$num_anchors"
  done
  exit 1
fi
