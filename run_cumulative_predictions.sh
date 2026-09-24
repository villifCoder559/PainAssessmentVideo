#!/usr/bin/env bash

usage() {
  echo "Usage: $0 --dataset {Biovid|UNBC|MINT} SAMPLE_ID [SAMPLE_ID ...] [--origin {target|source}] [--stage {1|2|4}] [--compare-native] [--video [SPEED]]" >&2
  echo "       $0 --dataset {Biovid|UNBC|MINT} --pick_rand_samples K [--origin {target|source}] [--stage {1|2|4}] [--compare-native] [--video [SPEED]]" >&2
  exit 2
}

if (( $# < 3 )) || [[ $1 != --dataset ]]; then
  usage
fi

dataset=$2
case "$dataset" in
  Biovid)
    max_sample_id=8700
    source_native_root="BIOVID_5FOLD_VideoMAEv2-S_FULL_1145473_ATTENTIVE_JEPA_lannister_1782457644/1782457647800_VIDEOMAE_v2_S_MEAN_SPATIAL_NONE_SLIDING_WINDOW_ATTENTIVE_JEPA"
    target_native_root="BIOVID_5FOLDcomplete_DFER_FULL/history_run_BIO_5_keep002_restartSched_FINAL_1070021_ATTENTIVE_JEPA_lannister_1782406974/1782406978358_DFER_MEAN_SPATIAL_NONE_SLIDING_WINDOW_ATTENTIVE_JEPA"
    ;;
  UNBC)
    max_sample_id=200
    source_native_root="UNBC_OPI_0to4_VIDEOMAE-S/history_run_UNBC_5FOLD_std_load/history_run_UNBC_5FOLD_S_batch16_1333517_ATTENTIVE_JEPA_targaryen_1781520074/1781520077852_VIDEOMAE_v2_S_MEAN_SPATIAL_NONE_SLIDING_WINDOW_ATTENTIVE_JEPA"
    target_native_root="UNBC_DFER_v2/1783348663339_DFER_MEAN_SPATIAL_NONE_SLIDING_WINDOW_ATTENTIVE_JEPA"
    ;;
  MINT)
    max_sample_id=3122
    source_native_root="MIntPAIN_VMAE-S_5_fold/1784112336728_VIDEOMAE_v2_S_MEAN_SPATIAL_NONE_SLIDING_WINDOW_ATTENTIVE_JEPA"
    target_native_root="MIntPAIN_DFER_5_fold/1783539017352_DFER_MEAN_SPATIAL_NONE_SLIDING_WINDOW_ATTENTIVE_JEPA"
    ;;
  *) usage ;;
esac

sample_value_in_range() {
  local value=$1
  [[ $value =~ ^[1-9][0-9]*$ ]] || return 1
  (( ${#value} <= ${#max_sample_id} )) || return 1
  (( 10#$value <= max_sample_id ))
}

shift 2
if [[ $1 == --pick_rand_samples ]]; then
  (( $# >= 2 )) || usage
  sample_count=$2
  sample_value_in_range "$sample_count" || usage
  mapfile -t sample_ids < <(
    shuf -i "1-$max_sample_id" -n "$sample_count"
  )
  echo "Selected sample IDs: ${sample_ids[*]}"
  shift 2
else
  sample_ids=()
  declare -A seen_sample_ids=()
  while (( $# > 0 )) && [[ $1 != --* ]]; do
    sample_value_in_range "$1" || usage
    if [[ ! ${seen_sample_ids[$1]+set} ]]; then
      sample_ids+=("$1")
      seen_sample_ids[$1]=1
    fi
    shift
  done
  (( ${#sample_ids[@]} > 0 )) || usage
fi

origin=source
stage=2
compare_native=0
origin_set=0
stage_set=0
video_set=0
video_arguments=()
while (( $# > 0 )); do
  case "$1" in
    --origin)
      (( origin_set == 0 && $# >= 2 )) || usage
      [[ $2 == target || $2 == source ]] || usage
      origin=$2
      origin_set=1
      shift 2
      ;;
    --stage)
      (( stage_set == 0 && $# >= 2 )) || usage
      [[ $2 == 1 || $2 == 2 || $2 == 4 ]] || usage
      stage=$2
      stage_set=1
      shift 2
      ;;
    --compare-native)
      (( compare_native == 0 )) || usage
      compare_native=1
      shift
      ;;
    --video)
      (( video_set == 0 )) || usage
      video_set=1
      video_arguments=(--video)
      shift
      if (( $# > 0 )) && [[ $1 != --* ]]; then
        video_arguments+=("$1")
        shift
      fi
      ;;
    *) usage ;;
  esac
done
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir" || exit 2

passed=0
failed=()

run_plot() {
  local sample_id=$1
  local experiment=$2
  shift 2

  echo "Running $experiment for sample $sample_id"
  if python3 "$script_dir/plot_cumulative_predictions.py" \
      "$experiment" "$sample_id" "$@" "${video_arguments[@]}"; then
    ((passed += 1))
  else
    failed+=("$sample_id: $experiment")
  fi
}

run_cross() {
  local native_args=()
  if (( compare_native )); then
    if [[ $origin == source ]]; then
      native_args=(--compare-native-root "$target_native_root")
    else
      native_args=(--compare-native-root "$source_native_root")
    fi
  elif [[ $origin == source ]]; then
    native_args=(--target-native-root "$target_native_root")
  fi
  run_plot "$1" "$2" --origin "$origin" --refinement-stage "$stage" "${native_args[@]}"
}

run_sample() {
  local sample_id=$1

  if (( compare_native )); then
    run_plot "$sample_id" "$source_native_root" --compare-native-root "$target_native_root"
  else
    run_plot "$sample_id" "$source_native_root"
    run_plot "$sample_id" "$target_native_root"
  fi

  case "$origin:$dataset" in
    target:Biovid)
      run_cross "$sample_id" "Cross_projection/unbcVmae_to_biovidDfer/refinement3_linear_cross-validation"
      run_cross "$sample_id" "Cross_projection/unbcVmae_to_biovidDfer/refinement3_mlp_cross-validation"
      run_cross "$sample_id" "Cross_projection/mintVMAE-bioDFER/refinement3_linear_cross-validation"
      run_cross "$sample_id" "Cross_projection/mintVMAE-bioDFER/refinement3_mlp_cross-validation"
      ;;
    target:UNBC)
      run_cross "$sample_id" "Cross_projection/cross-validation_BioVmae-unbcDFER_v2/refinement3_linear_100"
      run_cross "$sample_id" "Cross_projection/cross-validation_BioVmae-unbcDFER_v2/refinement3_mlp_100"
      run_cross "$sample_id" "Cross_projection/mintVMAE-unbcDFER/refinement3_mlp_cross-validation"
      run_cross "$sample_id" "Cross_projection/mintVMAE-unbcDFER/refinement3_linear_cross-validation"
      ;;
    target:MINT)
      run_cross "$sample_id" "Cross_projection/bioVmae_to_mintDfer/refinement3_linear_cross-validation"
      run_cross "$sample_id" "Cross_projection/bioVmae_to_mintDfer/refinement3_mlp_cross-validation"
      run_cross "$sample_id" "Cross_projection/unbcVMAE-mintDFER/refinement3_linear_cross-validation"
      run_cross "$sample_id" "Cross_projection/unbcVMAE-mintDFER/refinement3_mlp_cross-validation"
      ;;
    source:Biovid)
      run_cross "$sample_id" "Cross_projection/cross-validation_BioVmae-unbcDFER_v2/refinement3_linear_100"
      run_cross "$sample_id" "Cross_projection/cross-validation_BioVmae-unbcDFER_v2/refinement3_mlp_100"
      run_cross "$sample_id" "Cross_projection/bioVmae_to_mintDfer/refinement3_linear_cross-validation"
      run_cross "$sample_id" "Cross_projection/bioVmae_to_mintDfer/refinement3_mlp_cross-validation"
      ;;
    source:UNBC)
      run_cross "$sample_id" "Cross_projection/unbcVmae_to_biovidDfer/refinement3_linear_cross-validation"
      run_cross "$sample_id" "Cross_projection/unbcVmae_to_biovidDfer/refinement3_mlp_cross-validation"
      run_cross "$sample_id" "Cross_projection/unbcVMAE-mintDFER/refinement3_linear_cross-validation"
      run_cross "$sample_id" "Cross_projection/unbcVMAE-mintDFER/refinement3_mlp_cross-validation"
      ;;
    source:MINT)
      run_cross "$sample_id" "Cross_projection/mintVMAE-bioDFER/refinement3_linear_cross-validation"
      run_cross "$sample_id" "Cross_projection/mintVMAE-bioDFER/refinement3_mlp_cross-validation"
      run_cross "$sample_id" "Cross_projection/mintVMAE-unbcDFER/refinement3_linear_cross-validation"
      run_cross "$sample_id" "Cross_projection/mintVMAE-unbcDFER/refinement3_mlp_cross-validation"
      ;;
  esac
}

for sample_id in "${sample_ids[@]}"; do
  run_sample "$sample_id"
done

echo
echo "Passed: $passed"
echo "Failed: ${#failed[@]}"

if (( ${#failed[@]} > 0 )); then
  echo "Failed experiments:"
  printf '  %s\n' "${failed[@]}"
  exit 1
fi
