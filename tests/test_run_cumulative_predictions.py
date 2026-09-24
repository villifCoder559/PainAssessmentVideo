import os
import shlex
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "run_cumulative_predictions.sh"
PLOTTER = REPO_ROOT / "plot_cumulative_predictions.py"

REFINEMENT_ARGS = ("--origin", "source", "--refinement-stage", "2")
BIOVID_DFER = "BIOVID_5FOLDcomplete_DFER_FULL/history_run_BIO_5_keep002_restartSched_FINAL_1070021_ATTENTIVE_JEPA_lannister_1782406974/1782406978358_DFER_MEAN_SPATIAL_NONE_SLIDING_WINDOW_ATTENTIVE_JEPA"
UNBC_DFER = "UNBC_DFER_v2/1783348663339_DFER_MEAN_SPATIAL_NONE_SLIDING_WINDOW_ATTENTIVE_JEPA"
MINT_DFER = "MIntPAIN_DFER_5_fold/1783539017352_DFER_MEAN_SPATIAL_NONE_SLIDING_WINDOW_ATTENTIVE_JEPA"
EXPECTED_EXPERIMENTS = {
    "Biovid": [
        ("BIOVID_5FOLD_VideoMAEv2-S_FULL_1145473_ATTENTIVE_JEPA_lannister_1782457644/1782457647800_VIDEOMAE_v2_S_MEAN_SPATIAL_NONE_SLIDING_WINDOW_ATTENTIVE_JEPA",),
        (BIOVID_DFER,),
        ("Cross_projection/cross-validation_BioVmae-unbcDFER_v2/refinement3_linear_100", *REFINEMENT_ARGS, "--target-native-root", BIOVID_DFER),
        ("Cross_projection/cross-validation_BioVmae-unbcDFER_v2/refinement3_mlp_100", *REFINEMENT_ARGS, "--target-native-root", BIOVID_DFER),
        ("Cross_projection/bioVmae_to_mintDfer/refinement3_linear_cross-validation", *REFINEMENT_ARGS, "--target-native-root", BIOVID_DFER),
        ("Cross_projection/bioVmae_to_mintDfer/refinement3_mlp_cross-validation", *REFINEMENT_ARGS, "--target-native-root", BIOVID_DFER),
    ],
    "UNBC": [
        ("UNBC_OPI_0to4_VIDEOMAE-S/history_run_UNBC_5FOLD_std_load/history_run_UNBC_5FOLD_S_batch16_1333517_ATTENTIVE_JEPA_targaryen_1781520074/1781520077852_VIDEOMAE_v2_S_MEAN_SPATIAL_NONE_SLIDING_WINDOW_ATTENTIVE_JEPA",),
        (UNBC_DFER,),
        ("Cross_projection/unbcVmae_to_biovidDfer/refinement3_linear_cross-validation", *REFINEMENT_ARGS, "--target-native-root", UNBC_DFER),
        ("Cross_projection/unbcVmae_to_biovidDfer/refinement3_mlp_cross-validation", *REFINEMENT_ARGS, "--target-native-root", UNBC_DFER),
        ("Cross_projection/unbcVMAE-mintDFER/refinement3_linear_cross-validation", *REFINEMENT_ARGS, "--target-native-root", UNBC_DFER),
        ("Cross_projection/unbcVMAE-mintDFER/refinement3_mlp_cross-validation", *REFINEMENT_ARGS, "--target-native-root", UNBC_DFER),
    ],
    "MINT": [
        ("MIntPAIN_VMAE-S_5_fold/1784112336728_VIDEOMAE_v2_S_MEAN_SPATIAL_NONE_SLIDING_WINDOW_ATTENTIVE_JEPA",),
        (MINT_DFER,),
        ("Cross_projection/mintVMAE-bioDFER/refinement3_linear_cross-validation", *REFINEMENT_ARGS, "--target-native-root", MINT_DFER),
        ("Cross_projection/mintVMAE-bioDFER/refinement3_mlp_cross-validation", *REFINEMENT_ARGS, "--target-native-root", MINT_DFER),
        ("Cross_projection/mintVMAE-unbcDFER/refinement3_linear_cross-validation", *REFINEMENT_ARGS, "--target-native-root", MINT_DFER),
        ("Cross_projection/mintVMAE-unbcDFER/refinement3_mlp_cross-validation", *REFINEMENT_ARGS, "--target-native-root", MINT_DFER),
    ],
}

SOURCE_CROSS_EXPERIMENTS = {
    "Biovid": {
        "Cross_projection/cross-validation_BioVmae-unbcDFER_v2/refinement3_linear_100",
        "Cross_projection/cross-validation_BioVmae-unbcDFER_v2/refinement3_mlp_100",
        "Cross_projection/bioVmae_to_mintDfer/refinement3_linear_cross-validation",
        "Cross_projection/bioVmae_to_mintDfer/refinement3_mlp_cross-validation",
    },
    "UNBC": {
        "Cross_projection/unbcVmae_to_biovidDfer/refinement3_linear_cross-validation",
        "Cross_projection/unbcVmae_to_biovidDfer/refinement3_mlp_cross-validation",
        "Cross_projection/unbcVMAE-mintDFER/refinement3_linear_cross-validation",
        "Cross_projection/unbcVMAE-mintDFER/refinement3_mlp_cross-validation",
    },
    "MINT": {
        "Cross_projection/mintVMAE-bioDFER/refinement3_linear_cross-validation",
        "Cross_projection/mintVMAE-bioDFER/refinement3_mlp_cross-validation",
        "Cross_projection/mintVMAE-unbcDFER/refinement3_linear_cross-validation",
        "Cross_projection/mintVMAE-unbcDFER/refinement3_mlp_cross-validation",
    },
}


def fake_environment(tmp_path, python_body="exit 0"):
    command_dir = tmp_path / "bin"
    command_dir.mkdir()
    call_log = tmp_path / "calls"
    fake_python = command_dir / "python3"
    fake_python.write_text(
        '#!/usr/bin/env bash\n'
        'printf "%q " "$@" >> "$CALL_LOG"\n'
        'printf "\\n" >> "$CALL_LOG"\n'
        f"{python_body}\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = os.environ.copy()
    environment["PATH"] = f"{command_dir}:{environment['PATH']}"
    environment["CALL_LOG"] = str(call_log)
    return environment, call_log


def logged_calls(call_log):
    return [
        shlex.split(line)
        for line in call_log.read_text(encoding="utf-8").splitlines()
    ]


@pytest.mark.parametrize("dataset", EXPECTED_EXPERIMENTS)
def test_dataset_runs_its_six_active_experiments_in_order(tmp_path, dataset):
    environment, call_log = fake_environment(tmp_path)

    result = subprocess.run(
        ["bash", str(RUNNER), "--dataset", dataset, "42"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    calls = logged_calls(call_log)
    assert len(calls) == 6
    assert [tuple(call[1:]) for call in calls] == [
        (experiment, "42", *arguments)
        for experiment, *arguments in EXPECTED_EXPERIMENTS[dataset]
    ]
    assert all(Path(call[0]) == PLOTTER for call in calls)
    assert "Passed: 6" in result.stdout
    assert "Failed: 0" in result.stdout


@pytest.mark.parametrize("dataset", EXPECTED_EXPERIMENTS)
def test_compare_native_mode_replaces_two_native_runs_with_one_pair(tmp_path, dataset):
    environment, call_log = fake_environment(tmp_path)

    result = subprocess.run(
        ["bash", str(RUNNER), "--dataset", dataset, "42", "--compare-native",
         "--video"],
        cwd=tmp_path, env=environment, text=True, capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    calls = logged_calls(call_log)
    expected = EXPECTED_EXPERIMENTS[dataset]
    assert len(calls) == 5
    assert calls[0][1:] == [
        expected[0][0], "42", "--compare-native-root", expected[1][0], "--video",
    ]
    for call, (experiment, *arguments) in zip(calls[1:], expected[2:]):
        assert call[1:3] == [experiment, "42"]
        assert call[3:7] == ["--origin", "source", "--refinement-stage", "2"]
        assert call[7:] == [
            "--compare-native-root", expected[1][0], "--video",
        ]
    assert "Passed: 5" in result.stdout


def test_compare_native_mode_uses_videomae_root_for_target_origin(tmp_path):
    environment, call_log = fake_environment(tmp_path)

    result = subprocess.run(
        ["bash", str(RUNNER), "--dataset", "Biovid", "42", "--origin",
         "target", "--compare-native"],
        cwd=tmp_path, env=environment, text=True, capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    calls = logged_calls(call_log)
    assert len(calls) == 5
    assert calls[0][1:] == [
        EXPECTED_EXPERIMENTS["Biovid"][0][0], "42",
        "--compare-native-root", BIOVID_DFER,
    ]
    assert all(
        call[3:] == [
            "--origin", "target", "--refinement-stage", "2",
            "--compare-native-root", EXPECTED_EXPERIMENTS["Biovid"][0][0],
        ]
        for call in calls[1:]
    )


@pytest.mark.parametrize("stage", ["1", "2", "4"])
def test_stage_is_forwarded_only_to_cross_experiments(tmp_path, stage):
    environment, call_log = fake_environment(tmp_path)

    result = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "--dataset",
            "UNBC",
            "42",
            "--stage",
            stage,
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    calls = logged_calls(call_log)
    assert all(call[3:] == [] for call in calls[:2])
    assert all(
        call[3:] == ["--origin", "source", "--refinement-stage", stage,
                     "--target-native-root", UNBC_DFER]
        for call in calls[2:]
    )


@pytest.mark.parametrize("dataset", SOURCE_CROSS_EXPERIMENTS)
def test_source_origin_runs_the_four_outgoing_cross_experiments(tmp_path, dataset):
    environment, call_log = fake_environment(tmp_path)

    result = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "--dataset",
            dataset,
            "42",
            "--origin",
            "source",
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    calls = logged_calls(call_log)
    assert len(calls) == 6
    assert [tuple(call[1:]) for call in calls[:2]] == [
        (experiment, "42", *arguments)
        for experiment, *arguments in EXPECTED_EXPERIMENTS[dataset][:2]
    ]
    assert {call[1] for call in calls[2:]} == SOURCE_CROSS_EXPERIMENTS[dataset]
    assert all(
        call[3:] == ["--origin", "source", "--refinement-stage", "2",
                     "--target-native-root", EXPECTED_EXPERIMENTS[dataset][1][0]]
        for call in calls[2:]
    )


@pytest.mark.parametrize(
    ("dataset", "sample_id"),
    [("Biovid", "8700"), ("UNBC", "200"), ("MINT", "3122")],
)
def test_source_origin_direct_sample_uses_the_selected_dataset(tmp_path, dataset, sample_id):
    environment, call_log = fake_environment(tmp_path)

    result = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "--dataset",
            dataset,
            sample_id,
            "--origin",
            "source",
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert {call[2] for call in logged_calls(call_log)} == {sample_id}


def test_manual_samples_run_once_in_input_order_with_options(tmp_path):
    environment, call_log = fake_environment(tmp_path)

    result = subprocess.run(
        ["bash", str(RUNNER), "--dataset", "UNBC", "12", "34", "12",
         "--stage", "4", "--video", "0.5"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    calls = logged_calls(call_log)
    assert [call[2] for call in calls] == ["12"] * 6 + ["34"] * 6
    assert all(call[-2:] == ["--video", "0.5"] for call in calls)
    assert all("--refinement-stage" not in call for call in calls[:2] + calls[6:8])
    assert all(call[call.index("--refinement-stage") + 1] == "4"
               for call in calls[2:6] + calls[8:])
    assert "Passed: 12" in result.stdout


def test_invalid_later_manual_sample_prevents_all_runs(tmp_path):
    environment, call_log = fake_environment(tmp_path)

    result = subprocess.run(
        ["bash", str(RUNNER), "--dataset", "UNBC", "12", "201", "34"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "Usage:" in result.stderr
    assert not call_log.exists()


@pytest.mark.parametrize(
    "video_arguments",
    [("--video",), ("--video", "0.5"), ("--video", "2")],
)
def test_forwards_video_arguments_to_every_experiment(tmp_path, video_arguments):
    environment, call_log = fake_environment(tmp_path)

    result = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "--dataset",
            "UNBC",
            "42",
            *video_arguments,
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert all(
        tuple(call[-len(video_arguments):]) == video_arguments
        for call in logged_calls(call_log)
    )


@pytest.mark.parametrize(
    ("trailing_arguments", "video_arguments", "origin", "stage"),
    [
        (("--origin", "source", "--stage", "1", "--video", "0.5"), ("--video", "0.5"), "source", "1"),
        (("--video", "2", "--stage", "4", "--origin", "source"), ("--video", "2"), "source", "4"),
        (("--stage", "1", "--video", "--origin", "target"), ("--video",), "target", "1"),
    ],
)
def test_stage_origin_and_video_accept_any_trailing_order(
    tmp_path, trailing_arguments, video_arguments, origin, stage
):
    environment, call_log = fake_environment(tmp_path)

    result = subprocess.run(
        ["bash", str(RUNNER), "--dataset", "UNBC", "42", *trailing_arguments],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    calls = logged_calls(call_log)
    assert all(tuple(call[3:]) == video_arguments for call in calls[:2])
    assert all(
        tuple(call[3:])
        == ("--origin", origin, "--refinement-stage", stage,
            *(("--target-native-root", UNBC_DFER) if origin == "source" else ()),
            *video_arguments)
        for call in calls[2:]
    )


@pytest.mark.parametrize(
    ("dataset", "maximum"),
    [("Biovid", 8700), ("UNBC", 200), ("MINT", 3122)],
)
def test_random_mode_runs_unique_samples_in_the_dataset_range(
    tmp_path, dataset, maximum
):
    environment, call_log = fake_environment(tmp_path)

    result = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "--dataset",
            dataset,
            "--pick_rand_samples",
            "4",
            "--origin",
            "source",
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    calls = logged_calls(call_log)
    sample_ids = [int(call[2]) for call in calls[::6]]
    assert len(calls) == 24
    assert len(sample_ids) == len(set(sample_ids)) == 4
    assert all(1 <= sample_id <= maximum for sample_id in sample_ids)
    assert f"Selected sample IDs: {' '.join(map(str, sample_ids))}" in result.stdout


@pytest.mark.parametrize("sample_count", ["1", "200"])
def test_random_mode_accepts_dataset_size_boundaries(tmp_path, sample_count):
    environment, call_log = fake_environment(tmp_path)

    result = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "--dataset",
            "UNBC",
            "--pick_rand_samples",
            sample_count,
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    calls = logged_calls(call_log)
    assert len(calls) == int(sample_count) * 6
    assert len({call[2] for call in calls}) == int(sample_count)


def test_rejects_invalid_arguments_before_running_experiments(tmp_path):
    environment, call_log = fake_environment(tmp_path)
    invalid_arguments = [
        [],
        ["--dataset"],
        ["--dataset", "Unknown", "42"],
        ["--dataset", "UNBC"],
        ["--dataset", "UNBC", "0"],
        ["--dataset", "UNBC", "-1"],
        ["--dataset", "UNBC", "1.5"],
        ["--dataset", "UNBC", "many"],
        ["--dataset", "UNBC", "18446744073709551617"],
        ["--dataset", "Biovid", "8701"],
        ["--dataset", "UNBC", "201"],
        ["--dataset", "MINT", "3123"],
        ["--dataset", "UNBC", "--video"],
        ["--dataset", "UNBC", "--pick_rand_samples"],
        ["--dataset", "UNBC", "--pick_rand_samples", "0"],
        ["--dataset", "UNBC", "--pick_rand_samples", "-1"],
        ["--dataset", "UNBC", "--pick_rand_samples", "1.5"],
        ["--dataset", "UNBC", "--pick_rand_samples", "many"],
        ["--dataset", "UNBC", "--pick_rand_samples", "18446744073709551617"],
        ["--dataset", "UNBC", "--pick_rand_samples", "201"],
        ["--dataset", "UNBC", "--pick_rand_samples", "2", "42"],
        ["--dataset", "UNBC", "42", "extra"],
        ["--dataset", "UNBC", "42", "--video", "1", "extra"],
        ["--dataset", "UNBC", "42", "--video", "--unknown"],
        ["--dataset", "UNBC", "42", "--origin"],
        ["--dataset", "UNBC", "42", "--origin", "native"],
        ["--dataset", "UNBC", "42", "--origin", "source", "--origin", "target"],
        ["--dataset", "UNBC", "42", "--stage"],
        ["--dataset", "UNBC", "42", "--stage", "3"],
        ["--dataset", "UNBC", "42", "--stage", "two"],
        ["--dataset", "UNBC", "42", "--stage", "1", "--stage", "2"],
        ["--dataset", "UNBC", "42", "--video", "--video"],
    ]

    for arguments in invalid_arguments:
        result = subprocess.run(
            ["bash", str(RUNNER), *arguments],
            cwd=tmp_path,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode == 2, (arguments, result.stdout, result.stderr)
        assert "Usage:" in result.stderr
        assert not call_log.exists()


def test_random_mode_continues_after_failures_and_lists_sample_and_experiment(
    tmp_path,
):
    environment, call_log = fake_environment(
        tmp_path,
        '[[ "$2" != *unbcVmae_to_biovidDfer/refinement3_mlp_cross-validation ]]',
    )
    fake_shuf = Path(environment["PATH"].split(os.pathsep)[0]) / "shuf"
    fake_shuf.write_text("#!/usr/bin/env bash\nprintf '11\\n12\\n'\n", encoding="utf-8")
    fake_shuf.chmod(0o755)

    result = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "--dataset",
            "UNBC",
            "--pick_rand_samples",
            "2",
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert len(logged_calls(call_log)) == 12
    assert "Passed: 10" in result.stdout
    assert "Failed: 2" in result.stdout
    assert "11: Cross_projection/unbcVmae_to_biovidDfer/refinement3_mlp_cross-validation" in result.stdout
    assert "12: Cross_projection/unbcVmae_to_biovidDfer/refinement3_mlp_cross-validation" in result.stdout


def test_stage_four_failures_continue_and_return_status_one(tmp_path):
    environment, call_log = fake_environment(
        tmp_path,
        '[[ " $* " != *" --refinement-stage 4 "* ]]',
    )

    result = subprocess.run(
        ["bash", str(RUNNER), "--dataset", "UNBC", "42", "--stage", "4"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert len(logged_calls(call_log)) == 6
    assert "Passed: 2" in result.stdout
    assert "Failed: 4" in result.stdout
