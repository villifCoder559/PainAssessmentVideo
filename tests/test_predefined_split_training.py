from pathlib import Path
import sys
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeModel:
    def __init__(self):
        self.train_calls = []
        self.test_calls = []

    def train(self, **kwargs):
        self.train_calls.append(kwargs)
        with open(kwargs["train_csv_path"], "a", encoding="utf-8") as csv_file:
            csv_file.write("trained-copy-only\n")
        return {
            "dict_results": {
                "epochs": 1,
                "best_model_idx": 1,
                "best_model_state": {"weight": 42},
                "list_train_performance_metric": [0.4, 0.6],
                "list_val_performance_metric": [0.5, 0.8],
                "train_confusion_matricies": [],
                "val_confusion_matricies": [],
            },
            "count_y_train": [2],
            "count_y_val": [1],
            "count_subject_ids_train": [2],
            "count_subject_ids_val": [1],
        }

    def test_pretrained_model(self, **kwargs):
        self.test_calls.append(kwargs)
        return {
            "tested_csv": kwargs["csv_path"],
            "tested_state": kwargs["state_dict"],
        }


def _write(path, marker):
    path.write_text(marker, encoding="utf-8")
    return str(path)


def test_predefined_split_trains_and_tests_once_using_run_local_copies(tmp_path):
    from custom.predefined_training import run_predefined_split

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    train_source = _write(source_dir / "train.csv", "train-source\n")
    val_source = _write(source_dir / "val.csv", "val-source\n")
    test_source = _write(source_dir / "test.csv", "test-source\n")
    augmented_train = _write(tmp_path / "augmented_train.csv", "train-augmented\n")
    source_contents = {
        path: Path(path).read_bytes()
        for path in (train_source, val_source, test_source, augmented_train)
    }
    model = FakeModel()

    results = run_predefined_split(
        model_advanced=model,
        predefined_csv_splits={
            "train": train_source,
            "val": val_source,
            "test": test_source,
        },
        augmented_train_csv_path=augmented_train,
        train_folder_path=str(tmp_path / "run" / "train_HEAD"),
        seed_random_state=42,
        train_kwargs={
            "num_epochs": 2,
            "criterion": "criterion",
            "key_for_early_stopping": "val_accuracy",
        },
        test_kwargs={
            "criterion": "criterion",
            "concatenate_temporal": False,
        },
        reduce_logs=lambda train_result: train_result["dict_results"],
        set_seed=lambda _: None,
    )

    assert len(model.train_calls) == 1
    assert len(model.test_calls) == 1
    train_call = model.train_calls[0]
    test_call = model.test_calls[0]
    assert Path(train_call["train_csv_path"]).read_text(encoding="utf-8").startswith(
        "train-augmented\n"
    )
    assert Path(train_call["val_csv_path"]).read_text(encoding="utf-8") == "val-source\n"
    assert Path(test_call["csv_path"]).read_text(encoding="utf-8") == "test-source\n"
    assert train_call["saving_path"].endswith("k0_cross_val_sub_0")
    assert test_call["state_dict"] == {"weight": 42}
    assert all(Path(path).read_bytes() == content for path, content in source_contents.items())
    staged_source = Path(train_call["saving_path"]) / "source_train.csv"
    assert staged_source.read_text(encoding="utf-8") == "train-source\n"
    assert set(results) == {"k0_cross_val_sub_0", "k0_cross_val_final"}
    assert results["k0_cross_val_sub_0"]["test"]["tested_state"] == {"weight": 42}
    assert results["k0_cross_val_final"]["best_model"]["best_model_state"] is None


def test_run_train_test_propagates_augmentation_config_to_predefined_runner(
    tmp_path, monkeypatch
):
    import custom.scripts as scripts

    captured = {}

    class FakeAdvancedModel:
        def __init__(self, **_):
            self.dataset = SimpleNamespace(total_classes=2)
            self.T_S_S_shape = [1, 1, 1]

        def free_gpu_memory(self):
            pass

    def capture_predefined_run(**kwargs):
        captured.update(kwargs)
        return {
            "k0_cross_val_sub_0": {},
            "k0_cross_val_final": {},
        }

    monkeypatch.setattr(scripts, "Model_Advanced", FakeAdvancedModel)
    monkeypatch.setattr(scripts, "run_predefined_split", capture_predefined_run)
    monkeypatch.setattr(
        scripts.tools,
        "get_dataset_type",
        lambda _: scripts.helper.CUSTOM_DATASET_TYPE.BASE,
    )
    monkeypatch.setattr(scripts.tools, "save_dict_k_fold_results", lambda **_: None)

    enum_value = SimpleNamespace(name="NAME", value="VALUE")
    augmentation_config = {"hflip": 0.5}
    scripts.run_train_test(
        model_type=enum_value,
        pooling_embedding_reduction=enum_value,
        pooling_clips_reduction=enum_value,
        sample_frame_strategy=enum_value,
        path_csv_dataset=str(tmp_path / "train.csv"),
        path_video_dataset=str(tmp_path / "videos"),
        head=enum_value,
        stride_window_in_video=16,
        features_folder_saving_path=str(tmp_path / "features"),
        head_params={},
        k_fold=3,
        global_foder_name=str(tmp_path / "history"),
        batch_size_training=2,
        epochs=1,
        criterion="criterion",
        lr=0.001,
        seed_random_state=42,
        is_plot_dataset_distribution=False,
        is_round_output_loss=False,
        is_shuffle_video_chunks=False,
        is_shuffle_training_batch=True,
        init_network="default",
        key_for_early_stopping="val_accuracy",
        regularization_lambda_L1=0.0,
        regularization_lambda_L2=0.0,
        clip_length=16,
        target_metric_best_model="val_accuracy",
        early_stopping="early-stopping",
        concatenate_temp_dim=False,
        stop_after_kth_fold=[1, 1],
        n_workers=0,
        num_clips_per_video=1,
        clip_grad_norm=None,
        label_smooth=0.0,
        stride_inside_window=1,
        dict_augmented=augmentation_config,
        use_sdpa=False,
        prefetch_factor=2,
        soft_labels=0.0,
        adapter_dict=None,
        load_dataset_in_memory=False,
        predefined_csv_splits={
            "train": str(tmp_path / "train.csv"),
            "val": str(tmp_path / "val.csv"),
            "test": str(tmp_path / "test.csv"),
        },
    )

    assert captured["train_kwargs"]["dict_augmented"] == augmentation_config
    assert captured["test_kwargs"]["dict_augmented"] == augmentation_config
