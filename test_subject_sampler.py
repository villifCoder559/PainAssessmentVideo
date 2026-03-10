"""Test SubjectBatchSampler implementation."""
import sys
import numpy as np
import pandas as pd
import custom.helper as helper

# Set step_shift manually since we're not loading from a folder
helper.step_shift = 200

from custom.dataset import SubjectBatchSampler


def make_test_df(n_subjects=10, n_levels=4, samples_per_subject_level=5, add_augmented=False):
    """Create a synthetic DataFrame mimicking the real dataset structure."""
    rows = []
    sample_id = 1
    for subj_id in range(1, n_subjects + 1):
        for level in range(n_levels):
            for _ in range(samples_per_subject_level):
                rows.append({
                    'subject_id': subj_id,
                    'subject_name': f'subj_{subj_id}',
                    'class_id': level,
                    'class_name': f'level_{level}',
                    'sample_id': sample_id,
                    'sample_name': f'sample_{sample_id}',
                })
                sample_id += 1

    if add_augmented:
        orig_rows = list(rows)
        for row in orig_rows:
            aug_row = row.copy()
            aug_row['sample_id'] = row['sample_id'] + helper.step_shift  # hflip augmentation
            aug_row['sample_name'] = f"aug_{row['sample_name']}"
            rows.append(aug_row)

    df = pd.DataFrame(rows)
    return df


def test_basic_creation():
    """Test basic sampler creation and iteration."""
    print("=" * 60)
    print("TEST 1: Basic creation and iteration")
    print("=" * 60)
    df = make_test_df(n_subjects=10, n_levels=4, samples_per_subject_level=5)
    print(f"DataFrame: {len(df)} samples, {df['subject_id'].nunique()} subjects, {df['class_id'].nunique()} levels")

    sampler = SubjectBatchSampler(
        df=df, batch_size=40, min_subjects_per_level=3,
        include_augmented=True, require_all_levels=True, shuffle=True
    )
    print(f"Number of batches: {len(sampler)}")

    batches = list(sampler)
    print(f"Yielded {len(batches)} batches")
    print(f"Batch sizes: {[len(b) for b in batches]}")

    for i, batch in enumerate(batches):
        batch_df = df.loc[batch]
        levels_in_batch = batch_df['class_id'].unique()
        print(f"\n  Batch {i}: size={len(batch)}, levels={sorted(levels_in_batch)}")
        for level in sorted(levels_in_batch):
            level_samples = batch_df[batch_df['class_id'] == level]
            unique_subjects = level_samples['subject_id'].nunique()
            print(f"    Level {level}: {len(level_samples)} samples, {unique_subjects} unique subjects")
            assert unique_subjects >= 3, f"FAIL: Level {level} has only {unique_subjects} subjects (need 3)"

    print("\nPASSED: All batches have >= 3 unique subjects per level")


def test_class_balance():
    """Test that each pain level gets equal samples."""
    print("\n" + "=" * 60)
    print("TEST 2: Class balance")
    print("=" * 60)
    df = make_test_df(n_subjects=8, n_levels=4, samples_per_subject_level=6)
    sampler = SubjectBatchSampler(
        df=df, batch_size=32, min_subjects_per_level=2, shuffle=True
    )

    for i, batch in enumerate(list(sampler)):
        batch_df = df.loc[batch]
        level_counts = batch_df['class_id'].value_counts().sort_index()
        expected = 32 // 4  # 8 per level
        for level, count in level_counts.items():
            assert count == expected, f"FAIL: Batch {i}, level {level}: got {count}, expected {expected}"
        if i == 0:
            print(f"  Batch {i} level counts: {level_counts.to_dict()}")

    print(f"PASSED: All {len(sampler)} batches have exactly {32 // 4} samples per level")


def test_epoch_variation():
    """Test that batches differ between epochs."""
    print("\n" + "=" * 60)
    print("TEST 3: Epoch variation (random_state += 13)")
    print("=" * 60)
    df = make_test_df(n_subjects=10, n_levels=4, samples_per_subject_level=8)
    sampler = SubjectBatchSampler(
        df=df, batch_size=40, min_subjects_per_level=2, shuffle=True
    )

    epoch1_batches = list(sampler)  # This also triggers re-initialization
    epoch2_batches = list(sampler)

    epoch1_first = set(epoch1_batches[0])
    epoch2_first = set(epoch2_batches[0])

    differs = epoch1_first != epoch2_first
    print(f"  Epoch 1 first batch (first 5 indices): {epoch1_batches[0][:5]}")
    print(f"  Epoch 2 first batch (first 5 indices): {epoch2_batches[0][:5]}")
    print(f"  Batches differ between epochs: {differs}")
    assert differs, "FAIL: Batches should differ between epochs"
    print("PASSED: Epochs produce different batches")


def test_include_augmented_false():
    """Test that augmented samples are excluded when include_augmented=False."""
    print("\n" + "=" * 60)
    print("TEST 4: include_augmented=False filters augmented samples")
    print("=" * 60)
    df = make_test_df(n_subjects=8, n_levels=4, samples_per_subject_level=5, add_augmented=True)
    print(f"  Total samples (with augmented): {len(df)}")
    print(f"  Original samples: {len(df[df['sample_id'] <= helper.step_shift])}")
    print(f"  Augmented samples: {len(df[df['sample_id'] > helper.step_shift])}")

    sampler = SubjectBatchSampler(
        df=df, batch_size=32, min_subjects_per_level=2,
        include_augmented=False, shuffle=True
    )

    for batch in sampler:
        batch_df = df.loc[batch]
        augmented = batch_df[batch_df['sample_id'] > helper.step_shift]
        assert len(augmented) == 0, f"FAIL: Found {len(augmented)} augmented samples in batch"

    print("PASSED: No augmented samples in any batch")


def test_include_augmented_true():
    """Test that augmented samples CAN appear when include_augmented=True."""
    print("\n" + "=" * 60)
    print("TEST 5: include_augmented=True allows augmented samples")
    print("=" * 60)
    df = make_test_df(n_subjects=8, n_levels=4, samples_per_subject_level=2, add_augmented=True)

    sampler = SubjectBatchSampler(
        df=df, batch_size=32, min_subjects_per_level=2,
        include_augmented=True, shuffle=True
    )

    all_indices = []
    for batch in sampler:
        all_indices.extend(batch)

    all_batch_df = df.loc[all_indices]
    n_augmented = len(all_batch_df[all_batch_df['sample_id'] > helper.step_shift])
    print(f"  Augmented samples across all batches: {n_augmented}")
    assert n_augmented > 0, "FAIL: Expected some augmented samples"
    print("PASSED: Augmented samples are included")


def test_validation_error():
    """Test that ValueError is raised when subjects < min_subjects_per_level."""
    print("\n" + "=" * 60)
    print("TEST 6: Validation error on insufficient subjects")
    print("=" * 60)

    # Only 2 subjects, but require 5
    df = make_test_df(n_subjects=2, n_levels=4, samples_per_subject_level=10)
    try:
        SubjectBatchSampler(
            df=df, batch_size=20, min_subjects_per_level=5, shuffle=True
        )
        print("FAIL: Should have raised ValueError")
        sys.exit(1)
    except ValueError as e:
        print(f"  Correctly raised ValueError: {e}")
    print("PASSED: ValueError raised for insufficient subjects")


def test_batch_size_too_small_error():
    """Test ValueError when batch_size // n_levels < min_subjects_per_level."""
    print("\n" + "=" * 60)
    print("TEST 7: Validation error when batch too small for constraints")
    print("=" * 60)

    df = make_test_df(n_subjects=10, n_levels=4, samples_per_subject_level=5)
    try:
        # batch_size=8, n_levels=4 -> samples_per_level=2, but min_subjects=3
        SubjectBatchSampler(
            df=df, batch_size=8, min_subjects_per_level=3, shuffle=True
        )
        print("FAIL: Should have raised ValueError")
        sys.exit(1)
    except ValueError as e:
        print(f"  Correctly raised ValueError: {e}")
    print("PASSED: ValueError raised for batch too small")


def test_all_indices_valid():
    """Test that all yielded indices are valid DataFrame indices."""
    print("\n" + "=" * 60)
    print("TEST 8: All yielded indices are valid")
    print("=" * 60)
    df = make_test_df(n_subjects=10, n_levels=4, samples_per_subject_level=5)
    sampler = SubjectBatchSampler(
        df=df, batch_size=40, min_subjects_per_level=2, shuffle=True
    )

    valid_indices = set(df.index)
    for batch in sampler:
        for idx in batch:
            assert idx in valid_indices, f"FAIL: Index {idx} not in DataFrame"

    print("PASSED: All indices are valid")


if __name__ == '__main__':
    test_basic_creation()
    test_class_balance()
    test_epoch_variation()
    test_include_augmented_false()
    test_include_augmented_true()
    test_validation_error()
    test_batch_size_too_small_error()
    test_all_indices_valid()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
