from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from new_plot_tsne_post_head import run_tsne_and_plot


@pytest.mark.parametrize('reduction_method', ('tsne', 'umap'))
def test_explicit_plot_name_is_prefixed_with_model_and_dataset(
  tmp_path, reduction_method
):
  csv_path = tmp_path / 'samples.tsv'
  csv_path.write_text(
    'sample_id\tsubject_id\tsample_name\n'
    '0\t10\ta\n'
    '1\t10\tb\n'
    '2\t11\tc\n'
    '3\t11\td\n',
    encoding='utf-8',
  )
  data = {
    'model_pth_path': '/models/checkpoint.pt',
    'csv_path': str(csv_path),
    'video_embeddings': {
      'sample_ids': [0, 1, 2, 3],
      'labels': [0, 0, 1, 1],
    },
  }
  reduced = np.array([[0.0, 0.0], [0.1, 0.2], [1.0, 1.0], [1.1, 1.2]])

  run_tsne_and_plot(
    pkl_file=data,
    group_by='labels',
    cmap='viridis',
    png_output_name=str(tmp_path / 'plot.png'),
    reduced_embeddings=(reduced, np.arange(4)),
    reduction_method=reduction_method,
    log_path_folder=str(tmp_path),
  )

  assert (tmp_path / 'checkpoint_samples.tsv_plot.png').is_file()
