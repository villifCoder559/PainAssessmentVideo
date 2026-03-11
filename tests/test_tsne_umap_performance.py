import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import time
import sys
import os
import shutil
import tempfile

# Add the project root to sys.path to allow importing the script
sys.path.append(os.getcwd())

# Import the module to be tested
# Using __import__ because the filename might change or have special chars, 
# but here it is standard enough.
import new_plot_tsne_post_head as target_module

class MockTensor:
    """Mock class to simulate a PyTorch tensor with .cpu().numpy() methods."""
    def __init__(self, arr):
        self.arr = arr
    def cpu(self):
        return self
    def numpy(self):
        return self.arr

class TestTsneUmap(unittest.TestCase):

    def setUp(self):
        # Create synthetic data
        self.n_samples = 3000 # Enough to be meaningful, small enough to be fast
        self.n_features = 512
        self.synthetic_embeddings = np.random.rand(self.n_samples, self.n_features).astype(np.float32)
        
        # Structure the data as expected by the script:
        # embeddings is a list of batches, each batch is a list of tensors
        # Let's make it 2 batches of n_samples/2
        batch_size = self.n_samples // 2
        batch1 = [MockTensor(self.synthetic_embeddings[i]) for i in range(batch_size)]
        batch2 = [MockTensor(self.synthetic_embeddings[i]) for i in range(batch_size, self.n_samples)]
        
        self.mock_data = {
            'video_embeddings': {
                'sample_ids': list(range(self.n_samples)),
                'embeddings': [batch1, batch2],
                'labels': np.random.randint(0, 5, size=self.n_samples).tolist()
            },
            'config_model': {
                'model_advanced_params': {
                    'features_folder_saving_path': '/tmp/mock/path'
                }
            },
            'csv_path': '/tmp/mock.csv'
        }
        
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch('new_plot_tsne_post_head.get_valid_indices')
    def test_compute_tsne_performance(self, mock_get_valid):
        """Test t-SNE computation and measure time."""
        print("\n--- Testing t-SNE ---")
        # Mock get_valid_indices to return all indices, bypassing dataset checks
        mock_get_valid.return_value = list(range(self.n_samples))
        
        start_time = time.time()
        embeddings_2d = target_module.compute_valid_tsne_embeddings(self.mock_data)
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"t-SNE execution time for {self.n_samples} samples: {duration:.4f} seconds")
        
        self.assertEqual(embeddings_2d.shape, (self.n_samples, 2))
        return duration

    @patch('new_plot_tsne_post_head.get_valid_indices')
    def test_compute_umap_performance(self, mock_get_valid):
        """Test UMAP computation and measure time."""
        print("\n--- Testing UMAP ---")
        mock_get_valid.return_value = list(range(self.n_samples))
        
        start_time = time.time()
        embeddings_2d = target_module.compute_valid_umap_embeddings(self.mock_data)
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"UMAP execution time for {self.n_samples} samples: {duration:.4f} seconds")
        
        self.assertEqual(embeddings_2d.shape, (self.n_samples, 2))
        return duration

    @patch('new_plot_tsne_post_head.get_valid_indices')
    def test_comparison(self, mock_get_valid):
        """Run both and verify they run correctly."""
        # Reuse logic slightly for direct comparison in one test
        mock_get_valid.return_value = list(range(self.n_samples))
        
        t0 = time.time()
        target_module.compute_valid_tsne_embeddings(self.mock_data)
        t_tsne = time.time() - t0
        
        t0 = time.time()
        target_module.compute_valid_umap_embeddings(self.mock_data)
        t_umap = time.time() - t0
        
        print(f"\n[Comparison] Samples: {self.n_samples}, Features: {self.n_features}")
        print(f"t-SNE: {t_tsne:.4f} s")
        print(f"UMAP : {t_umap:.4f} s")
        if t_umap < t_tsne:
            print("=> UMAP was faster.")
        else:
            print("=> t-SNE was faster.")

    @patch('new_plot_tsne_post_head.determine_labels')
    @patch('new_plot_tsne_post_head.set_cmap')
    @patch('new_plot_tsne_post_head.get_valid_indices')
    def test_run_tsne_and_plot_integration(self, mock_get_valid, mock_set_cmap, mock_determine_labels):
        """Test the full plotting pipeline with mocks to ensure no crashes."""
        print("\n--- Testing Plotting Pipeline (Integration) ---")
        mock_get_valid.return_value = list(range(self.n_samples))
        mock_determine_labels.return_value = (np.array(self.mock_data['video_embeddings']['labels']), np.array(['subj1']*self.n_samples))
        mock_set_cmap.return_value = 'viridis'
        
        # Test t-SNE path
        output_name_tsne = os.path.join(self.test_dir, 'test_plot_tsne.png')
        target_module.run_tsne_and_plot(
            pkl_file=self.mock_data,
            group_by='labels',
            cmap='viridis',
            png_output_name=output_name_tsne,
            reduction_method='tsne',
            log_path_folder=self.test_dir
        )
        self.assertTrue(os.path.exists(output_name_tsne), "t-SNE plot file was not created.")

        # Test UMAP path
        output_name_umap = os.path.join(self.test_dir, 'test_plot_umap.png')
        target_module.run_tsne_and_plot(
            pkl_file=self.mock_data,
            group_by='labels',
            cmap='viridis',
            png_output_name=output_name_umap,
            reduction_method='umap',
            log_path_folder=self.test_dir
        )
        self.assertTrue(os.path.exists(output_name_umap), "UMAP plot file was not created.")
        print("Both t-SNE and UMAP plotting pipelines executed successfully.")

if __name__ == '__main__':
    unittest.main()
