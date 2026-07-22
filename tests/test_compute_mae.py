import unittest
import torch
import numpy as np
import sys
import os

# Add root path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from new_plot_res_from_server import compute_mae_per_class

class TestComputeMAE(unittest.TestCase):
    def test_simple_case_row_mae(self):
        print("\nTesting Simple 3x3 Row-wise MAE...")
        # Rows: True Class 0, 1, 2
        # Cols: Pred Class 0, 1, 2
        cm = torch.tensor([
            [10, 0, 0],  # Class 0: All correct. MAE = 0
            [0, 5, 5],   # Class 1: 5 correct, 5 pred as 2. Dist |1-2|=1. MAE = (5*0 + 5*1)/10 = 0.5
            [2, 0, 8]    # Class 2: 8 correct, 2 pred as 0. Dist |2-0|=2. MAE = (8*0 + 2*2)/10 = 0.4
        ])
        
        mae = compute_mae_per_class(cm)
        
        self.assertEqual(mae[0], 0.0)
        self.assertEqual(mae[1], 0.5)
        self.assertAlmostEqual(mae[2], 0.4, places=5)
        print("✅ Simple Row-wise MAE correct.")

    def test_empty_row(self):
        print("\nTesting Empty Row (Zero True Labels)...")
        cm = torch.tensor([
            [5, 5],
            [0, 0]  # No samples for True Class 1
        ])
        
        mae = compute_mae_per_class(cm)
        
        self.assertIsNone(mae[1])
        self.assertEqual(mae[0], 0.5) # (5*0 + 5*1)/10
        print("✅ Empty row handled correctly (None).")

    def test_column_wise(self):
        print("\nTesting Column-wise MAE...")
        # cm same as test 1
        cm = torch.tensor([
            [10, 0, 0],
            [0, 5, 5],
            [2, 0, 8]
        ])
        
        res = compute_mae_per_class(cm, compute_column_wise=True)
        
        # Check Col 0 (Predicted Class 0)
        # Entries: (0,0)=10, (1,0)=0, (2,0)=2
        # True classes: 0 and 2. Pred class: 0.
        # Errors: |0-0|=0, |2-0|=2.
        # Sum = 10*0 + 0*1 + 2*2 = 4
        # Count = 12
        # MAE = 4/12 = 1/3
        self.assertAlmostEqual(res['col_mae'][0], 1.0/3.0, places=5)
        
        # Check Col 1 (Predicted Class 1)
        # Entries: (0,1)=0, (1,1)=5, (2,1)=0
        # True class 1. Pred class 1.
        # Error = 0.
        self.assertEqual(res['col_mae'][1], 0.0)
        
        print("✅ Column-wise MAE correct.")

    def test_empty_column(self):
        cm = torch.tensor([
            [5, 0],
            [5, 0],
        ])

        result = compute_mae_per_class(cm, compute_column_wise=True)

        self.assertIsNone(result['col_mae'][1])

    def test_numpy_input(self):
        print("\nTesting Numpy Input...")
        cm = np.array([[10, 0], [0, 10]])
        mae = compute_mae_per_class(cm)
        self.assertEqual(mae[0], 0.0)
        self.assertEqual(mae[1], 0.0)
        print("✅ Numpy input handled correctly.")

if __name__ == '__main__':
    unittest.main()
