import torch
import torch.nn as nn
import sys
import os
import unittest

# Add root to path to import custom.loss
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from custom.loss import DisentangledLoss, RnCLossV2

class TestDisentangledLossV3(unittest.TestCase):
    def setUp(self):
        self.B = 8
        self.C = 16
        self.split_idx = 8
        self.D_pain = self.split_idx
        self.D_subj = self.C - self.split_idx
        
        # Simple MSE losses for testing
        self.loss_fn_pain = nn.MSELoss()
        self.loss_fn_subj = nn.MSELoss()
        
        # Random inputs
        self.features = torch.randn(self.B, self.C, requires_grad=True)
        # Assuming sub-losses handle these targets
        self.target_pain = torch.randn(self.B, 4) 
        self.target_subj = torch.randn(self.B, 4)

    def test_gradient_isolation_pain_v3(self):
        print("\nTesting Gradient Isolation V3: Pain Loss Only...")
        loss_mod = DisentangledLoss(
            loss_fns=[self.loss_fn_pain, self.loss_fn_subj],
            split_idx=self.split_idx,
            lambdas=[1.0, 0.0],
            ortho_lambda=0.0,
            return_dict=False
        )
        
        # We need to make sure targets match feature slice dimensions to avoid warnings
        t_pain = torch.randn(self.B, self.D_pain)
        t_subj = torch.randn(self.B, self.D_subj)
        
        if self.features.grad is not None:
            self.features.grad.zero_()
            
        total_loss = loss_mod(self.features, t_pain, t_subj)
        total_loss.backward()
        
        grad = self.features.grad
        grad_pain = grad[:, :self.split_idx]
        grad_subj = grad[:, self.split_idx:]
        
        self.assertTrue(torch.abs(grad_pain).sum().item() > 0, "Pain features should have gradient")
        self.assertTrue(torch.abs(grad_subj).sum().item() == 0.0, "Subject features should NOT have gradient")
        print("✅ Pain loss gradient is isolated correctly.")

    def test_gradient_isolation_subject_v3(self):
        print("\nTesting Gradient Isolation V3: Subject Loss Only...")
        loss_mod = DisentangledLoss(
            loss_fns=[self.loss_fn_pain, self.loss_fn_subj],
            split_idx=self.split_idx,
            lambdas=[0.0, 1.0],
            ortho_lambda=0.0,
            return_dict=False
        )
        
        t_pain = torch.randn(self.B, self.D_pain)
        t_subj = torch.randn(self.B, self.D_subj)
        
        if self.features.grad is not None:
            self.features.grad.zero_()
            
        total_loss = loss_mod(self.features, t_pain, t_subj)
        total_loss.backward()
        
        grad = self.features.grad
        grad_pain = grad[:, :self.split_idx]
        grad_subj = grad[:, self.split_idx:]
        
        self.assertTrue(torch.abs(grad_pain).sum().item() == 0.0, "Pain features should NOT have gradient")
        self.assertTrue(torch.abs(grad_subj).sum().item() > 0, "Subject features should have gradient")
        print("✅ Subject loss gradient is isolated correctly.")

    def test_return_dict_v3(self):
        print("\nTesting Return Dict Format V3...")
        loss_mod = DisentangledLoss(
            loss_fns=[self.loss_fn_pain, self.loss_fn_subj],
            split_idx=self.split_idx,
            lambdas=[0.7, 0.3],
            ortho_lambda=0.5,
            return_dict=True
        )
        
        t_pain = torch.randn(self.B, self.D_pain)
        t_subj = torch.randn(self.B, self.D_subj)
        
        out = loss_mod(self.features, t_pain, t_subj)
        self.assertIsInstance(out, dict)
        expected_keys = ['total', 'pain', 'subject', 'lambda_pain', 'lambda_subj', 'ortho']
        for key in expected_keys:
            self.assertIn(key, out, f"Missing key: {key}")
        
        self.assertEqual(out['lambda_pain'], 0.7)
        self.assertEqual(out['lambda_subj'], 0.3)
        print("✅ Return dict keys and lambdas verified.")

    def test_with_rnclossv2(self):
        print("\nTesting DisentangledLoss with RnCLossV2 component...")
        # RnCLossV2 expects features [2bs, dim] and labels [2bs, label_dim]
        # In DisentangledLoss, feat_pain will be [B, split_idx]
        # So B must be even if we want to simulate the 2bs logic, 
        # but RnCLossV2 just takes what it gets.
        
        rnc_pain = RnCLossV2()
        mse_subj = nn.MSELoss()
        
        loss_mod = DisentangledLoss(
            loss_fns=[rnc_pain, mse_subj],
            split_idx=self.split_idx,
            lambdas=[1.0, 1.0],
            ortho_lambda=0.1,
            return_dict=True
        )
        
        # For RnCLossV2, target should be [B, label_dim]
        t_pain = torch.randn(self.B, 1)
        t_subj = torch.randn(self.B, self.D_subj)
        
        out = loss_mod(self.features, t_pain, t_subj)
        out['total'].backward()
        
        self.assertTrue(self.features.grad is not None)
        print("✅ Works correctly with RnCLossV2 as a component.")

if __name__ == '__main__':
    unittest.main()
