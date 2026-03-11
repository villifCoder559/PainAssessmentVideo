import torch
import torch.nn as nn
import sys
import os
import unittest

# Add root to path to import custom.loss
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from custom.loss import DisentangledLoss

class TestDisentangledLoss(unittest.TestCase):
    def setUp(self):
        self.B = 4
        self.C = 10
        self.split_idx = 4
        self.V = 1
        
        # Simple MSE losses for testing
        self.loss_fn_pain = nn.MSELoss()
        self.loss_fn_subj = nn.MSELoss()
        
        # Random inputs
        self.features = torch.randn(self.B, self.C, requires_grad=True)
        # targets: [B, V, 2]
        self.targets = torch.randn(self.B, self.V, 2)

    def test_gradient_isolation_pain(self):
        print("\nTesting Gradient Isolation: Pain Loss Only...")
        # Weight only on pain, no ortho
        loss_mod = DisentangledLoss(
            loss_fns=[self.loss_fn_pain, self.loss_fn_subj],
            split_idx=self.split_idx,
            lambdas=[1.0, 0.0],
            ortho_lambda=0.0
        )
        
        # Reset grad
        if self.features.grad is not None:
            self.features.grad.zero_()
            
        total_loss = loss_mod(self.features, self.targets)
        total_loss.backward()
        
        grad = self.features.grad
        grad_pain = grad[:, :self.split_idx]
        grad_subj = grad[:, self.split_idx:]
        
        # Pain gradient should be non-zero
        self.assertTrue(torch.abs(grad_pain).sum().item() > 0, "Pain features should have gradient")
        
        # Subject gradient should be EXACTLY zero
        self.assertTrue(torch.abs(grad_subj).sum().item() == 0.0, f"Subject features should NOT have gradient, got norm {torch.norm(grad_subj)}")
        print("✅ Pain loss gradient is isolated correctly.")

    def test_gradient_isolation_subject(self):
        print("\nTesting Gradient Isolation: Subject Loss Only...")
        # Weight only on subject, no ortho
        loss_mod = DisentangledLoss(
            loss_fns=[self.loss_fn_pain, self.loss_fn_subj],
            split_idx=self.split_idx,
            lambdas=[0.0, 1.0],
            ortho_lambda=0.0
        )
        
        if self.features.grad is not None:
            self.features.grad.zero_()
            
        total_loss = loss_mod(self.features, self.targets)
        total_loss.backward()
        
        grad = self.features.grad
        grad_pain = grad[:, :self.split_idx]
        grad_subj = grad[:, self.split_idx:]
        
        # Pain gradient should be EXACTLY zero
        self.assertTrue(torch.abs(grad_pain).sum().item() == 0.0, "Pain features should NOT have gradient")
        
        # Subject gradient should be non-zero
        self.assertTrue(torch.abs(grad_subj).sum().item() > 0, "Subject features should have gradient")
        print("✅ Subject loss gradient is isolated correctly.")

    def test_orthogonality_gradient(self):
        print("\nTesting Orthogonality Gradient Flow...")
        # Only ortho loss
        loss_mod = DisentangledLoss(
            loss_fns=[self.loss_fn_pain, self.loss_fn_subj],
            split_idx=self.split_idx,
            lambdas=[0.0, 0.0],
            ortho_lambda=1.0
        )
        
        if self.features.grad is not None:
            self.features.grad.zero_()
            
        total_loss = loss_mod(self.features, self.targets)
        total_loss.backward()
        
        grad = self.features.grad
        grad_pain = grad[:, :self.split_idx]
        grad_subj = grad[:, self.split_idx:]
        
        # Both should have gradients because they interact in the covariance term
        self.assertTrue(torch.abs(grad_pain).sum().item() > 0, "Pain features should have gradient from ortho loss")
        self.assertTrue(torch.abs(grad_subj).sum().item() > 0, "Subject features should have gradient from ortho loss")
        print("✅ Orthogonality loss flows to both slices.")

    def test_combined_gradients(self):
        print("\nTesting Combined Gradients...")
        loss_mod = DisentangledLoss(
            loss_fns=[self.loss_fn_pain, self.loss_fn_subj],
            split_idx=self.split_idx,
            lambdas=[1.0, 1.0],
            ortho_lambda=0.0
        )
        
        if self.features.grad is not None:
            self.features.grad.zero_()
            
        total_loss = loss_mod(self.features, self.targets)
        total_loss.backward()
        
        grad = self.features.grad
        # Both should be active
        self.assertTrue(torch.abs(grad).sum().item() > 0)
        print("✅ Combined loss produces valid gradients.")

    def test_return_dict(self):
        print("\nTesting Return Dict...")
        loss_mod = DisentangledLoss(
            loss_fns=[self.loss_fn_pain, self.loss_fn_subj],
            split_idx=self.split_idx,
            lambdas=[1.0, 1.0],
            ortho_lambda=0.5,
            return_dict=True
        )
        
        out = loss_mod(self.features, self.targets)
        self.assertIsInstance(out, dict)
        self.assertIn('total', out)
        self.assertIn('pain', out)
        self.assertIn('subject', out)
        self.assertIn('ortho', out)
        print("✅ Return dict format is correct.")

if __name__ == '__main__':
    unittest.main()
