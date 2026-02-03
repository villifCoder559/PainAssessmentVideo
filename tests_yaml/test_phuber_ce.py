import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
import sys
import os

# Add root to path to import custom.loss
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from custom.loss import PHuberCrossEntropy

class TestPHuberCrossEntropy(unittest.TestCase):
    def setUp(self):
        self.tau = 10.0
        self.loss_fn = PHuberCrossEntropy(tau=self.tau, reduction='mean')
        self.threshold_p = 1.0 / self.tau

    def test_log_branch_correctness(self):
        print("\nTesting Log Branch Correctness...")
        # Create logits such that p_y > 1/tau (0.1)
        # p_y = 0.5 -> log loss expected
        logits = torch.tensor([[0.0, 0.0]], requires_grad=True) # p=0.5
        target = torch.tensor([0])
        
        loss = self.loss_fn(logits, target)
        
        # Manual calc
        p_y = 0.5
        expected_loss = -torch.log(torch.tensor(p_y))
        
        self.assertTrue(torch.allclose(loss, expected_loss, atol=1e-5), 
                        f"Log branch mismatch. Got {loss.item()}, expected {expected_loss.item()}")
        print("✅ Log branch output matches standard CE.")

    def test_linear_branch_correctness(self):
        print("\nTesting Linear Branch Correctness...")
        # Create logits such that p_y < 0.1
        # logits = [0, 10] -> p_0 approx exp(-10) which is tiny
        logits = torch.tensor([[0.0, 5.0]], requires_grad=True) 
        target = torch.tensor([0])
        
        # Manual p_y calculation
        probs = F.softmax(logits, dim=1)
        p_y = probs[0, 0].item()
        
        # Verify we are in linear region
        self.assertTrue(p_y < self.threshold_p, f"Test setup failure: p_y {p_y} not < {self.threshold_p}")
        
        loss = self.loss_fn(logits, target)
        
        # Manual calc: -tau * p + log(tau) + 1
        expected_loss = -self.tau * p_y + torch.log(torch.tensor(self.tau)) + 1.0
        
        self.assertTrue(torch.allclose(loss, torch.tensor(expected_loss), atol=1e-5),
                        f"Linear branch mismatch. Got {loss.item()}, expected {expected_loss}")
        print("✅ Linear branch output matches formula.")

    def test_gradient_flow_log_branch(self):
        print("\nTesting Gradient Flow (Log Branch)...")
        logits = torch.tensor([[0.0, 0.0]], requires_grad=True) # p=0.5
        target = torch.tensor([0])
        
        loss = self.loss_fn(logits, target)
        loss.backward()
        
        # CE gradient for class i: p_i - 1
        # here p_0 = 0.5, grad should be 0.5 - 1 = -0.5
        grad_target = logits.grad[0, 0]
        self.assertTrue(torch.abs(grad_target - (-0.5)) < 1e-5, f"Gradient incorrect. Got {grad_target}")
        print("✅ Log branch gradient is correct.")

    def test_gradient_flow_linear_branch(self):
        print("\nTesting Gradient Flow (Linear Branch)...")
        # Extremely small probability for target 0
        logits = torch.tensor([[-10.0, 10.0]], requires_grad=True) 
        target = torch.tensor([0])
        
        loss = self.loss_fn(logits, target)
        loss.backward()
        
        # In linear branch, dL/dp = -tau
        # dL/dz_i = dL/dp * dp/dz_i = -tau * p_i * (1 - p_i)
        # Since p_i is very small, 1-p_i approx 1, so gradient approx -tau * p_i
        
        probs = F.softmax(logits, dim=1)
        p_i = probs[0, 0]
        
        expected_grad_approx = -self.tau * p_i * (1 - p_i)
        grad_target = logits.grad[0, 0]
        
        # Relative error check
        rel_error = torch.abs((grad_target - expected_grad_approx) / expected_grad_approx)
        
        self.assertTrue(rel_error < 1e-4, f"Linear gradient mismatch. Got {grad_target}, expected approx {expected_grad_approx}")
        
        # Ensure it's not zero (crucial fix verification)
        self.assertTrue(torch.abs(grad_target) > 0, "Gradient should not be zero!")
        print("✅ Linear branch gradient flows and matches derivation.")

    def test_numerical_stability_edge_case(self):
        print("\nTesting Numerical Stability (Very hard example)...")
        # Target probability effectively 0 (e.g. logit diff > 80)
        logits = torch.tensor([[-100.0, 100.0]], requires_grad=True)
        target = torch.tensor([0])
        
        loss = self.loss_fn(logits, target)
        loss.backward()
        
        self.assertFalse(torch.isnan(loss).item(), "Loss is NaN")
        self.assertFalse(torch.isinf(loss).item(), "Loss is Inf")
        self.assertFalse(torch.isnan(logits.grad).any().item(), "Gradient is NaN")
        
        # Check if gradient is alive (should be very small but non-zero due to softmax derivative scaling by p, 
        # BUT the key is that dL/dp is constant -tau, avoiding the 1/p explosion of standard CE)
        # Wait, for standard CE, dL/dz = p - 1. For p=0, grad = -1. 
        # For PHuber, dL/dz = -tau * p * (1-p). If p -> 0, grad -> 0.
        # This is expected behavior for PHuber: it clamps the influence of outliers (hard examples) on the gradient magnitude relative to probability space, 
        # but in logit space?
        # Let's re-verify the PHuber gradient w.r.t logit z:
        # L = -tau * p + C
        # dL/dz = -tau * dp/dz = -tau * p * (1-p)
        # As p -> 0, gradient -> 0. This means extremely hard examples have VANISHING gradients in PHuber, 
        # unlike standard CE where gradient is constant -1. 
        # This makes it robust to label noise (outliers don't dominate).
        
        print(f"Loss value: {loss.item()}")
        print(f"Gradient value: {logits.grad[0,0].item()}")
        print("✅ Numerical stability check passed (no NaNs).")

if __name__ == '__main__':
    unittest.main()
