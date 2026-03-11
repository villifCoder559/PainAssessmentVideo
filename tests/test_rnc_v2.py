import torch
import torch.nn as nn
import time
import sys
import os
import argparse

# Add root to path to import custom.loss
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from custom.loss import RnCLoss, RnCLossV2

def test_correctness(bs, feat_dim):
    print(f"Testing Correctness (bs={bs}, feat_dim={feat_dim})...")
    label_dim = 1
    
    # Setup inputs
    features_src = torch.randn(bs, 2, feat_dim, requires_grad=True)
    labels_src = torch.randn(bs, label_dim)
    
    features_v2 = torch.cat([features_src[:, 0], features_src[:, 1]], dim=0).detach()
    features_v2.requires_grad = True
    
    labels_v2 = labels_src.repeat(2, 1) # [2bs, label_dim]
    
    # Initialize losses
    loss_v1 = RnCLoss()
    loss_v2 = RnCLossV2()
    
    # Forward V1
    out_v1 = loss_v1(features_src, labels_src)
    
    # Forward V2
    out_v2 = loss_v2(features_v2, labels_v2)
    
    diff = torch.abs(out_v1 - out_v2).item()
    print(f"Loss V1: {out_v1.item():.6f}")
    print(f"Loss V2: {out_v2.item():.6f}")
    print(f"Difference: {diff:.6e}")
    
    if diff < 1e-5:
        print("✅ Forward pass matches.")
    else:
        print("❌ Forward pass mismatch!")
    
    # Backward V1
    out_v1.backward()
    grad_v1 = features_src.grad # [bs, 2, D]
    grad_v1_reshaped = torch.cat([grad_v1[:, 0], grad_v1[:, 1]], dim=0)
    
    # Backward V2
    out_v2.backward()
    grad_v2 = features_v2.grad # [2bs, D]
    
    grad_diff = torch.max(torch.abs(grad_v1_reshaped - grad_v2)).item()
    print(f"Max Gradient Difference: {grad_diff:.6e}")
    
    if grad_diff < 1e-5:
        print("✅ Backward pass matches.")
    else:
        print("❌ Backward pass mismatch!")

def test_1d_labels(bs, feat_dim):
    print(f"\nTesting 1D Labels for RnCLossV2 (bs={bs}, feat_dim={feat_dim})...")
    
    # Setup inputs with 1D labels
    features = torch.randn(2*bs, feat_dim, requires_grad=True)
    labels_1d = torch.randn(2*bs) # 1D shape: [N]
    
    # Reference inputs with 2D labels for same data
    labels_2d = labels_1d.unsqueeze(-1) # 2D shape: [N, 1]
    
    loss_v2 = RnCLossV2()
    
    # Forward with 1D
    out_1d = loss_v2(features, labels_1d)
    
    # Forward with 2D
    # We detach and set grad again to compare gradients fairly
    features_2d = features.detach().clone()
    features_2d.requires_grad = True
    out_2d = loss_v2(features_2d, labels_2d)
    
    diff = torch.abs(out_1d - out_2d).item()
    print(f"Loss 1D: {out_1d.item():.6f}")
    print(f"Loss 2D: {out_2d.item():.6f}")
    print(f"Difference: {diff:.6e}")
    
    if diff < 1e-7:
        print("✅ 1D vs 2D Forward pass matches.")
    else:
        print("❌ 1D vs 2D Forward pass mismatch!")
        
    # Check gradient flow for 1D
    out_1d.backward()
    grad_1d = features.grad
    
    self_grad_norm = torch.norm(grad_1d).item()
    print(f"Gradient Norm (1D): {self_grad_norm:.6f}")
    
    if self_grad_norm > 0:
        print("✅ Gradient flows through features with 1D labels.")
    else:
        print("❌ No gradient flow through features with 1D labels!")
        
    # Compare gradients
    out_2d.backward()
    grad_2d = features_2d.grad
    grad_diff = torch.max(torch.abs(grad_1d - grad_2d)).item()
    print(f"Max Gradient Difference (1D vs 2D): {grad_diff:.6e}")
    
    if grad_diff < 1e-7:
        print("✅ 1D vs 2D Backward pass matches.")
    else:
        print("❌ 1D vs 2D Backward pass mismatch!")

def test_speed(bs, feat_dim, iters):
    print(f"\nTesting Speed (Forward Pass) - bs={bs}, feat_dim={feat_dim}, iters={iters}...")
    label_dim = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    features_src = torch.randn(bs, 2, feat_dim, device=device)
    labels_src = torch.randn(bs, label_dim, device=device)
    
    features_v2 = torch.cat([features_src[:, 0], features_src[:, 1]], dim=0)
    labels_v2 = labels_src.repeat(2, 1)
    
    loss_v1 = RnCLoss().to(device)
    loss_v2 = RnCLossV2().to(device)
    
    # Warmup
    for _ in range(5):
        loss_v1(features_src, labels_src)
        loss_v2(features_v2, labels_v2)
        
    # Measure V1
    if device.type == 'cuda': torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        loss_v1(features_src, labels_src)
    if device.type == 'cuda': torch.cuda.synchronize()
    t1 = time.time() - start
    
    # Measure V2
    if device.type == 'cuda': torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        loss_v2(features_v2, labels_v2)
    if device.type == 'cuda': torch.cuda.synchronize()
    t2 = time.time() - start
    
    print(f"V1 Time ({iters} iter): {t1:.4f}s")
    print(f"V2 Time ({iters} iter): {t2:.4f}s")
    print(f"Speedup: {t1/t2:.2f}x")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=32, help='Batch size (per view for V1)')
    parser.add_argument('--feat_dim', type=int, default=128, help='Feature dimension')
    parser.add_argument('--iters', type=int, default=20, help='Number of iterations for speed test')
    args = parser.parse_args()

    test_correctness(args.bs, args.feat_dim)
    test_1d_labels(args.bs, args.feat_dim)
    test_speed(args.bs, args.feat_dim, args.iters)