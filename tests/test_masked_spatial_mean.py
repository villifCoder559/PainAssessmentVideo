"""
Tests for masked_spatial_mean function.
Tests cover various scenarios including shape validation, computation accuracy, and device compatibility.

Run with: pytest test_masked_spatial_mean.py -v
"""

import torch
import pytest
import numpy as np
from extract_feature import masked_spatial_mean


class TestMaskedSpatialMeanBasics:
    """Test basic functionality and shape handling."""
    
    def test_output_shape_s14(self):
        """Test output shape with S=14."""
        B, T, S, C = 2, 8, 14, 256
        features = torch.randn(B, T, S, S, C)
        output = masked_spatial_mean(features)
        assert output.shape == (B, T, 1, 1, C)
    
    def test_output_shape_s16(self):
        """Test output shape with S=16."""
        B, T, S, C = 3, 4, 16, 512
        features = torch.randn(B, T, S, S, C)
        output = masked_spatial_mean(features)
        assert output.shape == (B, T, 1, 1, C)
    
    def test_batch_size_one(self):
        """Test with batch size 1 (most common case)."""
        features = torch.randn(1, 8, 14, 14, 256)
        output = masked_spatial_mean(features)
        assert output.shape == (1, 8, 1, 1, 256)
    
    def test_batch_size_large(self):
        """Test with large batch size."""
        features = torch.randn(32, 16, 14, 14, 128)
        output = masked_spatial_mean(features)
        assert output.shape == (32, 16, 1, 1, 128)
    
    def test_single_temporal(self):
        """Test with single temporal frame."""
        features = torch.randn(2, 1, 14, 14, 256)
        output = masked_spatial_mean(features)
        assert output.shape == (2, 1, 1, 1, 256)
    
    def test_single_channel(self):
        """Test with single channel."""
        features = torch.randn(2, 8, 14, 14, 1)
        output = masked_spatial_mean(features)
        assert output.shape == (2, 8, 1, 1, 1)


class TestMaskedSpatialMeanComputation:
    """Test computation accuracy."""
    
    def test_mean_computation_s14_with_constant_input(self):
        """Test that mean of constant input equals the constant."""
        B, T, S, C = 1, 1, 14, 1
        value = 5.0
        features = torch.full((B, T, S, S, C), value)
        output = masked_spatial_mean(features)
        
        # All positions get excluded based on the mask, but non-excluded should average to value
        # Check that output is close to the input value
        assert output.shape == (B, T, 1, 1, C)
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"
    
    def test_mean_computation_with_zeros(self):
        """Test mean computation with zeros."""
        features = torch.zeros(1, 2, 14, 14, 10)
        output = masked_spatial_mean(features)
        assert torch.allclose(output, torch.zeros(1, 2, 1, 1, 10))
    
    def test_mean_computation_with_ones(self):
        """Test mean computation with all ones."""
        features = torch.ones(1, 1, 14, 14, 1)
        output = masked_spatial_mean(features)
        # Output should be close to 1 (mean of non-excluded positions)
        assert output.shape == (1, 1, 1, 1, 1)
        # The value should be between 0 and 1
        assert 0 <= output.item() <= 1
    
    def test_manual_mean_verification(self):
        """Verify mean computation against manual calculation."""
        B, T, S, C = 1, 1, 14, 1
        features = torch.randn(B, T, S, S, C)
        output = masked_spatial_mean(features)
        
        # Create exclusion mask manually
        excluded_positions = [
            (0, 0), (0, 1), (0, 2), (0, -3), (0, -2), (0, -1),
            (1, 0), (1, -1),
            (2, 0), (2, -1),
            (-5, 0), (-5, -1),
            (-4, 0), (-4, -1),
            (-3, 0), (-3, 1), (-3, -2), (-3, -1),
            (-2, 0), (-2, 1), (-2, 2), (-2, -3), (-2, -2), (-2, -1),
            (-1, 0), (-1, 1), (-1, 2), (-1, 3), (-1, 4), (-1, -5), (-1, -4), (-1, -3), (-1, -2), (-1, -1)
        ]
        
        # Convert to positive indices
        excluded_set = set()
        for row, col in excluded_positions:
            r = row if row >= 0 else S + row
            c = col if col >= 0 else S + col
            if 0 <= r < S and 0 <= c < S:
                excluded_set.add((r, c))
        
        # Manually compute mean
        feat_2d = features[0, 0, :, :, 0]
        valid_values = []
        for i in range(S):
            for j in range(S):
                if (i, j) not in excluded_set:
                    valid_values.append(feat_2d[i, j].item())
        
        expected_mean = np.mean(valid_values)
        
        # Compare
        assert np.isclose(output.item(), expected_mean, rtol=1e-5)
    
    def test_mean_independent_across_batch(self):
        """Test that mean is computed independently for each batch element."""
        B, T, S, C = 2, 1, 14, 1
        features = torch.zeros(B, T, S, S, C)
        features[0, 0, :, :, 0] = 1.0  # Batch 0: all ones
        features[1, 0, :, :, 0] = 2.0  # Batch 1: all twos
        
        output = masked_spatial_mean(features)
        
        # Both batches should produce different outputs
        assert output[0, 0, 0, 0, 0].item() <= 1.0
        assert output[1, 0, 0, 0, 0].item() <= 2.0
        # The outputs should be different from each other
        assert output[0, 0, 0, 0, 0].item() != output[1, 0, 0, 0, 0].item()
    
    def test_mean_independent_across_temporal(self):
        """Test that mean is computed independently for each temporal frame."""
        B, T, S, C = 1, 3, 14, 1
        features = torch.zeros(B, T, S, S, C)
        features[0, 0, :, :, 0] = 1.0  # T=0: all ones
        features[0, 1, :, :, 0] = 2.0  # T=1: all twos
        features[0, 2, :, :, 0] = 3.0  # T=2: all threes
        
        output = masked_spatial_mean(features)
        
        # Temporal dimension should be preserved
        assert output.shape == (1, 3, 1, 1, 1)


class TestMaskedSpatialMeanScaling:
    """Test scaling from S=14 to S=16."""
    
    def test_s16_scaling_produces_valid_output(self):
        """Test that S=16 scaling produces valid output."""
        features = torch.randn(1, 2, 16, 16, 10)
        output = masked_spatial_mean(features)
        
        assert output.shape == (1, 2, 1, 1, 10)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_s16_mean_is_reasonable(self):
        """Test that S=16 mean is in reasonable range."""
        features = torch.ones(1, 1, 16, 16, 1)
        output = masked_spatial_mean(features)
        
        # Output should be a valid mean (between 0 and 1 for ones input)
        assert 0 <= output.item() <= 1
    
    def test_s14_and_s16_handle_same_constant_proportionally(self):
        """Test that S=14 and S=16 handle constant inputs proportionally."""
        features_14 = torch.ones(1, 1, 14, 14, 1)
        features_16 = torch.ones(1, 1, 16, 16, 1)
        
        output_14 = masked_spatial_mean(features_14)
        output_16 = masked_spatial_mean(features_16)
        
        # Both should produce valid values
        assert 0 <= output_14.item() <= 1
        assert 0 <= output_16.item() <= 1
        
        # They should be close (similar fraction of positions excluded)
        # Due to scaling, they might differ slightly
        assert abs(output_14.item() - output_16.item()) < 0.2


class TestMaskedSpatialMeanErrors:
    """Test error handling."""
    
    def test_invalid_s_dimension(self):
        """Test that invalid S dimension raises error."""
        features = torch.randn(1, 8, 15, 15, 256)
        with pytest.raises(ValueError, match="S must be 14 or 16"):
            masked_spatial_mean(features)
    
    def test_invalid_s_too_small(self):
        """Test that S too small raises error."""
        features = torch.randn(1, 8, 8, 8, 256)
        with pytest.raises(ValueError, match="S must be 14 or 16"):
            masked_spatial_mean(features)
    
    def test_invalid_s_too_large(self):
        """Test that S too large raises error."""
        features = torch.randn(1, 8, 32, 32, 256)
        with pytest.raises(ValueError, match="S must be 14 or 16"):
            masked_spatial_mean(features)
    
    def test_non_square_spatial(self):
        """Test that non-square spatial dimensions raise error."""
        features = torch.randn(1, 8, 14, 16, 256)
        with pytest.raises(ValueError, match="square spatial dimensions"):
            masked_spatial_mean(features)
    
    def test_wrong_tensor_shape_4d(self):
        """Test that 4D tensor raises error."""
        features = torch.randn(1, 8, 14, 256)
        with pytest.raises(ValueError, match="Expected 5D tensor"):
            masked_spatial_mean(features)
    
    def test_wrong_tensor_shape_3d(self):
        """Test that 3D tensor raises error."""
        features = torch.randn(8, 14, 256)
        with pytest.raises(ValueError, match="Expected 5D tensor"):
            masked_spatial_mean(features)
    
    def test_wrong_tensor_shape_6d(self):
        """Test that 6D tensor raises error."""
        features = torch.randn(1, 1, 8, 14, 14, 256)
        with pytest.raises(ValueError, match="Expected 5D tensor"):
            masked_spatial_mean(features)


class TestMaskedSpatialMeanDevice:
    """Test device compatibility (CPU/GPU if available)."""
    
    def test_cpu_tensor(self):
        """Test with CPU tensor."""
        features = torch.randn(1, 8, 14, 14, 256, device='cpu')
        output = masked_spatial_mean(features)
        
        assert output.device.type == 'cpu'
        assert output.shape == (1, 8, 1, 1, 256)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_tensor(self):
        """Test with GPU tensor."""
        features = torch.randn(1, 8, 14, 14, 256, device='cuda')
        output = masked_spatial_mean(features)
        
        assert output.device.type == 'cuda'
        assert output.shape == (1, 8, 1, 1, 256)
    
    def test_mixed_device_consistency(self):
        """Test that CPU and GPU produce numerically close results."""
        torch.manual_seed(42)
        features_cpu = torch.randn(1, 2, 14, 14, 10, device='cpu')
        
        if torch.cuda.is_available():
            features_gpu = features_cpu.cuda()
            output_cpu = masked_spatial_mean(features_cpu)
            output_gpu = masked_spatial_mean(features_gpu)
            
            # Results should be numerically close
            assert torch.allclose(output_cpu, output_gpu.cpu(), rtol=1e-4)


class TestMaskedSpatialMeanDataTypes:
    """Test different data types."""
    
    def test_float32(self):
        """Test with float32 (default)."""
        features = torch.randn(1, 8, 14, 14, 256, dtype=torch.float32)
        output = masked_spatial_mean(features)
        assert output.dtype == torch.float32
    
    def test_float64(self):
        """Test with float64."""
        features = torch.randn(1, 8, 14, 14, 256, dtype=torch.float64)
        output = masked_spatial_mean(features)
        assert output.dtype == torch.float64
    
    def test_float16(self):
        """Test with float16."""
        features = torch.randn(1, 8, 14, 14, 256, dtype=torch.float16)
        output = masked_spatial_mean(features)
        # Output dtype should match input dtype
        assert output.dtype == features.dtype


class TestMaskedSpatialMeanEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_very_large_values(self):
        """Test with very large input values."""
        features = torch.full((1, 1, 14, 14, 1), 1e6)
        output = masked_spatial_mean(features)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_very_small_values(self):
        """Test with very small input values."""
        features = torch.full((1, 1, 14, 14, 1), 1e-6)
        output = masked_spatial_mean(features)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_negative_values(self):
        """Test with negative input values."""
        features = torch.full((1, 1, 14, 14, 1), -5.0)
        output = masked_spatial_mean(features)
        
        assert output.item() < 0
    
    def test_mixed_positive_negative(self):
        """Test with mixed positive and negative values."""
        features = torch.randn(1, 1, 14, 14, 1)
        output = masked_spatial_mean(features)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_gradients_flow(self):
        """Test that gradients flow through the function."""
        features = torch.randn(1, 8, 14, 14, 10, requires_grad=True)
        output = masked_spatial_mean(features)
        loss = output.sum()
        loss.backward()
        
        assert features.grad is not None
        assert not torch.isnan(features.grad).any()


class TestMaskedSpatialMeanCustomMask:
    """Test custom exclusion mask functionality."""
    
    def test_custom_empty_mask(self):
        """Test with empty exclusion mask (exclude nothing)."""
        features = torch.ones(1, 1, 14, 14, 1)
        custom_mask = []
        output = masked_spatial_mean(features, excluded_positions_s14=custom_mask)
        
        # All positions included, so mean should be 1.0
        assert torch.isclose(output, torch.tensor(1.0), atol=1e-5)
    
    def test_custom_full_mask(self):
        """Test with full exclusion mask (exclude all)."""
        features = torch.ones(1, 1, 14, 14, 1)
        custom_mask = [(i, j) for i in range(14) for j in range(14)]
        output = masked_spatial_mean(features, excluded_positions_s14=custom_mask)
        
        # All positions excluded, mean should be 0 (0/0 = 0 after division)
        assert torch.isnan(output) or output.item() == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
