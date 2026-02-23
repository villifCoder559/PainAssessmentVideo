import torch
import sys
import os

# Add the project root to sys.path to allow imports from 'jepa'
sys.path.append(os.getcwd())

from jepa.src.models.attentive_pooler import AttentivePooler

def test_attentive_pooler_v2():
    print("Testing AttentivePooler with complete_block=2...")
    
    batch_size = 2
    seq_len = 10
    embed_dim = 128
    num_queries = 4
    num_heads = 4
    depth = 3

    # 1. Test standard flow (cross_block_after_transformers=False)
    print("\nCase 1: Standard flow (cross_block_after_transformers=False)")
    pooler = AttentivePooler(
        num_queries=num_queries,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_cross_heads=num_heads,
        depth=depth,
        complete_block=2,
        cross_block_after_transformers=False
    )
    
    x = torch.randn(batch_size, seq_len, embed_dim)
    mask = None # Mask is ignored in complete_block=2 as requested
    
    q, xattn = pooler(x, mask)
    
    print(f"Output q shape: {q.shape}")
    print(f"Output xattn: {xattn}")
    
    assert q.shape == (batch_size, num_queries, embed_dim), f"Expected shape {(batch_size, num_queries, embed_dim)}, got {q.shape}"
    assert xattn is None, "Expected xattn to be None"
    print("Case 1 passed!")

    # 2. Test cross_block_after_transformers=True
    print("\nCase 2: cross_block_after_transformers=True")
    pooler_after = AttentivePooler(
        num_queries=num_queries,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_cross_heads=num_heads,
        depth=depth,
        complete_block=2,
        cross_block_after_transformers=True
    )
    
    q_after, xattn_after = pooler_after(x, mask)
    
    print(f"Output q shape: {q_after.shape}")
    print(f"Output xattn: {xattn_after}")
    
    assert q_after.shape == (batch_size, num_queries, embed_dim), f"Expected shape {(batch_size, num_queries, embed_dim)}, got {q_after.shape}"
    assert xattn_after is None, "Expected xattn to be None"
    print("Case 2 passed!")

    # 3. Test with different q_k_v_dim
    print("\nCase 3: different q_k_v_dim")
    q_k_v_dim = 256
    pooler_qkv = AttentivePooler(
        num_queries=num_queries,
        embed_dim=embed_dim,
        q_k_v_dim=q_k_v_dim,
        num_heads=num_heads,
        num_cross_heads=num_heads,
        depth=depth,
        complete_block=2
    )
    
    q_qkv, _ = pooler_qkv(x, mask)
    assert q_qkv.shape == (batch_size, num_queries, embed_dim)
    print("Case 3 passed!")

if __name__ == "__main__":
    try:
        test_attentive_pooler_v2()
        print("\nAll tests passed successfully!")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
