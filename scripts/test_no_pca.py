"""
Test script to verify skip_pca functionality works correctly.

Tests both PCA and no-PCA modes with a small dataset.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from study_query_llm.algorithms.sweep import run_sweep, SweepConfig

def test_skip_pca():
    """Test that skip_pca flag works correctly."""
    print("=" * 80)
    print("Testing skip_pca functionality")
    print("=" * 80)
    
    # Create small synthetic dataset (need more samples than pca_dim)
    np.random.seed(42)
    n_samples = 100  # Increased to allow 64-dim PCA
    embedding_dim = 1536  # Typical embedding dimension
    
    texts = [f"Sample text {i}" for i in range(n_samples)]
    embeddings = np.random.randn(n_samples, embedding_dim).astype(np.float32)
    
    print(f"\nTest data:")
    print(f"  Samples: {n_samples}")
    print(f"  Embedding dimension: {embedding_dim}")
    
    # Test 1: With PCA (default)
    print("\n" + "-" * 80)
    print("Test 1: With PCA (skip_pca=False)")
    print("-" * 80)
    
    cfg_pca = SweepConfig(
        pca_dim=64,
        skip_pca=False,
        k_min=2,
        k_max=5,
        max_iter=50,
        n_restarts=2,
        compute_stability=True,
    )
    
    result_pca = run_sweep(texts, embeddings, cfg_pca)
    
    print(f"\nResults with PCA:")
    print(f"  PCA dim used: {result_pca.pca['pca_dim_used']}")
    print(f"  Skip PCA flag: {result_pca.pca.get('skip_pca', 'not set')}")
    print(f"  Z shape: {result_pca.Z.shape if result_pca.Z is not None else 'None'}")
    print(f"  K values tested: {sorted(result_pca.by_k.keys())}")
    print(f"  Components: {'Present' if result_pca.pca.get('components') is not None else 'None'}")
    
    # Verify PCA was applied
    assert result_pca.pca['pca_dim_used'] == 64, "PCA dimension should be 64"
    assert result_pca.pca.get('skip_pca') == False, "skip_pca should be False"
    assert result_pca.pca.get('components') is not None, "PCA components should be present"
    
    print("\n[PASS] Test 1: PCA mode works correctly")
    
    # Test 2: Without PCA (skip_pca=True)
    print("\n" + "-" * 80)
    print("Test 2: Without PCA (skip_pca=True)")
    print("-" * 80)
    
    cfg_no_pca = SweepConfig(
        pca_dim=64,  # This should be ignored
        skip_pca=True,
        k_min=2,
        k_max=5,
        max_iter=50,
        n_restarts=2,
        compute_stability=True,
    )
    
    result_no_pca = run_sweep(texts, embeddings, cfg_no_pca)
    
    print(f"\nResults without PCA:")
    print(f"  PCA dim used: {result_no_pca.pca['pca_dim_used']}")
    print(f"  Skip PCA flag: {result_no_pca.pca.get('skip_pca', 'not set')}")
    print(f"  Z shape: {result_no_pca.Z.shape if result_no_pca.Z is not None else 'None'}")
    print(f"  K values tested: {sorted(result_no_pca.by_k.keys())}")
    print(f"  Components: {'Present' if result_no_pca.pca.get('components') is not None else 'None'}")
    
    # Verify PCA was skipped
    assert result_no_pca.pca['pca_dim_used'] == embedding_dim, f"Should use full dimension ({embedding_dim})"
    assert result_no_pca.pca.get('skip_pca') == True, "skip_pca should be True"
    assert result_no_pca.pca.get('components') is None, "PCA components should be None"
    
    print("\n[PASS] Test 2: No-PCA mode works correctly")
    
    # Test 3: Compare results
    print("\n" + "-" * 80)
    print("Test 3: Comparing results")
    print("-" * 80)
    
    # Both should produce valid clustering results
    for k in ['2', '3', '4', '5']:
        obj_pca = result_pca.by_k[k]['objective']
        obj_no_pca = result_no_pca.by_k[k]['objective']
        
        print(f"\nK={k}:")
        print(f"  PCA objective: {obj_pca:.4f}")
        print(f"  No-PCA objective: {obj_no_pca:.4f}")
        print(f"  PCA representatives: {len(result_pca.by_k[k]['representatives'])}")
        print(f"  No-PCA representatives: {len(result_no_pca.by_k[k]['representatives'])}")
        
        # Verify both have valid results
        assert obj_pca > 0, "PCA objective should be positive"
        assert obj_no_pca > 0, "No-PCA objective should be positive"
        assert len(result_pca.by_k[k]['representatives']) == int(k), "Should have K representatives"
        assert len(result_no_pca.by_k[k]['representatives']) == int(k), "Should have K representatives"
    
    print("\n[PASS] Test 3: Both modes produce valid results")
    
    # Final summary
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED")
    print("=" * 80)
    print("\nSummary:")
    print(f"  - PCA mode: Uses {result_pca.pca['pca_dim_used']}-dimensional space")
    print(f"  - No-PCA mode: Uses {result_no_pca.pca['pca_dim_used']}-dimensional space")
    print(f"  - Both modes successfully cluster data into K=2 to K=5 clusters")
    print(f"  - skip_pca flag correctly controls dimensionality reduction")
    
    return True

if __name__ == "__main__":
    try:
        success = test_skip_pca()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
