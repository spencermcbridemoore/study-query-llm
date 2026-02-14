#!/usr/bin/env python3
"""
Validation Script: Confirm Embeddings Differ Across Summarizers

This script runs a small test to verify that different summarizers produce
different embeddings (fixing the bug where all summarizers used identical embeddings).

Usage:
    python scripts/validate_embedding_differences.py
"""

import os
import sys
import asyncio
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.embedding_service import EmbeddingService, EmbeddingRequest
from study_query_llm.services.summarization_service import SummarizationService, SummarizationRequest

# Configuration
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")

EMBEDDING_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

# Test with a small sample
TEST_TEXTS = [
    "The quick brown fox jumps over the lazy dog in the forest.",
    "Machine learning algorithms can identify patterns in large datasets.",
    "Climate change is affecting weather patterns around the world.",
    "The stock market experienced significant volatility yesterday.",
    "Quantum computers use qubits to perform complex calculations.",
]

SUMMARIZERS_TO_TEST = [
    None,  # Original text
    "gpt-4o-mini",
    "gpt-4o",
]


async def fetch_embeddings(texts, deployment, db):
    """Fetch embeddings for a list of texts."""
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        service = EmbeddingService(repository=repo)
        
        requests = [EmbeddingRequest(text=text, deployment=deployment) for text in texts]
        responses = await service.get_embeddings_batch(requests)
        
        return np.asarray([resp.vector for resp in responses], dtype=np.float64)


async def summarize_text(text, llm_deployment, db):
    """Summarize a single text."""
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        service = SummarizationService(repository=repo)
        
        request = SummarizationRequest(
            texts=[text],
            llm_deployment=llm_deployment,
            temperature=0.2,
            max_tokens=256,
        )
        
        result = await service.summarize_batch(request)
        if result.summaries and len(result.summaries) > 0:
            return result.summaries[0]
        return text  # Fallback


async def main():
    print("=" * 80)
    print("Embedding Difference Validation Test")
    print("=" * 80)
    print(f"\nTest configuration:")
    print(f"  Test texts: {len(TEST_TEXTS)}")
    print(f"  Summarizers: {SUMMARIZERS_TO_TEST}")
    print(f"  Embedding deployment: {EMBEDDING_DEPLOYMENT}")
    
    # Initialize database
    print("\n[1/4] Initializing database...")
    db = DatabaseConnectionV2(DATABASE_URL, enable_pgvector=True)
    db.init_db()
    print("  [OK] Database ready")
    
    # Store embeddings for each summarizer
    embeddings_by_summarizer = {}
    summarized_texts_by_summarizer = {}
    
    print("\n[2/4] Processing each summarizer...")
    for summarizer in SUMMARIZERS_TO_TEST:
        summarizer_name = "None" if summarizer is None else summarizer
        print(f"\n  Summarizer: {summarizer_name}")
        
        # Step 1: Summarize (if applicable)
        if summarizer is not None:
            print(f"    Summarizing {len(TEST_TEXTS)} texts...")
            summarized = []
            for i, text in enumerate(TEST_TEXTS):
                try:
                    summary = await summarize_text(text, summarizer, db)
                    summarized.append(summary)
                    try:
                        print(f"      [{i+1}/{len(TEST_TEXTS)}] Original ({len(text)} chars) -> Summary ({len(summary)} chars)")
                    except UnicodeEncodeError:
                        print(f"      [{i+1}/{len(TEST_TEXTS)}] Summarized (encoding issue in output)")
                except Exception as e:
                    print(f"      [{i+1}/{len(TEST_TEXTS)}] FAILED: {str(e)[:50]}, using original")
                    summarized.append(text)
            texts_to_embed = summarized
        else:
            print(f"    Using original texts (no summarization)")
            texts_to_embed = TEST_TEXTS
        
        summarized_texts_by_summarizer[summarizer_name] = texts_to_embed
        
        # Step 2: Embed
        print(f"    Embedding {len(texts_to_embed)} texts...")
        try:
            embeddings = await fetch_embeddings(texts_to_embed, EMBEDDING_DEPLOYMENT, db)
            embeddings_by_summarizer[summarizer_name] = embeddings
            print(f"    [OK] Got embeddings: shape {embeddings.shape}")
        except Exception as e:
            print(f"    [FAIL] FAILED: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Compare embeddings
    print("\n[3/4] Comparing embeddings across summarizers...")
    summarizer_names = list(embeddings_by_summarizer.keys())
    
    if len(summarizer_names) < 2:
        print("  [FAIL] ERROR: Need at least 2 summarizers to compare!")
        return
    
    print(f"\n  Pairwise comparisons:")
    all_different = True
    for i in range(len(summarizer_names)):
        for j in range(i + 1, len(summarizer_names)):
            name_a = summarizer_names[i]
            name_b = summarizer_names[j]
            
            emb_a = embeddings_by_summarizer[name_a]
            emb_b = embeddings_by_summarizer[name_b]
            
            # Check if shapes match
            if emb_a.shape != emb_b.shape:
                print(f"\n    {name_a} vs {name_b}: [ERROR] SHAPE MISMATCH")
                print(f"      {name_a} shape: {emb_a.shape}")
                print(f"      {name_b} shape: {emb_b.shape}")
                print(f"      Cannot compare - different number of embeddings!")
                continue
            
            identical = np.array_equal(emb_a, emb_b)
            max_diff = np.abs(emb_a - emb_b).max()
            mean_diff = np.abs(emb_a - emb_b).mean()
            
            status = "[FAIL] IDENTICAL" if identical else "[OK] DIFFERENT"
            print(f"\n    {name_a} vs {name_b}: {status}")
            print(f"      Max difference:  {max_diff:.6f}")
            print(f"      Mean difference: {mean_diff:.6f}")
            
            if identical:
                all_different = False
    
    # Show example text transformations
    print("\n[4/4] Example text transformations:")
    for i in range(min(2, len(TEST_TEXTS))):
        print(f"\n  Text {i+1}:")
        for summarizer_name in summarizer_names:
            text = summarized_texts_by_summarizer[summarizer_name][i]
            preview = text[:100] + "..." if len(text) > 100 else text
            print(f"    {summarizer_name:15s}: {preview}")
    
    # Final verdict
    print("\n" + "=" * 80)
    if all_different:
        print("[OK] SUCCESS: All summarizers produce DIFFERENT embeddings!")
        print("  The bug is fixed. You can now run the full experimental sweep.")
    else:
        print("[FAIL] FAILURE: Some summarizers produce IDENTICAL embeddings!")
        print("  The bug is NOT fixed. Do not run the full sweep yet.")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
