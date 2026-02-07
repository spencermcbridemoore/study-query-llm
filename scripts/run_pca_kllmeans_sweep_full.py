#!/usr/bin/env python3
"""
PCA KLLMeans Sweep - Full Data Source Support

This script runs the PCA KLLMeans sweep analysis with support for multiple data sources:
- "dictionary": Load from estela prompt dictionary (pickle/JSON file)
- "benchmark": Load benchmark dataset (e.g., 20 Newsgroups)
- "text_list": Load simple text list from file

Usage:
    python scripts/run_pca_kllmeans_sweep_full.py --data-source benchmark
    python scripts/run_pca_kllmeans_sweep_full.py --data-source dictionary --dict-file notebooks/estela_prompt_data.pkl
    python scripts/run_pca_kllmeans_sweep_full.py --data-source text_list --text-file path/to/texts.txt

Or set environment variables:
    DATA_SOURCE=benchmark python scripts/run_pca_kllmeans_sweep_full.py
"""

import os
import sys
import asyncio
import argparse
import pickle
import time
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from tqdm.asyncio import tqdm as async_tqdm

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.embedding_service import EmbeddingService, EmbeddingRequest, estimate_tokens, DEPLOYMENT_MAX_TOKENS
from study_query_llm.services.summarization_service import SummarizationService, SummarizationRequest
from study_query_llm.services.provenance_service import ProvenanceService
from study_query_llm.algorithms import SweepConfig, run_sweep

# Try to apply nest_asyncio for Jupyter compatibility
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass  # Not in Jupyter, no need for nest_asyncio


# ============================================================================
# Configuration
# ============================================================================

# Embedding deployment
EMBEDDING_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

# LLM deployments for summarization (3 LLMs + None = 4 runs)
LLM_SUMMARIZERS = [
    None,  # Non-LLM summaries (just use original representatives)
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-5-chat",  # GPT-5 chat deployment
]

# Sweep configuration
SWEEP_CONFIG = SweepConfig(
    pca_dim=64,
    k_min=2,
    k_max=10,
    max_iter=200,
    base_seed=0,
    n_restarts=20,  # Multiple restarts for stability analysis
    compute_stability=True,  # Enable stability metrics
    coverage_threshold=0.2,
    llm_interval=20,
    max_samples=10,
)

# Database URL (required)
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError(
        "DATABASE_URL environment variable must be set. "
        "Example: postgresql://user:pass@host/db?sslmode=require"
    )


# ============================================================================
# Data Loading Functions
# ============================================================================

def _is_prompt_key(key: str) -> bool:
    """Check if a key represents a prompt."""
    key_lower = key.lower()
    return "prompt" in key_lower


def flatten_prompt_dict(data, path=()):
    """Flatten nested prompt dictionary into a flat map of key tuples -> prompt strings."""
    flat = {}

    if isinstance(data, dict):
        for key, value in data.items():
            new_path = path + (key,)
            if isinstance(key, str) and _is_prompt_key(key) and isinstance(value, str):
                flat[new_path] = value
            else:
                flat.update(flatten_prompt_dict(value, new_path))
    elif isinstance(data, list):
        for i, value in enumerate(data):
            new_path = path + (f"[{i}]",)
            flat.update(flatten_prompt_dict(value, new_path))

    return flat


def _clean_texts(texts_list: List[str]) -> List[str]:
    """Clean and filter texts."""
    cleaned = []
    for text in texts_list:
        if text is None:
            continue
        if not isinstance(text, str):
            text = str(text)
        text = text.replace("\x00", "").strip()
        if text:  # Only keep non-empty strings
            cleaned.append(text)
    return cleaned


def load_20newsgroups_subset(n_samples=400, categories=None, random_state=42):
    """
    Load a subset of 20 Newsgroups dataset.
    
    Args:
        n_samples: Number of samples to load
        categories: List of category names (default: 4 categories)
        random_state: Random seed for reproducibility
        
    Returns:
        (texts, ground_truth_labels, dataset_name)
    """
    try:
        from sklearn.datasets import fetch_20newsgroups
    except ImportError:
        raise ImportError("scikit-learn is required for benchmark datasets. Install with: pip install scikit-learn")
    
    if categories is None:
        categories = ['alt.atheism', 'soc.religion.christian', 
                     'comp.graphics', 'rec.sport.hockey']
    
    newsgroups = fetch_20newsgroups(
        subset='train',
        categories=categories,
        shuffle=True,
        random_state=random_state,
        remove=('headers', 'footers', 'quotes')
    )
    
    # Sample n_samples
    rng = np.random.RandomState(random_state)
    indices = rng.choice(len(newsgroups.data), min(n_samples, len(newsgroups.data)), replace=False)
    texts = [newsgroups.data[i] for i in indices]
    labels = newsgroups.target[indices]
    
    # Filter short texts
    filtered_texts = []
    filtered_labels = []
    for t, l in zip(texts, labels):
        if len(t) > 50:
            filtered_texts.append(t)
            filtered_labels.append(l)
    
    return filtered_texts, np.array(filtered_labels), "20newsgroups_4cat"


def load_text_list_from_file(filepath: str) -> Tuple[List[str], Optional[np.ndarray], str]:
    """
    Load text list from file.
    
    Supports:
    - Text file: one text per line
    - JSON file: list of strings
    
    Args:
        filepath: Path to text or JSON file
        
    Returns:
        (texts, None, "text_list")
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Text file not found: {filepath}")
    
    if filepath.endswith('.json'):
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            texts = json.load(f)
        if not isinstance(texts, list):
            raise ValueError("JSON file must contain a list of strings")
    else:
        # Assume text file, one per line
        with open(filepath, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    
    return texts, None, "text_list"


def load_data(
    data_source: str,
    dict_file: Optional[str] = None,
    text_file: Optional[str] = None,
    benchmark_n_samples: int = 400,
) -> Tuple[List[str], Optional[np.ndarray], str]:
    """
    Load data based on data source type.
    
    Args:
        data_source: "dictionary", "benchmark", or "text_list"
        dict_file: Path to dictionary pickle/JSON file (for dictionary source)
        text_file: Path to text file (for text_list source)
        benchmark_n_samples: Number of samples for benchmark dataset
        
    Returns:
        (texts, ground_truth_labels, dataset_name)
    """
    if data_source == "dictionary":
        database_estela_dict = None
        
        # Try to load from pickle file first
        if dict_file and os.path.exists(dict_file):
            print(f"   Loading from pickle file: {dict_file}")
            with open(dict_file, "rb") as f:
                database_estela_dict = pickle.load(f)
        else:
            # Try to load from JSON file
            if dict_file and dict_file.endswith('.json') and os.path.exists(dict_file):
                print(f"   Loading from JSON file: {dict_file}")
                import json
                with open(dict_file, "r", encoding="utf-8") as f:
                    database_estela_dict = json.load(f)
            else:
                # Try environment variable
                prompt_dict_file = os.environ.get("PROMPT_DICT_FILE")
                if prompt_dict_file and os.path.exists(prompt_dict_file):
                    print(f"   Loading from pickle file (env): {prompt_dict_file}")
                    with open(prompt_dict_file, "rb") as f:
                        database_estela_dict = pickle.load(f)
                else:
                    prompt_dict_json = os.environ.get("PROMPT_DICT_JSON")
                    if prompt_dict_json and os.path.exists(prompt_dict_json):
                        print(f"   Loading from JSON file (env): {prompt_dict_json}")
                        import json
                        with open(prompt_dict_json, "r", encoding="utf-8") as f:
                            database_estela_dict = json.load(f)
                    else:
                        # Default location
                        default_file = "notebooks/estela_prompt_data.pkl"
                        if os.path.exists(default_file):
                            print(f"   Loading from default location: {default_file}")
                            with open(default_file, "rb") as f:
                                database_estela_dict = pickle.load(f)
                        else:
                            raise ValueError(
                                "No dictionary file found. Please provide --dict-file or set PROMPT_DICT_FILE environment variable."
                            )
        
        # Flatten prompts and extract texts
        flat_prompts = flatten_prompt_dict(database_estela_dict)
        texts = list(flat_prompts.values())
        texts = _clean_texts(texts)
        dataset_name = "prompt_dictionary"
        return texts, None, dataset_name
        
    elif data_source == "benchmark":
        texts, ground_truth_labels, dataset_name = load_20newsgroups_subset(
            n_samples=benchmark_n_samples,
            categories=None,
            random_state=42
        )
        return texts, ground_truth_labels, dataset_name
        
    elif data_source == "text_list":
        if not text_file:
            text_file = os.environ.get("TEXT_FILE")
            if not text_file:
                raise ValueError("--text-file or TEXT_FILE environment variable required for text_list source")
        texts, _, dataset_name = load_text_list_from_file(text_file)
        return texts, None, dataset_name
        
    else:
        raise ValueError(f"Unknown data_source: {data_source}. Must be 'dictionary', 'benchmark', or 'text_list'")


# ============================================================================
# Helper Functions
# ============================================================================

async def fetch_embeddings_async(
    texts_list: List[str],
    deployment: str,
    db: DatabaseConnectionV2,
) -> np.ndarray:
    """Fetch or create embeddings using EmbeddingService."""
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        service = EmbeddingService(repository=repo)

        # Create embedding requests
        requests = [
            EmbeddingRequest(text=text, deployment=deployment)
            for text in texts_list
        ]

        # Get embeddings (will use cache if available)
        responses = await service.get_embeddings_batch(requests)

        # Extract vectors
        embeddings = [resp.vector for resp in responses]

        return np.asarray(embeddings, dtype=np.float64)


def create_paraphraser_for_llm(
    llm_deployment: str, db: DatabaseConnectionV2
) -> callable:
    """Create a synchronous paraphraser function for a specific LLM deployment."""
    if llm_deployment is None:
        return None

    async def _paraphrase_batch_async(texts: List[str]) -> str:
        """Async wrapper for summarization.
        
        Combines multiple texts into a single summary for the cluster.
        Ensures all async resources are properly cleaned up.
        """
        try:
            with db.session_scope() as session:
                repo = RawCallRepository(session)
                service = SummarizationService(repository=repo)

                # Combine texts into a single prompt for summarization
                # Format: "Summarize the following texts into a single coherent summary:\n\nText 1: ...\n\nText 2: ..."
                combined_text = "Summarize the following texts into a single coherent summary:\n\n"
                for i, text in enumerate(texts, 1):
                    combined_text += f"Text {i}:\n{text}\n\n"
                
                # Summarize the combined text
                request = SummarizationRequest(
                    texts=[combined_text],  # Single combined text
                    llm_deployment=llm_deployment,
                    temperature=0.2,
                    max_tokens=256,  # Increased for combined summaries
                )

                result = await service.summarize_batch(request)
                # Return the single summary
                if result.summaries and len(result.summaries) > 0:
                    summary = result.summaries[0]
                else:
                    # Fallback: return first text if summarization fails
                    summary = texts[0] if texts else ""
            
            # Providers are now closed explicitly in summarize_batch's finally block
            # This ensures httpx clients are cleaned up before the event loop closes
            return summary
        except Exception as e:
            # Re-raise any errors
            raise

    def paraphrase_batch_sync(texts: List[str]) -> str:
        """Synchronous wrapper that returns a single summary string.
        
        Always creates a fresh event loop using asyncio.run() when called from
        ThreadPoolExecutor to ensure proper cleanup of async resources (httpx clients).
        This prevents "Task exception was never retrieved" errors from cleanup tasks.
        """
        # Clear any existing event loop in this thread to ensure asyncio.run() works
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                # Close and remove existing loop to allow asyncio.run() to create a new one
                loop.close()
            asyncio.set_event_loop(None)
        except RuntimeError:
            # No loop exists, which is fine
            pass
        
        # Always use asyncio.run() when in a thread pool executor
        # This creates a fresh loop and properly handles cleanup of all async resources
        # before closing the loop, preventing httpx client cleanup errors
        return asyncio.run(_paraphrase_batch_async(texts))

    return paraphrase_batch_sync


def save_results(
    all_results: Dict[str, Any],
    output_file: str = None,
    ground_truth_labels: Optional[np.ndarray] = None,
    dataset_name: str = "unknown",
) -> str:
    """Save results to a pickle file in the new format with metadata."""
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"pca_kllmeans_sweep_results_{timestamp}.pkl"

    # Prepare results for saving
    save_results = {}
    for summarizer_name, result in all_results.items():
        save_results[summarizer_name] = {
            "pca": result.pca,
            "by_k": {},
        }

        # Save distance matrices for later metric computation
        if result.Z is not None:
            save_results[summarizer_name]["Z"] = result.Z.tolist()
        if result.Z_norm is not None:
            save_results[summarizer_name]["Z_norm"] = result.Z_norm.tolist()
        if result.dist is not None:
            save_results[summarizer_name]["dist"] = result.dist.tolist()

        # Save all data for each K value
        for k, k_data in result.by_k.items():
            save_results[summarizer_name]["by_k"][k] = {
                "representatives": k_data.get("representatives", []),
                "labels": (
                    k_data.get("labels", []).tolist()
                    if hasattr(k_data.get("labels"), "tolist")
                    else k_data.get("labels", [])
                ),
                "labels_all": (
                    [
                        l.tolist() if hasattr(l, "tolist") else l
                        for l in k_data.get("labels_all", [])
                    ]
                    if k_data.get("labels_all") is not None
                    else None
                ),
                "objective": k_data.get("objective", {}),
                "objectives": k_data.get("objectives", []),
                "stability": k_data.get("stability"),
            }

    # Wrap in new format with metadata (compatible with notebook)
    final_data = {
        "summarizers": save_results,
        "ground_truth_labels": (
            ground_truth_labels.tolist()
            if ground_truth_labels is not None
            else None
        ),
        "dataset_name": dataset_name,
    }

    with open(output_file, "wb") as f:
        pickle.dump(final_data, f)

    return output_file


# ============================================================================
# Main Execution
# ============================================================================

async def main(
    data_source: str,
    dict_file: Optional[str] = None,
    text_file: Optional[str] = None,
    benchmark_n_samples: int = 400,
    output_file: Optional[str] = None,
):
    """Main execution function."""
    print("=" * 60)
    print("PCA KLLMeans Sweep Analysis")
    print("=" * 60)
    print(f"Data source: {data_source}")

    # Initialize database
    print("\n[INFO] Initializing database...")
    db = DatabaseConnectionV2(DATABASE_URL, enable_pgvector=True)
    db.init_db()
    print("[OK] Database initialized")

    # Load data
    print(f"\n[INFO] Loading data from {data_source}...")
    texts, ground_truth_labels, dataset_name = load_data(
        data_source=data_source,
        dict_file=dict_file,
        text_file=text_file,
        benchmark_n_samples=benchmark_n_samples,
    )
    print(f"[OK] Loaded {len(texts)} texts from {dataset_name}")

    # Filter texts that exceed token limit
    max_tokens = DEPLOYMENT_MAX_TOKENS.get(EMBEDDING_DEPLOYMENT)
    if max_tokens:
        print(f"\n[INFO] Filtering texts that exceed token limit ({max_tokens} tokens)...")
        valid_texts = []
        valid_indices = []
        filtered_texts = []
        filtered_indices = []
        
        for i, text in enumerate(texts):
            try:
                estimated = estimate_tokens(text, EMBEDDING_DEPLOYMENT)
                if estimated <= max_tokens:
                    valid_texts.append(text)
                    valid_indices.append(i)
                else:
                    filtered_texts.append((i, text, estimated))
                    filtered_indices.append(i)
            except Exception as e:
                # If estimation fails, include the text (better to try than skip)
                print(f"[WARN]  Failed to estimate tokens for text {i}: {e}")
                valid_texts.append(text)
                valid_indices.append(i)
        
        if filtered_texts:
            print(f"[WARN]  Filtered out {len(filtered_texts)} text(s) that exceed token limit:")
            for idx, text, est_tokens in filtered_texts[:5]:  # Show first 5
                print(f"   Index {idx}: {est_tokens} tokens (limit: {max_tokens})")
                print(f"      Preview: {text[:100]}...")
            if len(filtered_texts) > 5:
                print(f"   ... and {len(filtered_texts) - 5} more")
            
            # Update texts list to only include valid texts
            texts = valid_texts
            
            # Also filter ground truth labels if they exist
            if ground_truth_labels is not None:
                if isinstance(ground_truth_labels, np.ndarray):
                    ground_truth_labels = ground_truth_labels[valid_indices]
                else:
                    ground_truth_labels = [ground_truth_labels[i] for i in valid_indices]
                print(f"[INFO] Also filtered ground truth labels to match valid texts")
            
            print(f"[OK] Filtered texts: {len(texts)} valid, {len(filtered_texts)} filtered out")
        else:
            print(f"[OK] All {len(texts)} texts are within token limit")
    else:
        print(f"[WARN]  Unknown max tokens for {EMBEDDING_DEPLOYMENT}, skipping length validation")

    # Show samples
    print("\n[INFO] Sample texts:")
    for i, text in enumerate(texts[:3]):
        print(f"  {i+1}. {text[:100]}{'...' if len(text) > 100 else ''}")

    # Fetch embeddings
    print(f"\n[INFO] Fetching embeddings using {EMBEDDING_DEPLOYMENT}...")
    embeddings = await fetch_embeddings_async(texts, EMBEDDING_DEPLOYMENT, db)
    print(f"[OK] Got embeddings: shape {embeddings.shape}")

    # Create run group for provenance tracking
    print("\n[INFO] Creating run group for provenance tracking...")
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        provenance = ProvenanceService(repository=repo)

        run_group_id = provenance.create_run_group(
            algorithm="pca_kllmeans_sweep",
            name=f"pca_kllmeans_sweep_{dataset_name}_{EMBEDDING_DEPLOYMENT}",
            config={
                "data_source": data_source,
                "dataset_name": dataset_name,
                "embedding_deployment": EMBEDDING_DEPLOYMENT,
                "n_texts": len(texts),
                "k_range": f"{SWEEP_CONFIG.k_min}-{SWEEP_CONFIG.k_max}",
                "llm_summarizers": [
                    s if s else "None" for s in LLM_SUMMARIZERS
                ],
            },
        )
        print(f"[OK] Created run group: id={run_group_id}")

    # Run sweep for each LLM summarizer concurrently
    print("\n[INFO] Running sweeps concurrently...")
    
    async def run_single_sweep(llm_deployment: str) -> tuple[str, Any]:
        """Run a single sweep for a given LLM deployment."""
        summarizer_name = "None" if llm_deployment is None else llm_deployment
        
        # Create paraphraser and embedder
        paraphraser = create_paraphraser_for_llm(llm_deployment, db)
        
        # Create embedder function
        async def embedder_func(texts_list: List[str]) -> np.ndarray:
            """Embed texts using EmbeddingService."""
            return await fetch_embeddings_async(texts_list, EMBEDDING_DEPLOYMENT, db)
        
        def embedder_sync(texts_list: List[str]) -> np.ndarray:
            """Synchronous wrapper for embedder."""
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    return loop.run_until_complete(embedder_func(texts_list))
            except RuntimeError:
                pass
            return asyncio.run(embedder_func(texts_list))
        
        # Run sweep in thread pool executor (CPU-bound work)
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(
                executor,
                lambda: run_sweep(
                    texts, embeddings, SWEEP_CONFIG,
                    paraphraser=paraphraser,
                    embedder=embedder_sync if paraphraser else None
                )
            )
        
        print(f"[OK] Completed {summarizer_name}. Ks: {sorted([int(k) for k in result.by_k.keys()])}")
        return summarizer_name, result
    
    # Create tasks for all sweeps
    tasks = [
        run_single_sweep(llm_deployment)
        for llm_deployment in LLM_SUMMARIZERS
    ]
    
    # Run all sweeps concurrently with progress tracking
    results_list = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results and check for errors
    all_results = {}
    for result_item in results_list:
        if isinstance(result_item, Exception):
            print(f"[ERROR] Error in sweep: {result_item}")
            raise result_item
        summarizer_name, result = result_item
        all_results[summarizer_name] = result

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Data source: {data_source}")
    print(f"Dataset: {dataset_name}")
    print(f"Embedding deployment: {EMBEDDING_DEPLOYMENT}")
    print(f"Number of texts: {len(texts)}")
    print(f"K range: {SWEEP_CONFIG.k_min} to {SWEEP_CONFIG.k_max}")
    print(
        f"Summarizers tested: {len(LLM_SUMMARIZERS)} ({', '.join([s if s else 'None' for s in LLM_SUMMARIZERS])})"
    )
    print(f"\nResults structure: all_results[summarizer_name]['by_k'][k_value]")

    # Save results
    print("\n[INFO] Saving results...")
    saved_file = save_results(all_results, output_file, ground_truth_labels, dataset_name)
    print(f"[OK] Results saved to: {saved_file}")
    print(f"   Includes: representatives, labels, objectives, stability metrics, and distance matrices")
    print(f"\n   Example access (new format):")
    print(f"     loaded_data = pickle.load(open('{saved_file}', 'rb'))")
    print(f"     results = loaded_data['summarizers']")
    print(f"     ground_truth = loaded_data.get('ground_truth_labels')")
    print(f"     results['None']['by_k']['5']['stability']['silhouette']['mean']")
    print(f"     results['None']['by_k']['5']['representatives']")

    # Display results
    print(f"\n{'=' * 60}")
    print("RESULTS PREVIEW")
    print(f"{'=' * 60}")
    for summarizer_name, result in all_results.items():
        print(f"\n{summarizer_name}:")
        for k in sorted([int(k) for k in result.by_k.keys()])[:3]:  # Show first 3 K values
            k_data = result.by_k[str(k)]
            reps = k_data.get("representatives", [])
            print(f"  K={k}: {len(reps)} representatives")
            if k_data.get("stability"):
                stab = k_data["stability"]
                print(f"    Silhouette: {stab['silhouette']['mean']:.3f} ± {stab['silhouette']['std']:.3f}")
                print(f"    Stability ARI: {stab['stability_ari']['mean']:.3f} ± {stab['stability_ari']['std']:.3f}")

    print("\n[OK] Analysis complete!")
    return all_results, saved_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run PCA KLLMeans sweep with multiple data source support"
    )
    parser.add_argument(
        "--data-source",
        type=str,
        choices=["dictionary", "benchmark", "text_list"],
        default=os.environ.get("DATA_SOURCE", "benchmark"),
        help="Data source type: dictionary, benchmark, or text_list (default: benchmark)"
    )
    parser.add_argument(
        "--dict-file",
        type=str,
        default=None,
        help="Path to dictionary pickle/JSON file (for dictionary source)"
    )
    parser.add_argument(
        "--text-file",
        type=str,
        default=None,
        help="Path to text file (for text_list source)"
    )
    parser.add_argument(
        "--benchmark-n-samples",
        type=int,
        default=400,
        help="Number of samples for benchmark dataset (default: 400)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output pickle file path (default: auto-generated with timestamp)"
    )
    
    args = parser.parse_args()
    
    # Run the main function
    results, output_file = asyncio.run(main(
        data_source=args.data_source,
        dict_file=args.dict_file,
        text_file=args.text_file,
        benchmark_n_samples=args.benchmark_n_samples,
        output_file=args.output_file,
    ))
    print(f"\n[INFO] Results saved to: {output_file}")
