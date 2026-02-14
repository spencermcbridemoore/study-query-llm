#!/usr/bin/env python3
"""
Experimental PCA KLLMeans Sweep - Nested Parameter Exploration

This script runs nested experimental sweeps across multiple dimensions:
- entry_max: [100, 200, 300, 400, 500] - Maximum number of input samples
- benchmark sources: Different 20newsgroups category combinations
- label_max: [2, 6, 10, 18, 30] - Maximum unique ground truth labels allowed
- embedding/paraphraser: Combinations of embedding engines and LLM summarizers
- k values: 2 to 10

For each label_max, samples are collected with constraint:
- Sample until either entry_max entries OR label_max unique labels
- Once label_max is hit, only accept samples matching existing labels until entry_max is reached

Usage:
    python scripts/run_experimental_sweep.py
"""

import os
import sys
import asyncio
import argparse
import pickle
import time
import glob
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Output directory for pickle files
OUTPUT_DIR = Path(__file__).parent.parent / "experimental_results"
OUTPUT_DIR.mkdir(exist_ok=True)

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

# LLM deployments for summarization
LLM_SUMMARIZERS = [
    None,  # Non-LLM summaries (just use original representatives)
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-5-chat",
]

# Experimental parameters
ENTRY_MAX_VALUES = [100, 200, 300, 400, 500]
LABEL_MAX_VALUES = [1, 2, 3, 4, 5, 6]

# Benchmark sources: Multiple categorized text datasets
# Only include datasets with at least 6 categories
BENCHMARK_SOURCES = [
    # 20 Newsgroups variations (at least 6 categories)
    {
        "name": "20newsgroups_6cat",
        "type": "20newsgroups",
        "categories": ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'rec.sport.hockey', 
                      'sci.space', 'talk.politics.misc'],
    },
    {
        "name": "20newsgroups_10cat",
        "type": "20newsgroups",
        "categories": ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'rec.sport.hockey',
                      'sci.space', 'talk.politics.misc', 'comp.sys.mac.hardware', 'rec.autos',
                      'sci.crypt', 'talk.religion.misc'],
    },
    # Reuters-21578 (via Hugging Face datasets) - has many categories
    {
        "name": "reuters",
        "type": "reuters",
        "categories": None,  # All categories
    },
    # DBpedia (via Hugging Face datasets) - has 14 categories
    {
        "name": "dbpedia",
        "type": "dbpedia",
        "categories": None,  # 14 categories
    },
    # Yahoo Answers (via Hugging Face datasets) - has 10 categories
    {
        "name": "yahoo_answers",
        "type": "yahoo_answers",
        "categories": None,  # 10 categories
    },
    # BBC News (via Hugging Face datasets) - has 5 categories (may need to check)
    # Note: BBC News typically has 5 categories, so might not meet 6+ requirement
    # {
    #     "name": "bbc_news",
    #     "type": "bbc_news",
    #     "categories": None,
    # },
    # TREC-6 (via Hugging Face datasets) - has 6 categories
    {
        "name": "trec",
        "type": "trec",
        "categories": None,  # 6 categories
    },
    # News Category Dataset (via Hugging Face datasets) - has many categories
    {
        "name": "news_category",
        "type": "news_category",
        "categories": None,  # Many categories
    },
]

# Sweep configuration
SWEEP_CONFIG = SweepConfig(
    pca_dim=64,
    k_min=2,
    k_max=10,
    max_iter=200,
    base_seed=0,
    n_restarts=20,
    compute_stability=True,
    coverage_threshold=0.2,
    llm_interval=20,
    max_samples=10,
)

# Database URL (required)
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_20newsgroups_full(categories=None, random_state=42):
    """
    Load full 20 Newsgroups dataset (no sampling limit).
    
    Args:
        categories: List of category names
        random_state: Random seed for reproducibility
        
    Returns:
        (texts, ground_truth_labels, category_names)
    """
    try:
        from sklearn.datasets import fetch_20newsgroups
    except ImportError:
        raise ImportError("scikit-learn is required for benchmark datasets. Install with: pip install scikit-learn")
    
    newsgroups = fetch_20newsgroups(
        subset='train',
        categories=categories,
        shuffle=True,
        random_state=random_state,
        remove=('headers', 'footers', 'quotes')
    )
    
    # Filter texts: minimum 10 chars, maximum 1000 chars
    filtered_texts = []
    filtered_labels = []
    for t, l in zip(newsgroups.data, newsgroups.target):
        if len(t) > 10 and len(t) <= 1000:
            filtered_texts.append(t)
            filtered_labels.append(l)
    
    # Get category names
    category_names = [newsgroups.target_names[i] for i in sorted(set(filtered_labels))]
    
    return filtered_texts, np.array(filtered_labels), category_names


def load_reuters_full(random_state=42):
    """
    Load full Reuters-21578 dataset via Hugging Face datasets.
    
    Args:
        random_state: Random seed for reproducibility
        
    Returns:
        (texts, ground_truth_labels, category_names)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Hugging Face datasets library required for Reuters. Install with: pip install datasets")
    
    try:
        dataset = load_dataset("reuters21578", "ModApte", split="train")
        
        # Extract texts and labels
        texts = []
        labels = []
        label_to_id = {}
        next_id = 0
        
        for item in dataset:
            text = item.get('text', '')
            if text and len(text) > 10 and len(text) <= 1000:
                # Reuters has multiple topics per document
                topics = item.get('topics', [])
                if topics:
                    # Use first topic as primary label
                    topic = topics[0]
                    if topic not in label_to_id:
                        label_to_id[topic] = next_id
                        next_id += 1
                    texts.append(text)
                    labels.append(label_to_id[topic])
        
        category_names = sorted(label_to_id.keys())
        return texts, np.array(labels), category_names
    except Exception as e:
        raise ValueError(f"Failed to load Reuters dataset: {e}. You may need to install: pip install datasets")


def load_ag_news_full(random_state=42):
    """
    Load full AG News dataset (4 categories: World, Sports, Business, Sci/Tech).
    
    Args:
        random_state: Random seed for reproducibility
        
    Returns:
        (texts, ground_truth_labels, category_names)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Hugging Face datasets library required. Install with: pip install datasets")
    
    dataset = load_dataset("ag_news", split="train")
    
    texts = []
    labels = []
    
    for item in dataset:
        # AG News has 'text' and 'label' fields
        text = item.get('text', '')
        label = item.get('label', -1)
        
        if text and len(text) > 10 and len(text) <= 1000 and label >= 0:
            texts.append(text)
            labels.append(label)
    
    category_names = ['World', 'Sports', 'Business', 'Sci/Tech']
    return texts, np.array(labels), category_names


def load_dbpedia_full(random_state=42):
    """
    Load full DBpedia dataset (14 ontology classes).
    
    Args:
        random_state: Random seed for reproducibility
        
    Returns:
        (texts, ground_truth_labels, category_names)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Hugging Face datasets library required. Install with: pip install datasets")
    
    dataset = load_dataset("dbpedia_14", split="train")
    
    texts = []
    labels = []
    
    for item in dataset:
        # DBpedia has 'content' and 'label' fields
        text = item.get('content', '')
        label = item.get('label', -1)
        
        if text and len(text) > 10 and len(text) <= 1000 and label >= 0:
            texts.append(text)
            labels.append(label)
    
    category_names = [
        'Company', 'EducationalInstitution', 'Artist', 'Athlete',
        'OfficeHolder', 'MeanOfTransportation', 'Building', 'NaturalPlace',
        'Village', 'SportsTeam', 'Information', 'Animal', 'Plant', 'Album'
    ]
    return texts, np.array(labels), category_names


def load_yahoo_answers_full(random_state=42):
    """
    Load full Yahoo Answers dataset (10 categories).
    
    Args:
        random_state: Random seed for reproducibility
        
    Returns:
        (texts, ground_truth_labels, category_names)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Hugging Face datasets library required. Install with: pip install datasets")
    
    dataset = load_dataset("yahoo_answers_topics", split="train")
    
    texts = []
    labels = []
    
    for item in dataset:
        # Yahoo Answers has 'question_title', 'question_content', 'best_answer' and 'topic' fields
        # Combine question and answer for full context
        title = item.get('question_title', '')
        content = item.get('question_content', '')
        answer = item.get('best_answer', '')
        label = item.get('topic', -1)
        
        # Combine into single text
        text = f"{title}\n{content}\n{answer}".strip()
        
        if text and len(text) > 10 and len(text) <= 1000 and label >= 0:
            texts.append(text)
            labels.append(label)
    
    category_names = [
        'Society & Culture', 'Science & Mathematics', 'Health', 'Education & Reference',
        'Computers & Internet', 'Sports', 'Business & Finance', 'Entertainment & Music',
        'Family & Relationships', 'Politics & Government'
    ]
    return texts, np.array(labels), category_names


def load_trec_full(random_state=42):
    """
    Load full TREC dataset (6 question classification categories).
    
    Args:
        random_state: Random seed for reproducibility
        
    Returns:
        (texts, ground_truth_labels, category_names)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Hugging Face datasets library required. Install with: pip install datasets")
    
    dataset = load_dataset("trec", split="train")
    
    texts = []
    labels = []
    
    for item in dataset:
        text = item.get('text', '')
        label = item.get('label-coarse', -1)  # Use coarse-grained labels (6 categories)
        
        if text and len(text) > 10 and len(text) <= 1000 and label >= 0:  # TREC questions can be shorter
            texts.append(text)
            labels.append(label)
    
    category_names = [
        'Description', 'Entity', 'Abbreviation', 'Human', 'Location', 'Numeric'
    ]
    return texts, np.array(labels), category_names


def load_news_category_full(random_state=42):
    """
    Load full News Category Dataset (many categories).
    
    Args:
        random_state: Random seed for reproducibility
        
    Returns:
        (texts, ground_truth_labels, category_names)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Hugging Face datasets library required. Install with: pip install datasets")
    
    dataset = load_dataset("news_category", split="train")
    
    texts = []
    labels = []
    label_to_id = {}
    next_id = 0
    
    for item in dataset:
        # News Category has 'headline', 'short_description', 'category' fields
        headline = item.get('headline', '')
        description = item.get('short_description', '')
        category = item.get('category', '')
        
        text = f"{headline}\n{description}".strip()
        
        if text and len(text) > 10 and len(text) <= 1000 and category:
            if category not in label_to_id:
                label_to_id[category] = next_id
                next_id += 1
            texts.append(text)
            labels.append(label_to_id[category])
    
    category_names = sorted(label_to_id.keys())
    return texts, np.array(labels), category_names


def load_benchmark_dataset(source_config: Dict[str, Any], random_state: int = 42):
    """
    Load a benchmark dataset based on source configuration.
    
    Args:
        source_config: Dictionary with 'name', 'type', and optionally 'categories'
        random_state: Random seed for reproducibility
        
    Returns:
        (texts, ground_truth_labels, category_names)
    """
    dataset_type = source_config.get('type', '20newsgroups')
    
    if dataset_type == '20newsgroups':
        return load_20newsgroups_full(
            categories=source_config.get('categories'),
            random_state=random_state
        )
    elif dataset_type == 'reuters':
        return load_reuters_full(random_state=random_state)
    elif dataset_type == 'ag_news':
        return load_ag_news_full(random_state=random_state)
    elif dataset_type == 'dbpedia':
        return load_dbpedia_full(random_state=random_state)
    elif dataset_type == 'yahoo_answers':
        return load_yahoo_answers_full(random_state=random_state)
    elif dataset_type == 'trec':
        return load_trec_full(random_state=random_state)
    elif dataset_type == 'news_category':
        return load_news_category_full(random_state=random_state)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def sample_with_label_constraint(
    texts: List[str],
    labels: np.ndarray,
    entry_max: int,
    label_max: int,
    random_state: int = 42
) -> Tuple[List[str], np.ndarray]:
    """
    Sample texts with label_max constraint.
    
    Strategy:
    1. Sample until either entry_max entries OR label_max unique labels
    2. Once label_max is hit, only accept samples matching existing labels until entry_max is reached
    
    Args:
        texts: Full list of texts
        labels: Ground truth labels (aligned with texts)
        entry_max: Maximum number of entries to sample
        label_max: Maximum number of unique labels allowed
        random_state: Random seed
        
    Returns:
        (sampled_texts, sampled_labels)
    """
    rng = np.random.RandomState(random_state)
    n_total = len(texts)
    
    # Shuffle indices
    indices = np.arange(n_total)
    rng.shuffle(indices)
    
    sampled_texts = []
    sampled_labels = []
    seen_labels = set()
    
    for idx in indices:
        text = texts[idx]
        label = labels[idx]
        
        # Check if we've hit entry_max
        if len(sampled_texts) >= entry_max:
            break
        
        # Phase 1: Accept any label until we hit label_max
        if len(seen_labels) < label_max:
            sampled_texts.append(text)
            sampled_labels.append(label)
            seen_labels.add(label)
        else:
            # Phase 2: Only accept labels we've already seen
            if label in seen_labels:
                sampled_texts.append(text)
                sampled_labels.append(label)
    
    return sampled_texts, np.array(sampled_labels)


# ============================================================================
# Helper Functions
# ============================================================================

async def fetch_embeddings_async(
    texts_list: List[str],
    deployment: str,
    db: DatabaseConnectionV2,
    timeout: float = 600.0,  # 10 minutes timeout
) -> np.ndarray:
    """Fetch embeddings asynchronously with timeout."""
    async def _fetch():
        with db.session_scope() as session:
            repo = RawCallRepository(session)
            service = EmbeddingService(repository=repo)
            
            # Create embedding requests (one per text)
            requests = [
                EmbeddingRequest(text=text, deployment=deployment)
                for text in texts_list
            ]
            
            # Get embeddings (will use cache if available)
            responses = await service.get_embeddings_batch(requests)
            
            # Extract vectors
            embeddings = [resp.vector for resp in responses]
            
            return np.asarray(embeddings, dtype=np.float64)
    
    try:
        return await asyncio.wait_for(_fetch(), timeout=timeout)
    except asyncio.TimeoutError:
        raise TimeoutError(f"Embedding fetch timed out after {timeout} seconds for {len(texts_list)} texts")


def create_paraphraser_for_llm(
    llm_deployment: str, db: DatabaseConnectionV2
) -> callable:
    """Create a synchronous paraphraser function for a specific LLM deployment."""
    if llm_deployment is None:
        return None

    async def _paraphrase_batch_async(texts: List[str]) -> str:
        """Async wrapper for summarization with timeout."""
        async def _summarize():
            with db.session_scope() as session:
                repo = RawCallRepository(session)
                service = SummarizationService(repository=repo)

                combined_text = "Summarize the following texts into a single coherent summary:\n\n"
                for i, text in enumerate(texts, 1):
                    combined_text += f"Text {i}:\n{text}\n\n"
                
                request = SummarizationRequest(
                    texts=[combined_text],
                    llm_deployment=llm_deployment,
                    temperature=0.2,
                    max_tokens=256,
                )

                result = await service.summarize_batch(request)
                if result.summaries and len(result.summaries) > 0:
                    return result.summaries[0]
                else:
                    return texts[0] if texts else ""
        
        try:
            return await asyncio.wait_for(_summarize(), timeout=300.0)  # 5 minutes timeout
        except asyncio.TimeoutError:
            print(f"      [WARN] Summarization timed out, using first text as fallback")
            return texts[0] if texts else ""

    def paraphrase_batch_sync(texts: List[str]) -> str:
        """Synchronous wrapper that returns a single summary string."""
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                loop.close()
            asyncio.set_event_loop(None)
        except RuntimeError:
            pass
        
        return asyncio.run(_paraphrase_batch_async(texts))

    return paraphrase_batch_sync


def save_results(
    result: Any,
    output_file: str,
    ground_truth_labels: Optional[np.ndarray] = None,
    dataset_name: str = "unknown",
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Save results to a pickle file."""
    save_data = {
        "pca": result.pca,
        "by_k": {},
    }
    
    if result.Z is not None:
        save_data["Z"] = result.Z.tolist()
    if result.Z_norm is not None:
        save_data["Z_norm"] = result.Z_norm.tolist()
    if result.dist is not None:
        save_data["dist"] = result.dist.tolist()

    for k, k_data in result.by_k.items():
        save_data["by_k"][k] = {
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

    final_data = {
        "result": save_data,
        "ground_truth_labels": (
            ground_truth_labels.tolist()
            if ground_truth_labels is not None
            else None
        ),
        "dataset_name": dataset_name,
        "metadata": metadata or {},
    }

    with open(output_file, "wb") as f:
        pickle.dump(final_data, f)

    return output_file


# ============================================================================
# Main Execution
# ============================================================================

async def run_single_sweep(
    texts: List[str],
    embeddings: np.ndarray,
    llm_deployment: Optional[str],
    db: DatabaseConnectionV2,
) -> Any:
    """Run a single sweep for a given LLM deployment."""
    summarizer_name = "None" if llm_deployment is None else llm_deployment
    
    paraphraser = create_paraphraser_for_llm(llm_deployment, db)
    
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
    
    return result


async def main():
    """Main execution function."""
    print("=" * 80)
    print("Experimental PCA KLLMeans Sweep - Nested Parameter Exploration")
    print("=" * 80)
    
    # Initialize database
    print("\n[INFO] Initializing database...")
    db = DatabaseConnectionV2(DATABASE_URL, enable_pgvector=True)
    db.init_db()
    print("[OK] Database initialized")
    
    total_runs = (
        len(ENTRY_MAX_VALUES) *
        len(BENCHMARK_SOURCES) *
        len(LABEL_MAX_VALUES) *
        len(LLM_SUMMARIZERS)
    )
    print(f"\n[INFO] Total experimental runs: {total_runs}")
    print(f"   entry_max values: {ENTRY_MAX_VALUES}")
    print(f"   benchmark sources: {len(BENCHMARK_SOURCES)}")
    print(f"   label_max values: {LABEL_MAX_VALUES}")
    print(f"   summarizers: {len(LLM_SUMMARIZERS)}")
    
    run_count = 0
    start_time = time.time()
    
    # Load full benchmark datasets once
    print("\n[INFO] Loading full benchmark datasets...")
    print("   Filtering: 10 < length <= 1000 characters")
    print("   Requirement: At least 500 valid datapoints per dataset")
    benchmark_data = {}
    for source in BENCHMARK_SOURCES:
        print(f"   Loading {source['name']} ({source.get('type', 'unknown')})...")
        try:
            texts, labels, category_names = load_benchmark_dataset(source, random_state=42)
            
            # Validate minimum datapoint requirement
            if len(texts) < 500:
                print(f"      [WARN] Only {len(texts)} valid datapoints (need at least 500)")
                print(f"      Skipping this dataset for this run")
                continue
            
            benchmark_data[source['name']] = {
                'texts': texts,
                'labels': labels,
                'category_names': category_names,
                'source_config': source,
            }
            print(f"      [OK] Loaded {len(texts)} texts, {len(set(labels))} unique labels")
        except Exception as e:
            print(f"      [WARN] Failed to load {source['name']}: {e}")
            print(f"      Skipping this dataset for this run")
    
    # Nested loops
    for entry_max in ENTRY_MAX_VALUES:
        print(f"\n{'='*80}")
        print(f"ENTRY_MAX: {entry_max}")
        print(f"{'='*80}")
        
        for benchmark_source in BENCHMARK_SOURCES:
            source_name = benchmark_source['name']
            
            # Skip sources that failed to load
            if source_name not in benchmark_data:
                print(f"\n  BENCHMARK: {source_name} (SKIP - failed to load)")
                continue
            
            print(f"\n  BENCHMARK: {source_name}")
            
            full_texts = benchmark_data[source_name]['texts']
            full_labels = benchmark_data[source_name]['labels']
            
            for label_max in LABEL_MAX_VALUES:
                print(f"\n    LABEL_MAX: {label_max}")
                
                # Sample with label_max constraint
                sampled_texts, sampled_labels = sample_with_label_constraint(
                    full_texts, full_labels, entry_max, label_max, random_state=42
                )
                
                actual_entry_count = len(sampled_texts)
                actual_label_count = len(set(sampled_labels))
                
                print(f"      Sampled: {actual_entry_count} entries, {actual_label_count} unique labels")
                
                # Filter texts that exceed token limit
                max_tokens = DEPLOYMENT_MAX_TOKENS.get(EMBEDDING_DEPLOYMENT)
                if max_tokens:
                    valid_texts = []
                    valid_indices = []
                    for i, text in enumerate(sampled_texts):
                        try:
                            estimated = estimate_tokens(text, EMBEDDING_DEPLOYMENT)
                            if estimated <= max_tokens:
                                valid_texts.append(text)
                                valid_indices.append(i)
                        except Exception:
                            valid_texts.append(text)
                            valid_indices.append(i)
                    
                    if len(valid_texts) < len(sampled_texts):
                        print(f"      Filtered: {len(valid_texts)} valid, {len(sampled_texts) - len(valid_texts)} filtered out")
                        sampled_texts = valid_texts
                        sampled_labels = sampled_labels[valid_indices]
                
                # Fetch embeddings
                print(f"      Fetching embeddings...")
                try:
                    embeddings = await fetch_embeddings_async(sampled_texts, EMBEDDING_DEPLOYMENT, db)
                    print(f"      [OK] Fetched {len(embeddings)} embeddings")
                except TimeoutError as e:
                    print(f"      [ERROR] Embedding fetch timed out: {e}")
                    print(f"      [INFO] Skipping this configuration...")
                    continue
                except Exception as e:
                    print(f"      [ERROR] Embedding fetch failed: {e}")
                    print(f"      [INFO] Skipping this configuration...")
                    import traceback
                    traceback.print_exc()
                    continue
                
                # Run sweep for each LLM summarizer sequentially (save separate pickle for each)
                # Run one at a time to avoid overwhelming the system
                for llm_deployment in LLM_SUMMARIZERS:
                    run_count += 1
                    summarizer_name = "None" if llm_deployment is None else llm_deployment
                    
                    # Check if this run already exists (skip if found)
                    existing_files = list(glob.glob(
                        str(OUTPUT_DIR / (
                            f"experimental_sweep_"
                            f"entry{entry_max}_"
                            f"{source_name}_"
                            f"labelmax{label_max}_"
                            f"{summarizer_name.replace('-', '_')}_*.pkl"
                        ))
                    ))
                    if existing_files:
                        print(f"\n        [{run_count}/{total_runs}] Summarizer: {summarizer_name} (SKIP - already exists)")
                        continue
                    
                    print(f"\n        [{run_count}/{total_runs}] Summarizer: {summarizer_name}")
                    
                    # Run sweep - wrap in try/except to ensure file is saved even if DB fails
                    result = None
                    try:
                        # Add timeout for sweep execution (30 minutes)
                        result = await asyncio.wait_for(
                            run_single_sweep(sampled_texts, embeddings, llm_deployment, db),
                            timeout=1800.0
                        )
                    except asyncio.TimeoutError:
                        print(f"        [ERROR] Sweep execution timed out after 30 minutes")
                        print(f"        [INFO] Skipping this run and continuing...")
                        continue
                    except Exception as e:
                        print(f"        [WARN] Sweep execution failed: {e}")
                        print(f"        [INFO] Will attempt to save any partial results...")
                        import traceback
                        traceback.print_exc()
                        # If sweep failed, we can't save results, but continue to next
                        continue
                    
                    # Generate output filename
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_file = str(OUTPUT_DIR / (
                        f"experimental_sweep_"
                        f"entry{entry_max}_"
                        f"{source_name}_"
                        f"labelmax{label_max}_"
                        f"{summarizer_name.replace('-', '_')}_"
                        f"{timestamp}.pkl"
                    ))
                    
                    # Save results with metadata - ensure this happens even if DB operations fail
                    metadata = {
                        "entry_max": entry_max,
                        "label_max": label_max,
                        "actual_entry_count": actual_entry_count,
                        "actual_label_count": actual_label_count,
                        "benchmark_source": source_name,
                        "categories": benchmark_source.get('categories'),
                        "summarizer": summarizer_name,
                        "embedding_deployment": EMBEDDING_DEPLOYMENT,
                        "sweep_config": {
                            "pca_dim": SWEEP_CONFIG.pca_dim,
                            "k_min": SWEEP_CONFIG.k_min,
                            "k_max": SWEEP_CONFIG.k_max,
                            "max_iter": SWEEP_CONFIG.max_iter,
                            "n_restarts": SWEEP_CONFIG.n_restarts,
                        },
                    }
                    
                    try:
                        saved_file = save_results(
                            result,
                            output_file,
                            ground_truth_labels=sampled_labels,
                            dataset_name=source_name,
                            metadata=metadata,
                        )
                        print(f"        [OK] Saved to: {saved_file}")
                    except Exception as e:
                        print(f"        [ERROR] Failed to save results file: {e}")
                        print(f"        [WARN] Results were computed but could not be saved!")
                        import traceback
                        traceback.print_exc()
                        # Continue anyway - don't let file save failure stop the experiment
                    
                    # Progress update
                    elapsed = time.time() - start_time
                    avg_time = elapsed / run_count
                    remaining = (total_runs - run_count) * avg_time
                    print(f"        Progress: {run_count}/{total_runs} ({100*run_count/total_runs:.1f}%)")
                    print(f"        ETA: {remaining/3600:.1f} hours")
    
    total_elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"[OK] All experimental runs completed!")
    print(f"   Total runs: {run_count}")
    print(f"   Total time: {total_elapsed/3600:.2f} hours")
    print(f"{'='*80}")


if __name__ == "__main__":
    asyncio.run(main())
