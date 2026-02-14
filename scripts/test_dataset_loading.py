#!/usr/bin/env python3
"""Test dataset loading for the three problematic datasets."""

from datasets import load_dataset

print("Testing dataset loading...\n")

# Test Reuters - try alternative version
print("1. Testing Reuters alternatives...")
reuters_alternatives = [
    ("malteos/cmu_reuters", None),
    ("reuters21578", "ModApte"),
]
for repo, config in reuters_alternatives:
    try:
        if config:
            dataset = load_dataset(repo, config, split="train")
        else:
            dataset = load_dataset(repo, split="train")
        print(f"   [OK] {repo} loaded: {len(dataset)} samples")
        break
    except Exception as e:
        print(f"   [FAIL] {repo}: {str(e)[:80]}")

# Test TREC - try alternative version
print("\n2. Testing TREC alternatives...")
trec_alternatives = [
    ("CogComp/trec", None),
    ("trec", None),
]
for repo, config in trec_alternatives:
    try:
        if config:
            dataset = load_dataset(repo, config, split="train")
        else:
            dataset = load_dataset(repo, split="train")
        print(f"   [OK] {repo} loaded: {len(dataset)} samples")
        break
    except Exception as e:
        print(f"   [FAIL] {repo}: {str(e)[:80]}")

# Test News Category
print("\n3. Testing News Category alternatives...")
news_alternatives = [
    ("Fraser/news-category-dataset", None),
    ("heegyu/news-category-dataset", None),
]
for repo, config in news_alternatives:
    try:
        if config:
            dataset = load_dataset(repo, config, split="train")
        else:
            dataset = load_dataset(repo, split="train")
        print(f"   [OK] {repo} loaded: {len(dataset)} samples")
        print(f"   Sample fields: {list(dataset[0].keys())}")
        break
    except Exception as e:
        print(f"   [FAIL] {repo}: {str(e)[:80]}")

print("\nDone!")
