"""Shared Hugging Face dataset loaders for sweep experiments."""

from __future__ import annotations

import numpy as np


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
        raise ImportError(
            "Hugging Face datasets library required. Install with: pip install datasets"
        )

    dataset = load_dataset("dbpedia_14", split="train")

    texts = []
    labels = []

    for item in dataset:
        text = item.get("content", "")
        label = item.get("label", -1)

        if text and len(text) > 10 and len(text) <= 1000 and label >= 0:
            texts.append(text)
            labels.append(label)

    category_names = [
        "Company",
        "EducationalInstitution",
        "Artist",
        "Athlete",
        "OfficeHolder",
        "MeanOfTransportation",
        "Building",
        "NaturalPlace",
        "Village",
        "SportsTeam",
        "Information",
        "Animal",
        "Plant",
        "Album",
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
        raise ImportError(
            "Hugging Face datasets library required. Install with: pip install datasets"
        )

    dataset = load_dataset("yahoo_answers_topics", split="train")

    texts = []
    labels = []

    for item in dataset:
        title = item.get("question_title", "")
        content = item.get("question_content", "")
        answer = item.get("best_answer", "")
        label = item.get("topic", -1)

        text = f"{title}\n{content}\n{answer}".strip()

        if text and len(text) > 10 and len(text) <= 1000 and label >= 0:
            texts.append(text)
            labels.append(label)

    category_names = [
        "Society & Culture",
        "Science & Mathematics",
        "Health",
        "Education & Reference",
        "Computers & Internet",
        "Sports",
        "Business & Finance",
        "Entertainment & Music",
        "Family & Relationships",
        "Politics & Government",
    ]
    return texts, np.array(labels), category_names
