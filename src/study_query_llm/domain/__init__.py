"""
Domain models for problem representation and classification.

Provides conceptual structures (e.g., physics problem hierarchy)
that can be populated by parsers, extractors, or other tools.
"""

from .representation_hierarchy import (
    Level,
    SymmetryType,
    ConservationStatus,
    QuantityStatus,
    SolverType,
    Lagrangian,
    SymmetryEntry,
    SymmetryStructure,
    Quantity,
    HyperEdge,
    ConstraintHypergraph,
    SolutionStep,
    SolutionPath,
    PiGroup,
    DimensionalAnalysis,
    RepresentationHierarchy,
    build_orbital_example,
)

__all__ = [
    "Level",
    "SymmetryType",
    "ConservationStatus",
    "QuantityStatus",
    "SolverType",
    "Lagrangian",
    "SymmetryEntry",
    "SymmetryStructure",
    "Quantity",
    "HyperEdge",
    "ConstraintHypergraph",
    "SolutionStep",
    "SolutionPath",
    "PiGroup",
    "DimensionalAnalysis",
    "RepresentationHierarchy",
    "build_orbital_example",
]
