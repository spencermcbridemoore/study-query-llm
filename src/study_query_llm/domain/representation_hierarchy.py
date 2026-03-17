"""
Representation Hierarchy for Physics Problems
==============================================

Four abstraction levels forming a refinement lattice:

    Level 0: Lagrangian/Hamiltonian  (generative object)
        ↓  apply Noether's theorem
    Level 1: Symmetry/Conservation    (problem-class structure)
        ↓  instantiate with specific variables + ICs
    Level 2: Constraint Hypergraph    (problem-instance structure)
        ↓  choose traversal strategy
    Level 3: Solution Path            (executable computation)

Each downward arrow REQUIRES additional information (injection).
Each upward arrow is a lossy projection (forgetting).

Dimensional analysis is orthogonal — it reads the dimension matrix,
not the equations, and can be applied at levels 0–2.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import sympy as sp


# ── Enums ──────────────────────────────────────────────────────────────

class Level(Enum):
    """Abstraction level in the hierarchy."""
    LAGRANGIAN  = 0  # Full dynamics, most compact
    SYMMETRY    = 1  # Conservation structure (Noether)
    HYPERGRAPH  = 2  # Named variables + law-hyperedges
    SOLUTION    = 3  # Ordered traversal / execution plan

    def __gt__(self, other):  return self.value > other.value
    def __ge__(self, other):  return self.value >= other.value
    def __lt__(self, other):  return self.value < other.value
    def __le__(self, other):  return self.value <= other.value

    def implies(self, other: Level) -> bool:
        """A higher level implies all lower levels.
        Having a solution path (3) implies you have the hypergraph (2),
        which implies you know conservation laws (1), which implies
        you have the Lagrangian (0)."""
        return self >= other

    def requires_to_reach(self, target: Level) -> set[str]:
        """What additional information is needed to go from self → target."""
        injections = {
            (0, 1): {"symmetry_analysis"},       # Apply Noether to L
            (1, 2): {"variable_instantiation",    # Name specific quantities
                     "initial_conditions",         # Specify ICs
                     "known_unknown_partition"},    # Which are given vs sought
            (2, 3): {"traversal_strategy"},        # Choose solution method
        }
        if target <= self:
            return set()  # Upward = projection, no new info needed
        needed = set()
        for step in range(self.value, target.value):
            needed |= injections.get((step, step + 1), set())
        return needed


class SymmetryType(Enum):
    """Types of continuous symmetry relevant to Noether."""
    ROTATION    = auto()  # SO(2), SO(3)
    TRANSLATION_TIME   = auto()  # ℝ time
    TRANSLATION_SPACE  = auto()  # ℝ^n spatial
    BOOST       = auto()  # Galilean / Lorentz
    GAUGE       = auto()  # U(1), SU(N), etc.


class ConservationStatus(Enum):
    CONSERVED = auto()
    BROKEN    = auto()


class QuantityStatus(Enum):
    GIVEN   = auto()
    UNKNOWN = auto()
    DERIVED = auto()  # computed during solution


class SolverType(Enum):
    """Level-3 solution strategy variants."""
    PROPAGATOR = auto()  # Sussman-style cell network
    GROEBNER   = auto()  # Polynomial elimination
    SYMPY_EL   = auto()  # Euler-Lagrange via SymPy
    SMT_Z3     = auto()  # Constraint satisfaction
    NUMERICAL  = auto()  # ODE integration


# ── Level 0: Lagrangian / Hamiltonian ──────────────────────────────────

@dataclass
class Lagrangian:
    """Level 0 — the generative object.

    Everything downstream derives from this. The Lagrangian IS the physics.
    """
    level = Level.LAGRANGIAN

    expr: sp.Expr                          # ℒ(q, q̇, t)
    generalized_coords: list[sp.Symbol]    # q_i
    generalized_velocities: list[sp.Symbol]  # q̇_i
    parameters: dict[sp.Symbol, Optional[float]] = field(default_factory=dict)
    name: str = ""

    @property
    def hamiltonian(self) -> sp.Expr:
        """Legendre transform: H = Σ pᵢq̇ᵢ − L"""
        H = -self.expr
        for q, qdot in zip(self.generalized_coords, self.generalized_velocities):
            p = sp.diff(self.expr, qdot)  # conjugate momentum
            H += p * qdot
        return sp.simplify(H)

    def euler_lagrange(self) -> list[sp.Expr]:
        """∂L/∂qᵢ − d/dt(∂L/∂q̇ᵢ) = 0  →  returns the LHS expressions."""
        t = sp.Symbol('t')
        eoms = []
        for q, qdot in zip(self.generalized_coords, self.generalized_velocities):
            dL_dq = sp.diff(self.expr, q)
            dL_dqdot = sp.diff(self.expr, qdot)
            # For symbolic analysis, we note which terms vanish
            eoms.append(sp.simplify(dL_dq - sp.diff(dL_dqdot, t)))
        return eoms

    def cyclic_coordinates(self) -> set[sp.Symbol]:
        """Coordinates q where ∂L/∂q = 0  →  conjugate momentum conserved."""
        return {
            q for q in self.generalized_coords
            if sp.simplify(sp.diff(self.expr, q)) == 0
        }

    def is_time_independent(self) -> bool:
        t = sp.Symbol('t')
        return sp.diff(self.expr, t) == 0

    # ── Downward projection: L0 → L1 ──
    def extract_symmetries(self) -> SymmetryStructure:
        """Apply Noether's theorem: cyclic coords → conservation laws.
        This is the L0 → L1 arrow."""
        symmetries = []
        cyclic = self.cyclic_coordinates()

        for q in self.generalized_coords:
            # Heuristic mapping from coordinate name to symmetry type
            # (In a full implementation, you'd do Lie derivative analysis)
            sym_type = _guess_symmetry_type(q)
            conserved_name = _noether_map(sym_type)
            status = (ConservationStatus.CONSERVED if q in cyclic
                      else ConservationStatus.BROKEN)
            symmetries.append(SymmetryEntry(
                symmetry=sym_type,
                status=status,
                conserved_quantity=conserved_name if status == ConservationStatus.CONSERVED else None,
                reason=f"∂L/∂{q} = 0" if status == ConservationStatus.CONSERVED
                       else f"∂L/∂{q} ≠ 0"
            ))

        # Time translation
        if self.is_time_independent():
            symmetries.append(SymmetryEntry(
                symmetry=SymmetryType.TRANSLATION_TIME,
                status=ConservationStatus.CONSERVED,
                conserved_quantity="energy",
                reason="∂L/∂t = 0"
            ))
        else:
            symmetries.append(SymmetryEntry(
                symmetry=SymmetryType.TRANSLATION_TIME,
                status=ConservationStatus.BROKEN,
                conserved_quantity=None,
                reason="∂L/∂t ≠ 0"
            ))

        return SymmetryStructure(
            symmetries=symmetries,
            symmetry_group=_format_group(symmetries),
            source_lagrangian=self
        )


# ── Level 1: Symmetry / Conservation (Noether) ────────────────────────

@dataclass
class SymmetryEntry:
    symmetry: SymmetryType
    status: ConservationStatus
    conserved_quantity: Optional[str]  # e.g. "angular_momentum", "energy"
    reason: str

    def __repr__(self):
        arrow = "→" if self.status == ConservationStatus.CONSERVED else "↛"
        qty = self.conserved_quantity or "N/A"
        return f"{self.symmetry.name} {arrow} {qty} ({self.reason})"


@dataclass
class SymmetryStructure:
    """Level 1 — problem-class knowledge, not instance-specific.

    Applies to ALL central-force problems, not just this one with r₀=8e6.
    """
    level = Level.SYMMETRY

    symmetries: list[SymmetryEntry]
    symmetry_group: str  # e.g. "SO(2) × ℝ"
    source_lagrangian: Optional[Lagrangian] = None

    @property
    def conserved_quantities(self) -> set[str]:
        return {s.conserved_quantity for s in self.symmetries
                if s.status == ConservationStatus.CONSERVED and s.conserved_quantity}

    @property
    def broken_symmetries(self) -> list[SymmetryEntry]:
        return [s for s in self.symmetries if s.status == ConservationStatus.BROKEN]

    def implies_conservation_of(self, quantity: str) -> bool:
        return quantity in self.conserved_quantities

    # ── Upward projection: L1 → L0 (lossy — can't reconstruct L from symmetries alone)
    def project_up(self) -> Optional[Lagrangian]:
        """Return source Lagrangian if available. Cannot reconstruct from
        symmetries alone — that's the lossy direction."""
        return self.source_lagrangian

    # ── Downward: L1 → L2 requires instantiation ──
    def instantiate(self,
                    quantities: list[Quantity],
                    laws: list[HyperEdge],
                    initial_conditions: Optional[dict] = None
                    ) -> ConstraintHypergraph:
        """Inject problem-instance data to descend to Level 2.

        This is where abstract conservation laws become concrete:
        'angular momentum conserved' → L = r₀v₀ with r₀ GIVEN, L UNKNOWN.
        """
        return ConstraintHypergraph(
            quantities=quantities,
            hyperedges=laws,
            initial_conditions=initial_conditions or {},
            source_symmetry=self
        )


# ── Level 2: Constraint Hypergraph ────────────────────────────────────

@dataclass
class Quantity:
    """A named physical variable with units and known/unknown status."""
    name: str
    symbol: sp.Symbol
    status: QuantityStatus
    units: str = ""
    value: Optional[float] = None
    depth: Optional[int] = None  # topological depth in the graph

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, Quantity) and self.name == other.name

    def __repr__(self):
        val = f" = {self.value}" if self.value is not None else ""
        return f"{self.name}({self.status.name}{val})"


@dataclass
class HyperEdge:
    """A physical law connecting input quantities to an output quantity.

    This is a directed hyperedge: inputs → output via some equation.
    """
    name: str                    # e.g. "vis-viva", "Kepler III"
    inputs: set[str]             # names of input quantities
    output: str                  # name of output quantity
    equation: Optional[sp.Expr] = None  # symbolic form (LHS - RHS = 0)
    degree: int = 1              # polynomial degree in output variable
    is_linear: bool = True

    def __repr__(self):
        ins = ", ".join(sorted(self.inputs))
        lin = "" if self.is_linear else " [nonlinear]"
        return f"{self.name}: {{{ins}}} → {self.output}{lin}"


@dataclass
class ConstraintHypergraph:
    """Level 2 — the problem instance, fully specified but unsolved.

    This is what a physics educator means by 'the problem structure.'
    Variables are named, laws are explicit, but no solution order is imposed.
    """
    level = Level.HYPERGRAPH

    quantities: list[Quantity]
    hyperedges: list[HyperEdge]
    initial_conditions: dict[str, str]  # e.g. {"ic_type": "turning_point"}
    source_symmetry: Optional[SymmetryStructure] = None

    def __post_init__(self):
        self._qty_map = {q.name: q for q in self.quantities}
        self._compute_depths()

    @property
    def givens(self) -> set[Quantity]:
        return {q for q in self.quantities if q.status == QuantityStatus.GIVEN}

    @property
    def unknowns(self) -> set[Quantity]:
        return {q for q in self.quantities if q.status == QuantityStatus.UNKNOWN}

    @property
    def chain_depth(self) -> int:
        """Maximum topological depth — a difficulty proxy for PER."""
        return max((q.depth or 0) for q in self.quantities)

    def _compute_depths(self):
        """Topological sort to assign propagation depth to each quantity."""
        # Givens are depth 0
        for q in self.quantities:
            if q.status == QuantityStatus.GIVEN:
                q.depth = 0

        changed = True
        while changed:
            changed = False
            for edge in self.hyperedges:
                input_depths = []
                for inp_name in edge.inputs:
                    q = self._qty_map.get(inp_name)
                    if q and q.depth is not None:
                        input_depths.append(q.depth)
                    else:
                        break
                else:
                    # All inputs have depths assigned
                    out_q = self._qty_map.get(edge.output)
                    if out_q and out_q.depth is None:
                        out_q.depth = max(input_depths) + 1
                        changed = True

    def hub_nodes(self, min_degree: int = 2) -> list[Quantity]:
        """Quantities that feed into many laws (high out-degree)."""
        degree = {q.name: 0 for q in self.quantities}
        for edge in self.hyperedges:
            for inp in edge.inputs:
                degree[inp] = degree.get(inp, 0) + 1
        return [self._qty_map[n] for n, d in degree.items()
                if d >= min_degree and n in self._qty_map]

    def leaf_nodes(self) -> list[Quantity]:
        """Quantities that are produced but never consumed."""
        consumed = set()
        for edge in self.hyperedges:
            consumed |= edge.inputs
        produced = {edge.output for edge in self.hyperedges}
        return [self._qty_map[n] for n in (produced - consumed)
                if n in self._qty_map]

    def subgraph_for(self, target: str) -> ConstraintHypergraph:
        """Extract the minimal subgraph needed to solve for `target`."""
        needed_edges = []
        needed_vars = set()
        frontier = {target}
        while frontier:
            var = frontier.pop()
            needed_vars.add(var)
            for edge in self.hyperedges:
                if edge.output == var:
                    needed_edges.append(edge)
                    for inp in edge.inputs:
                        if inp not in needed_vars:
                            frontier.add(inp)
        return ConstraintHypergraph(
            quantities=[q for q in self.quantities if q.name in needed_vars],
            hyperedges=needed_edges,
            initial_conditions=self.initial_conditions,
            source_symmetry=self.source_symmetry
        )

    # ── Upward: L2 → L1 (project out instance details) ──
    def project_up(self) -> Optional[SymmetryStructure]:
        """Forget variable names, values, ICs — return conservation structure."""
        return self.source_symmetry

    # ── Downward: L2 → L3 (impose traversal) ──
    def solve_with(self, strategy: SolverType) -> SolutionPath:
        """Choose a solution method to descend to Level 3."""
        # Compute the topological firing order
        ordered_edges = sorted(
            self.hyperedges,
            key=lambda e: self._qty_map.get(e.output, Quantity("?", sp.Symbol("?"), QuantityStatus.UNKNOWN)).depth or 999
        )
        return SolutionPath(
            strategy=strategy,
            steps=[
                SolutionStep(
                    order=i,
                    edge=edge,
                    inputs_available={inp for inp in edge.inputs
                                      if self._qty_map.get(inp, Quantity("?", sp.Symbol("?"), QuantityStatus.UNKNOWN)).status == QuantityStatus.GIVEN
                                      or any(e.output == inp for e in ordered_edges[:i])},
                    output=edge.output
                )
                for i, edge in enumerate(ordered_edges)
            ],
            source_hypergraph=self
        )

    def __repr__(self):
        givens = ", ".join(q.name for q in self.givens)
        unknowns = ", ".join(q.name for q in self.unknowns)
        return (f"Hypergraph(givens={{{givens}}}, unknowns={{{unknowns}}}, "
                f"depth={self.chain_depth}, edges={len(self.hyperedges)})")


# ── Level 3: Solution Path ────────────────────────────────────────────

@dataclass
class SolutionStep:
    order: int
    edge: HyperEdge
    inputs_available: set[str]
    output: str

    def __repr__(self):
        return f"Step {self.order}: {self.edge.name} → {self.output}"


@dataclass
class SolutionPath:
    """Level 3 — an ordered traversal of the constraint graph.

    This is the executable computation plan.
    """
    level = Level.SOLUTION

    strategy: SolverType
    steps: list[SolutionStep]
    source_hypergraph: Optional[ConstraintHypergraph] = None

    @property
    def firing_order(self) -> list[str]:
        """The sequence of quantities computed, in order."""
        return [s.output for s in self.steps]

    @property
    def parallelizable_groups(self) -> list[list[SolutionStep]]:
        """Group steps that can fire simultaneously (same depth)."""
        if not self.source_hypergraph:
            return [[s] for s in self.steps]
        groups: dict[int, list[SolutionStep]] = {}
        for step in self.steps:
            q = self.source_hypergraph._qty_map.get(step.output)
            d = q.depth if q else 0
            groups.setdefault(d, []).append(step)
        return [groups[k] for k in sorted(groups)]

    # ── Upward: L3 → L2 (strip ordering) ──
    def project_up(self) -> Optional[ConstraintHypergraph]:
        """Forget the traversal order — return the unordered constraint structure."""
        return self.source_hypergraph

    def __repr__(self):
        steps = " → ".join(self.firing_order)
        return f"SolutionPath({self.strategy.name}: {steps})"


# ── Dimensional Analysis (Orthogonal Axis) ─────────────────────────────

@dataclass
class PiGroup:
    """A single dimensionless Π group."""
    name: str
    expression: str       # e.g. "v₀²r₀/GM"
    value: Optional[float] = None
    redundant_with: Optional[str] = None  # e.g. "Π₁"

@dataclass
class DimensionalAnalysis:
    """Orthogonal to the 4-level hierarchy — reads the dimension matrix."""
    quantities: list[str]
    dimension_matrix_rank: int
    n_pi_groups: int          # = n_quantities - rank
    pi_groups: list[PiGroup]

    @property
    def redundancies(self) -> list[tuple[str, str]]:
        """Pairs of Π groups that are algebraically identical."""
        return [(g.name, g.redundant_with) for g in self.pi_groups
                if g.redundant_with]

    def classify_orbit(self, pi1_value: float) -> str:
        """Π₁ = v₀²r₀/GM classifies orbit type before solving."""
        if pi1_value < 2:
            return "bound_elliptic"
        elif pi1_value == 2:
            return "parabolic_escape"
        else:
            return "hyperbolic"


# ── The Full Hierarchy Container ───────────────────────────────────────

@dataclass
class RepresentationHierarchy:
    """Container for the full 4-level stack for a single physics problem.

    Encodes the implication structure:
        solution.implies(hypergraph)  →  True
        hypergraph.implies(symmetry)  →  True
        symmetry.implies(lagrangian)  →  True

    And the injection requirements:
        lagrangian → symmetry:   needs symmetry_analysis
        symmetry → hypergraph:   needs variable_instantiation + ICs
        hypergraph → solution:   needs traversal_strategy
    """
    lagrangian: Optional[Lagrangian] = None
    symmetry: Optional[SymmetryStructure] = None
    hypergraph: Optional[ConstraintHypergraph] = None
    solution: Optional[SolutionPath] = None
    dimensional: Optional[DimensionalAnalysis] = None

    @property
    def highest_level(self) -> Level:
        """What's the most specific representation we currently have?"""
        if self.solution:    return Level.SOLUTION
        if self.hypergraph:  return Level.HYPERGRAPH
        if self.symmetry:    return Level.SYMMETRY
        if self.lagrangian:  return Level.LAGRANGIAN
        raise ValueError("Empty hierarchy — no representations loaded")

    def has_level(self, level: Level) -> bool:
        return {
            Level.LAGRANGIAN: self.lagrangian,
            Level.SYMMETRY:   self.symmetry,
            Level.HYPERGRAPH: self.hypergraph,
            Level.SOLUTION:   self.solution,
        }[level] is not None

    def can_derive(self, target: Level) -> tuple[bool, set[str]]:
        """Can we reach `target` from what we have? What's missing?"""
        current = self.highest_level
        if current.implies(target):
            return True, set()
        needed = current.requires_to_reach(target)
        return False, needed

    def traverse_down(self) -> list[str]:
        """Trace the full derivation chain from L0 → L3."""
        chain = []
        if self.lagrangian:
            chain.append(f"L0: {self.lagrangian.name or 'Lagrangian'}")
        if self.symmetry:
            chain.append(f"L1: G = {self.symmetry.symmetry_group}")
            chain.append(f"    conserved: {self.symmetry.conserved_quantities}")
        if self.hypergraph:
            chain.append(f"L2: {self.hypergraph}")
        if self.solution:
            chain.append(f"L3: {self.solution}")
        return chain


# ── Helper functions ───────────────────────────────────────────────────

def _guess_symmetry_type(coord: sp.Symbol) -> SymmetryType:
    """Heuristic: coordinate name → symmetry type."""
    name = str(coord).lower()
    if name in ('theta', 'φ', 'phi', 'θ'):
        return SymmetryType.ROTATION
    elif name in ('x', 'y', 'z', 'r'):
        return SymmetryType.TRANSLATION_SPACE
    elif name == 't':
        return SymmetryType.TRANSLATION_TIME
    return SymmetryType.TRANSLATION_SPACE

def _noether_map(sym: SymmetryType) -> str:
    """Noether's theorem: symmetry → conserved quantity name."""
    return {
        SymmetryType.ROTATION:          "angular_momentum",
        SymmetryType.TRANSLATION_TIME:  "energy",
        SymmetryType.TRANSLATION_SPACE: "linear_momentum",
        SymmetryType.BOOST:             "center_of_mass_velocity",
        SymmetryType.GAUGE:             "charge",
    }[sym]

def _format_group(symmetries: list[SymmetryEntry]) -> str:
    """Format the symmetry group as a string."""
    conserved = [s for s in symmetries if s.status == ConservationStatus.CONSERVED]
    parts = []
    for s in conserved:
        if s.symmetry == SymmetryType.ROTATION:
            parts.append("SO(2)")
        elif s.symmetry == SymmetryType.TRANSLATION_TIME:
            parts.append("ℝ")
        elif s.symmetry == SymmetryType.TRANSLATION_SPACE:
            parts.append("ℝⁿ")
    return " × ".join(parts) if parts else "∅"


# ═══════════════════════════════════════════════════════════════════════
# EXAMPLE: Satellite at periapsis — the running example from the writeup
# ═══════════════════════════════════════════════════════════════════════

def build_orbital_example() -> RepresentationHierarchy:
    """Build the complete 4-level hierarchy for the orbital mechanics problem."""

    # ── Level 0: Lagrangian ──
    r, rdot, theta, thetadot = sp.symbols('r rdot theta thetadot')
    m, GM = sp.symbols('m GM', positive=True)

    L_expr = sp.Rational(1, 2) * m * (rdot**2 + r**2 * thetadot**2) + GM * m / r

    lagrangian = Lagrangian(
        expr=L_expr,
        generalized_coords=[r, theta],
        generalized_velocities=[rdot, thetadot],
        parameters={GM: 4e14, m: 1},  # specific energy: m=1
        name="Central force (Keplerian)"
    )

    # ── Level 1: Symmetry structure (derived from L0) ──
    symmetry = lagrangian.extract_symmetries()

    # ── Level 2: Constraint hypergraph (inject problem instance) ──
    quantities = [
        Quantity("r₀",  sp.Symbol("r_0"),  QuantityStatus.GIVEN,   "m",     8e6),
        Quantity("v₀",  sp.Symbol("v_0"),  QuantityStatus.GIVEN,   "m/s",   9000),
        Quantity("GM",  sp.Symbol("GM"),   QuantityStatus.GIVEN,   "m³/s²", 4e14),
        Quantity("ε",   sp.Symbol("eps"),  QuantityStatus.UNKNOWN, "m²/s²"),
        Quantity("L",   sp.Symbol("L"),    QuantityStatus.UNKNOWN, "m²/s"),
        Quantity("a",   sp.Symbol("a"),    QuantityStatus.UNKNOWN, "m"),
        Quantity("e",   sp.Symbol("e_"),   QuantityStatus.UNKNOWN, ""),
        Quantity("T",   sp.Symbol("T"),    QuantityStatus.UNKNOWN, "s"),
    ]

    laws = [
        HyperEdge("E₁ vis-viva",        {"v₀", "GM", "r₀"},     "ε",  degree=1),
        HyperEdge("E₂ angular momentum", {"r₀", "v₀"},           "L",  degree=1),
        HyperEdge("E₃ semi-major axis",  {"GM", "ε"},            "a",  degree=1),
        HyperEdge("E₄ eccentricity",     {"r₀", "a"},            "e",  degree=1),
        HyperEdge("E₅ Kepler III",       {"a", "GM"},            "T",  degree=3, is_linear=False),
    ]

    hypergraph = symmetry.instantiate(
        quantities=quantities,
        laws=laws,
        initial_conditions={"ic_type": "turning_point", "constraint": "v₀ ⊥ r₀"}
    )

    # ── Level 3: Solution path (choose propagator strategy) ──
    solution = hypergraph.solve_with(SolverType.PROPAGATOR)

    # ── Orthogonal: Dimensional analysis ──
    dimensional = DimensionalAnalysis(
        quantities=["GM", "r₀", "v₀", "a", "T", "ε", "L", "e"],
        dimension_matrix_rank=2,
        n_pi_groups=6,
        pi_groups=[
            PiGroup("Π₁", "v₀²r₀/GM",        1.620),
            PiGroup("Π₂", "T²GM/r₀³",         None),
            PiGroup("Π₃", "εr₀/GM",           -0.190),
            PiGroup("Π₄", "L²/(GMr₀)",         1.620, redundant_with="Π₁"),
            PiGroup("Π₅", "e",                  0.619),
            PiGroup("Π₆", "a/r₀",              2.631),
        ]
    )

    return RepresentationHierarchy(
        lagrangian=lagrangian,
        symmetry=symmetry,
        hypergraph=hypergraph,
        solution=solution,
        dimensional=dimensional
    )


# ═══════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    # Windows console may not support Unicode; use UTF-8 if available
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    h = build_orbital_example()

    print("=" * 70)
    print("REPRESENTATION HIERARCHY - Satellite at Periapsis")
    print("=" * 70)

    # Traverse the full chain
    print("\n-- Full derivation chain --")
    for line in h.traverse_down():
        print(f"  {line}")

    # Implication checks
    print("\n-- Implication checks --")
    print(f"  Solution implies Hypergraph?  {Level.SOLUTION.implies(Level.HYPERGRAPH)}")
    print(f"  Symmetry implies Solution?    {Level.SYMMETRY.implies(Level.SOLUTION)}")
    print(f"  Lagrangian implies Lagrangian? {Level.LAGRANGIAN.implies(Level.LAGRANGIAN)}")

    # What's needed to go from L1 -> L3?
    print("\n-- Injection requirements --")
    needed = Level.SYMMETRY.requires_to_reach(Level.SOLUTION)
    print(f"  L1 → L3 requires: {needed}")
    needed = Level.LAGRANGIAN.requires_to_reach(Level.HYPERGRAPH)
    print(f"  L0 → L2 requires: {needed}")

    # Hypergraph structure
    print("\n-- Hypergraph (Level 2) --")
    print(f"  {h.hypergraph}")
    print(f"  Hub nodes (degree ≥ 2): {h.hypergraph.hub_nodes()}")
    print(f"  Leaf nodes: {h.hypergraph.leaf_nodes()}")

    # Solution path
    print("\n-- Solution Path (Level 3) --")
    print(f"  {h.solution}")
    print(f"  Parallelizable groups:")
    for group in h.solution.parallelizable_groups:
        print(f"    {[str(s) for s in group]}")

    # Dimensional analysis
    print("\n-- Dimensional Analysis (orthogonal) --")
    print(f"  Π groups: {h.dimensional.n_pi_groups}")
    print(f"  Redundancies: {h.dimensional.redundancies}")
    print(f"  Orbit type (Π₁=1.62): {h.dimensional.classify_orbit(1.62)}")
    print(f"  Orbit type (Π₁=2.5):  {h.dimensional.classify_orbit(2.5)}")

    # Subgraph extraction
    print("\n-- Minimal subgraph for eccentricity --")
    sub = h.hypergraph.subgraph_for("e")
    print(f"  {sub}")
    for edge in sub.hyperedges:
        print(f"    {edge}")

    # Symmetry details
    print("\n-- Symmetry Structure (Level 1) --")
    for s in h.symmetry.symmetries:
        print(f"  {s}")
    print(f"  Angular momentum conserved? {h.symmetry.implies_conservation_of('angular_momentum')}")
    print(f"  Linear momentum conserved?  {h.symmetry.implies_conservation_of('linear_momentum')}")

    # Can-derive checks
    print("\n-- Can-derive checks --")
    can, missing = h.can_derive(Level.SOLUTION)
    print(f"  Can derive L3 from current state? {can}, missing: {missing}")
