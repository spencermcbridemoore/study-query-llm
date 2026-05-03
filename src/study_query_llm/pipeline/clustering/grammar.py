"""Bundled clustering method-name grammar (Slice 2 Wave 1).

Parses ``+<chain>+<fit-mode>`` suffixes without importing the registry so runners
and ``runner_common`` stay cycle-free.
"""

from __future__ import annotations

_FIT_MODE_TOKENS = frozenset({"sweep", "fixed-k", "fixed-eps", "fixed"})
_CHAIN_TOKENS = frozenset({"normalize", "pca", "approx"})
_ALLOWED_RUNTIME_CHAINS: frozenset[tuple[str, ...]] = frozenset(
    {
        (),
        ("normalize",),
        ("pca",),
        ("normalize", "pca"),
    }
)


def _fold_approx_tokens(base_algorithm: str, raw_middle: list[str]) -> list[str]:
    """Fold ``approx`` to runtime normalize for the spherical-kmeans family."""
    out: list[str] = []
    i = 0
    while i < len(raw_middle):
        tok = raw_middle[i]
        if tok == "approx":
            if base_algorithm != "spherical-kmeans":
                raise ValueError(
                    f"token 'approx' is only valid for spherical-kmeans family; "
                    f"got base_algorithm={base_algorithm!r}"
                )
            out.append("normalize")
            i += 1
            continue
        if tok in ("normalize", "pca"):
            out.append(tok)
            i += 1
            continue
        raise ValueError(f"unknown preprocessing token {tok!r} in method name")
    return out


def _validate_runtime_chain(base_algorithm: str, chain: tuple[str, ...]) -> None:
    if chain not in _ALLOWED_RUNTIME_CHAINS:
        raise ValueError(
            f"invalid preprocessing chain {chain!r} for {base_algorithm!r}; "
            f"allowed chains: {sorted(_ALLOWED_RUNTIME_CHAINS)}"
        )


def _validate_base_fit_combo(base_algorithm: str, fit_mode_token: str) -> None:
    if base_algorithm == "dbscan" and fit_mode_token != "fixed-eps":
        raise ValueError(
            f"dbscan family requires fit-mode token 'fixed-eps', got {fit_mode_token!r}"
        )
    if base_algorithm in {"kmeans", "gmm", "agglomerative"} and fit_mode_token not in {
        "fixed-k",
        "sweep",
    }:
        raise ValueError(
            f"{base_algorithm} family expects fit-mode 'fixed-k' or 'sweep', "
            f"got {fit_mode_token!r}"
        )
    if base_algorithm == "spherical-kmeans" and fit_mode_token != "fixed-k":
        raise ValueError(
            f"spherical-kmeans family requires fit-mode 'fixed-k', got {fit_mode_token!r}"
        )
    if base_algorithm == "hdbscan" and fit_mode_token not in {"fixed", "sweep"}:
        raise ValueError(
            f"hdbscan family expects fit-mode 'fixed' or 'sweep', got {fit_mode_token!r}"
        )


def parse_method_name(name: str) -> tuple[str, tuple[str, ...], str]:
    """Decompose a bundled clustering method name into ``(base_algorithm, chain, fit_mode_token)``.

    Splits on ``+``. The first token is the base algorithm. Tokens between the
    first and last that are in ``{'normalize', 'pca'}`` (after folding ``approx``
    for spherical-kmeans) form the runtime preprocessing chain in declaration
    order. The last token is the fit-mode suffix (``sweep``, ``fixed-k``,
    ``fixed-eps``, or ``fixed``).

    For ``spherical-kmeans+approx+...``, the ``approx`` annotation folds to a
    runtime ``normalize`` prefix (the returned chain is what preprocessing
    applies, not the documentation token).

    Raises:
        ValueError: on empty input, unknown tokens, or grammar violations.
    """
    raw = str(name or "").strip()
    if not raw or raw == "+":
        raise ValueError("method name must be non-empty")
    parts = raw.split("+")
    if len(parts) < 2:
        raise ValueError(f"method name must contain at least one '+' separator: {name!r}")

    last = parts[-1].strip().lower()
    if last not in _FIT_MODE_TOKENS:
        raise ValueError(
            f"last token must be a fit-mode suffix {sorted(_FIT_MODE_TOKENS)}, got {last!r}"
        )

    base_algorithm = parts[0].strip().lower()
    raw_middle = [p.strip().lower() for p in parts[1:-1]]
    for tok in raw_middle:
        if tok not in _CHAIN_TOKENS:
            raise ValueError(f"unknown preprocessing token {tok!r} in method name {name!r}")

    folded = _fold_approx_tokens(base_algorithm, list(raw_middle))
    chain_list = folded

    # Validate order: only allowed subsequences
    if chain_list == ["pca"]:
        runtime_chain: tuple[str, ...] = ("pca",)
    elif chain_list == ["normalize"]:
        runtime_chain = ("normalize",)
    elif chain_list == ["normalize", "pca"]:
        runtime_chain = ("normalize", "pca")
    elif chain_list == []:
        runtime_chain = ()
    elif chain_list == ["pca", "normalize"]:
        raise ValueError("'pca' cannot precede 'normalize' in the preprocessing chain")
    else:
        raise ValueError(f"invalid preprocessing token sequence: {chain_list!r}")

    _validate_runtime_chain(base_algorithm, runtime_chain)
    _validate_base_fit_combo(base_algorithm, last)

    return base_algorithm, runtime_chain, last
