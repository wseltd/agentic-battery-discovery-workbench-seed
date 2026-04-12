from __future__ import annotations

"""Domain keyword registry for request routing.

Compile-time constants derived from Q1 of the research pack.  Every keyword
is lowercase-normalised at definition time.  No NLP, no fuzzy matching,
no config loading -- these are frozen sets used by the deterministic
routing gate (T006).

Design choice: flat frozensets with a linear lookup in classify_token.
An alternative would be a merged dict for O(1) per-token lookup, but the
sets are small (~80 terms total) and keeping them separate makes disjointness
testing trivial.  The linear scan over three small sets is negligible.
"""

# ---------------------------------------------------------------------------
# Small-molecule cue terms (route -> "small_molecule")
# Source: research-pack.md Q1 "Small-molecule" keyword list
# ---------------------------------------------------------------------------
SMALL_MOLECULE_KEYWORDS: frozenset[str] = frozenset({
    "smiles",
    "inchi",
    "ligand",
    "docking",
    "admet",
    "clogp",
    "logp",
    "tpsa",
    "hbd",
    "hba",
    "qed",
    "pains",
    "scaffold",
    "linker",
    "r-group",
    "lead optimisation",
    "hit generation",
    "sar",
    "qsar",
    "ic50",
    "ec50",
    "ki",
    "lipinski",
    "fragment",
    "safe",
    "fragment remasking",
    "pmo benchmark",
})

# ---------------------------------------------------------------------------
# Inorganic crystalline materials cue terms (route -> "inorganic_materials")
# Source: research-pack.md Q1 "Inorganic crystalline materials" keyword list
# ---------------------------------------------------------------------------
INORGANIC_MATERIALS_KEYWORDS: frozenset[str] = frozenset({
    "crystal",
    "unit cell",
    "lattice",
    "fractional coordinates",
    "space group",
    "cif",
    "poscar",
    "primitive cell",
    "niggli reduction",
    "convex hull",
    "energy above hull",
    "formation energy",
    "phonon",
    "k-points",
    "vasp",
    "bulk modulus",
    "band gap",
    "magnetic density",
    "structure relaxation",
    "mattergen",
    "mattersim",
    "s.u.n. structures",
})

# ---------------------------------------------------------------------------
# Unsupported domain cue terms (route -> "unsupported")
# Source: research-pack.md Q1 "Unsupported" keyword list
# ---------------------------------------------------------------------------
UNSUPPORTED_KEYWORDS: frozenset[str] = frozenset({
    "polymer",
    "protein",
    "biologics",
    "mixture",
    "solution",
    "device",
    "process design",
    "synthesis planning",
    "mof",
    "cof",
})

# ---------------------------------------------------------------------------
# Ambiguity terms -- present in user requests but must NOT auto-route.
# These require a clarifying question from the confidence scorer (T007).
# Source: research-pack.md Q1 "Ambiguity cases"
# ---------------------------------------------------------------------------
AMBIGUITY_KEYWORDS: frozenset[str] = frozenset({
    "catalyst",
    "battery electrolyte",
    "semiconductor",
    "molecular magnet",
    "crystalline magnet",
    "perovskite",
    "nanoparticle",
})

# ---------------------------------------------------------------------------
# Structured constraint cue patterns, keyed by domain.
# These are phrases that signal a constraint clause rather than a domain
# identity, but they reinforce routing when found alongside domain keywords.
# Source: research-pack.md Q1 "Structured constraint cues"
# ---------------------------------------------------------------------------
STRUCTURED_CONSTRAINT_CUES: dict[str, frozenset[str]] = {
    "small_molecule": frozenset({
        "mw",
        "logp",
        "tpsa",
        "hbd",
        "hba",
        "smarts",
        "substructure",
        "ring count",
        "pains avoidance",
    }),
    "inorganic_materials": frozenset({
        "element list",
        "stoichiometry",
        "space-group number",
        "max atoms",
        "symmetry system",
        "stability threshold",
        "ev/atom",
    }),
}

# ---------------------------------------------------------------------------
# Internal lookup table -- built once at import time from the three domain
# sets.  Maps each keyword to its domain string.
# ---------------------------------------------------------------------------
_TOKEN_TO_DOMAIN: dict[str, str] = {}
for _kw in SMALL_MOLECULE_KEYWORDS:
    _TOKEN_TO_DOMAIN[_kw] = "small_molecule"
for _kw in INORGANIC_MATERIALS_KEYWORDS:
    _TOKEN_TO_DOMAIN[_kw] = "inorganic_materials"
for _kw in UNSUPPORTED_KEYWORDS:
    _TOKEN_TO_DOMAIN[_kw] = "unsupported"


def classify_token(token: str) -> str | None:
    """Return the domain for a single keyword token, or None if unrecognised.

    Ambiguity keywords explicitly return None -- they must never auto-route.
    Lookup is case-insensitive; the token is lowered before matching.

    Parameters
    ----------
    token:
        A single keyword or short phrase extracted from a user request.

    Returns
    -------
    ``"small_molecule"``, ``"inorganic_materials"``, ``"unsupported"``,
    or ``None`` if the token is ambiguous or not in any domain set.
    """
    normalised = token.lower()
    if normalised in AMBIGUITY_KEYWORDS:
        return None
    return _TOKEN_TO_DOMAIN.get(normalised)
