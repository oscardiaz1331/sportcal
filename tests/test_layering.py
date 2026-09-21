"""Enforce the dependency rules of docs/architecture.md mechanically, so the layout cannot
silently rot: imports are read with `ast`, nothing is executed."""
import ast
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent / "sportcal"


def _imports(path, module_level_only):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = tree.body if module_level_only else list(ast.walk(tree))
    out = set()
    for n in nodes:
        if isinstance(n, ast.Import):
            out |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            out.add(n.module)
    return {m for m in out if m.startswith("sportcal")}


def _violations(layer, forbidden, module_level_only=False):
    bad = []
    for f in (PKG / layer).rglob("*.py"):
        for m in _imports(f, module_level_only):
            if any(m == x or m.startswith(x + ".") for x in forbidden):
                bad.append("{} imports {}".format(f.relative_to(PKG.parent), m))
    return bad


def test_core_is_sport_agnostic_and_depends_on_nothing_above_it():
    assert _violations("core", ["sportcal.sports", "sportcal.lab", "sportcal.product"]) == []


def test_sports_are_pure_geometry():
    assert _violations("sports", ["sportcal.lab", "sportcal.product"]) == []


def test_product_never_imports_lab_at_module_level():
    """Lab code pulls training-only dependencies; lazy imports inside constructors are the
    documented, temporary exception (see SegDltEstimator)."""
    assert _violations("product", ["sportcal.lab"], module_level_only=True) == []


def test_sports_do_not_import_each_other():
    """A sport is self-contained: shared geometry belongs in core/, never borrowed from another sport."""
    names = [d.name for d in (PKG / "sports").iterdir() if d.is_dir() and (d / "__init__.py").exists()]
    assert {"hockey", "soccer"} <= set(names)
    bad = []
    for sport in names:
        for f in (PKG / "sports" / sport).rglob("*.py"):
            for m in _imports(f, False):
                if any(m.startswith("sportcal.sports." + o) for o in names if o != sport):
                    bad.append("{} imports {}".format(f.relative_to(PKG.parent), m))
    assert bad == []


def test_a_sport_lab_never_imports_another_sports_lab():
    """lab/hockey and lab/soccer are independent; what both need lives in core/ or lab/common/."""
    labs = [d.name for d in (PKG / "lab").iterdir() if d.is_dir() and d.name not in ("common", "__pycache__")]
    bad = []
    for lab in labs:
        for f in (PKG / "lab" / lab).rglob("*.py"):
            for m in _imports(f, False):
                if any(m.startswith("sportcal.lab." + o) for o in labs if o != lab):
                    bad.append("{} imports {}".format(f.relative_to(PKG.parent), m))
    assert bad == []
