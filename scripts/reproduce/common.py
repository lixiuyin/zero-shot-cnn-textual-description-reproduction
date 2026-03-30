"""Shared paths and helpers for reproduction scripts (Ba et al. ICCV 2015).

Checkpoint resolution strategy
------------------------------
Each checkpoint key maps to an **exact** glob pattern (one file per key).
No fallback heuristics — if the pattern doesn't match, the key is unresolved.

For cross-validation, ``resolve_cv_checkpoints`` looks inside ``fold{i}/``
subdirectories using the same patterns.  ``resolve_with_cv`` is the main
entry point: it returns *both* the root checkpoint and any per-fold
checkpoints so that callers can average across folds when available.
"""
from __future__ import annotations

from pathlib import Path

# Code root = parent of scripts/
CODE_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = CODE_ROOT / "results"

# Default data paths (corrected paths based on actual directory structure)
DEFAULT_WIKIPEDIA_BIRDS = "data/wikipedia/birds.jsonl"
DEFAULT_WIKIPEDIA_FLOWERS = "data/wikipedia/flowers.jsonl"

# Default checkpoint directory (relative to CODE_ROOT).
DEFAULT_CHECKPOINT_DIR = "checkpoints"

# ---------------------------------------------------------------------------
# Exact-match pattern table
# ---------------------------------------------------------------------------
# Every key maps to ONE glob pattern that should match at most one file.
# Naming convention: {model}_{loss}_{dataset}_{layer}_{n_unseen}[_tr{ratio}].pt
#
# Default conv layer is conv5_3.  conv4_3 / pool5 have their own explicit keys.
# ---------------------------------------------------------------------------
CHECKPOINT_PATTERNS: dict[str, str] = {
    # ── Table 1: model-type comparison (BCE, default layer) ───────────────
    # CUB
    "fc_bce_cub":              "fc_bce_cub_fc_*.pt",
    "conv_bce_cub":            "conv_bce_cub_conv5_3_*.pt",
    "fc_conv_bce_cub":         "fc_conv_bce_cub_conv5_3_*.pt",
    # Flowers
    "fc_bce_flowers":          "fc_bce_flowers_fc_*.pt",
    "conv_bce_flowers":        "conv_bce_flowers_conv5_3_*.pt",
    "fc_conv_bce_flowers":     "fc_conv_bce_flowers_conv5_3_*.pt",

    # ── Table 1 extended: hinge / euclidean across model types ────────────
    # CUB – fc
    "fc_hinge_cub":            "fc_hinge_cub_fc_*.pt",
    "fc_euclidean_cub":        "fc_euclidean_cub_fc_*.pt",
    # CUB – conv (euclidean not supported for conv-only)
    "conv_hinge_cub":          "conv_hinge_cub_conv5_3_*.pt",
    # CUB – fc+conv
    "fc_conv_hinge_cub":       "fc_conv_hinge_cub_conv5_3_*.pt",
    "fc_conv_euclidean_cub":   "fc_conv_euclidean_cub_conv5_3_*.pt",
    # Flowers – fc
    "fc_hinge_flowers":        "fc_hinge_flowers_fc_*.pt",
    "fc_euclidean_flowers":    "fc_euclidean_flowers_fc_*.pt",
    # Flowers – conv
    "conv_hinge_flowers":      "conv_hinge_flowers_conv5_3_*.pt",
    # Flowers – fc+conv
    "fc_conv_hinge_flowers":   "fc_conv_hinge_flowers_conv5_3_*.pt",
    "fc_conv_euclidean_flowers": "fc_conv_euclidean_flowers_conv5_3_*.pt",

    # ── Table 3: conv-layer ablation (CUB, fc+conv, BCE) ─────────────────
    "fc_conv_bce_cub_conv4_3": "fc_conv_bce_cub_conv4_3_*.pt",
    "fc_conv_bce_cub_conv5_3": "fc_conv_bce_cub_conv5_3_*.pt",
    "fc_conv_bce_cub_pool5":   "fc_conv_bce_cub_pool5_*.pt",

    # ── Table 4: supervised baseline (50/50 split, n_unseen=0) ────────────
    "fc_bce_cub_5050":         "fc_bce_cub_fc_0_tr0.5.pt",
    "fc_conv_bce_cub_5050":    "fc_conv_bce_cub_conv5_3_0_tr0.5.pt",
    "fc_bce_flowers_5050":     "fc_bce_flowers_fc_0_tr0.5.pt",
    "fc_conv_bce_flowers_5050": "fc_conv_bce_flowers_conv5_3_0_tr0.5.pt",
}


def _resolve_in_dir(directory: Path, key: str) -> str:
    """Match a single checkpoint in *directory* using ``CHECKPOINT_PATTERNS``.

    Returns the path string, or empty string if not found.
    For patterns with wildcards, prefers files WITHOUT ``_0_`` (non-50/50)
    unless the key itself contains ``5050``.
    """
    pattern = CHECKPOINT_PATTERNS.get(key)
    if not pattern:
        return ""

    matches = list(directory.glob(pattern))
    if not matches:
        return ""

    if len(matches) == 1:
        return str(matches[0])

    # Multiple matches (e.g. both _40.pt and _0_tr0.5.pt hit the wildcard).
    # For 5050 keys, prefer the _0_ file; otherwise exclude it.
    is_5050 = "5050" in key
    if is_5050:
        preferred = [f for f in matches if "_0_" in f.name]
    else:
        preferred = [f for f in matches if "_0_" not in f.name]

    chosen = preferred if preferred else matches
    chosen.sort(key=lambda p: p.name)
    return str(chosen[0])


def resolve_checkpoint(key: str, checkpoint_dir: str = "", explicit: str = "") -> str:
    """Resolve a single checkpoint by key.

    Resolution order:
      1. *explicit* path (if given and exists)
      2. Pattern match in *checkpoint_dir* (or DEFAULT_CHECKPOINT_DIR)

    No fallback heuristics.
    """
    if explicit:
        p = Path(explicit)
        if p.exists():
            print(f"  [CKPT] {key} → {p} (explicit)")
            return str(p)
        # Try relative to CODE_ROOT
        p2 = CODE_ROOT / p
        if p2.exists():
            print(f"  [CKPT] {key} → {p2} (explicit)")
            return str(p2)

    base = Path(checkpoint_dir) if checkpoint_dir else Path(DEFAULT_CHECKPOINT_DIR)
    if not base.is_absolute():
        base = CODE_ROOT / base

    if base.exists() and base.is_dir():
        result = _resolve_in_dir(base, key)
        if result:
            print(f"  [CKPT] {key} → {Path(result).name}")
        else:
            print(f"  [CKPT] {key} → NOT FOUND (pattern: {CHECKPOINT_PATTERNS.get(key, '?')})")
        return result

    return ""


def resolve_cv_checkpoints(
    key: str,
    n_folds: int = 0,
    checkpoint_dir: str = "",
) -> list[str]:
    """Return per-fold checkpoint paths from ``fold{i}/`` subdirectories.

    Args:
        key: Checkpoint key (e.g. ``"fc_bce_cub"``).
        n_folds: Max folds to search (0 = auto-detect from ``fold*/`` dirs).
        checkpoint_dir: Override for checkpoints root directory.

    Returns:
        List of resolved paths (only includes folds where a match was found).
    """
    base = Path(checkpoint_dir) if checkpoint_dir else Path(DEFAULT_CHECKPOINT_DIR)
    if not base.is_absolute():
        base = CODE_ROOT / base

    if not base.exists():
        return []

    fold_dirs = sorted(base.glob("fold*/"), key=lambda p: p.name)
    if n_folds > 0:
        fold_dirs = fold_dirs[:n_folds]

    paths = []
    for fold_dir in fold_dirs:
        p = _resolve_in_dir(fold_dir, key)
        if p:
            paths.append(p)

    if paths:
        print(f"  [CKPT] {key} → {len(paths)} CV folds: {', '.join(Path(p).parent.name + '/' + Path(p).name for p in paths)}")

    return paths


def resolve_with_cv(
    key: str,
    n_folds: int = 0,
    checkpoint_dir: str = "",
    explicit: str = "",
) -> tuple[str, list[str]]:
    """Resolve both root and per-fold checkpoints for a key.

    Returns:
        ``(root_path, fold_paths)`` — either or both may be empty.
        Callers should prefer fold_paths (average across folds) when available,
        falling back to root_path for single-run evaluation.
    """
    root = resolve_checkpoint(key, checkpoint_dir, explicit)
    folds = resolve_cv_checkpoints(key, n_folds, checkpoint_dir)
    return root, folds


# Paper-style: single column ~3.3in, two-column figure ~6.6in; font ~9pt
FIG_SINGLE_COL_INCH = 3.3
FIG_TWO_COL_INCH = 6.6
FIG_DPI = 150


def get_tables_dir() -> Path:
    d = RESULTS_ROOT / "tables"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_tex_dir() -> Path:
    d = RESULTS_ROOT / "tex"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_figures_dir() -> Path:
    d = RESULTS_ROOT / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_table_csv(tables_dir: Path, table_id: int) -> tuple[list[str], list[list[str]]] | None:
    """Read existing table CSV file. Returns (headers, rows) or None if file doesn't exist."""
    import csv
    path = tables_dir / f"Table{table_id}.csv"
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
        if not rows:
            return None
        return rows[0], rows[1:]


def write_table_csv(tables_dir: Path, table_id: int, headers: list[str], rows: list[list[str]]) -> Path:
    import csv
    path = tables_dir / f"Table{table_id}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    return path


def write_table_latex(
    tables_dir: Path,
    table_id: int,
    caption: str,
    label: str,
    header_rows: list[list[str]],
    data_rows: list[list[str]],
    col_align: str = "l",
) -> Path:
    """Write LaTeX table (booktabs) matching paper layout. header_rows: list of rows for \\thead; data_rows: body."""
    path = tables_dir / f"Table{table_id}.tex"
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{" + caption + "}",
        "\\label{tab:" + label + "}",
        "\\begin{tabular}{" + col_align + "}",
        "\\toprule",
    ]
    for row in header_rows:
        lines.append(" & ".join(str(c) for c in row) + " \\\\")
    lines.append("\\midrule")
    for row in data_rows:
        lines.append(" & ".join(str(c) for c in row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def compile_table_to_pdf(tex_dir: Path, table_id: int, xelatex: str = "xelatex") -> bool:
    """
    Compile a single table LaTeX file to PDF using xelatex.
    PDF is saved to results/ root, TEX files are in tex/ subdirectory.

    Args:
        tex_dir: Directory containing Table{table_id}.tex (results/tex/)
        table_id: Table number (1, 2, 3, 4, etc.)
        xelatex: xelatex command (default: "xelatex")

    Returns:
        True if compilation succeeded, False otherwise
    """
    import subprocess

    tex_path = tex_dir / f"Table{table_id}.tex"
    if not tex_path.exists():
        return False

    name = tex_path.stem
    content = tex_path.read_text(encoding="utf-8")

    preamble = r"""
\documentclass[11pt]{article}
\usepackage{booktabs}
\usepackage{fontspec}
\usepackage{geometry}
\geometry{margin=1in}
\begin{document}
"""

    footer = r"""
\end{document}
"""

    wrapper = tex_dir / f"{name}_pdf.tex"
    wrapper.write_text(preamble + content + footer, encoding="utf-8")

    # Output PDF to results root, not tex subdirectory
    results_root = tex_dir.parent
    try:
        result = subprocess.run(
            [xelatex, "-interaction=nonstopmode", "-output-directory", str(results_root), wrapper.name],
            cwd=tex_dir,
            capture_output=True,
            timeout=60,
        )
        # Check if compilation succeeded
        if result.returncode != 0:
            print(f"XeLaTeX compilation failed with return code {result.returncode}")
            if result.stderr:
                print(f"Error output:\n{result.stderr.decode('utf-8', errors='ignore')}")
            if result.stdout:
                print(f"Standard output:\n{result.stdout.decode('utf-8', errors='ignore')[:500]}")
            if wrapper.exists():
                wrapper.unlink()
            return False
    except FileNotFoundError:
        print(f"XeLaTeX not found. Please install LaTeX (e.g., 'brew install mactex' on macOS)")
        if wrapper.exists():
            wrapper.unlink()
        return False
    except subprocess.TimeoutExpired:
        print(f"XeLaTeX compilation timed out after 60 seconds")
        if wrapper.exists():
            wrapper.unlink()
        return False

    # Clean up auxiliary files in results root
    for ext in (".aux", ".log"):
        p = results_root / f"{name}_pdf{ext}"
        if p.exists():
            p.unlink()

    # Rename PDF to final name
    pdf_out = results_root / f"{name}_pdf.pdf"
    target = results_root / f"{name}.pdf"
    if pdf_out.exists():
        pdf_out.rename(target)
        if wrapper.exists():
            wrapper.unlink()
        return True

    if wrapper.exists():
        wrapper.unlink()
    return False


def validate_data_path(cub_root: str, wikipedia_path: str, script_name: str = "Script") -> bool:
    """
    Validate that required data paths exist.

    Returns True if all paths exist, False otherwise.
    Prints informative error messages if paths are missing.
    """
    missing = []
    if cub_root and not Path(cub_root).exists():
        missing.append(f"CUB images directory: {cub_root}")
    if wikipedia_path and not Path(wikipedia_path).exists():
        missing.append(f"Wikipedia JSONL: {wikipedia_path}")

    if missing:
        print(f"{script_name}: Missing required data paths:")
        for path in missing:
            print(f"  - {path}")
        return False
    return True


def get_device(device_arg: str):
    """
    Get torch device with fallback.
    """
    import torch
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
