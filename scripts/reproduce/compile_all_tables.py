"""
Generate a complete LaTeX document with all tables.
This creates a single AllTables.tex file that includes all tables with proper document structure.
Output is always written to the default tex dir (Code/results/tex/AllTables.tex).
Usage:
    python scripts/reproduce/compile_all_tables.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.reproduce.common import get_tables_dir, get_tex_dir, read_table_csv


def generate_complete_latex() -> Path:
    """
    Generate a complete LaTeX document with all tables.
    Reads CSV from Code/results/tables, writes AllTables.tex to Code/results/tex.

    Returns:
        Path to the generated AllTables.tex file
    """
    tex_dir = get_tex_dir()
    tables_dir = get_tables_dir()

    # Read all tables
    tables = []
    for table_id in [1, 2, 3, 4]:
        result = read_table_csv(tables_dir, table_id)
        if result:
            headers, rows = result
            tables.append((table_id, headers, rows))

    # Generate complete LaTeX document
    latex_content = r"""\documentclass[11pt]{article}
\usepackage{booktabs}
\usepackage{fontspec}
\usepackage{geometry}
\usepackage{multirow}
\geometry{margin=1in}

\begin{document}


"""

    # Add each table
    for table_id, headers, rows in tables:
        # Determine table content based on table_id
        if table_id == 1:
            latex_content += generate_table1(headers, rows)
        elif table_id == 2:
            latex_content += generate_table2(headers, rows)
        elif table_id == 3:
            latex_content += generate_table3(headers, rows)
        elif table_id == 4:
            latex_content += generate_table4(headers, rows)

    latex_content += r"""
\end{document}
"""

    # Write to file
    output_path = tex_dir / "AllTables.tex"
    output_path.write_text(latex_content, encoding="utf-8")
    return output_path


def generate_table1(headers: list[str], rows: list[list[str]]) -> str:
    """Generate Table 1 LaTeX content."""
    # Filter out note row if present
    data_rows = [r for r in rows if not r[0].startswith(r"\midrule") and r[0] != r"\textit{Note:"]

    content = r"""
\begin{table}[t]
\centering
\scriptsize
\caption{ROC-AUC and PR-AUC performance on CUB-200-2011 and Oxford Flowers-102.}
\label{tab:table1}
\begin{tabular}{llcccccccccccc}
\toprule
Dataset & Model & \multicolumn{6}{c}{ROC-AUC} & \multicolumn{6}{c}{PR-AUC} \\
\cmidrule(lr){3-8} \cmidrule(lr){9-14}
 & & \multicolumn{2}{c}{unseen} & \multicolumn{2}{c}{seen} & \multicolumn{2}{c}{mean} & \multicolumn{2}{c}{unseen} & \multicolumn{2}{c}{seen} & \multicolumn{2}{c}{mean} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8} \cmidrule(lr){9-10} \cmidrule(lr){11-12} \cmidrule(lr){13-14}
 & & Paper & Ours & Paper & Ours & Paper & Ours & Paper & Ours & Paper & Ours & Paper & Ours \\
\midrule
"""
    for row in data_rows:
        content += " & ".join(str(c) for c in row) + r" \\" + "\n"

    content += r""" \bottomrule
\end{tabular}
\end{table}

"""
    return content


def generate_table2(headers: list[str], rows: list[list[str]]) -> str:
    """Generate Table 2 LaTeX content (objective functions: BCE, Hinge, Euclidean)."""
    content = r"""
\begin{table}[t]
\centering
\scriptsize
\caption{Model performance using various objective functions on CUB-200-2011 dataset.}
\label{tab:table2}
\begin{tabular}{lrrrrrr}
\toprule
"""
    content += " & ".join(str(c) for c in headers) + r" \\ \midrule" + "\n"

    for row in rows:
        content += " & ".join(str(c) for c in row) + r" \\" + "\n"

    content += r""" \bottomrule
\end{tabular}
\end{table}

"""
    return content


def generate_table3(headers: list[str], rows: list[list[str]]) -> str:
    """Generate Table 3 LaTeX content."""
    content = r"""
\begin{table}[t]
\centering
\scriptsize
\caption{Performance comparison using different intermediate ConvLayers from VGG on CUB-200-2011 (fc+conv models).}
\label{tab:table3}
\begin{tabular}{lrrrrrr}
\toprule
"""
    content += " & ".join(str(c) for c in headers) + r" \\ \midrule" + "\n"

    for row in rows:
        content += " & ".join(str(c) for c in row) + r" \\" + "\n"

    content += r""" \bottomrule
\end{tabular}
\end{table}

"""
    return content


def generate_table4(headers: list[str], rows: list[list[str]]) -> str:
    """Generate Table 4 LaTeX content."""
    content = r"""
\begin{table}[t]
\centering
\scriptsize
\caption{Performance trained on full dataset with 50/50 train/test split per class (Top-1 accuracy).}
\label{tab:table4}
\begin{tabular}{lrrrr}
\toprule
"""
    content += " & ".join(str(c) for c in headers) + r" \\ \midrule" + "\n"

    for row in rows:
        content += " & ".join(str(c) for c in row) + r" \\" + "\n"

    content += r""" \bottomrule
\end{tabular}
\end{table}

"""
    return content


def main():
    output_path = generate_complete_latex()
    print(f"✓ Generated complete LaTeX document: {output_path}")
    print(f"\nTo compile to PDF:")
    print(f"  cd {output_path.parent}")
    print(f"  xelatex AllTables.tex")
    print(f"\nOr use the provided script:")
    print(f"  ../compile_all_tables.sh")


if __name__ == "__main__":
    main()
