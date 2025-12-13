from pathlib import Path

from matplotlib.pyplot import Axes


def find_project_root(marker="psd"):
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent / marker
    raise FileNotFoundError(f"Could not find {marker} in any parent directory.")


def style_pvalue(pvalue: float) -> str:
    if pvalue <= 0.005:
        return "<0.005"
    elif pvalue > 0.05:
        return f"{pvalue:.2f}"
    else:
        return "<0.05"
