"""Simple Configuration for matplotlib plotting."""

import matplotlib as mpl
from matplotlib.pyplot import Axes

mpl.rcParams.update(
    {
        "figure.figsize": (6, 4),
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "ytick.major.width": 1,
        "xtick.major.width": 1,
        "savefig.format": "pdf",
        "axes.linewidth": 1.2,
        "savefig.bbox": "tight",
        "lines.linewidth": 1.5,
        "legend.fontsize": 8,
    }
)


def style_spines(ax: Axes, offset: int = 3) -> None:
    """Apply padding for bottom and left spine.

    Parameters
    ----------
    ax : Axes
        Plot for drawing
    offset : int
        Padding value
    """
    for side in ["bottom", "left"]:
        ax.spines[side].set_position(("outward", offset))
