import pathlib
from contextlib import contextmanager

import matplotlib as mpl
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import nixio
import numpy as np
import plotly.graph_objects as go
from IPython import embed
from plotly.subplots import make_subplots


@contextmanager
def dark_xkcd():
    """
    A custom context manager that combines dark_background, xkcd,
    and fixes the path effects to use a black outline (k)
    so fonts remain crisp on a dark background.
    """
    rc_fix = {
        "path.effects": [patheffects.withStroke(linewidth=4, foreground="k")],
    }

    with plt.xkcd():
        with plt.style.context("dark_background"):
            with mpl.rc_context(rc_fix):
                yield


def spectral(
    dataset: list[pathlib.Path],
    contrast: float | None = None,
    negative_frequencies: bool = False,
):
    nix_file = nixio.File.open(str(dataset[0]), "r")
    block = nix_file.blocks[0]
    das = block.data_arrays
    contrast_suffix = f"_contrast_{contrast}" if contrast else ""
    pxx = das[f"pxx{contrast_suffix}"][:]
    pyy = das[f"pyy{contrast_suffix}"][:]
    pxy = das[f"pxy{contrast_suffix}"][:]
    coherence = das[f"coherence{contrast_suffix}"][:]
    transfer = das[f"transfer{contrast_suffix}"][:]
    f = das[f"frequency{contrast_suffix}"][:]

    fig = make_subplots(
        6,
        2,
        specs=[
            [{"rowspan": 2}, {"rowspan": 3}],
            [None, None],
            [{"rowspan": 2}, None],
            [None, {"rowspan": 3}],
            [{"rowspan": 2}, None],
            [None, None],
        ],
        shared_xaxes=True,
    )
    fig.add_trace(
        go.Scattergl(x=f, y=pyy, mode="markers+lines", name="pyy"), row=1, col=1
    )
    fig.add_trace(
        go.Scattergl(x=f, y=pxx, mode="markers+lines", name="pxx"), row=3, col=1
    )
    fig.add_trace(
        go.Scattergl(x=f, y=np.abs(pxy), mode="markers+lines", name="pxy"), row=5, col=1
    )
    fig.add_trace(
        go.Scattergl(x=f, y=coherence, mode="markers+lines", name="coherence"),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scattergl(x=f, y=transfer, mode="markers+lines", name="transfer"),
        row=4,
        col=2,
    )
    x_range = [-350, 350] if negative_frequencies else [-1, 350]
    fig.update_xaxes(range=x_range)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(type="log", row=3, col=1)
    fig.update_yaxes(type="log", row=5, col=1)
    fig.update_yaxes(range=[0.0, 0.02], row=4, col=2)
    nix_file.close()

    return fig


def rate(dataset: list[pathlib.Path], contrast: float | None = None):
    nix_file = nixio.File.open(str(dataset[0]), "r")
    block = nix_file.blocks[0]
    das = block.data_arrays
    contrast_suffix = f"_contrast_{contrast}" if contrast else ""
    time = das[f"time{contrast_suffix}"][:]
    rate = das[f"mean_rate{contrast_suffix}"][:]
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=time, y=rate, mode="markers+lines", name="rate"))
    nix_file.close()
    return fig


def rate_comparision(
    datasets: list[pathlib.Path], names=list[str], contrast: float | None = None
):
    fig = go.Figure()
    for i, dataset in enumerate(datasets):
        nix_file = nixio.File.open(str(dataset), "r")
        block = nix_file.blocks[0]
        das = block.data_arrays
        contrast_suffix = f"_contrast_{contrast}" if contrast else ""
        time = das[f"time{contrast_suffix}"][:]
        rate = das[f"mean_rate{contrast_suffix}"][:]
        fig.add_trace(go.Scattergl(x=time, y=rate, mode="markers+lines", name=names[i]))
    nix_file.close()
    return fig


def comparision(
    dataset_methods_jax,
    dataset_methods_numba,
    contrast: float | None = None,
    negative_frequencies: bool = False,
    names: list[str] = ["jax", "numba"],
):
    fig = make_subplots(
        6,
        2,
        specs=[
            [{"rowspan": 2}, {"rowspan": 3}],
            [None, None],
            [{"rowspan": 2}, None],
            [None, {"rowspan": 3}],
            [{"rowspan": 2}, None],
            [None, None],
        ],
        shared_xaxes="all",
    )
    color = ["blue", "magenta"]
    for i, dataset in enumerate([dataset_methods_jax, dataset_methods_numba]):
        print(dataset)
        nix_file = nixio.File.open(str(dataset[0]), "r")
        block = nix_file.blocks[0]
        das = block.data_arrays

        contrast_suffix = f"_contrast_{contrast}" if contrast else ""
        pxx = das[f"pxx{contrast_suffix}"][:]
        pyy = das[f"pyy{contrast_suffix}"][:]
        pxy = das[f"pxy{contrast_suffix}"][:]
        coherence = das[f"coherence{contrast_suffix}"][:]
        transfer = das[f"transfer{contrast_suffix}"][:]
        f = das[f"frequency{contrast_suffix}"][:]

        fig.add_trace(
            go.Scattergl(
                x=f,
                y=pyy,
                mode="markers+lines",
                name=f"pyy {names[i]}",
                line_color=color[i],
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scattergl(
                x=f,
                y=pxx,
                mode="markers+lines",
                name=f"pxx {names[i]}",
                line_color=color[i],
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scattergl(
                x=f,
                y=np.abs(pxy),
                mode="markers+lines",
                name=f"pxy {names[i]}",
                line_color=color[i],
            ),
            row=5,
            col=1,
        )
        fig.add_trace(
            go.Scattergl(
                x=f,
                y=coherence,
                mode="markers+lines",
                name=f"coherence {names[i]}",
                line_color=color[i],
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scattergl(
                x=f,
                y=transfer,
                mode="markers+lines",
                name=f"transfer {names[i]}",
                line_color=color[i],
            ),
            row=4,
            col=2,
        )
    x_range = [-350, 350] if negative_frequencies else [-1, 350]
    fig.update_xaxes(range=x_range)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(type="log", row=3, col=1)
    fig.update_yaxes(type="log", row=5, col=1)
    fig.update_yaxes(range=[0.0, 0.02], row=4, col=2)
    nix_file.close()
    return fig


def get_coherence(
    dataset, contrast: float | None = None, lower: float = -50, upper: float = 350
):
    nix_file = nixio.File.open(str(dataset[0]), "r")
    block = nix_file.blocks[0]
    das = block.data_arrays
    contrast_suffix = f"_contrast_{contrast}" if contrast else ""
    coherence = das[f"coherence{contrast_suffix}"][:]
    f = das[f"frequency{contrast_suffix}"][:]
    mask = (f >= lower) & (f <= upper)
    return f[mask], coherence[mask]


if __name__ == "__main__":
    import plotly.io as pio

    pio.renderers.default = "browser"
    method = "fft_without"
    datafolder = pathlib.Path("psd/data/punit/duration/")
    dataset_methods = sorted(datafolder.rglob(f"{method}*.nix"))
    fig = spectral(dataset_methods, contrast=0.1, negative_frequencies=True)
    fig.show()
