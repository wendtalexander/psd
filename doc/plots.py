import pathlib

import nixio
import numpy as np
import plotly.graph_objects as go
from IPython import embed
from plotly.subplots import make_subplots


def spectral(
    dataset: list[pathlib.Path],
    contrast: float = 0.1,
    negative_frequencies: bool = False,
):
    nix_file = nixio.File.open(str(dataset[0]), "r")
    block = nix_file.blocks[0]
    das = block.data_arrays
    pxx = das[f"pxx_contrast_{contrast}"][:]
    pyy = das[f"pyy_contrast_{contrast}"][:]
    pxy = das[f"pxy_contrast_{contrast}"][:]
    coherence = das[f"coherence_contrast_{contrast}"][:]
    transfer = das[f"transfer_contrast_{contrast}"][:]
    f = das[f"frequency_contrast_{contrast}"][:]

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

    return fig


def jax_vs_numba(
    dataset_methods_jax,
    dataset_methods_numba,
    contrast: float = 0.1,
    negative_frequencies: bool = False,
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
    names = ["jax", "numba"]
    color = ["blue", "magenta"]
    for i, dataset in enumerate([dataset_methods_jax, dataset_methods_numba]):
        print(dataset)
        nix_file = nixio.File.open(str(dataset[0]), "r")
        block = nix_file.blocks[0]
        das = block.data_arrays
        pxx = das[f"pxx_contrast_{contrast}"][:]
        pyy = das[f"pyy_contrast_{contrast}"][:]
        pxy = das[f"pxy_contrast_{contrast}"][:]
        coherence = das[f"coherence_contrast_{contrast}"][:]
        transfer = das[f"transfer_contrast_{contrast}"][:]
        f = das[f"frequency_contrast_{contrast}"][:]

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
    return fig


if __name__ == "__main__":
    import plotly.io as pio

    pio.renderers.default = "browser"
    method = "fft"
    datafolder = pathlib.Path("psd/data/punit/jax/")
    dataset_methods = sorted(datafolder.rglob(f"{method}*.nix"))
    fig = spectral(dataset_methods, negative_frequencies=True)
    fig.show()
