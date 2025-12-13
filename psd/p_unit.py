import gc
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import nixio
from IPython.terminal.embed import embed
from jaxon.models import punit
from jaxon.params import load
from jaxon.stimuli.noise import whitenoise
from nixio.execption import DuplicateName
from rich.progress import track

from psd.utils import setup_rich
from psd.utils.general import find_project_root
from psd.utils.logging import setup_logging

log = logging.getLogger(__name__)
setup_logging(log)


@dataclass
class SimulationConfig:
    save_path: Path
    cell: str
    eodf: float
    duration: float = 2.0
    trials: int = 100_000
    contrasts: list[float] = field(default_factory=lambda: [0.1])
    batch_size: int = 2000
    nperseg: int = 2**15
    fs: int = 30_000
    jax_key: int = 42
    wh_low: float = 0.0
    wh_high: float = 300.0
    sigma: float = 0.001
    ktime: float = 4


def calc_spike_rate(binary_spike_train, kernel):
    return jsp.signal.fftconvolve(binary_spike_train, kernel, "same")


def gauss_kernel(sigma, dt, k_time):
    x = jnp.arange(-k_time * sigma, k_time * sigma, dt)
    y = jnp.exp(-0.5 * (x / sigma) ** 2) / jnp.sqrt(2.0 * jnp.pi) / sigma
    return y


def simulation(config: SimulationConfig, params: punit.PUnitParams):
    log.debug(f"Processing nix File {config.save_path.name}")
    k = jax.random.PRNGKey(config.jax_key)
    keys = jax.random.split(k, config.trials * len(config.contrasts) * 2).reshape(
        2, len(config.contrasts), config.trials, -1
    )

    simulate = jax.vmap(jax.jit(punit.simulate_spikes), in_axes=[0, 0, None])
    white_noise = jax.vmap(whitenoise, in_axes=[0, None, None, None, None])
    # spectral_by_hand_func = jax.vmap(
    #     jax.jit(spectral_by_hand, static_argnames=["config"]), in_axes=[None, 0, 0]
    # )
    spike_rate = jax.vmap(jax.jit(calc_spike_rate), in_axes=[0, None])
    ktime = jnp.arange(
        -config.ktime * config.sigma, config.ktime * config.sigma, 1 / config.fs
    )
    time = jnp.arange(0, config.duration, 1 / config.fs)
    baseline = jnp.sin(2 * jnp.pi * config.eodf * time)[jnp.newaxis, :]
    baseline = jnp.repeat(baseline, config.batch_size, axis=0)
    n_freq_bins = config.nperseg // 2 + 1
    kernel = gauss_kernel(config.sigma, 1 / config.fs, config.ktime)

    for con, contrast in enumerate(config.contrasts):
        pyys = jnp.zeros(n_freq_bins)
        pxxs = jnp.zeros(n_freq_bins)
        pxys = jnp.zeros(n_freq_bins, dtype=jnp.complex64)
        rates = jnp.zeros(config.nperseg)

        for batch in track(
            jnp.arange(0, config.trials, config.batch_size),
            description=f"Processing Contrast {contrast}",
        ):
            wh = white_noise(
                keys[0, con, batch : batch + config.batch_size, :],
                config.wh_low,
                config.wh_high,
                config.fs,
                config.duration,
            )
            stimulus = baseline + (baseline * (wh * contrast))
            spikes = simulate(
                keys[1, con, batch : batch + config.batch_size, :], stimulus, params
            )
            f, pyy, pxx, pxy = spectral_by_hand(config, spikes, wh)
            rates += spike_rate(spikes[:, -config.nperseg :], kernel).sum(axis=0)
            pyys += pyy
            pxxs += pxx
            pxys += pxy

        pyys /= config.trials
        pxxs /= config.trials
        pxys /= config.trials
        rates /= config.trials

        coh = jnp.abs(pxys) ** 2 / (pxxs * pyys)
        das = [pyys, pxxs, pxys, rates, coh]
        x_axis = [f, f, f, time, f]
        x_axis_label = ["f", "f", "f", "time", "f"]
        x_axis_unit = ["Hz", "Hz", "Hz", "s", "Hz"]

        das_names = [
            f"stimulus_power_spectrum_contrast_{contrast}",
            f"spikes_power_spectra_contrast_{contrast}",
            f"cross_power_spectra_contrast_{contrast}",
            f"mean_rate_contrast_{contrast}",
            f"coherence_contrast_{contrast}",
        ]
        das_type = [
            "stimulus.power.spectrum",
            "spikes.power.spectra",
            "cross.power.spectra",
            "mean_rate",
            "coherence",
        ]

        with nixio.File(str(config.save_path), "a") as file:
            try:
                block: nixio.Block = file.create_block("result", "result.block")
            except DuplicateName:
                block = file.blocks["result"]

            for arr in range(len(das)):
                a: nixio.DataArray = block.create_data_array(
                    das_names[arr], das_type[arr], data=das[arr]
                )
                a.append_range_dimension(
                    x_axis[arr], x_axis_label[arr], x_axis_unit[arr]
                )
        gc.collect()


def main() -> None:
    models = load.punit_params()
    for model in models:
        cell = model.pop("cell")
        eodf = model.pop("EODf")
        embed()
        exit()
        savepath: Path = find_project_root() / "data" / "simulation" / cell
        if not savepath.exists():
            savepath.mkdir(parents=True, exist_ok=True)

        nix_file_path: Path = savepath / f"{cell}.nix"
        if nix_file_path.is_file():
            log.debug("Found nix File deleting it")
            nix_file_path.unlink()
        config = SimulationConfig(save_path=nix_file_path, cell=cell, eodf=eodf)
        model["deltat"] = 1 / config.fs
        params = punit.PUnitParams(**model)
        simulation(config, params)
        sys.exit()


if __name__ == "__main__":
    main()
