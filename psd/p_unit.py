import gc
import logging
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.pyplot as plt
import nixio
from IPython.terminal.embed import embed
from jaxon import dsp
from jaxon.models import punit
from jaxon.params import load
from jaxon.stimuli.noise import whitenoise
from nixio.exceptions import DuplicateName
from rich.progress import track

from psd import spectral_methods
from psd.utils import setup_rich
from psd.utils.general import find_project_root
from psd.utils.logging import setup_logging
from psd.utils.spectral import SpectralResults
from psd.utils.white_noise import whitenoise as whitenoise_jan

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


def simulation(config: SimulationConfig, params: punit.PUnitParams):
    log.debug(f"Processing nix File {config.save_path.name}")
    k = jax.random.PRNGKey(config.jax_key)
    keys = jax.random.split(k, config.trials * len(config.contrasts) * 2).reshape(
        2, len(config.contrasts), config.trials, -1
    )

    simulate = jax.vmap(jax.jit(punit.simulate_spikes), in_axes=[0, 0, None])
    white_noise = jax.vmap(whitenoise, in_axes=[0, None, None, None, None])
    spike_rate = jax.vmap(jax.jit(dsp.rate.spike_rate), in_axes=[0, None])
    ktime = jnp.arange(
        -config.ktime * config.sigma, config.ktime * config.sigma, 1 / config.fs
    )
    time = jnp.arange(0, config.duration, 1 / config.fs)
    baseline = jnp.sin(2 * jnp.pi * config.eodf * time)[jnp.newaxis, :]
    baseline = jnp.repeat(baseline, config.batch_size, axis=0)
    n_freq_bins = config.nperseg // 2 + 1
    kernel = dsp.kernels.gauss_kernel(config.sigma, 1 / config.fs, config.ktime)

    for con, contrast in enumerate(config.contrasts):
        f = jnp.fft.rfftfreq(config.nperseg)

        spectral_by_hand = SpectralResults.zeros(
            config.nperseg, config.fs, oneside=False
        )

        for batch in jnp.arange(0, config.trials, config.batch_size):
            wh = white_noise(
                keys[0, con, batch : batch + config.batch_size, :],
                config.wh_low,
                config.wh_high,
                config.fs,
                config.duration,
            )
            # wh_jan = jnp.array(
            #     [
            #         whitenoise_jan(
            #             config.wh_low,
            #             config.wh_high,
            #             1 / config.fs,
            #             config.duration,
            #         )[:-1]
            #         for _ in range(config.batch_size)
            #     ]
            # )
            stimulus = baseline + (baseline * (wh * contrast))
            spikes = simulate(
                keys[1, con, batch : batch + config.batch_size, :], stimulus, params
            )
            rate = spike_rate(spikes[:, -config.nperseg :], kernel).sum(axis=0)

            pyy, pxx, pxy = spectral_methods.fft(config, spikes, wh)
            plt.semilogy(spectral_by_hand.f, pyy)
            plt.xlim(-30, 30)
            plt.savefig("psd_jan.pdf")
            plt.close()

            spectral_by_hand = spectral_by_hand.update(
                pyy=pyy, pxx=pxx, pxy=pxy, rate=rate
            )

        spectral_by_hand = spectral_by_hand.norm(config.trials)

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
    models: list[punit.PUnitParams] = load.punit_params()
    for model in models:
        savepath: Path = find_project_root() / "data" / "methods" / model.cell

        if not savepath.exists():
            savepath.mkdir(parents=True, exist_ok=True)

        nix_file_path: Path = savepath / f"{model.cell}.nix"
        if nix_file_path.is_file():
            log.debug("Found nix File deleting it")
            nix_file_path.unlink()
        config = SimulationConfig(
            save_path=nix_file_path, cell=model.cell, eodf=model.EODf
        )
        model.deltat = 1 / config.fs
        simulation(config, model)
        sys.exit()


if __name__ == "__main__":
    main()
