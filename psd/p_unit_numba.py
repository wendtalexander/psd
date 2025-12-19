import gc
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from IPython.terminal.embed import embed
from jaxon import dsp
from jaxon.models import punit
from jaxon.params import load
from jaxon.stimuli.noise import whitenoise
from rich.progress import track

from psd import spectral_methods
from psd.punit_model_numba import simulate
from psd.sim_config import SimulationConfig
from psd.utils import setup_rich
from psd.utils.general import find_project_root
from psd.utils.logging import setup_logging
from psd.utils.spectral import SpectralResults
from psd.utils.white_noise import whitenoise as whitenoise_jan

log = logging.getLogger(__name__)
setup_logging(log)


def simulation(config: SimulationConfig, params: punit.PUnitParams):
    log.debug(f"Processing nix File {config.save_path.name}")
    spike_rate = jax.vmap(jax.jit(dsp.rate.spike_rate), in_axes=[0, None])
    ktime = jnp.arange(
        -config.ktime * config.sigma, config.ktime * config.sigma, 1 / config.fs
    )
    time = np.arange(0, config.duration, 1 / config.fs)
    baseline = np.sin(2 * jnp.pi * config.eodf * time)
    kernel = dsp.kernels.gauss_kernel(config.sigma, 1 / config.fs, config.ktime)
    params = asdict(params)
    params.pop("cell")
    params.pop("EODf")

    for con, contrast in enumerate(config.contrasts):
        f = jnp.fft.rfftfreq(config.nperseg)

        spectral_fft = SpectralResults.zeros(
            config.nperseg, config.fs, name="fft", oneside=False
        )
        spectral_welch = SpectralResults.zeros(
            config.nperseg, config.fs, name="welch_segments"
        )
        spectral_convolved_spikes_fft = SpectralResults.zeros(
            config.nperseg, config.fs, name="rate_fft", oneside=False
        )
        spectral_convolved_spikes_welch_segments = SpectralResults.zeros(
            config.nperseg, config.fs, name="rate_welch_segments"
        )

        for trial in track(range(config.trials), description="Trials"):
            wh = whitenoise_jan(
                config.wh_low,
                config.wh_high,
                1 / config.fs,
                config.duration,
            )[:-1]

            stimulus = baseline + (baseline * (wh * contrast))
            # with jax.default_device(cpu_device):

            spikes_numba = simulate(stimulus, **params)
            spikes = np.zeros_like(time)
            spike_index = np.clip(
                np.round(spikes_numba * config.fs), 0, len(time) - 1
            ).astype(int)
            spikes[spike_index] = 1
            spikes = spikes[np.newaxis, :]
            wh = wh[np.newaxis, :]

            rate = spike_rate(spikes[:, -config.nperseg :], kernel)
            rate_sum = rate.sum(axis=0)

            pyy, pxx, pxy = spectral_methods.fft(config, spikes, wh)
            spectral_fft = spectral_fft.update(pyy=pyy, pxx=pxx, pxy=pxy, rate=rate_sum)

            pyy, pxx, pxy = spectral_methods.fft(config, rate, wh)
            spectral_convolved_spikes_fft = spectral_convolved_spikes_fft.update(
                pyy=pyy, pxx=pxx, pxy=pxy, rate=rate_sum
            )

            pyy, pxx, pxy = spectral_methods.welch_segments(config, spikes, wh)
            spectral_welch = spectral_welch.update(
                pyy=pyy, pxx=pxx, pxy=pxy, rate=rate_sum
            )

            pyy, pxx, pxy = spectral_methods.welch_segments(config, rate, wh)
            spectral_convolved_spikes_welch_segments = (
                spectral_convolved_spikes_welch_segments.update(
                    pyy=pyy, pxx=pxx, pxy=pxy, rate=rate_sum
                )
            )

        methods = [
            spectral_fft,
            spectral_welch,
            spectral_convolved_spikes_welch_segments,
            spectral_convolved_spikes_fft,
        ]
        for i, method in enumerate(methods):
            m = method.norm(config.trials)
            m = m.coherence_and_transfer()
            m.save(config, contrast)

        gc.collect()


def main() -> None:
    models: list[punit.PUnitParams] = load.punit_params()
    for model in models:
        savepath: Path = find_project_root() / "data" / "numba" / model.cell

        if not savepath.exists():
            savepath.mkdir(parents=True, exist_ok=True)

        nix_files = savepath.rglob("*.nix")
        for nix_file in nix_files:
            if nix_file.is_file():
                log.debug("Found nix File deleting it")
                nix_file.unlink()
        config = SimulationConfig(save_path=savepath, cell=model.cell, eodf=model.EODf)
        model.deltat = 1 / config.fs
        simulation(config, model)
        sys.exit()


if __name__ == "__main__":
    main()
