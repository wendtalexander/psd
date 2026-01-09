import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from IPython.terminal.embed import embed
from jaxon import dsp
from jaxon.models import punit
from jaxon.params import load
from rich.progress import track

from psd.punit.model_numba import simulate
from psd.spectral_methods import Config, SpectralMethods
from psd.utils import setup_rich
from psd.utils.general import find_project_root
from psd.utils.logging import setup_logging
from psd.utils.white_noise import whitenoise as whitenoise_jan

log = logging.getLogger(__name__)
setup_logging(log)


def simulation(config: Config, params: punit.PUnitParams):
    log.debug(f"Processing nix File {config.savepath.name}")
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

    cpu_device = jax.devices("cpu")[0]
    with jax.default_device(cpu_device):
        for con, contrast in enumerate(config.contrasts):
            # sm = SpectralMethods(config)
            sm = SpectralMethods(config, methods=["welch", "fft_segments"])

            # for trial in track(range(config.trials), description="Trials"):
            for trial in range(config.trials):
                wh = whitenoise_jan(
                    config.wh_low,
                    config.wh_high,
                    1 / config.fs,
                    config.duration,
                )[:-1]

                stimulus = baseline + (baseline * (wh * contrast))

                spikes_numba = simulate(stimulus, **params)
                spikes = jnp.zeros_like(time)
                spike_index = jnp.clip(
                    jnp.round(spikes_numba * config.fs), 0, len(time) - 1
                ).astype(int)
                spikes.at[spike_index].set(1)
                spikes = spikes[jnp.newaxis, :]
                wh = wh[jnp.newaxis, :]

                rate = spike_rate(spikes[:, -config.nperseg :], kernel)

                sm.update(spikes, wh, rate)

            sm.norm()
            sm.coherence_and_transfer()
            sm.save(contrast)


def main() -> None:
    models: list[punit.PUnitParams] = load.punit_params()
    for model in models:
        savepath: Path = (
            find_project_root() / "data" / "punit" / "long_numba" / model.cell
        )

        if not savepath.exists():
            savepath.mkdir(parents=True, exist_ok=True)

        nix_files = savepath.rglob("*.nix")
        for nix_file in nix_files:
            if nix_file.is_file():
                log.debug("Found nix File deleting it")
                nix_file.unlink()
        config = Config(
            trials=1,
            duration=20_000,
            savepath=savepath,
            cell=model.cell,
            eodf=model.EODf,
        )
        model.deltat = 1 / config.fs
        simulation(config, model)
        sys.exit()


if __name__ == "__main__":
    main()
