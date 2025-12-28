import logging
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
from IPython.terminal.embed import embed
from jaxon import dsp
from jaxon.models import punit
from jaxon.params import load
from jaxon.stimuli.noise import whitenoise
from rich.progress import track

from psd.spectral_methods import Config, SpectralMethods
from psd.utils import setup_rich
from psd.utils.general import find_project_root
from psd.utils.logging import setup_logging

log = logging.getLogger(__name__)
setup_logging(log)


def simulation(config: Config, params: punit.PUnitParams):
    log.debug(f"Processing nix File {config.savepath.name}")
    k = jax.random.PRNGKey(config.jax_key)
    keys = jax.random.split(k, config.trials * len(config.contrasts) * 2).reshape(
        2, len(config.contrasts), config.trials, -1
    )
    simulate = jax.vmap(jax.jit(punit.simulate_spikes), in_axes=[0, 0, None])
    white_noise = jax.vmap(whitenoise, in_axes=[0, None, None, None, None])
    spike_rate = jax.vmap(jax.jit(dsp.rate.spike_rate), in_axes=[0, None])
    time = jnp.arange(0, config.duration, 1 / config.fs)
    baseline = jnp.sin(2 * jnp.pi * config.eodf * time)[jnp.newaxis, :]
    baseline = jnp.repeat(baseline, config.batch_size, axis=0)
    kernel = dsp.kernels.gauss_kernel(config.sigma, 1 / config.fs, config.ktime)
    cpu_device = jax.devices("cpu")[0]
    for con, contrast in enumerate(config.contrasts):
        sm = SpectralMethods(config)
        # for batch in jnp.arange(0, config.trials, config.batch_size):
        for batch in track(
            jnp.arange(0, config.trials, config.batch_size), description="Batches"
        ):
            wh = white_noise(
                keys[0, con, batch : batch + config.batch_size, :],
                config.wh_low,
                config.wh_high,
                config.fs,
                config.duration,
            )
            stimulus = baseline + (baseline * (wh * contrast))
            with jax.default_device(cpu_device):
                spikes = simulate(
                    keys[1, con, batch : batch + config.batch_size, :], stimulus, params
                )
                rate = spike_rate(spikes[:, -config.nperseg :], kernel)
                sm.update(spikes, wh, rate)
        sm.norm()
        sm.coherence_and_transfer()
        sm.save(contrast=contrast)


def main() -> None:
    models: list[punit.PUnitParams] = load.punit_params()
    for model in models:
        basepath: Path = find_project_root() / "data" / "punit" / "nperseg" / model.cell
        npersegs = [2**15, 2**16, 2**17, 2**18, 2**19, 2**20, 2**21]
        for nperseg in npersegs:
            savepath = basepath / f"{nperseg}"
            if not savepath.exists():
                savepath.mkdir(parents=True, exist_ok=True)

            nix_files = savepath.rglob("*.nix")
            for nix_file in nix_files:
                if nix_file.is_file():
                    log.debug("Found nix File deleting it")
                    nix_file.unlink()

            config = Config(
                fs=30_000,
                duration=100,
                batch_size=25,
                trials=100,
                nperseg=nperseg,
                savepath=savepath,
                cell=model.cell,
                eodf=model.EODf,
            )
            model.deltat = 1 / config.fs
            simulation(config, model)
        sys.exit()


if __name__ == "__main__":
    main()
