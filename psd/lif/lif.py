import logging
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
from IPython.terminal.embed import embed
from jaxon import dsp
from jaxon.models import lif
from jaxon.params import load
from jaxon.stimuli.noise import whitenoise
from rich.progress import track

from psd.spectral_methods import Config, SpectralMethods
from psd.utils import setup_rich
from psd.utils.general import find_project_root
from psd.utils.logging import setup_logging

log = logging.getLogger(__name__)
setup_logging(log)


def simulation(config: Config, params: lif.LIFParams):
    log.debug(f"Processing nix File {config.savepath.name}")
    k = jax.random.PRNGKey(config.jax_key)
    keys = jax.random.split(k, config.trials * len(config.contrasts) * 2).reshape(
        2, config.trials, -1
    )
    simulate = jax.vmap(jax.jit(lif.simulate_spikes), in_axes=[0, 0, None])
    white_noise = jax.vmap(whitenoise, in_axes=[0, None, None, None, None])
    spike_rate = jax.vmap(jax.jit(dsp.rate.spike_rate), in_axes=[0, None])
    kernel = dsp.kernels.gauss_kernel(config.sigma, 1 / config.fs, config.ktime)

    sm = SpectralMethods(config)
    # for batch in jnp.arange(0, config.trials, config.batch_size):
    for batch in track(
        jnp.arange(0, config.trials, config.batch_size), description="Batches"
    ):
        wh = white_noise(
            keys[0, batch : batch + config.batch_size, :],
            config.wh_low,
            config.wh_high,
            config.fs,
            config.duration,
        )
        spikes = simulate(keys[1, batch : batch + config.batch_size, :], wh, params)
        rate = spike_rate(spikes[:, -config.nperseg :], kernel)
        sm.update(spikes, wh, rate)
    sm.norm()
    sm.coherence_and_transfer()
    sm.save()


def main() -> None:
    savepath: Path = find_project_root() / "data" / "lif"
    model = lif.LIFParams(
        fs=30_000,
        threshold=1.0,
        v_offset=0.2,
        tau_mem=0.01,
        v_base=0.0,
        noise_strength=0.01,
    )

    if not savepath.exists():
        savepath.mkdir(parents=True, exist_ok=True)

    nix_files = savepath.rglob("*.nix")
    for nix_file in nix_files:
        if nix_file.is_file():
            log.debug("Found nix File deleting it")
            nix_file.unlink()
    config = Config(savepath=savepath, cell="None", eodf="None")
    simulation(config, model)


if __name__ == "__main__":
    main()
