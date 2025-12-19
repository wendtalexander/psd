import jax.numpy as jnp
import jax.scipy as jsp
from IPython import embed
from scipy import signal

from psd.p_unit_jax import SimulationConfig


def welch_segments(config: SimulationConfig, spikes: jnp.ndarray, stimulus):
    spikes = spikes[:, -config.nperseg :]
    stimulus = stimulus[:, -config.nperseg :]
    f, pyy = jsp.signal.welch(
        stimulus,
        fs=config.fs,
        nperseg=config.nperseg,
        nfft=None,
        noverlap=None,
    )
    _, pxx = jsp.signal.welch(
        spikes - jnp.mean(spikes, axis=1, keepdims=True),
        fs=config.fs,
        nperseg=config.nperseg,
        detrend=False,
        nfft=None,
        noverlap=None,
    )
    _, pxy = jsp.signal.csd(
        spikes - jnp.mean(spikes, axis=1, keepdims=True),
        stimulus,
        fs=config.fs,
        nperseg=config.nperseg,
        detrend=False,
        nfft=None,
        noverlap=None,
    )

    return pyy.sum(axis=0), pxx.sum(axis=0), pxy.sum(axis=0)


def fft(config: SimulationConfig, spikes: jnp.ndarray, stimulus: jnp.ndarray):
    spikes = spikes[:, -config.nperseg :]
    stimulus = stimulus[:, -config.nperseg :]

    dt = 1 / config.fs
    fft_pxx = jnp.fft.fft(spikes - jnp.mean(spikes, axis=1, keepdims=True)) * dt
    pxx = jnp.abs(fft_pxx) ** 2

    fft_pyy = jnp.fft.fft(stimulus) * dt
    pyy = jnp.abs(fft_pyy) ** 2

    fft_pxy = fft_pxx * jnp.conj(fft_pyy)
    pxy = fft_pxy
    return pyy.sum(axis=0), pxx.sum(axis=0), pxy.sum(axis=0)


def spectral_scipy(config: SimulationConfig, spikes: jnp.ndarray, stimulus):
    spikes = spikes[:, -config.nperseg :]
    stimulus = stimulus[:, -config.nperseg :]
    f, pyy = signal.welch(
        stimulus,
        fs=config.fs,
        nperseg=config.nperseg,
        nfft=None,
        noverlap=None,
    )
    _, pxx = signal.welch(
        spikes - jnp.mean(spikes, axis=1, keepdims=True),
        fs=config.fs,
        nperseg=config.nperseg,
        detrend=False,
        nfft=None,
        noverlap=None,
    )
    _, pxy = signal.csd(
        spikes - jnp.mean(spikes, axis=1, keepdims=True),
        stimulus,
        fs=config.fs,
        nperseg=config.nperseg,
        detrend=False,
        nfft=None,
        noverlap=None,
    )

    return pyy.sum(axis=0), pxx.sum(axis=0), pxy.sum(axis=0)
