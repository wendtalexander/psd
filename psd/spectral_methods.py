import jax.numpy as jnp
import jax.scipy as jsp
from scipy import signal

from psd.p_unit import SimulationConfig


def spectral(config: SimulationConfig, spikes: jnp.ndarray, stimulus):
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

    return f, pyy.sum(axis=0), pxx.sum(axis=0), pxy.sum(axis=0)


def spectral_by_hand(
    config: SimulationConfig, spikes: jnp.ndarray, stimulus: jnp.ndarray
):
    spikes = spikes[:, -config.nperseg :]
    stimulus = stimulus[:, -config.nperseg :]

    tau = spikes.shape[1] * config.fs
    fft_pxx = jnp.fft.fft(spikes - jnp.mean(spikes, axis=1, keepdims=True))
    pxx = (jnp.abs(fft_pxx) ** 2) / tau

    fft_pyy = jnp.fft.fft(stimulus)
    pyy = (jnp.abs(fft_pyy) ** 2) / tau  # multiply by dt

    fft_pxy = fft_pxx * jnp.conj(fft_pyy)
    pxy = fft_pxy / tau

    f = jnp.fft.rfftfreq(spikes.shape[1], d=1 / config.fs)

    return f, pyy.sum(axis=0), pxx.sum(axis=0), pxy.sum(axis=0)


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

    return f, pyy.sum(axis=0), pxx.sum(axis=0), pxy.sum(axis=0)
