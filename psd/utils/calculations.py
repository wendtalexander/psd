import nixio
import numpy as np
import numpy.typing as npt
from IPython import embed
from scipy import signal


def get_bandwidth(coherence, freqs):
    max_coh = np.nanargmax(coherence)
    threshold = coherence[max_coh] * 0.5
    new_f = np.arange(0, 1_000, 0.001)
    new_coherence = np.interp(new_f, freqs[:], coherence[:])
    crossings = new_f[(new_coherence > threshold) & (np.roll(new_coherence, 1) < threshold)]
    new_coherence[(new_coherence < threshold) | (new_f > 1000)] = 0

    bandpass = np.where(new_coherence > 0)[0]
    try:
        f1_index, f2_index = bandpass[0], bandpass[-1]
    except IndexError:
        embed()
        exit()

    return threshold, new_f[f1_index], new_f[f2_index]


def mean_spike_rate_and_modulation(
    spikes: list[nixio.DataArray], time: npt.ArrayLike, fs: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    spike_rate = np.zeros((len(spikes), time.data.shape[0]))
    for i, spikes_trial in enumerate(spikes):
        spike_rate[i, :] = get_rate(time, spikes_trial[:], dt=1 / fs)
    return np.mean(spike_rate, axis=0), np.std(spike_rate, axis=0)


def get_rate(
    time,
    spikes,
    dt,
    sigma=0.001,
    k_time=4.0,
):
    k = gauss_kernel(sigma, dt, k_time)
    rate = np.zeros_like(time)
    indeces = np.floor(spikes / dt).astype(np.int32)
    indeces = np.clip(indeces, 0, len(rate) - 1)
    rate[indeces] = 1.0
    return signal.convolve(rate, k, mode="same")


def gauss_kernel(sigma, dt, k_time):
    x = np.arange(-k_time * sigma, k_time * sigma, dt)
    return np.exp(-0.5 * (x / sigma) ** 2) / np.sqrt(2.0 * np.pi) / sigma
