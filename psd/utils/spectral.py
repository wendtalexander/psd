from dataclasses import dataclass, replace

import jax.numpy as jnp


@dataclass
class SpectralResults:
    pxxs: jnp.ndarray
    pyys: jnp.ndarray
    pxys: jnp.ndarray
    rates: jnp.ndarray
    coherence: jnp.ndarray
    transfer: jnp.ndarray
    f: jnp.ndarray

    @classmethod
    def zeros(cls, nperseg, fs, oneside: True):
        n_freqs = (nperseg // 2) + 1
        f = jnp.fft.rfftfreq(nperseg, d=1 / fs)
        n_freqs = f.shape[0]
        if not oneside:
            f = jnp.fft.fftfreq(nperseg, d=1 / fs)
            n_freqs = f.shape[0]

        return cls(
            pxxs=jnp.zeros(n_freqs),
            pyys=jnp.zeros(n_freqs),
            pxys=jnp.zeros(n_freqs, dtype=jnp.complex64),
            transfer=jnp.zeros(n_freqs, dtype=jnp.complex64),
            coherence=jnp.zeros(n_freqs),
            rates=jnp.zeros(nperseg),
            f=f,
        )

    def norm(self, trials):
        return replace(
            self,
            pxxs=self.pxxs / trials,
            pyys=self.pyys / trials,
            pxys=self.pxys / trials,
            rates=self.rates / trials,
        )

    def update(self, pxx, pyy, pxy, rate):
        return replace(
            self,
            pxxs=self.pxxs + pxx,
            pyys=self.pyys + pyy,
            pxys=self.pxys + pxy,
            rates=self.rates + rate,
        )
