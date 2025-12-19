from dataclasses import dataclass, replace

import jax.numpy as jnp
import nixio
from IPython.terminal.embed import embed
from nixio.exceptions import DuplicateName


@dataclass
class SpectralResults:
    pxxs: jnp.ndarray
    pyys: jnp.ndarray
    pxys: jnp.ndarray
    rates: jnp.ndarray
    coherence: jnp.ndarray
    transfer: jnp.ndarray
    f: jnp.ndarray
    time: jnp.ndarray
    name: str

    @classmethod
    def zeros(cls, nperseg: int, fs: float, name: str, oneside: bool = True):
        n_freqs = (nperseg // 2) + 1
        f = jnp.fft.rfftfreq(nperseg, d=1 / fs)
        n_freqs = f.shape[0]
        time = jnp.arange(nperseg) / fs
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
            time=time,
            name=name,
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

    def coherence_and_transfer(self):
        coherence = (jnp.abs(self.pxys) ** 2) / (self.pxxs * self.pyys)
        transfer = jnp.abs(self.pxys / self.pyys)
        return replace(self, coherence=coherence, transfer=transfer)

    def save(self, config, contrast):
        das = [
            self.pyys,
            self.pxxs,
            self.pxys,
            self.rates,
            self.coherence,
            self.transfer,
            self.f,
            self.time,
        ]
        das_names = [
            f"{self.name}_pyy_contrast_{contrast}",
            f"{self.name}_pxx_contrast_{contrast}",
            f"{self.name}_pxy_contrast_{contrast}",
            f"{self.name}_rate_contrast_{contrast}",
            f"{self.name}_coherence_contrast_{contrast}",
            f"{self.name}_transfer_contrast_{contrast}",
            "f",
            "time",
        ]
        das_type = [
            "stimulus.power.spectrum",
            "spikes.power.spectra",
            "cross.power.spectra",
            "mean.rate",
            "coherence",
            "transfer",
            "f",
            "time",
        ]
        name = self.name + "_" + config.save_path.name + ".nix"
        nix_file_name = config.save_path / name

        with nixio.File(str(nix_file_name), "a") as file:
            try:
                block: nixio.Block = file.create_block("result", "result.block")
            except DuplicateName:
                block = file.blocks["result"]

            for arr in range(len(das_names)):
                block.create_data_array(das_names[arr], das_type[arr], data=das[arr])
