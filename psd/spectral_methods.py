import pathlib
from dataclasses import dataclass, replace

import jax.numpy as jnp
import jax.scipy as jsp
import nixio
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
    def zeros(
        cls, nperseg: int, fs: float, name: str, negative_frequencies: bool = False
    ):
        n_freqs = (nperseg // 2) + 1
        f = jnp.fft.rfftfreq(nperseg, d=1 / fs)
        n_freqs = f.shape[0]
        time = jnp.arange(nperseg) / fs
        if negative_frequencies:
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


@dataclass
class Config:
    nperseg: int
    fs: float
    savepath: str | pathlib.Path
    sigma: float  # sigma Gaus
    ktime: float  # kernel time


@dataclass(frozen=True)
class MethodInfo:
    func_name: str
    negative_freq: bool
    use_rate: bool


class SpectralMethods:
    _REGISTRY = {
        "fft": MethodInfo("fft", True, False),
        "rate_fft": MethodInfo("fft", True, True),
        "welch_segments": MethodInfo("welch_segments", False, False),
        "rate_welch_segments": MethodInfo("welch_segments", False, True),
    }

    def __init__(self, config: Config, methods: list[str] | None) -> None:
        self.config = config

        if methods is None:
            self.methods = list(self._REGISTRY.keys())
        else:
            self.methods = methods

        for m in self.methods:
            if m not in self._REGISTRY:
                raise ValueError(
                    f"Unknown method: '{m}', Valid options: {list(self._REGISTRY.keys())}"
                )
        self.spectral_results = self.zeros()

    def zeros(self) -> list[SpectralResults]:
        return [
            SpectralResults.zeros(
                self.config.nperseg,
                self.config.fs,
                method_name,
                self._REGISTRY[method_name].negative_freq,
            )
            for method_name in self.methods
        ]

    def update(self, spikes, stimulus, rate) -> None:
        spectral_result: list[SpectralResults] = []
        for i, method in enumerate(self.methods):
            reg = self._REGISTRY[method]
            method_func = getattr(self, reg.func_name)

            if reg.use_rate:
                pxx, pyy, pxy = method_func(rate, stimulus)
            else:
                pxx, pyy, pxy = method_func(spikes, stimulus)

            res = self.spectral_results[i].update(pxx, pyy, pxy, rate)
            spectral_result.append(res)

        self.spectral_results = spectral_result

    def norm(self) -> None:
        pass

    def coherence_and_transfer(self) -> None:
        pass

    def save(self) -> None:
        pass

    def welch_segments(self, spikes: jnp.ndarray, stimulus):
        spikes = spikes[:, -self.config.nperseg :]
        stimulus = stimulus[:, -self.config.nperseg :]
        f, pyy = jsp.signal.welch(
            stimulus,
            fs=self.config.fs,
            nperseg=self.config.nperseg,
            nfft=None,
            noverlap=None,
        )
        _, pxx = jsp.signal.welch(
            spikes - jnp.mean(spikes, axis=1, keepdims=True),
            fs=self.config.fs,
            nperseg=self.config.nperseg,
            detrend=False,
            nfft=None,
            noverlap=None,
        )
        _, pxy = jsp.signal.csd(
            spikes - jnp.mean(spikes, axis=1, keepdims=True),
            stimulus,
            fs=self.config.fs,
            nperseg=self.config.nperseg,
            detrend=False,
            nfft=None,
            noverlap=None,
        )

        return pyy.sum(axis=0), pxx.sum(axis=0), pxy.sum(axis=0)

    def fft(self, spikes: jnp.ndarray, stimulus: jnp.ndarray):
        spikes = spikes[:, -self.config.nperseg :]
        stimulus = stimulus[:, -self.config.nperseg :]

        dt = 1 / self.config.fs
        fft_pxx = jnp.fft.fft(spikes - jnp.mean(spikes, axis=1, keepdims=True)) * dt
        pxx = jnp.abs(fft_pxx) ** 2

        fft_pyy = jnp.fft.fft(stimulus) * dt
        pyy = jnp.abs(fft_pyy) ** 2

        fft_pxy = fft_pxx * jnp.conj(fft_pyy)
        pxy = fft_pxy
        return pyy.sum(axis=0), pxx.sum(axis=0), pxy.sum(axis=0)
