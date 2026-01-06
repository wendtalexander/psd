import pathlib
from dataclasses import dataclass, field, fields, replace

import jax.numpy as jnp
import jax.scipy as jsp
import nixio
from IPython.terminal.embed import embed
from nixio.exceptions import DuplicateName


@dataclass
class Config:
    savepath: pathlib.Path
    cell: str | None = None
    eodf: float | None = None
    duration: int = 2
    trials: int = 10_000
    contrasts: list[float] = field(default_factory=lambda: [0.1])
    batch_size: int = 2000
    nperseg: int = 2**15
    fs: int = 30_000
    jax_key: int = 42
    wh_low: float = 0.0
    wh_high: float = 300.0
    sigma: float = 0.001
    ktime: float = 4


@dataclass
class SpectralResults:
    pxxs: jnp.ndarray = field(
        metadata={"nix_name": "pxx", "nix_type": "spike.power.spectra"}
    )
    pyys: jnp.ndarray = field(
        metadata={"nix_name": "pyy", "nix_type": "stimulus.power.spectra"}
    )

    pxys: jnp.ndarray = field(
        metadata={"nix_name": "pxy", "nix_type": "cross.power.spectra"}
    )

    rates: jnp.ndarray = field(
        metadata={"nix_name": "mean_rate", "nix_type": "spike.mean.rate"}
    )
    coherence: jnp.ndarray = field(
        metadata={"nix_name": "coherence", "nix_type": "coherence.stimulus.spikes"}
    )

    transfer: jnp.ndarray = field(
        metadata={"nix_name": "transfer", "nix_type": "transfer.stimulus.spikes"}
    )

    f: jnp.ndarray = field(
        metadata={"nix_name": "frequency", "nix_type": "frequency.spectra"}
    )

    time: jnp.ndarray = field(metadata={"nix_name": "time", "nix_type": "time.rate"})

    name: str = "fft"

    @classmethod
    def zeros(
        cls, nperseg: int, fs: float, name: str, negative_frequencies: bool = False
    ):
        if negative_frequencies:
            f = jnp.fft.fftfreq(nperseg, d=1 / fs)  # fft
            f = jnp.fft.fftshift(f)
        else:
            f = jnp.fft.rfftfreq(nperseg, d=1 / fs)  # rfft
        n_freqs = f.shape[0]
        time = jnp.arange(nperseg) / fs

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

    def save(self, savepath: pathlib.Path, contrast: float | None):
        name = self.name + "_" + savepath.name + ".nix"
        nix_file_name = savepath / name

        with nixio.File(str(nix_file_name), "a") as file:
            try:
                block: nixio.Block = file.create_block("result", "result.block")
            except DuplicateName:
                block = file.blocks["result"]

            for f in fields(self):
                meta = f.metadata
                # dont save the name
                if "nix_type" not in meta:
                    continue
                if contrast:
                    da_name = f"{meta['nix_name']}_contrast_{contrast}"
                else:
                    da_name = f"{meta['nix_name']}"
                da = getattr(self, f.name)
                block.create_data_array(da_name, meta["nix_type"], data=da)


@dataclass(frozen=True)
class MethodInfo:
    func_name: str
    negative_freq: bool
    use_rate: bool


class SpectralMethods:
    _REGISTRY = {
        "fft": MethodInfo("fft", True, False),
        "fft_hanning": MethodInfo("fft_hanning", True, False),
        "rate_fft": MethodInfo("fft", True, True),
        "welch_segments": MethodInfo("welch_segments", False, False),
        "rate_welch_segments": MethodInfo("welch_segments", False, True),
        "fft_without_mean_substraction": MethodInfo(
            "fft_without_mean_substraction", True, False
        ),
        "welch": MethodInfo("welch", False, False),
        "fft_segments": MethodInfo("fft_segments", True, False),
    }

    def __init__(self, config: Config, methods: list[str] | None = None) -> None:
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

            signal = rate if reg.use_rate else spikes
            pyy, pxx, pxy = getattr(self, reg.func_name)(signal, stimulus)
            res = self.spectral_results[i].update(pxx, pyy, pxy, rate.sum(axis=0))
            spectral_result.append(res)

        self.spectral_results = spectral_result

    def norm(self) -> None:
        spectral_result: list[SpectralResults] = []
        for result in self.spectral_results:
            spectral_result.append(result.norm(self.config.trials))
        self.spectral_results = spectral_result

    def coherence_and_transfer(self) -> None:
        spectral_result: list[SpectralResults] = []
        for result in self.spectral_results:
            spectral_result.append(result.coherence_and_transfer())
        self.spectral_results = spectral_result

    def save(self, contrast: float | None = None) -> None:
        for result in self.spectral_results:
            result.save(self.config.savepath, contrast)

    def _segment_signal(self, spikes: jnp.ndarray, stimulus: jnp.ndarray):
        spikes = spikes[:, -self.config.nperseg :]
        stimulus = stimulus[:, -self.config.nperseg :]
        return spikes, stimulus

    def _segment_signal_trials(self, spikes: jnp.ndarray, stimulus: jnp.ndarray):
        segments = jnp.floor(stimulus.shape[1] / self.config.nperseg).astype(int)
        segments_index = int(segments * self.config.nperseg)
        stimulus = stimulus[:, :segments_index].reshape(
            -1, segments, self.config.nperseg
        )
        spikes = spikes[:, :segments_index].reshape(-1, segments, self.config.nperseg)
        return spikes, stimulus

    def _calc_fft(self, spikes, stimulus):
        scale = 1.0 / (self.config.fs * spikes.shape[0])
        fft_pxx = jnp.fft.fftshift(jnp.fft.fft(spikes))
        pxx = scale * jnp.abs(fft_pxx) ** 2

        fft_pyy = jnp.fft.fftshift(jnp.fft.fft(stimulus))
        pyy = scale * jnp.abs(fft_pyy) ** 2
        pxy = scale * fft_pxx * jnp.conj(fft_pyy)

        return pyy, pxx, pxy

    def _calc_fft_window(self, spikes, stimulus):
        win = jnp.hanning(spikes.shape[1])

        scale = 1.0 / (self.config.fs * (win * win).sum())
        fft_pxx = jnp.fft.fftshift(jnp.fft.fft(spikes * win))
        pxx = scale * jnp.abs(fft_pxx) ** 2

        fft_pyy = jnp.fft.fftshift(jnp.fft.fft(stimulus * win))
        pyy = scale * jnp.abs(fft_pyy) ** 2

        pxy = scale * fft_pxx * jnp.conj(fft_pyy)

        return pyy, pxx, pxy

    def welch_segments(self, spikes: jnp.ndarray, stimulus):
        spikes, stimulus = self._segment_signal(spikes, stimulus)
        f, pyy = jsp.signal.welch(
            stimulus,
            fs=self.config.fs,
            nperseg=self.config.nperseg,
            nfft=None,
            noverlap=0,
        )
        _, pxx = jsp.signal.welch(
            spikes - jnp.mean(spikes, axis=1, keepdims=True),
            fs=self.config.fs,
            nperseg=self.config.nperseg,
            detrend=False,
            nfft=None,
            noverlap=0,
        )
        _, pxy = jsp.signal.csd(
            spikes - jnp.mean(spikes, axis=1, keepdims=True),
            stimulus,
            fs=self.config.fs,
            nperseg=self.config.nperseg,
            detrend=False,
            nfft=None,
            noverlap=0,
        )

        return pyy.sum(axis=0), pxx.sum(axis=0), pxy.sum(axis=0)

    def fft(self, spikes: jnp.ndarray, stimulus: jnp.ndarray):
        spikes, stimulus = self._segment_signal(spikes, stimulus)
        pyy, pxx, pxy = self._calc_fft(
            spikes - jnp.mean(spikes, axis=1, keepdims=True), stimulus
        )

        return pyy.sum(axis=0), pxx.sum(axis=0), pxy.sum(axis=0)

    def fft_hanning(self, spikes: jnp.ndarray, stimulus: jnp.ndarray):
        spikes, stimulus = self._segment_signal(spikes, stimulus)
        pyy, pxx, pxy = self._calc_fft_window(
            spikes - jnp.mean(spikes, axis=1, keepdims=True), stimulus
        )
        return pyy.sum(axis=0), pxx.sum(axis=0), pxy.sum(axis=0)

    def fft_without_mean_substraction(self, spikes: jnp.ndarray, stimulus: jnp.ndarray):
        spikes, stimulus = self._segment_signal(spikes, stimulus)
        pyy, pxx, pxy = self._calc_fft(spikes, stimulus)

        return pyy.sum(axis=0), pxx.sum(axis=0), pxy.sum(axis=0)

    def welch(self, spikes: jnp.ndarray, stimulus):
        _, pyy = jsp.signal.welch(
            stimulus,
            fs=self.config.fs,
            nperseg=self.config.nperseg,
            noverlap=self.config.nperseg // 2,
            detrend=False,
        )
        _, pxx = jsp.signal.welch(
            spikes - jnp.mean(spikes),
            fs=self.config.fs,
            nperseg=self.config.nperseg,
            noverlap=self.config.nperseg // 2,
            detrend=False,
        )
        _, pxy = jsp.signal.csd(
            spikes - jnp.mean(spikes),
            stimulus,
            fs=self.config.fs,
            nperseg=self.config.nperseg,
            noverlap=self.config.nperseg // 2,
            detrend=False,
        )

        return pyy.sum(axis=0), pxx.sum(axis=0), pxy.sum(axis=0)

    def fft_segments(self, spikes: jnp.ndarray, stimulus: jnp.ndarray):
        spikes, stimulus = self._segment_signal_trials(spikes, stimulus)
        pyy, pxx, pxy = self._calc_fft(
            spikes - jnp.mean(spikes, axis=-1, keepdims=True), stimulus
        )
        return pyy.sum(axis=(0, 1)), pxx.sum(axis=(0, 1)), pxy.sum(axis=(0, 1))
