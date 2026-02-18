import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class STFT(nn.Module):
    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        window: str = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = center
        self.normalized = normalized
        self.onesided = onesided

        if window == "hann":
            self.register_buffer("window", torch.hann_window(win_length))
        elif window == "hamming":
            self.register_buffer("window", torch.hamming_window(win_length))
        elif window == "boxcar":
            self.register_buffer("window", torch.ones(win_length))
        else:
            raise ValueError(f"Unknown window type: {window}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        stft_result = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            normalized=self.normalized,
            onesided=self.onesided,
            return_complex=True,
        )
        mag = torch.abs(stft_result)
        phase = torch.angle(stft_result)
        return mag, phase

    def inverse(self, mag: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        complex_spec = mag * torch.exp(1j * phase)
        return torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            normalized=self.normalized,
            onesided=self.onesided,
        )


class MelFilterbank(nn.Module):
    def __init__(
        self,
        n_fft: int = 512,
        n_mels: int = 80,
        sample_rate: int = 16000,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else sample_rate / 2

        mel_points = torch.linspace(
            self._hz_to_mel(self.fmin),
            self._hz_to_mel(self.fmax),
            n_mels + 2,
        )
        hz_points = self._mel_to_hz(mel_points)
        bin_points = torch.floor((n_fft + 1) * hz_points / sample_rate).long()

        filters = torch.zeros(n_mels, n_fft // 2 + 1)
        for i in range(n_mels):
            left = bin_points[i].item()
            center = bin_points[i + 1].item()
            right = bin_points[i + 2].item()

            for j in range(left, center):
                filters[i, j] = (j - left) / (center - left)
            for j in range(center, right):
                filters[i, j] = (right - j) / (right - center)

        self.register_buffer("filters", filters)

    @staticmethod
    def _hz_to_mel(hz: float) -> float:
        return 2595 * math.log10(1 + hz / 700)

    @staticmethod
    def _mel_to_hz(mel: float) -> float:
        return 700 * (10 ** (mel / 2595) - 1)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        return F.linear(spec, self.filters)


class MelSpectrogram(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        n_mels: int = 80,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        power: float = 2.0,
    ):
        super().__init__()
        self.stft = STFT(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            center=True,
        )
        self.mel_filterbank = MelFilterbank(
            n_fft=n_fft,
            n_mels=n_mels,
            sample_rate=sample_rate,
            fmin=fmin,
            fmax=fmax,
        )
        self.power = power

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mag, _ = self.stft(x)
        mel_spec = self.mel_filterbank(mag)
        if self.power == 2.0:
            mel_spec = mel_spec**2
        elif self.power != 1.0:
            mel_spec = torch.pow(mel_spec, self.power)
        return mel_spec


class MFCC(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        n_mels: int = 80,
        n_mfcc: int = 13,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        dct_type: int = 2,
        norm: str = "ortho",
    ):
        super().__init__()
        self.mel_spec = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            power=1.0,
        )
        self.n_mfcc = n_mfcc
        self.register_buffer(
            "dct_matrix", self._create_dct_matrix(n_mels, n_mfcc, dct_type, norm)
        )

    def _create_dct_matrix(
        self, n_input: int, n_output: int, dct_type: int, norm: str
    ) -> torch.Tensor:
        dct_mat = torch.zeros(n_input, n_output)
        for k in range(n_output):
            dct_mat[:, k] = torch.cos(
                math.pi * (k + 1) * torch.arange(n_input) / n_input
            )
        if norm == "ortho":
            dct_mat *= math.sqrt(2.0 / n_input)
            dct_mat[0, :] /= math.sqrt(2)
        return dct_mat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mel_spec = self.mel_spec(x)
        log_mel = torch.log(torch.clamp(mel_spec, min=1e-10))
        mfcc = F.linear(log_mel, self.dct_matrix.T)
        return mfcc


class SpectrogramExtractor(nn.Module):
    def __init__(
        self,
        spec_type: str = "mel",
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        n_mels: int = 80,
        n_mfcc: int = 13,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
    ):
        super().__init__()
        self.spec_type = spec_type
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        if spec_type == "stft":
            self.spec_layer = STFT(n_fft, hop_length, win_length)
        elif spec_type == "mel":
            self.spec_layer = MelSpectrogram(
                sample_rate, n_fft, hop_length, win_length, n_mels, fmin, fmax
            )
        elif spec_type == "mfcc":
            self.spec_layer = MFCC(
                sample_rate,
                n_fft,
                hop_length,
                win_length,
                n_mels,
                n_mfcc,
                fmin,
                fmax,
            )
        else:
            raise ValueError(f"Unknown spec_type: {spec_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spec_layer(x)


class InverseSpectrogram(nn.Module):
    def __init__(
        self,
        spec_type: str = "mel",
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        n_mels: int = 80,
    ):
        super().__init__()
        self.spec_type = spec_type
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        if spec_type == "stft" or spec_type == "mel":
            self.stft = STFT(n_fft, hop_length, win_length)
            if spec_type == "mel":
                self.mel_filterbank = MelFilterbank(n_fft, n_mels, sample_rate)
        else:
            raise ValueError(f"Inverse spectrogram not supported for {spec_type}")

    def forward(
        self, spec: torch.Tensor, phase: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.spec_type == "mel":
            mel_to_linear = torch.pinverse(self.mel_filterbank.filters.float())
            linear_spec = F.linear(spec, mel_to_linear)
            linear_spec = torch.clamp(linear_spec, min=1e-10)
            if phase is None:
                phase = torch.zeros_like(linear_spec)
            audio = self.stft.inverse(torch.sqrt(linear_spec), phase)
        else:
            if phase is None:
                phase = torch.zeros_like(spec)
            audio = self.stft.inverse(spec, phase)
        return audio


__all__ = [
    "STFT",
    "MelFilterbank",
    "MelSpectrogram",
    "MFCC",
    "SpectrogramExtractor",
    "InverseSpectrogram",
]
