"""
Speech Feature Extraction

Speech-specific feature extraction for ASR and speech analysis:
- Formant tracking
- Pitch extraction (YIN, autocorrelation)
- Energy-based features
- RASTA filtering
- Speech/nonspeech classification features
"""

from typing import Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np
from scipy import signal


@dataclass
class SpeechFeatureConfig:
    """Configuration for speech feature extraction."""

    sample_rate: int = 16000
    frame_length: int = 25
    frame_shift: int = 10
    n_fft: int = 512
    n_mels: int = 40
    n_filts: int = 26
    n_formants: int = 5
    min_freq: float = 50.0
    max_freq: float = 8000.0
    use_rasta: bool = True


class PitchExtractor:
    """Extract pitch (F0) from speech signals.

    Implements multiple pitch extraction methods:
    - Autocorrelation
    - YIN algorithm
    - Cross-correlation
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length: int = 25,
        frame_shift: int = 10,
        min_freq: float = 50.0,
        max_freq: float = 500.0,
        method: str = "autocorrelation",
    ):
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.method = method

        self.win_length = int(sample_rate * frame_length / 1000)
        self.hop_length = int(sample_rate * frame_shift / 1000)

        self.min_lag = int(sample_rate / max_freq)
        self.max_lag = int(sample_rate / min_freq)

    def _autocorrelation(self, frame: torch.Tensor) -> float:
        """Compute pitch using autocorrelation."""
        frame = frame - frame.mean()

        acf = torch.correlate(frame, frame, mode="full")
        acf = acf[len(acf) // 2 :]

        peak_idx = self.min_lag + torch.argmax(acf[self.min_lag : self.max_lag])

        if peak_idx >= len(acf):
            return 0.0

        return self.sample_rate / peak_idx.item() if peak_idx > 0 else 0.0

    def _yin(self, frame: torch.Tensor) -> float:
        """Compute pitch using YIN algorithm."""
        frame = frame - frame.mean()

        tau_max = self.max_lag

        diff = torch.zeros(tau_max).to(frame.device)
        for tau in range(1, tau_max):
            diff[tau] = torch.sum((frame[tau:] - frame[:-tau]) ** 2)

        diff[0] = 1.0

        running_sum = torch.zeros(tau_max).to(frame.device)
        running_sum[0] = diff[0]
        for tau in range(1, tau_max):
            running_sum[tau] = running_sum[tau - 1] + diff[tau]

        cmnd = torch.zeros(tau_max).to(frame.device)
        for tau in range(2, tau_max):
            cmnd[tau] = diff[tau] * tau / running_sum[tau]

        tau_estimate = self.min_lag + torch.argmin(cmnd[self.min_lag : self.max_lag])

        if tau_estimate >= len(cmnd):
            return 0.0

        better_tau = tau_estimate
        if better_tau < len(cmnd) - 1:
            x0 = cmnd[better_tau - 1]
            x1 = cmnd[better_tau]
            x2 = cmnd[better_tau + 1]
            if x1 != 0:
                better_tau = tau_estimate.float() + (x2 - x0) / (2 * (2 * x1 - x2 - x0))

        if better_tau > 0:
            return self.sample_rate / better_tau
        return 0.0

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract pitch contour.

        Args:
            audio: Audio waveform (n_samples,)

        Returns:
            Pitch contour (n_frames,)
        """
        if audio.dim() > 1:
            audio = audio.squeeze()

        window = torch.hann_window(self.win_length).to(audio.device)

        frames = audio.unfold(0, self.win_length, self.hop_length)
        n_frames = frames.shape[0]

        pitches = torch.zeros(n_frames).to(audio.device)

        for i in range(n_frames):
            frame = frames[i] * window

            if self.method == "autocorrelation":
                pitches[i] = self._autocorrelation(frame)
            elif self.method == "yin":
                pitches[i] = self._yin(frame)
            else:
                pitches[i] = self._autocorrelation(frame)

        return pitches


class FormantExtractor:
    """Extract formants (F1-F5) from speech signals.

    Uses linear predictive coding (LPC) to estimate formant frequencies.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length: int = 25,
        frame_shift: int = 10,
        n_formants: int = 5,
        lpc_order: int = 16,
    ):
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.n_formants = n_formants
        self.lpc_order = lpc_order

        self.win_length = int(sample_rate * frame_length / 1000)
        self.hop_length = int(sample_rate * frame_shift / 1000)

    def _lpc(self, frame: torch.Tensor, order: int) -> torch.Tensor:
        """Compute LPC coefficients using Levinson-Durbin."""
        n = len(frame)

        r = torch.zeros(order + 1).to(frame.device)
        for i in range(order + 1):
            r[i] = torch.sum(frame[i:] * frame[: n - i])

        a = torch.zeros(order + 1).to(frame.device)
        a[0] = 1.0

        e = torch.zeros(order + 1).to(frame.device)
        e[0] = r[0]

        k = torch.zeros(order).to(frame.device)

        for i in range(1, order + 1):
            sigma = 0.0
            for j in range(1, i):
                sigma += a[j] * r[i - j]

            k[i - 1] = (r[i] - sigma) / (e[i - 1] + 1e-10)

            a_new = a.clone()
            for j in range(1, i):
                a[j] = a[j] - k[i - 1] * a_new[i - j]

            a[i] = k[i - 1]
            e[i] = (1 - k[i - 1] ** 2) * e[i - 1]

        return a

    def _find_peaks(
        self, power_spec: torch.Tensor, freqs: torch.Tensor, n_peaks: int
    ) -> list:
        """Find peaks in the power spectrum."""
        peaks = []

        for i in range(1, len(power_spec) - 1):
            if power_spec[i] > power_spec[i - 1] and power_spec[i] > power_spec[i + 1]:
                peaks.append((freqs[i].item(), power_spec[i].item()))

        peaks.sort(key=lambda x: x[1], reverse=True)
        return peaks[:n_peaks]

    def __call__(self, audio: torch.Tensor) -> dict:
        """Extract formant frequencies.

        Args:
            audio: Audio waveform (n_samples,)

        Returns:
            Dictionary with formant trajectories
        """
        if audio.dim() > 1:
            audio = audio.squeeze()

        window = torch.hann_window(self.win_length).to(audio.device)

        frames = audio.unfold(0, self.win_length, self.hop_length)
        n_frames = frames.shape[0]

        formants = {f"F{i}": [] for i in range(1, self.n_formants + 1)}
        formants["bandwidth"] = []

        for i in range(n_frames):
            frame = frames[i] * window

            lpc_coeffs = self._lpc(frame, self.lpc_order)

            roots = torch.roots(lpc_coeffs)
            roots = roots[torch.imag(roots) >= 0]

            angles = torch.atan2(torch.imag(roots), torch.real(roots))
            freqs = angles * self.sample_rate / (2 * torch.pi)

            power_spec = 1.0 / (
                torch.abs(1 - torch.exp(-1j * angles)[:, None]) ** 2 + 1e-10
            )

            peaks = self._find_peaks(power_spec, freqs, self.n_formants)

            for j, (freq, _) in enumerate(peaks):
                if j < self.n_formants:
                    formants[f"F{j + 1}"].append(freq)

            for j in range(len(peaks), self.n_formants):
                formants[f"F{j + 1}"].append(0.0)

            formants["bandwidth"].append(0.0)

        for key in formants:
            formants[key] = torch.tensor(formants[key])

        return formants


class EnergyExtractor:
    """Extract energy features from speech signals."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length: int = 25,
        frame_shift: int = 10,
    ):
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.frame_shift = frame_shift

        self.win_length = int(sample_rate * frame_length / 1000)
        self.hop_length = int(sample_rate * frame_shift / 1000)

    def __call__(self, audio: torch.Tensor) -> dict:
        """Extract energy features.

        Args:
            audio: Audio waveform (n_samples,)

        Returns:
            Dictionary with energy features
        """
        if audio.dim() > 1:
            audio = audio.squeeze()

        window = torch.hann_window(self.win_length).to(audio.device)

        frames = audio.unfold(0, self.win_length, self.hop_length)
        frames = frames * window

        energy = torch.sum(frames**2, dim=1)

        log_energy = torch.log(energy + 1e-10)

        delta_energy = torch.zeros_like(energy)
        delta_energy[1:-1] = (energy[2:] - energy[:-2]) / 2

        return {
            "energy": energy,
            "log_energy": log_energy,
            "delta_energy": delta_energy,
        }


class RASTAFilter:
    """RASTA (RelAtive SpecTrAl) filtering.

    Applies RASTA processing to PLP or mel features to
    remove slowly varying channel effects.
    """

    def __init__(
        self,
        filter_order: int = 5,
        sampling_rate: int = 100,
    ):
        self.filter_order = filter_order
        self.sampling_rate = sampling_rate

        self._create_filter()

    def _create_filter(self):
        """Create RASTA filter coefficients."""
        num = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
        denom = np.array([1.0, -0.94])

        self.num = num / num.sum()
        self.denom = denom

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """Apply RASTA filtering.

        Args:
            features: Input features (n_features, n_frames)

        Returns:
            RASTA-filtered features (n_features, n_frames)
        """
        device = features.device
        num = torch.tensor(self.num, dtype=torch.float32).to(device)
        denom = torch.tensor(self.denom, dtype=torch.float32).to(device)

        filtered = torch.zeros_like(features)

        for i in range(features.shape[0]):
            feat = features[i].cpu().numpy()
            filtered[i] = torch.tensor(
                signal.lfilter(self.num, self.denom, feat),
                dtype=torch.float32,
            ).to(device)

        return filtered


class SpeechFeatureExtractor:
    """Complete speech feature extraction pipeline.

    Combines multiple speech analysis features:
    - MFCC
    - Pitch
    - Formants
    - Energy
    - Delta and delta-delta features
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length: int = 25,
        frame_shift: int = 10,
        n_mfcc: int = 13,
        n_mels: int = 40,
        use_delta: bool = True,
        use_delta_delta: bool = True,
        use_rasta: bool = True,
    ):
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.use_delta = use_delta
        self.use_delta_delta = use_delta_delta
        self.use_rasta = use_rasta

        self.win_length = int(sample_rate * frame_length / 1000)
        self.hop_length = int(sample_rate * frame_shift / 1000)

        self._create_mel_filterbank()

        self.pitch_extractor = PitchExtractor(
            sample_rate=sample_rate,
            frame_length=frame_length,
            frame_shift=frame_shift,
        )

        self.formant_extractor = FormantExtractor(
            sample_rate=sample_rate,
            frame_length=frame_length,
            frame_shift=frame_shift,
        )

        self.energy_extractor = EnergyExtractor(
            sample_rate=sample_rate,
            frame_length=frame_length,
            frame_shift=frame_shift,
        )

        self.rasa_filter = RASTAFilter() if use_rasta else None

    def _create_mel_filterbank(self):
        """Create mel filterbank."""
        n_freqs = self.win_length // 2 + 1

        fmin_mel = 0
        fmax_mel = 2595 * np.log10(1 + self.sample_rate / 2 / 700)

        mel_points = np.linspace(fmin_mel, fmax_mel, self.n_mels + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)

        bin_points = np.floor(
            (self.win_length + 1) * hz_points / self.sample_rate
        ).astype(int)

        filterbank = np.zeros((self.n_mels, n_freqs))
        for i in range(self.n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]

            for j in range(left, center):
                filterbank[i, j] = (j - left) / (center - left)
            for j in range(center, right):
                filterbank[i, j] = (right - j) / (right - center)

        self.mel_filterbank = torch.tensor(filterbank, dtype=torch.float32)

    def _compute_mfcc(self, spec: torch.Tensor) -> torch.Tensor:
        """Compute MFCC from mel spectrogram."""
        log_spec = torch.log(spec + 1e-10)

        n_inputs = log_spec.shape[0]
        dct_basis = torch.zeros(self.n_mfcc, n_inputs)
        for i in range(self.n_mfcc):
            dct_basis[i] = torch.cos(
                torch.pi
                * i
                * (torch.arange(n_inputs, device=spec.device) + 0.5)
                / n_inputs
            )

        mfcc = torch.matmul(dct_basis, log_spec)

        return mfcc

    def _compute_deltas(self, features: torch.Tensor) -> torch.Tensor:
        """Compute delta features."""
        n_frames = features.shape[1]

        deltas = torch.zeros_like(features)

        for i in range(n_frames):
            if i == 0:
                deltas[:, i] = features[:, 1] - features[:, 0]
            elif i == n_frames - 1:
                deltas[:, i] = features[:, -1] - features[:, -2]
            else:
                deltas[:, i] = (features[:, i + 1] - features[:, i - 1]) / 2

        return deltas

    def __call__(self, audio: torch.Tensor) -> dict:
        """Extract all speech features.

        Args:
            audio: Audio waveform (n_samples,)

        Returns:
            Dictionary containing all speech features
        """
        if audio.dim() > 1:
            audio = audio.squeeze()

        features = {}

        window = torch.hann_window(self.win_length).to(audio.device)

        spec = torch.stft(
            audio,
            n_fft=self.win_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )
        spec = torch.abs(spec) ** 2

        mel_spec = torch.matmul(self.mel_filterbank.to(audio.device), spec)
        mel_spec = torch.log(mel_spec + 1e-10)

        features["mel_spectrogram"] = mel_spec

        mfcc = self._compute_mfcc(mel_spec)
        features["mfcc"] = mfcc

        if self.use_delta:
            delta_mfcc = self._compute_deltas(mfcc)
            features["delta_mfcc"] = delta_mfcc

            if self.use_delta_delta:
                delta_delta_mfcc = self._compute_deltas(delta_mfcc)
                features["delta_delta_mfcc"] = delta_delta_mfcc

        pitch = self.pitch_extractor(audio)
        features["pitch"] = pitch

        formants = self.formant_extractor(audio)
        features["formants"] = formants

        energy = self.energy_extractor(audio)
        features["energy"] = energy

        return features
