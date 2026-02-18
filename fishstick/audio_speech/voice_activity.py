"""
Voice Activity Detection (VAD)

Voice activity detection modules for detecting speech vs non-speech:
- Energy-based VAD
- Spectral entropy VAD
- Neural network-based VAD
- Hybrid VAD combining multiple methods
"""

from typing import Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VADMethod(Enum):
    """Voice activity detection methods."""

    ENERGY = "energy"
    SPECTRAL_ENTROPY = "spectral_entropy"
    NEURAL = "neural"
    HYBRID = "hybrid"


@dataclass
class VADConfig:
    """Configuration for voice activity detection."""

    sample_rate: int = 16000
    frame_length: int = 25
    frame_shift: int = 10
    energy_threshold: float = 0.5
    entropy_threshold: float = 0.5
    min_speech_duration: float = 0.1
    min_silence_duration: float = 0.2
    smoothing_window: int = 5
    use_neural: bool = True
    hidden_size: int = 128
    num_layers: int = 2


class EnergyVAD:
    """Energy-based voice activity detection.

    Detects speech based on signal energy level.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length: int = 25,
        frame_shift: int = 10,
        threshold: float = 0.5,
        noise_floor: float = 0.01,
    ):
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.threshold = threshold
        self.noise_floor = noise_floor

        self.win_length = int(sample_rate * frame_length / 1000)
        self.hop_length = int(sample_rate * frame_shift / 1000)

        self.noise_level = None

    def _estimate_noise_level(self, audio: torch.Tensor, num_frames: int = 10) -> float:
        """Estimate noise level from initial frames."""
        frames = self._extract_frames(audio)

        if len(frames) < num_frames:
            return 0.0

        energies = torch.sum(frames**2, dim=1)
        noise_energy = torch.median(energies[:num_frames])

        return noise_energy.item()

    def _extract_frames(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract frames from audio."""
        window = torch.hann_window(self.win_length).to(audio.device)

        frames = audio.unfold(0, self.win_length, self.hop_length)
        frames = frames * window

        return frames

    def __call__(
        self,
        audio: torch.Tensor,
        threshold: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Detect voice activity.

        Args:
            audio: Audio waveform (n_samples,) or (batch, n_samples)
            threshold: Optional threshold override

        Returns:
            Tuple of (frame_energy, vad_decision)
            - frame_energy: Energy per frame
            - vad_decision: Binary decision per frame (1=voice, 0=noise)
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        device = audio.device
        threshold = threshold or self.threshold

        frames = self._extract_frames(audio[0])

        energy = torch.sum(frames**2, dim=1)

        energy_norm = (energy - energy.min()) / (energy.max() - energy.min() + 1e-10)

        vad = (energy_norm > threshold).float()

        return energy, vad

    def set_threshold_adaptive(self, audio: torch.Tensor, factor: float = 1.5):
        """Set adaptive threshold based on audio content."""
        energy, _ = self(audio)

        threshold = torch.mean(energy) + factor * torch.std(energy)
        self.threshold = threshold.item() / energy.max()

        return self.threshold


class SpectralEntropyVAD:
    """Spectral entropy-based voice activity detection.

    Detects speech based on spectral entropy which is
    lower for speech than for noise.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length: int = 25,
        frame_shift: int = 10,
        n_fft: int = 512,
        threshold: float = 0.5,
    ):
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.n_fft = n_fft
        self.threshold = threshold

        self.win_length = int(sample_rate * frame_length / 1000)
        self.hop_length = int(sample_rate * frame_shift / 1000)

    def _compute_spectral_entropy(self, spec: torch.Tensor) -> torch.Tensor:
        """Compute spectral entropy."""
        spec_power = torch.abs(spec) ** 2

        prob = spec_power / (spec_power.sum(dim=0, keepdim=True) + 1e-10)

        entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=0)

        entropy_norm = entropy / torch.log(torch.tensor(spec.shape[0]).to(spec.device))

        return entropy_norm

    def __call__(
        self,
        audio: torch.Tensor,
        threshold: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Detect voice activity using spectral entropy.

        Args:
            audio: Audio waveform (n_samples,)
            threshold: Optional threshold override

        Returns:
            Tuple of (spectral_entropy, vad_decision)
        """
        if audio.dim() > 1:
            audio = audio.squeeze()

        device = audio.device
        threshold = threshold or self.threshold

        window = torch.hann_window(self.win_length).to(device)

        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )

        entropy = self._compute_spectral_entropy(spec)

        vad = (entropy > threshold).float()

        return entropy, vad


class NeuralVAD(nn.Module):
    """Neural network-based voice activity detection.

    Uses a bidirectional LSTM to classify frames as speech or non-speech.
    """

    def __init__(
        self,
        input_size: int = 40,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            features: Input features (batch, seq_len, input_size)

        Returns:
            VAD probabilities (batch, seq_len, 1)
        """
        lstm_out, _ = self.lstm(features)
        vad_prob = self.fc(lstm_out)

        return vad_prob.squeeze(-1)


class NeuralVADWrapper:
    """Wrapper for neural VAD with feature extraction."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length: int = 25,
        frame_shift: int = 10,
        n_mels: int = 40,
        hidden_size: int = 128,
        num_layers: int = 2,
    ):
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.n_mels = n_mels

        self.win_length = int(sample_rate * frame_length / 1000)
        self.hop_length = int(sample_rate * frame_shift / 1000)

        self.mel_spec = self._create_mel_transform()

        self.vad_model = NeuralVAD(
            input_size=n_mels,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vad_model.to(self.device)

    def _create_mel_transform(self):
        """Create mel spectrogram transform."""
        from fishstick.audio_speech.spectral_features import SpectralFeatures

        return SpectralFeatures(
            sample_rate=self.sample_rate,
            n_fft=self.win_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )

    def _extract_features(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract mel spectrogram features."""
        spec = self.mel_spec(audio.unsqueeze(0))

        if isinstance(spec, dict):
            spec = spec.get("mel_spectrogram", list(spec.values())[0])

        return spec.T.unsqueeze(0)

    def train_model(
        self,
        audio_clips: list,
        labels: list,
        epochs: int = 10,
        lr: float = 0.001,
    ):
        """Train the neural VAD model.

        Args:
            audio_clips: List of audio tensors
            labels: List of VAD labels (binary)
            epochs: Number of training epochs
            lr: Learning rate
        """
        self.vad_model.train()

        optimizer = torch.optim.Adam(self.vad_model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        for epoch in range(epochs):
            total_loss = 0

            for audio, label in zip(audio_clips, labels):
                features = self._extract_features(audio).to(self.device)
                label_tensor = torch.tensor([label], dtype=torch.float32).to(
                    self.device
                )

                optimizer.zero_grad()

                output = self.vad_model(features)

                loss = criterion(output, label_tensor)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(
                f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(audio_clips):.4f}"
            )

    def __call__(
        self,
        audio: torch.Tensor,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Detect voice activity.

        Args:
            audio: Audio waveform (n_samples,)
            threshold: Decision threshold

        Returns:
            Tuple of (vad_probabilities, vad_decision)
        """
        self.vad_model.eval()

        features = self._extract_features(audio).to(self.device)

        with torch.no_grad():
            vad_prob = self.vad_model(features)

        vad = (vad_prob > threshold).float()

        return vad_prob.squeeze(0).cpu(), vad.squeeze(0).cpu()


class HybridVAD:
    """Hybrid voice activity detection combining multiple methods."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length: int = 25,
        frame_shift: int = 10,
        energy_weight: float = 0.3,
        entropy_weight: float = 0.3,
        neural_weight: float = 0.4,
    ):
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.frame_shift = frame_shift

        self.energy_vad = EnergyVAD(
            sample_rate=sample_rate,
            frame_length=frame_length,
            frame_shift=frame_shift,
        )

        self.entropy_vad = SpectralEntropyVAD(
            sample_rate=sample_rate,
            frame_length=frame_length,
            frame_shift=frame_shift,
        )

        self.energy_weight = energy_weight
        self.entropy_weight = entropy_weight
        self.neural_weight = neural_weight

        self.neural_vad = None

    def set_neural_vad(self, neural_vad: NeuralVADWrapper):
        """Set neural VAD model."""
        self.neural_vad = neural_vad

    def _normalize_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """Normalize scores to [0, 1] range."""
        scores_min = scores.min()
        scores_max = scores.max()

        if scores_max - scores_min > 1e-10:
            return (scores - scores_min) / (scores_max - scores_min)
        return torch.zeros_like(scores)

    def __call__(
        self,
        audio: torch.Tensor,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Detect voice activity using hybrid method.

        Args:
            audio: Audio waveform (n_samples,)
            threshold: Decision threshold

        Returns:
            Tuple of (combined_scores, vad_decision)
        """
        energy, vad_energy = self.energy_vad(audio)
        entropy, vad_entropy = self.entropy_vad(audio)

        energy_norm = self._normalize_scores(energy)
        entropy_norm = self._normalize_scores(entropy)

        if self.neural_vad is not None:
            neural_prob, vad_neural = self.neural_vad(audio, threshold)

            neural_norm = self._normalize_scores(neural_prob)

            combined = (
                self.energy_weight * energy_norm
                + self.entropy_weight * entropy_norm
                + self.neural_weight * neural_norm
            )
        else:
            combined = (
                self.energy_weight * energy_norm + self.entropy_weight * entropy_norm
            )

            total_weight = self.energy_weight + self.entropy_weight
            combined = combined / total_weight

        vad = (combined > threshold).float()

        return combined, vad


class VADPostProcessor:
    """Post-processing for VAD output.

    Applies smoothing and removes short speech/silence segments.
    """

    def __init__(
        self,
        min_speech_duration: float = 0.1,
        min_silence_duration: float = 0.2,
        sample_rate: int = 16000,
        frame_shift: int = 10,
    ):
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration
        self.sample_rate = sample_rate
        self.frame_shift = frame_shift

        self.min_speech_frames = int(min_speech_duration * 1000 / frame_shift)
        self.min_silence_frames = int(min_silence_duration * 1000 / frame_shift)

    def _apply_smoothing(self, vad: torch.Tensor, window_size: int = 5) -> torch.Tensor:
        """Apply smoothing using moving average."""
        if window_size <= 1:
            return vad

        kernel = torch.ones(window_size) / window_size

        smoothed = torch.conv1d(
            vad.unsqueeze(0).unsqueeze(0).float(),
            kernel.unsqueeze(0).unsqueeze(0).float(),
            padding=window_size // 2,
        )

        return (smoothed.squeeze() > 0.5).float()

    def _remove_short_segments(self, vad: torch.Tensor) -> torch.Tensor:
        """Remove short speech or silence segments."""
        result = vad.clone()

        i = 0
        while i < len(vad):
            if vad[i] == 1:
                j = i
                while j < len(vad) and vad[j] == 1:
                    j += 1

                if j - i < self.min_speech_frames:
                    result[i:j] = 0

                i = j
            else:
                j = i
                while j < len(vad) and vad[j] == 0:
                    j += 1

                if j - i < self.min_silence_frames:
                    result[i:j] = 1

                i = j

        return result

    def __call__(self, vad: torch.Tensor, smooth: bool = True) -> torch.Tensor:
        """Apply post-processing to VAD output.

        Args:
            vad: Raw VAD decisions
            smooth: Whether to apply smoothing

        Returns:
            Processed VAD decisions
        """
        if smooth:
            vad = self._apply_smoothing(vad)

        vad = self._remove_short_segments(vad)

        return vad


def create_vad(
    method: VADMethod = VADMethod.HYBRID,
    sample_rate: int = 16000,
    **kwargs,
) -> Union[EnergyVAD, SpectralEntropyVAD, HybridVAD]:
    """Factory function to create VAD detector.

    Args:
        method: VAD method to use
        sample_rate: Audio sample rate
        **kwargs: Additional arguments for VAD

    Returns:
        VAD detector instance
    """
    frame_length = kwargs.get("frame_length", 25)
    frame_shift = kwargs.get("frame_shift", 10)

    if method == VADMethod.ENERGY:
        return EnergyVAD(
            sample_rate=sample_rate,
            frame_length=frame_length,
            frame_shift=frame_shift,
            threshold=kwargs.get("energy_threshold", 0.5),
        )

    elif method == VADMethod.SPECTRAL_ENTROPY:
        return SpectralEntropyVAD(
            sample_rate=sample_rate,
            frame_length=frame_length,
            frame_shift=frame_shift,
            threshold=kwargs.get("entropy_threshold", 0.5),
        )

    elif method == VADMethod.NEURAL:
        return NeuralVADWrapper(
            sample_rate=sample_rate,
            frame_length=frame_length,
            frame_shift=frame_shift,
            n_mels=kwargs.get("n_mels", 40),
            hidden_size=kwargs.get("hidden_size", 128),
            num_layers=kwargs.get("num_layers", 2),
        )

    elif method == VADMethod.HYBRID:
        hybrid = HybridVAD(
            sample_rate=sample_rate,
            frame_length=frame_length,
            frame_shift=frame_shift,
            energy_weight=kwargs.get("energy_weight", 0.3),
            entropy_weight=kwargs.get("entropy_weight", 0.3),
            neural_weight=kwargs.get("neural_weight", 0.4),
        )

        if kwargs.get("use_neural", True):
            neural = NeuralVADWrapper(
                sample_rate=sample_rate,
                frame_length=frame_length,
                frame_shift=frame_shift,
                n_mels=kwargs.get("n_mels", 40),
                hidden_size=kwargs.get("hidden_size", 128),
                num_layers=kwargs.get("num_layers", 2),
            )
            hybrid.set_neural_vad(neural)

        return hybrid

    return EnergyVAD(sample_rate=sample_rate)
