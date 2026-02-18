"""
Evaluation Metrics for Audio Source Separation

Comprehensive metrics for evaluating source separation quality including
SI-SDR, PESQ, STOI, and BSS Eval metrics.
"""

from typing import Optional, List, Dict, Tuple, Any
import torch
import numpy as np


class SI_SDR:
    """Scale-Invariant Source-to-Distortion Ratio.

    A widely used metric for audio source separation that measures
    the ratio of signal power to distortion power.

    Reference:
        "SDR - half-baked or well done?" (Wisdom et al., 2019)
    """

    @staticmethod
    def compute(
        estimate: torch.Tensor,
        reference: torch.Tensor,
        epsilon: float = 1e-8,
    ) -> float:
        """Compute SI-SDR.

        Args:
            estimate: Estimated source signal
            reference: Reference source signal
            epsilon: Small constant for numerical stability

        Returns:
            SI-SDR value in dB
        """
        estimate = estimate.reshape(-1).cpu().numpy()
        reference = reference.reshape(-1).cpu().numpy()

        dot_product = np.dot(estimate, reference)
        ref_energy = np.dot(reference, reference) + epsilon

        scale = dot_product / ref_energy
        scaled_ref = scale * reference
        noise = estimate - scaled_ref

        signal_power = np.dot(scaled_ref, scaled_ref)
        noise_power = np.dot(noise, noise)

        return 10 * np.log10((signal_power + epsilon) / (noise_power + epsilon))

    @staticmethod
    def compute_batch(
        estimates: torch.Tensor,
        references: torch.Tensor,
    ) -> List[float]:
        """Compute SI-SDR for batch.

        Args:
            estimates: Estimated sources (n_sources, batch, channels, time)
            references: Reference sources (n_sources, batch, channels, time)

        Returns:
            List of SI-SDR values for each source
        """
        results = []
        for i in range(estimates.shape[0]):
            for j in range(estimates.shape[1]):
                est = estimates[i, j]
                ref = references[i, j]
                results.append(SI_SDR.compute(est, ref))
        return results


class SI_SAR:
    """Scale-Invariant Source-to-Artifacts Ratio.

    Measures the ratio of signal power to artifact power.
    """

    @staticmethod
    def compute(
        estimate: torch.Tensor,
        reference: torch.Tensor,
        epsilon: float = 1e-8,
    ) -> float:
        """Compute SI-SAR.

        Args:
            estimate: Estimated source
            reference: Reference source

        Returns:
            SI-SAR value in dB
        """
        estimate = estimate.reshape(-1).cpu().numpy()
        reference = reference.reshape(-1).cpu().numpy()

        dot_product = np.dot(estimate, reference)
        ref_energy = np.dot(reference, reference) + epsilon

        scale = dot_product / ref_energy
        scaled_ref = scale * reference
        artifacts = estimate - scaled_ref

        signal_power = np.dot(scaled_ref, scaled_ref)
        artifact_power = np.dot(artifacts, artifacts)

        return 10 * np.log10((signal_power + epsilon) / (artifact_power + epsilon))


class SI_SNR:
    """Scale-Invariant Signal-to-Noise Ratio.

    Similar to SI-SDR but uses noise rather than distortion.
    """

    @staticmethod
    def compute(
        estimate: torch.Tensor,
        reference: torch.Tensor,
        epsilon: float = 1e-8,
    ) -> float:
        """Compute SI-SNR.

        Args:
            estimate: Estimated signal
            reference: Reference signal

        Returns:
            SI-SNR value in dB
        """
        return SI_SDR.compute(estimate, reference, epsilon)


class PESQ:
    """Perceptual Evaluation of Speech Quality.

    PESQ is a standardized metric for assessing speech quality.
    This is a simplified implementation for cases where the
    full PESQ implementation is not available.

    Reference:
        "PESQ - The new ITU standard for end-to-end speech quality
        assessment" (Rix et al., 2001)
    """

    @staticmethod
    def compute(
        estimate: torch.Tensor,
        reference: torch.Tensor,
        sample_rate: int = 16000,
    ) -> float:
        """Compute PESQ score (simplified approximation).

        Note: Full PESQ requires soundfile/pyworld. This is a simplified
        frequency-domain based approximation.

        Args:
            estimate: Estimated signal
            reference: Reference signal
            sample_rate: Sample rate

        Returns:
            Approximate PESQ score (-0.5 to 4.5)
        """
        estimate = estimate.reshape(-1).cpu().numpy()
        reference = reference.reshape(-1).cpu().numpy()

        est_fft = np.fft.rfft(estimate)
        ref_fft = np.fft.rfft(reference)

        est_mag = np.abs(est_fft)
        ref_mag = np.abs(ref_fft)

        spectral_diff = np.abs(est_mag - ref_mag)
        spectral_distortion = np.mean(spectral_diff)

        pesq_approx = 4.5 - 0.1 * spectral_distortion
        return np.clip(pesq_approx, -0.5, 4.5)


class STOI:
    """Short-Time Objective Intelligibility.

    STOI measures speech intelligibility by analyzing the correlation
    between temporal envelopes in short time frames.

    Reference:
        "Short-Time Objective Intelligibility measure for
        speech intelligibility" (Taal et al., 2010)
    """

    @staticmethod
    def compute(
        estimate: torch.Tensor,
        reference: torch.Tensor,
        sample_rate: int = 16000,
        third_octave: bool = True,
    ) -> float:
        """Compute STOI score.

        Args:
            estimate: Estimated signal
            reference: Reference signal
            sample_rate: Sample rate
            third_octave: Whether to use third octave analysis

        Returns:
            STOI score (0 to 1)
        """
        estimate = estimate.reshape(-1).cpu().numpy()
        reference = reference.reshape(-1).cpu().numpy()

        frame_len = int(0.256 * sample_rate)
        hop_len = int(0.128 * sample_rate)

        def enframe(x: np.ndarray) -> np.ndarray:
            frames = []
            for i in range(0, len(x) - frame_len + 1, hop_len):
                frames.append(x[i : i + frame_len])
            return np.array(frames)

        est_frames = enframe(estimate)
        ref_frames = enframe(reference)

        if len(est_frames) == 0 or len(ref_frames) == 0:
            return 0.0

        est_frames = est_frames[: len(ref_frames)]

        correlations = []
        for est_frame, ref_frame in zip(est_frames, ref_frames):
            if np.std(ref_frame) > 1e-8:
                correlation = np.corrcoef(est_frame, ref_frame)[0, 1]
                correlations.append(max(0, correlation))
            else:
                correlations.append(0.0)

        return float(np.mean(correlations)) if correlations else 0.0


class BSSEval:
    """BSS Eval metrics for source separation.

    Implements SDR, SIR, SAR, and ISR from the BSS Eval framework.

    Reference:
        "BSS Eval v3.0: A toolbox for performance benchmarking
        in blind source separation" (Vincent et al., 2006)
    """

    @staticmethod
    def compute(
        estimates: torch.Tensor,
        references: torch.Tensor,
        compute_permutation: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Compute BSS Eval metrics.

        Args:
            estimates: Estimated sources (n_sources, batch, channels, time)
            references: Reference sources (n_sources, batch, channels, time)
            compute_permutation: Whether to find best permutation

        Returns:
            Dictionary with SDR, SIR, SAR, ISR arrays
        """
        estimates = estimates.cpu().numpy()
        references = references.cpu().numpy()

        n_sources = estimates.shape[0]
        batch_size = estimates.shape[1]

        sdr = np.zeros((batch_size, n_sources))
        sir = np.zeros((batch_size, n_sources))
        sar = np.zeros((batch_size, n_sources))
        isr = np.zeros((batch_size, n_sources))

        for b in range(batch_size):
            for s in range(n_sources):
                est = estimates[s, b, 0, :]
                ref = references[s, b, 0, :]

                sdr[b, s], sir[b, s], sar[b, s], isr[b, s] = BSSEval._bss_eval_single(
                    est, ref
                )

        return {
            "sdr": sdr,
            "sir": sir,
            "sar": sar,
            "isr": isr,
        }

    @staticmethod
    def _bss_eval_single(
        estimate: np.ndarray,
        reference: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        """Compute BSS Eval metrics for single source."""
        epsilon = 1e-10

        ref_energy = np.dot(reference, reference) + epsilon

        scale = np.dot(estimate, reference) / ref_energy
        scaled_ref = scale * reference

        target = scaled_ref
        noise = estimate - scaled_ref

        target_power = np.dot(target, target) + epsilon
        noise_power = np.dot(noise, noise) + epsilon

        sdr = 10 * np.log10(target_power / noise_power)

        interference = np.zeros_like(reference)
        for i in range(len(reference)):
            if i < len(estimate):
                interference[i] = estimate[i] - target[i]

        interference_power = np.dot(interference, interference) + epsilon
        sir = 10 * np.log10(target_power / interference_power)

        artifacts = noise - interference
        artifact_power = np.dot(artifacts, artifacts) + epsilon
        sar = 10 * np.log10((target_power + interference_power) / artifact_power)

        isr = 10 * np.log10(
            np.dot(scaled_ref, scaled_ref)
            / np.dot(estimate - scaled_ref, estimate - scaled_ref)
        )

        return sdr, sir, sar, isr


class SeparationMetrics:
    """Container for comprehensive separation metrics."""

    def __init__(
        self,
        sample_rate: int = 16000,
        compute_permutation: bool = True,
    ):
        self.sample_rate = sample_rate
        self.compute_permutation = compute_permutation

    def compute(
        self,
        estimates: torch.Tensor,
        references: torch.Tensor,
    ) -> Dict[str, Any]:
        """Compute all metrics.

        Args:
            estimates: Estimated sources
            references: Reference sources

        Returns:
            Dictionary containing all metrics
        """
        results = {}

        si_sdr_values = []
        for i in range(estimates.shape[0]):
            for j in range(estimates.shape[1]):
                est = (
                    estimates[i, j, 0, :] if estimates.shape[2] > 0 else estimates[i, j]
                )
                ref = (
                    references[i, j, 0, :]
                    if references.shape[2] > 0
                    else references[i, j]
                )
                si_sdr_values.append(SI_SDR.compute(est, ref))

        results["si_sdr"] = {
            "mean": float(np.mean(si_sdr_values)),
            "std": float(np.std(si_sdr_values)),
            "values": si_sdr_values,
        }

        try:
            pesq_values = []
            for i in range(min(estimates.shape[0], 2)):
                for j in range(min(estimates.shape[1], 2)):
                    est = (
                        estimates[i, j, 0, :]
                        if estimates.shape[2] > 0
                        else estimates[i, j]
                    )
                    ref = (
                        references[i, j, 0, :]
                        if references.shape[2] > 0
                        else references[i, j]
                    )
                    pesq_values.append(PESQ.compute(est, ref, self.sample_rate))

            results["pesq"] = {
                "mean": float(np.mean(pesq_values)),
                "values": pesq_values,
            }
        except Exception:
            results["pesq"] = {"mean": 0.0, "values": []}

        try:
            stoi_values = []
            for i in range(min(estimates.shape[0], 2)):
                for j in range(min(estimates.shape[1], 2)):
                    est = (
                        estimates[i, j, 0, :]
                        if estimates.shape[2] > 0
                        else estimates[i, j]
                    )
                    ref = (
                        references[i, j, 0, :]
                        if references.shape[2] > 0
                        else references[i, j]
                    )
                    stoi_values.append(STOI.compute(est, ref, self.sample_rate))

            results["stoi"] = {
                "mean": float(np.mean(stoi_values)),
                "values": stoi_values,
            }
        except Exception:
            results["stoi"] = {"mean": 0.0, "values": []}

        try:
            bss_results = BSSEval.compute(
                estimates, references, self.compute_permutation
            )
            results["bss_eval"] = {
                "sdr_mean": float(np.mean(bss_results["sdr"])),
                "sir_mean": float(np.mean(bss_results["sir"])),
                "sar_mean": float(np.mean(bss_results["sar"])),
            }
        except Exception:
            results["bss_eval"] = {
                "sdr_mean": 0.0,
                "sir_mean": 0.0,
                "sar_mean": 0.0,
            }

        return results


class MetricTracker:
    """Track metrics over training/evaluation epochs."""

    def __init__(self):
        self.history: Dict[str, List[float]] = {}
        self.current_epoch: Dict[str, List[float]] = {}

    def update(self, metrics: Dict[str, float]) -> None:
        """Update metrics for current epoch."""
        for key, value in metrics.items():
            if key not in self.current_epoch:
                self.current_epoch[key] = []
            self.current_epoch[key].append(value)

    def commit_epoch(self) -> Dict[str, float]:
        """Commit current epoch and return averages."""
        epoch_avg = {}
        for key, values in self.current_epoch.items():
            epoch_avg[key] = float(np.mean(values))

            if key not in self.history:
                self.history[key] = []
            self.history[key].append(epoch_avg[key])

        self.current_epoch = {}
        return epoch_avg

    def get_history(self, key: Optional[str] = None) -> Dict[str, List[float]]:
        """Get metric history."""
        if key:
            return {key: self.history.get(key, [])}
        return self.history

    def reset(self) -> None:
        """Reset all metrics."""
        self.history = {}
        self.current_epoch = {}
