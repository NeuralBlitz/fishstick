"""
Advanced Filter Design

Design and implementation of various digital filters for signal
processing including FIR, IIR, and learnable filter banks.
"""

from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FIRFilterDesign(nn.Module):
    """FIR (Finite Impulse Response) filter design.

    Designs FIR filters using windowing methods.
    """

    SUPPORTED_WINDOWS = ["hann", "hamming", "blackman", "kaiser", "rectangular"]

    def __init__(
        self,
        num_taps: int = 64,
        cutoff_freq: float = 0.5,
        window: str = "hann",
        filter_type: str = "lowpass",
    ):
        super().__init__()
        self.num_taps = num_taps
        self.cutoff_freq = cutoff_freq
        self.window = window
        self.filter_type = filter_type

        self.coeffs = self._design_filter()

    def _get_window(self) -> torch.Tensor:
        """Generate window function."""
        n = torch.arange(self.num_taps)

        if self.window == "hann":
            window = 0.5 * (1 - torch.cos(2 * np.pi * n / (self.num_taps - 1)))
        elif self.window == "hamming":
            window = 0.54 - 0.46 * torch.cos(2 * np.pi * n / (self.num_taps - 1))
        elif self.window == "blackman":
            window = (
                0.42
                - 0.5 * torch.cos(2 * np.pi * n / (self.num_taps - 1))
                + 0.08 * torch.cos(4 * np.pi * n / (self.num_taps - 1))
            )
        elif self.window == "kaiser":
            beta = 8.0
            window = torch.i0(
                beta * torch.sqrt(1 - (2 * n / (self.num_taps - 1) - 1) ** 2)
            )
            window = window / window.max()
        else:
            window = torch.ones(self.num_taps)

        return window

    def _design_filter(self) -> nn.Parameter:
        """Design FIR filter coefficients."""
        n = torch.arange(self.num_taps)
        n_center = (self.num_taps - 1) / 2

        window = self._get_window()

        if self.filter_type == "lowpass":
            coefficients = torch.sin(2 * np.pi * self.cutoff_freq * (n - n_center)) / (
                np.pi * (n - n_center + 1e-8)
            )
        elif self.filter_type == "highpass":
            coefficients = -torch.sin(2 * np.pi * self.cutoff_freq * (n - n_center)) / (
                np.pi * (n - n_center + 1e-8)
            )
            coefficients[n_center.to(int)] = 1 - 2 * self.cutoff_freq
        elif self.filter_type == "bandpass":
            fc1, fc2 = self.cutoff_freq * 0.5, self.cutoff_freq * 1.5
            coefficients = torch.sin(2 * np.pi * fc2 * (n - n_center)) / (
                np.pi * (n - n_center + 1e-8)
            ) - torch.sin(2 * np.pi * fc1 * (n - n_center)) / (
                np.pi * (n - n_center + 1e-8)
            )
        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}")

        coefficients[n_center.to(int)] = (
            2 * self.cutoff_freq
            if self.filter_type == "lowpass"
            else 1 - 2 * self.cutoff_freq
        )

        coefficients = coefficients * window

        return nn.Parameter(coefficients)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply FIR filter.

        Args:
            x: Input signal

        Returns:
            Filtered signal
        """
        return F.conv1d(
            x.unsqueeze(0),
            self.coeffs.unsqueeze(0).unsqueeze(0),
            padding=self.num_taps // 2,
        ).squeeze(0)


class IIRFilterDesign(nn.Module):
    """IIR (Infinite Impulse Response) filter design.

    Implements classic IIR filter designs (Butterworth, Chebyshev).
    """

    def __init__(
        self,
        num_poles: int = 4,
        cutoff_freq: float = 0.5,
        filter_type: str = "lowpass",
        ripple: float = 0.5,
    ):
        super().__init__()
        self.num_poles = num_poles
        self.cutoff_freq = cutoff_freq
        self.filter_type = filter_type
        self.ripple = ripple

        self.coeffs_b, self.coeffs_a = self._design_iir()

    def _design_iir(self) -> Tuple[nn.Parameter, nn.Parameter]:
        """Design IIR filter coefficients."""
        from scipy.signal import butter, cheby1

        nyquist = 0.5
        wc = self.cutoff_freq / nyquist

        if wc >= 1:
            wc = 0.99

        try:
            if self.filter_type == "butterworth":
                b, a = butter(self.num_poles, wc, btype="low", analog=False)
            elif self.filter_type == "chebyshev":
                b, a = cheby1(
                    self.num_poles, self.ripple, wc, btype="low", analog=False
                )
            else:
                b, a = butter(self.num_poles, wc, btype="low", analog=False)
        except:
            b = np.array([1.0])
            a = np.array([1.0])

        b_tensor = torch.tensor(b, dtype=torch.float32)
        a_tensor = torch.tensor(a, dtype=torch.float32)

        return nn.Parameter(b_tensor), nn.Parameter(a_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply IIR filter using difference equation."""
        b = self.coeffs_b
        a = self.coeffs_a

        a_norm = a / a[0]
        b_norm = b / a[0]

        output = torch.zeros_like(x)

        for i in range(x.shape[-1]):
            if i == 0:
                output[..., i] = b_norm[0] * x[..., i]
            else:
                acc = torch.zeros(x.shape[0] if x.dim() > 1 else 1, device=x.device)
                for j in range(min(i + 1, len(b_norm))):
                    if x.dim() == 1:
                        acc += b_norm[j] * x[..., i - j]
                    else:
                        acc += b_norm[j] * x[..., i - j]

                for j in range(1, min(i + 1, len(a_norm))):
                    acc -= a_norm[j] * output[..., i - j]

                output[..., i] = acc

        return output


class LearnableFIRFilter(nn.Module):
    """FIR filter with learnable coefficients."""

    def __init__(
        self,
        num_taps: int = 64,
        in_channels: int = 1,
        out_channels: int = 1,
    ):
        super().__init__()
        self.num_taps = num_taps
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.coeffs = nn.Parameter(torch.randn(out_channels, in_channels, num_taps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply learnable FIR filter.

        Args:
            x: Input signal

        Returns:
            Filtered output
        """
        return F.conv1d(x, self.coeffs, padding=self.num_taps // 2, groups=1)


class AdaptiveFilter(nn.Module):
    """Adaptive filter with LMS/RLS learning."""

    def __init__(
        self,
        filter_length: int = 32,
        learning_rate: float = 0.01,
        algorithm: str = "lms",
    ):
        super().__init__()
        self.filter_length = filter_length
        self.learning_rate = learning_rate
        self.algorithm = algorithm

        self.coeffs = nn.Parameter(torch.randn(filter_length))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply adaptive filter."""
        if x.shape[-1] < self.filter_length:
            x = F.pad(x, (self.filter_length - x.shape[-1], 0))

        output = F.conv1d(
            x.unsqueeze(0).unsqueeze(0),
            self.coeffs.unsqueeze(0).unsqueeze(0),
            padding=0,
        ).squeeze()

        return output

    def update(self, x: torch.Tensor, error: torch.Tensor):
        """Update filter coefficients."""
        if self.algorithm == "lms":
            x_window = x[..., : self.filter_length]
            update = self.learning_rate * error * x_window
            with torch.no_grad():
                self.coeffs.add_(update)


class WindowedSincFilter(nn.Module):
    """Windowed sinc filter for optimal FIR design."""

    def __init__(
        self,
        num_taps: int = 65,
        cutoff: float = 0.5,
        window_type: str = "kaiser",
        beta: float = 8.0,
    ):
        super().__init__()
        self.num_taps = num_taps
        self.cutoff = cutoff
        self.window_type = window_type
        self.beta = beta

        self.coeffs = self._compute_coeffs()

    def _compute_coeffs(self) -> nn.Parameter:
        """Compute windowed sinc coefficients."""
        n = torch.arange(self.num_taps)
        n_center = (self.num_taps - 1) / 2

        sinc = torch.sin(2 * np.pi * self.cutoff * (n - n_center)) / (
            np.pi * (n - n_center + 1e-8)
        )
        sinc[n_center.to(int)] = 2 * np.pi * self.cutoff

        if self.window_type == "kaiser":
            window = torch.i0(
                self.beta * torch.sqrt(1 - (2 * n / (self.num_taps - 1) - 1) ** 2)
            )
            window = window / window.max()
        elif self.window_type == "hann":
            window = 0.5 * (1 - torch.cos(2 * np.pi * n / (self.num_taps - 1)))
        else:
            window = torch.ones(self.num_taps)

        coeffs = sinc * window

        return nn.Parameter(coeffs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply windowed sinc filter."""
        return F.conv1d(
            x.unsqueeze(0),
            self.coeffs.unsqueeze(0).unsqueeze(0),
            padding=self.num_taps // 2,
        ).squeeze(0)


class FilterBankLayer(nn.Module):
    """Complete filter bank as a neural network layer."""

    def __init__(
        self,
        in_channels: int,
        num_filters: int,
        filter_length: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.filter_length = filter_length

        self.filters = nn.Parameter(
            torch.randn(num_filters, in_channels, filter_length)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply filter bank.

        Args:
            x: Input signal

        Returns:
            Filtered outputs
        """
        outputs = []

        for i in range(self.num_filters):
            filtered = F.conv1d(
                x, self.filters[i : i + 1], padding=self.filter_length // 2
            )
            outputs.append(filtered)

        return torch.cat(outputs, dim=1)


class GaborFilterDesign(nn.Module):
    """Gabor filter design for time-frequency analysis."""

    def __init__(
        self,
        num_filters: int = 16,
        center_freqs: Optional[List[float]] = None,
        bandwidths: Optional[List[float]] = None,
    ):
        super().__init__()
        self.num_filters = num_filters

        if center_freqs is None:
            center_freqs = np.linspace(0.05, 0.45, num_filters).tolist()
        if bandwidths is None:
            bandwidths = [1.0] * num_filters

        self.center_freqs = nn.Parameter(
            torch.tensor(center_freqs, dtype=torch.float32)
        )
        self.bandwidths = nn.Parameter(torch.tensor(bandwidths, dtype=torch.float32))

    def _generate_gabor(
        self,
        length: int,
        center_freq: float,
        bandwidth: float,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate Gabor filter kernel."""
        t = torch.arange(length, dtype=torch.float32, device=device)
        t = (t - length / 2) / (length / 4)

        envelope = torch.exp(-0.5 * (t / bandwidth) ** 2)

        oscillation = torch.cos(2 * np.pi * center_freq * t)

        gabor = envelope * oscillation

        return gabor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gabor filter bank."""
        batch_size = x.shape[0]
        length = x.shape[-1]
        device = x.device

        filter_length = min(length, 31)
        filter_length = filter_length if filter_length % 2 == 1 else filter_length + 1

        outputs = []

        for i in range(self.num_filters):
            center_freq = torch.abs(self.center_freqs[i])
            bandwidth = torch.abs(self.bandwidths[i]) + 0.1

            kernel = self._generate_gabor(filter_length, center_freq, bandwidth, device)

            kernel = kernel.view(1, 1, -1)

            filtered = F.conv1d(x, kernel, padding=filter_length // 2)
            outputs.append(filtered)

        return torch.cat(outputs, dim=1)


class ComplexGaborFilter(nn.Module):
    """Complex Gabor filter for analytic signal filtering."""

    def __init__(
        self,
        num_filters: int = 8,
        center_freq: float = 0.1,
    ):
        super().__init__()
        self.num_filters = num_filters
        self.center_freq = center_freq

        self.real_kernel = nn.Parameter(torch.randn(num_filters, 1, 17))
        self.imag_kernel = nn.Parameter(torch.randn(num_filters, 1, 17))

        nn.init.normal_(self.real_kernel, mean=0, std=0.1)
        nn.init.normal_(self.imag_kernel, mean=0, std=0.1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply complex Gabor filter.

        Args:
            x: Input signal

        Returns:
            Tuple of (magnitude, phase)
        """
        real_outputs = []
        imag_outputs = []

        for i in range(self.num_filters):
            real_out = F.conv1d(x, self.real_kernel[i : i + 1], padding=8)
            imag_out = F.conv1d(x, self.imag_kernel[i : i + 1], padding=8)

            real_outputs.append(real_out)
            imag_outputs.append(imag_out)

        real = torch.cat(real_outputs, dim=1)
        imag = torch.cat(imag_outputs, dim=1)

        magnitude = torch.sqrt(real**2 + imag**2)
        phase = torch.atan2(imag, real)

        return magnitude, phase


class PolyphaseFilterBank(nn.Module):
    """Polyphase filter bank for efficient multirate processing."""

    def __init__(
        self,
        num_branches: int = 4,
        filter_length: int = 64,
    ):
        super().__init__()
        self.num_branches = num_branches
        self.filter_length = filter_length

        self.polyphase_filters = nn.ModuleList(
            [nn.Conv1d(1, 1, filter_length, groups=1) for _ in range(num_branches)]
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Apply polyphase decomposition.

        Args:
            x: Input signal

        Returns:
            List of polyphase components
        """
        outputs = []

        for i in range(self.num_branches):
            downsampled = x[..., i :: self.num_branches]

            filtered = self.polyphase_filters[i](downsampled)

            outputs.append(filtered)

        return outputs


class QuadratureMirrorFilter(nn.Module):
    """Quadrature Mirror Filter (QMF) bank.

    Perfect reconstruction filter bank using QMFs.
    """

    def __init__(self, filter_length: int = 32):
        super().__init__()
        self.filter_length = filter_length

        h = np.sin(np.pi * (np.arange(filter_length) + 0.5) / filter_length)
        h = h / (h.sum() + 1e-8)

        self.h_low = nn.Parameter(torch.tensor(h, dtype=torch.float32))
        self.h_high = nn.Parameter(
            torch.tensor(((-1) ** np.arange(filter_length)) * h, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply QMF decomposition.

        Args:
            x: Input signal

        Returns:
            Tuple of (lowpass, highpass) components
        """
        lowpass = F.conv1d(
            x,
            self.h_low.unsqueeze(0).unsqueeze(0),
            padding=self.filter_length // 2,
            stride=2,
        )

        highpass = F.conv1d(
            x,
            self.h_high.unsqueeze(0).unsqueeze(0),
            padding=self.filter_length // 2,
            stride=2,
        )

        return lowpass, highpass


class AllpassFilter(nn.Module):
    """Allpass filter for phase manipulation."""

    def __init__(self, num_sections: int = 4):
        super().__init__()
        self.num_sections = num_sections

        self.allpass_coeffs = nn.Parameter(torch.randn(num_sections) * 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply allpass filter chain."""
        output = x

        for i in range(self.num_sections):
            coeff = torch.tanh(self.allpass_coeffs[i])

            delayed = F.pad(output[..., :-1], (1, 0))

            output = delayed + coeff * (output - F.pad(delayed[..., :-1], (1, 0)))

        return output


class GradientFilter(nn.Module):
    """Gradient filter for edge detection in signals."""

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size

        grad_1d = (
            torch.tensor([-1, 0, 1], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        )

        self.register_buffer("gradient_kernel", grad_1d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gradient filter."""
        return F.conv1d(x, self.gradient_kernel, padding=self.kernel_size // 2)
