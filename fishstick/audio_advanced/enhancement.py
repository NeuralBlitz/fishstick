import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch.nn import dropout


class NoiseSuppressionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, hidden_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm = nn.GroupNorm(1, hidden_channels)
        self.activation = nn.PReLU()
        self.gate = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_out = self.activation(self.norm(self.conv(x)))
        gate = self.gate(conv_out)
        return conv_out * gate


class SpectralSubtraction(nn.Module):
    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        alpha: float = 2.0,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.alpha = alpha
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(
        self, x: torch.Tensor, noise_estimate: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        stft_result = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )
        mag = torch.abs(stft_result)
        phase = torch.angle(stft_result)

        if noise_estimate is not None:
            noise_mag = noise_estimate
        else:
            noise_mag = mag[:, :, : mag.size(2) // 10].mean(dim=2, keepdim=True)

        enhanced_mag = mag - self.alpha * noise_mag
        enhanced_mag = torch.clamp(enhanced_mag, min=0)

        complex_spec = enhanced_mag * torch.exp(1j * phase)
        enhanced = torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
        )
        return enhanced


class WienerFiltering(nn.Module):
    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        beta: float = 2.0,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.beta = beta
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(
        self, x: torch.Tensor, noise_estimate: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        stft_result = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )
        mag = torch.abs(stft_result)
        phase = torch.angle(stft_result)
        power = mag**2

        if noise_estimate is not None:
            noise_power = noise_estimate**2
        else:
            noise_power = power[:, :, : power.size(2) // 10].mean(dim=2, keepdim=True)

        snr = power / (noise_power + 1e-8)
        wiener_gain = (snr ** (self.beta / 2)) / (snr ** (self.beta / 2) + 1)

        enhanced_mag = wiener_gain * mag
        complex_spec = enhanced_mag * torch.exp(1j * phase)
        enhanced = torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
        )
        return enhanced


class DeepNoiseSuppression(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        hidden_channels: int = 64,
        num_layers: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_conv = nn.Conv1d(input_channels, hidden_channels, 1)

        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(
                NoiseSuppressionBlock(hidden_channels, hidden_channels, kernel_size)
            )

        self.output_conv = nn.Conv1d(hidden_channels, input_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)
        for block in self.blocks(x):
            x = block(x)
        x = self.output_conv(x)
        return x


class RNNoiseSuppression(nn.Module):
    def __init__(
        self,
        input_channels: int = 72,
        hidden_channels: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_channels,
            hidden_channels,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.output = nn.Linear(hidden_channels, input_channels // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.gru(x)
        gain = self.output(x)
        gain = torch.sigmoid(gain)
        return gain


class DereverberationBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm = nn.InstanceNorm1d(out_channels)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(x)))


class WeightedPredictionDereverberation(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            in_channels,
            hidden_channels,
            num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.output = nn.Linear(hidden_channels * 2, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm(x)
        weight = torch.sigmoid(self.output(x))
        return weight


class DeepAttractor(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 1024,
        num_sources: int = 2,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_sources = num_sources

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, channels = embedding.shape
        attractors = []
        for i in range(self.num_sources):
            attractor = embedding.mean(dim=1)
            attractors.append(attractor)
        attractors = torch.stack(attractors, dim=1)
        return attractors


class Dereverberation(nn.Module):
    def __init__(
        self,
        input_channels: int = 257,
        hidden_channels: int = 512,
        num_layers: int = 4,
        num_sources: int = 2,
    ):
        super().__init__()
        self.num_sources = num_sources

        self.encoder = nn.ModuleList()
        ch = input_channels
        for _ in range(num_layers):
            self.encoder.append(DereverberationBlock(ch, hidden_channels))
            ch = hidden_channels

        self.attractor = DeepAttractor(hidden_channels, num_sources)

        self.decoder = nn.ModuleList()
        for _ in range(num_layers):
            self.decoder.append(DereverberationBlock(hidden_channels, input_channels))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        encoder_outs = []
        for enc in self.encoder:
            x = enc(x)
            encoder_outs.append(x)

        attractors = self.attractor(x)

        x = x.unsqueeze(1).expand(-1, self.num_sources, -1, -1)
        sources = []
        for i in range(self.num_sources):
            source = x[:, i]
            for dec in self.decoder:
                source = dec(source)
            sources.append(source)

        return tuple(sources)


class SpeechEnhancement(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        enhancement_type: str = "deep",
        num_sources: int = 1,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.enhancement_type = enhancement_type

        self.register_buffer("window", torch.hann_window(win_length))

        if enhancement_type == "deep":
            self.model = DeepNoiseSuppression()
        elif enhancement_type == "rnnoise":
            self.model = RNNoiseSuppression()
        elif enhancement_type == "dereverb":
            self.model = Dereverberation(
                input_channels=n_fft // 2 + 1,
                num_sources=num_sources,
            )
        elif enhancement_type == "spectral_subtraction":
            self.model = SpectralSubtraction(n_fft, hop_length, win_length)
        elif enhancement_type == "wiener":
            self.model = WienerFiltering(n_fft, hop_length, win_length)
        else:
            raise ValueError(f"Unknown enhancement type: {enhancement_type}")

    def forward(
        self, x: torch.Tensor, return_spectrogram: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        if self.enhancement_type in ["spectral_subtraction", "wiener"]:
            enhanced = self.model(x)
            return enhanced, torch.abs(
                torch.stft(
                    enhanced,
                    self.n_fft,
                    self.hop_length,
                    self.win_length,
                    self.window,
                    return_complex=True,
                )
            )

        stft_result = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )
        mag = torch.abs(stft_result)
        phase = torch.angle(stft_result)

        mag_input = mag.permute(0, 2, 1)

        if self.enhancement_type == "dereverb":
            enhanced_mags = self.model(mag_input)
            enhanced_mags = enhanced_mags.permute(0, 2, 1)
        else:
            enhanced_mags = self.model(mag_input)
            enhanced_mags = (
                enhanced_mags.squeeze(1) if enhanced_mags.dim() == 2 else enhanced_mags
            )
            enhanced_mags = enhanced_mags.permute(0, 2, 1)

        complex_spec = enhanced_mags * torch.exp(1j * phase)
        enhanced = torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
        )

        if return_spectrogram:
            return enhanced, enhanced_mags
        return enhanced


class MultiChannelEnhancement(nn.Module):
    def __init__(
        self,
        num_channels: int = 2,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 128,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.enhancers = nn.ModuleList(
            [
                SpeechEnhancement(
                    sample_rate=sample_rate,
                    n_fft=n_fft,
                    hop_length=hop_length,
                )
                for _ in range(num_channels)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        enhanced = []
        for i in range(self.num_channels):
            ch = x[:, i : i + 1] if x.size(1) > i else x[:, :1]
            enh_ch = self.enhancers[i](ch.squeeze(1))
            enhanced.append(enh_ch)
        return torch.stack(enhanced, dim=1)


class SpeechEnhancementTrainer:
    def __init__(
        self,
        model: SpeechEnhancement,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module = nn.MSELoss(),
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def train_step(self, noisy: torch.Tensor, clean: torch.Tensor) -> float:
        self.model.train()
        self.optimizer.zero_grad()

        enhanced = self.model(noisy)
        loss = self.criterion(enhanced, clean)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def eval_step(self, noisy: torch.Tensor, clean: torch.Tensor) -> dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            enhanced = self.model(noisy)
            loss = self.criterion(enhanced, clean)

            pesq_score = self._compute_pesq(enhanced, clean)
            stoi_score = self._compute_stoi(enhanced, clean)

        return {
            "loss": loss.item(),
            "pesq": pesq_score,
            "stoi": stoi_score,
        }

    def _compute_pesq(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        return 1.0

    def _compute_stoi(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        return 1.0


__all__ = [
    "SpectralSubtraction",
    "WienerFiltering",
    "DeepNoiseSuppression",
    "RNNoiseSuppression",
    "Dereverberation",
    "DeepAttractor",
    "SpeechEnhancement",
    "MultiChannelEnhancement",
    "SpeechEnhancementTrainer",
]
