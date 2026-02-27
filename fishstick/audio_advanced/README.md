# Audio Advanced

Advanced audio processing: spectrograms, source separation, speech enhancement.

## Spectrogram Features

```python
from fishstick.audio_advanced import (
    STFT, MelSpectrogram, MFCC, MelFilterbank
)

# Short-Time Fourier Transform
stft = STFT(n_fft=2048, hop_length=512)
spec = stft(audio_tensor)  # [batch, freq, time, complex]

# Mel spectrogram
mel_spec = MelSpectrogram(
    sample_rate=16000,
    n_mels=80,
    n_fft=2048
)(audio_tensor)

# MFCC features
mfcc = MFCC(n_mfcc=13)(audio_tensor)
```

## Source Separation

```python
from fishstick.audio_advanced import (
    WaveUNet, TasNet, DCCRN, DPRNN, SourceSeparator
)

# Wave-U-Net for source separation
model = WaveUNet(
    n_sources=2,
    n_audio_channels=1,
    n_filters=32
)
sources = model(mixed_audio)  # [n_sources, batch, samples]

# Dual-Path RNN
model = DPRNN(
    n_sources=2,
    n_features=256,
    n_layers=6
)
sources = model(mixed_audio)
```

## Speech Enhancement

```python
from fishstick.audio_advanced import (
    DeepNoiseSuppression,
    RNNoiseSuppression,
    Dereverberation,
    SpeechEnhancement,
)

# Deep noise suppression
enhancer = DeepNoiseSuppression(
    sample_rate=16000,
    window_size=512
)
enhanced_audio = enhancer(noisy_audio)

# Full speech enhancement pipeline
enhancement = SpeechEnhancement(
    model_type="dccrn",
    sample_rate=16000
)
clean_audio = enhancement(noisy_audio)
```
