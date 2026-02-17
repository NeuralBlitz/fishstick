# audio - Audio Processing Module

## Overview

The `audio` module provides comprehensive audio processing capabilities including feature extraction, preprocessing, augmentation, and deep learning models for audio tasks.

## Purpose and Scope

This module enables:
- Audio loading and preprocessing
- Feature extraction (MFCC, mel spectrogram, chroma, etc.)
- Data augmentation for audio
- Audio classification and generation models
- Speech recognition components

## Key Classes and Functions

### Preprocessing (`preprocessing.py`)

#### `AudioLoader`
Load and decode audio files with automatic resampling.

```python
from fishstick.audio import AudioLoader

loader = AudioLoader(sample_rate=16000, mono=True)
waveform = loader("audio.wav")  # Returns tensor
```

#### `AudioNormalizer`
Normalize audio to target dB level.

```python
from fishstick.audio import AudioNormalizer

normalizer = AudioNormalizer(target_db=-20.0)
normalized = normalizer(waveform)
```

#### `TimeStretch`
Time stretch audio with configurable rate.

```python
from fishstick.audio import TimeStretch

stretcher = TimeStretch(min_rate=0.8, max_rate=1.2)
stretched = stretcher(waveform, rate=1.1)  # 10% faster
```

#### `PitchShift`
Pitch shift audio in semitones.

```python
from fishstick.audio import PitchShift

shifter = PitchShift(min_steps=-2, max_steps=2)
shifted = shifter(waveform, sample_rate=16000, steps=1)  # Up 1 semitone
```

#### `AddNoise`
Add noise at specified SNR.

```python
from fishstick.audio import AddNoise

noiser = AddNoise(min_snr=10, max_snr=30)
noisy = noiser(waveform, snr=20)  # 20 dB SNR
```

#### `RandomCrop` and `PadToLength`
Crop or pad audio to target length.

#### `AudioTransform`
Composable audio transformations.

```python
from fishstick.audio import AudioTransform, AddNoise, AudioNormalizer

pipeline = AudioTransform([
    AddNoise(min_snr=20, max_snr=40),
    AudioNormalizer(target_db=-20.0)
])
processed = pipeline(waveform)
```

#### `create_augmentation_pipeline`
Factory for common augmentation pipelines.

```python
from fishstick.audio import create_augmentation_pipeline

augmenter = create_augmentation_pipeline(
    sample_rate=16000,
    add_noise=True,
    pitch_shift=True,
    time_stretch=True
)
augmented = augmenter(waveform)
```

### Feature Extraction (`features.py`)

#### `MelSpectrogram`
Extract mel spectrogram features.

```python
from fishstick.audio import MelSpectrogram

mel_extractor = MelSpectrogram(
    sample_rate=16000,
    n_fft=400,
    hop_length=160,
    n_mels=80
)
mel_spec = mel_extractor(waveform)  # [n_mels, time]
```

#### `MFCC`
Extract Mel-Frequency Cepstral Coefficients.

```python
from fishstick.audio import MFCC

mfcc_extractor = MFCC(
    sample_rate=16000,
    n_fft=400,
    hop_length=160,
    n_mfcc=13
)
mfcc = mfcc_extractor(waveform)  # [n_mfcc, time]
```

#### `Spectrogram`
Extract standard spectrogram.

```python
from fishstick.audio import Spectrogram

spec_extractor = Spectrogram(n_fft=400, hop_length=160)
spec = spec_extractor(waveform)
```

#### `ChromaFeatures`
Extract chroma features for music analysis.

```python
from fishstick.audio import ChromaFeatures

chroma_extractor = ChromaFeatures(
    sample_rate=16000,
    n_fft=4096,
    n_chroma=12
)
chroma = chroma_extractor(waveform)  # [12, time]
```

#### `SpectralContrast`
Extract spectral contrast features.

### Models (`models.py`)

#### `AudioClassifier`
Convolutional audio classifier.

```python
from fishstick.audio import AudioClassifier

model = AudioClassifier(
    input_channels=1,
    num_classes=10,
    hidden_dim=128,
    num_layers=3
)
logits = model(mel_spectrogram)  # [batch, num_classes]
```

#### `WaveNetAudio`
WaveNet-style model for audio generation.

```python
from fishstick.audio import WaveNetAudio

wavenet = WaveNetAudio(
    input_dim=256,
    hidden_dim=512,
    num_layers=8
)
output = wavenet(audio_tokens)
```

#### `TransformerAudio`
Transformer model for audio processing.

```python
from fishstick.audio import TransformerAudio

model = TransformerAudio(
    input_dim=80,      # mel bands
    num_classes=10,
    d_model=256,
    nhead=4,
    num_layers=6
)
logits = model(mel_spec)  # mel_spec: [batch, time, 80]
```

#### `VQVAE`
Vector Quantized VAE for audio.

```python
from fishstick.audio import VQVAE

vqvae = VQVAE(
    input_dim=80,
    hidden_dim=256,
    num_embeddings=256,
    embedding_dim=64
)
recon, quant, commit_loss = vqvae(mel_spec)
```

#### `AudioAutoencoder`
Autoencoder for audio compression.

#### `SpeechRecognizer`
End-to-end speech recognition model.

```python
from fishstick.audio import SpeechRecognizer

asr = SpeechRecognizer(
    input_dim=80,
    num_classes=32,
    d_model=256,
    nhead=4,
    num_layers=6
)
outputs = asr(mel_spec)  # [batch, time, num_classes]
```

## Dependencies

- `torch`: PyTorch tensors and neural networks
- `numpy`: Numerical operations
- `scipy`: Signal processing (optional, fallback)

Optional:
- `torchaudio`: Efficient audio loading

## Usage Examples

### Complete Audio Pipeline

```python
from fishstick.audio import (
    AudioLoader, MelSpectrogram, AudioNormalizer,
    AddNoise, AudioClassifier
)

# Load audio
loader = AudioLoader(sample_rate=16000)
waveform = loader("speech.wav")

# Extract features
mel_extractor = MelSpectrogram(sample_rate=16000, n_mels=80)
mel = mel_extractor(waveform)

# Normalize
normalizer = AudioNormalizer(target_db=-20.0)
mel_normalized = normalizer(mel)

# Classify
classifier = AudioClassifier(input_channels=1, num_classes=10)
logits = classifier(mel_normalized.unsqueeze(0).unsqueeze(0))
```

### Data Augmentation

```python
from fishstick.audio import create_augmentation_pipeline, AudioLoader

loader = AudioLoader(sample_rate=16000)
augmenter = create_augmentation_pipeline(
    sample_rate=16000,
    add_noise=True,
    pitch_shift=True,
    time_stretch=True
)

waveform = loader("audio.wav")
augmented = augmenter(waveform)
```

### Training an Audio Classifier

```python
import torch
from torch.utils.data import DataLoader
from fishstick.audio import AudioClassifier, MelSpectrogram, AudioLoader

# Setup
loader = AudioLoader(sample_rate=16000)
mel_extractor = MelSpectrogram(sample_rate=16000)
model = AudioClassifier(input_channels=1, num_classes=10)
optimizer = torch.optim.Adam(model.parameters())

# Training loop
for epoch in range(10):
    for batch in dataloader:
        waveforms, labels = batch
        
        # Extract features
        mels = torch.stack([mel_extractor(w) for w in waveforms])
        mels = mels.unsqueeze(1)  # Add channel dim
        
        # Forward pass
        logits = model(mels)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```
