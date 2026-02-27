# Audio Separation

Comprehensive audio source separation: music, speaker, and general source separation.

## Time-Frequency Masking

```python
from fishstick.audio_separation import (
    TimeFrequencyMask,
    IdealBinaryMask,
    IdealRatioMask,
    PhaseSensitiveMask,
    WienerFilter,
)

# Ideal ratio mask
mask = IdealRatioMask()
estimated_mask = mask.compute(estimated_spec, target_spec)

# Wiener filtering
wiener = WienerFilter()
sources = wiener(mixed_audio, n_sources=2)
```

## Deep Clustering

```python
from fishstick.audio_separation import (
    DeepClustering,
    EmbeddingExtractor,
    DiscriminativeLoss,
)

model = DeepClustering(
    n_sources=2,
    embedding_dim=20,
    n_layers=4
)
embeddings = model(mixed_spectrogram)
sources = model.separate(mixed_audio)
```

## Waveform Separation

```python
from fishstick.audio_separation import (
    ConvTasNet,
    DualPathRNN,
    TemporalConvNet,
)

# Conv-TasNet
model = ConvTasNet(
    n_sources=2,
    n_encoder_channels=256,
    n_masker_channels=512
)
sources = model(mixed_audio)
```

## Speaker Separation

```python
from fishstick.audio_separation import (
    SpeakerSeparator,
    SpeakerBeam,
    SpeakerRecognition,
)

# Speaker extraction
separator = SpeakerSeparator(
    n_speakers=2,
    sample_rate=16000
)
isolated = separator(mixed_audio, speaker_embeddings)
```

## Music Separation

```python
from fishstick.audio_separation import (
    MusicSeparator,
    Demucs,
    OpenUnmix,
)

# Demucs for music source separation
demucs = Demucs(
    sources=["drums", "bass", "other", "vocals"],
    sample_rate=44100
)
stems = demucs(mixed_audio)
```

## Metrics

```python
from fishstick.audio_separation import (
    SI_SDR, SI_SAR, SI_SNR, PESQ, STOI
)

# Calculate SI-SDR
metric = SI_SDR()
score = metric(estimated_source, ground_truth_source)
```
