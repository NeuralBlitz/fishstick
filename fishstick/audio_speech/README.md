# Audio & Speech

Speech processing: features, VAD, speaker recognition, synthesis.

## Spectral Features

```python
from fishstick.audio_speech import (
    SpectralFeatures,
    ChromaFeatures,
    SpectralContrast,
    TonnetzFeatures,
)

# Extract spectral features
features = SpectralFeatures(sample_rate=16000)
centroid, rolloff, flux, flatness = features(audio_tensor)

# Chroma features
chroma = ChromaFeatures(n_chroma=12)
chroma_features = chroma(audio_tensor)
```

## Speech Features

```python
from fishstick.audio_speech import (
    SpeechFeatureExtractor,
    PitchExtractor,
    FormantExtractor,
    EnergyExtractor,
)

# Complete speech feature extraction
extractor = SpeechFeatureExtractor(sample_rate=16000)
features = extractor(audio_tensor)

# Pitch extraction
pitch_extractor = PitchExtractor(method="yin")
f0 = pitch_extractor(audio_tensor)
```

## Voice Activity Detection

```python
from fishstick.audio_speech import (
    EnergyVAD,
    NeuralVAD,
    HybridVAD,
    create_vad,
)

# Energy-based VAD
vad = EnergyVAD(threshold=0.01)
is_speech = vad(audio_tensor)

# Neural VAD
vad = NeuralVAD(model_path="vad_model.pt")
is_speech = vad(audio_tensor)

# Create VAD from config
vad = create_vad(method="hybrid", sample_rate=16000)
```

## Speaker Recognition

```python
from fishstick.audio_speech import (
    SpeakerEncoder,
    RawNetEncoder,
    SpeakerVerification,
    SpeakerIdentification,
    AngularMarginLoss,
)

# Speaker embedding extraction
encoder = SpeakerEncoder(embedding_dim=512)
embeddings = encoder(audio_tensor)

# Speaker verification
verifier = SpeakerVerification(threshold=0.5)
is_same = verifier.verify(enrollment_audio, test_audio)

# Speaker identification
identifier = SpeakerIdentification(n_speakers=100)
speaker_id = identifier.identify(audio_tensor)
```

## Audio Synthesis

```python
from fishstick.audio_speech import (
    GriffinLimVocoder,
    MelGANVocoder,
    WaveNetVocoder,
    create_vocoder,
)

# Griffin-Lim vocoder
vocoder = GriffinLimVocoder(n_fft=1024, hop_length=256)
audio = vocoder(mel_spectrogram)

# Neural vocoder
vocoder = create_vocoder("melgan", checkpoint="melgan.pt")
audio = vocoder(mel_spectrogram)
```
