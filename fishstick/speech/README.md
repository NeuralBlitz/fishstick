# Speech Processing

Speech recognition, audio processing, and speech models.

## Installation

```bash
pip install fishstick[speech]
```

## Overview

The `speech` module provides speech recognition models and audio processing utilities.

## Usage

```python
from fishstick.speech import Wav2Vec2Model, WhisperModel, SpeechRecognizer

# Wav2Vec2
wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
recognizer = SpeechRecognizer(wav2vec)
text = recognizer.recognize(audio)

# Whisper
whisper = WhisperModel.from_pretrained("openai/whisper-base")
text = whisper.transcribe(audio)
```

## Models

| Model | Description |
|-------|-------------|
| `Wav2Vec2Model` | Wav2Vec 2.0 for ASR |
| `HubertModel` | HuBERT for self-supervised speech |
| `ConformerModel` | Conformer for ASR |
| `WhisperModel` | OpenAI Whisper |
| `AudioSpectrogramTransformer` | Audio spectrogram transformer |

## Utilities

| Class | Description |
|-------|-------------|
| `FeatureExtractor` | Extract audio features |
| `AudioPreprocessor` | Preprocess audio |
| `CTCLossWrapper` | CTC loss wrapper |
| `SpeechDataset` | Speech dataset |
| `SpeechRecognizer` | Speech-to-text recognizer |

## Examples

See `examples/speech/` for complete examples.
