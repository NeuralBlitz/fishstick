# Diffusion Models

Diffusion probabilistic models, schedulers, and utilities for image generation.

## Installation

```bash
pip install fishstick
```

## Quick Start

```python
import torch
from fishstick.diffusion import (
    TextToImagePipeline,
    DDPMScheduler,
    DDIMScheduler,
    UNetModel,
)

# Create a text-to-image pipeline
pipeline = TextToImagePipeline(
    num_inference_steps=50,
    guidance_scale=7.5,
)

# Generate an image from text
image = pipeline(
    prompt="a beautiful sunset over the ocean",
    negative_prompt="blurry, low quality",
)

# Custom training with DDPM
scheduler = DDPMScheduler(num_train_timesteps=1000)
unet = UNetModel(in_channels=4, out_channels=4)

# Training loop
for batch in dataloader:
    noise = torch.randn_like(batch)
    timesteps = torch.randint(0, 1000, (batch.shape[0],))
    noisy = scheduler.add_noise(batch, noise, timesteps)
    prediction = unet(noisy, timesteps)
```

## API Reference

### Schedulers

| Class | Description |
|-------|-------------|
| `DDPMScheduler` | Denoising Diffusion Probabilistic Models scheduler |
| `DDIMScheduler` | DDIM scheduler for faster sampling |
| `DPMSolverMultistepScheduler` | DPM-Solver++ for fast sampling |
| `EulerDiscreteScheduler` | Euler method discrete scheduler |
| `LMSDiscreteScheduler` | Linear Multistep scheduler |

### UNet Components

| Class | Description |
|-------|-------------|
| `ResBlock` | Residual block with time embedding |
| `AttentionBlock` | Multi-head self-attention block |
| `TimestepEmbedding` | Sinusoidal timestep embedding |
| `TimestepEmbedSequential` | Sequential with timestep embedding |
| `UNetModel` | Complete U-Net for diffusion |
| `classifier_free_guidance` | Utility for CFG sampling |

### Latent Diffusion

| Class | Description |
|-------|-------------|
| `VAEEncoder` | Variational Autoencoder encoder |
| `VAEDecoder` | Variational Autoencoder decoder |
| `LatentDiffusionModel` | Latent diffusion model |
| `TextToImagePipeline` | Complete text-to-image pipeline |

## Examples

### Training a Diffusion Model

```python
import torch
from fishstick.diffusion import (
    LatentDiffusionModel,
    DDPMScheduler,
)

model = LatentDiffusionModel(
    in_channels=3,
    latent_channels=4,
    hidden_channels=128,
)

scheduler = DDPMScheduler(num_train_timesteps=1000)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

model.train()
for epoch in range(100):
    for batch in train_loader:
        # Sample random timesteps
        t = torch.randint(0, 1000, (batch.shape[0],))
        
        # Add noise
        noise = torch.randn_like(batch)
        noisy = scheduler.add_noise(batch, noise, t)
        
        # Predict noise
        pred = model(noisy, t)
        
        # Compute loss
        loss = torch.nn.functional.mse_loss(pred, noise)
        
        loss.backward()
        optimizer.step()
```

### Fast Sampling with DDIM

```python
from fishstick.diffusion import (
    TextToImagePipeline,
    DDIMScheduler,
)

pipeline = TextToImagePipeline(
    scheduler=DDIMScheduler(),
    num_inference_steps=20,  # Fewer steps = faster
)

image = pipeline(prompt="cyberpunk city at night")
```

### Classifier-Free Guidance

```python
from fishstick.diffusion import (
    TextToImagePipeline,
    classifier_free_guidance,
)

pipeline = TextToImagePipeline(guidance_scale=7.5)
images = pipeline(prompt="a cat", negative_prompt="dog")
```
