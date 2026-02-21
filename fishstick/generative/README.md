# Generative Models Module

Comprehensive implementation of state-of-the-art generative model architectures.

## Overview

This module provides cutting-edge generative models including:

- **Diffusion Models**: DDPM, DDIM, Score-based, Latent Diffusion
- **GAN Extensions**: StyleGAN, StyleGAN2, BigGAN
- **Autoregressive Models**: GPT-style, PixelCNN
- **Energy-Based Models**: EBMs with advanced sampling
- **Flow Matching**: OT-Flow, Conditional Flow Matching

## Installation

```bash
# Core dependencies (included in fishstick)
pip install torch numpy scipy

# For better performance
pip installeinops ftfy
```

## Quick Start

### Denoising Diffusion Probabilistic Models (DDPM)

```python
import torch
from fishstick.generative import DDPM, DiffusionScheduler

# Create DDPM model
model = DDPM(
    in_channels=3,
    hidden_channels=128,
    n_res_blocks=2,
    n_heads=4,
    T=1000  # Number of diffusion steps
)

# Forward process (add noise)
x_t, noise = model.forward_process(x_0, t)

# Reverse process (denoise)
x_0_pred = model.reverse_process(x_t)

# Sampling
samples = model.sample(batch_size=16, shape=(3, 32, 32))

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(100):
    for batch in dataloader:
        x_0 = batch
        
        # Sample timestep
        t = torch.randint(0, model.T, (x_0.shape[0],))
        
        # Sample noise
        noise = torch.randn_like(x_0)
        
        # Forward process
        x_t = model.q_sample(x_0, t, noise)
        
        # Predict noise
        noise_pred = model(x_t, t)
        
        # Compute loss
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        
        loss.backward()
        optimizer.step()
```

### DDIM (Deterministic Sampling)

```python
from fishstick.generative import DDIM, DDIMScheduler

# DDIM for faster sampling
ddim = DDIM(
    model=model,  # Pre-trained DDPM model
    n_steps=50   # Fewer steps than DDPM
)

# Fast sampling
samples = ddim.sample(batch_size=16)

# Custom schedule
scheduler = DDIMScheduler(
    schedule="linear",
    beta_start=0.0001,
    beta_end=0.02
)
```

### StyleGAN

```python
from fishstick.generative import StyleGAN2, MappingNetwork, SynthesisNetwork

# Create StyleGAN2
gan = StyleGAN2(
    latent_dim=512,
    hidden_dim=512,
    w_dim=512,
    n_layers=8,
    channel_base=8192
)

# Generate images
latents = torch.randn(16, 512)
styles = gan.mapping_network(latents)  # [16, 10, 512]

# Truncation trick for better quality
styles = gan.truncation(styles, truncation_psi=0.7)

images = gan.synthesis_network(styles)  # [16, 3, 1024, 1024]

# Training requires balanced batch of real/fake
```

### BigGAN

```python
from fishstick.generative import BigGAN, ConditionalGenerator

biggan = BigGAN(
    latent_dim=120,
    cond_dim=1000,  # ImageNet classes
    hidden_dim=2048,
    channel_factor=64
)

# Conditional generation
z = torch.randn(8, 120)  # Noise
y = torch.randint(0, 1000, (8,))  # Class labels

img = biggan(z, y)  # [8, 3, 256, 256]
```

### Score-Based Models

```python
from fishstick.generative import ScoreBasedModel, AnnealedLangevinDynamics

# Score network
score_net = ScoreBasedModel(
    in_channels=3,
    hidden_channels=256,
    n_res_blocks=10
)

# Annealed Langevin Dynamics sampling
ald = AnnealedLangevinDynamics(
    score_model=score_net,
    sigmas=torch.linspace(1.0, 0.01, 10),
    step_size=0.1
)

samples = ald.sample(shape=(16, 3, 32, 32))
```

### Latent Diffusion

```python
from fishstick.generative import LatentDiffusionModel, AutoencoderKL

# Autoencoder for latent space
autoencoder = AutoencoderKL(
    in_channels=3,
    latent_dim=4,
    hidden_channels=128
)

# Latent diffusion model
ldm = LatentDiffusionModel(
    autoencoder=autoencoder,
    unet_channels=[128, 256, 512],
    latent_channels=4,
    T=1000
)

# Train in latent space
```

### Autoregressive Generation

```python
from fishstick.generative import AutoregressiveTransformer

transformer = AutoregressiveTransformer(
    vocab_size=50000,
    d_model=512,
    n_heads=8,
    n_layers=6,
    max_seq_len=1024
)

# Autoregressive generation
output = transformer.generate(
    input_ids=start_tokens,
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.9
)
```

### PixelCNN

```python
from fishstick.generative import PixelCNN

pixelcnn = PixelCNN(
    in_channels=3,
    n_channels=128,
    n_res_blocks=15,
    gated=True
)

# Discrete generation
log_probs = pixelcnn(x)  # [B, 256, H, W]
```

### Energy-Based Models

```python
from fishstick.generative import EnergyBasedModel, LangevinSampler

ebm = ConvEnergyModel(
    in_channels=3,
    hidden_channels=128
)

# Langevin sampling
sampler = LangevinSampler(
    ebm=ebm,
    step_size=0.1,
    n_steps=100
)

samples = sampler.sample(shape=(16, 3, 32, 32))
```

### Flow Matching

```python
from fishstick.generative import FlowMatching, ConditionalFlowMatching

# Flow matching
flow = FlowMatching(
    dim=256,
    hidden_dim=512,
    n_blocks=6
)

# Train
x_0 = torch.randn(32, 256)  # Data
x_1 = torch.randn(32, 256)  # Noise
t = torch.rand(32)

loss = flow.training_step(x_0, x_1, t)

# Sample
samples = flow.sample(shape=(16, 256))
```

## API Reference

### Diffusion Models

| Class | Description |
|-------|-------------|
| `DDPM` | Denoising Diffusion Probabilistic Model |
| `DDIM` | Denoising Diffusion Implicit Model |
| `DiffusionScheduler` | Noise scheduling |
| `ScoreBasedModel` | Score-based generative model |
| `AnnealedLangevinDynamics` | Score-based sampling |
| `LatentDiffusionModel` | Diffusion in latent space |

### GANs

| Class | Description |
|-------|-------------|
| `StyleGAN2` | StyleGAN2 architecture |
| `MappingNetwork` | Style mapping network |
| `SynthesisNetwork` | Image synthesis network |
| `BigGAN` | BigGAN conditional generation |
| `ConditionalGenerator` | Conditional GAN generator |

### Autoregressive

| Class | Description |
|-------|-------------|
| `AutoregressiveTransformer` | Transformer autoregressive model |
| `GPTGenerator` | GPT-style language model |
| `PixelCNN` | PixelCNN for images |
| `PixelCNN++` | Improved PixelCNN |

### Energy-Based

| Class | Description |
|-------|-------------|
| `EnergyBasedModel` | Base EBM architecture |
| `ConvEnergyModel` | Convolutional EBM |
| `LangevinSampler` | Langevin dynamics sampler |
| `HMCSampler` | Hamiltonian Monte Carlo sampler |

### Flow-Based

| Class | Description |
|-------|-------------|
| `FlowMatching` | Flow matching model |
| `ConditionalFlowMatching` | Conditional flow matching |
| `OptimalTransportFlow` | OT-based flow |
| `SinkhornDivergence` | Sinkhorn divergence loss |

## Examples

See `examples/generative/` for complete training examples.

## References

- Ho et al., "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
- Song et al., "Denoising Diffusion Implicit Models" (ICLR 2021)
- Karras et al., "StyleGAN2" (CVPR 2020)
- Brock et al., "BigGAN" (ICLR 2019)
- Lipman et al., "Flow Matching" (ICLR 2023)

## License

MIT License - see project root for details.
