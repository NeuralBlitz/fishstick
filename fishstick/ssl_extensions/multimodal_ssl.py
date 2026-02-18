"""
Multi-Modal Self-Supervised Learning

Extended implementations:
- CLIP-style image-text contrastive learning
- Multi-modal projector and encoder
- Audio-visual SSL
- Cross-modal retrieval
"""

from typing import Optional, Tuple, Dict, Any, List, Callable, Union
import copy
import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from fishstick.ssl_extensions.base import (
    EMAUpdater,
    MemoryBank,
    stop_gradient,
    gather_from_all,
    L2Normalize,
    PositionalEmbedding2D,
)


class CLIPTextEncoder(nn.Module):
    """Text encoder for CLIP-style models.
    
    Args:
        vocab_size: Size of vocabulary
        max_seq_len: Maximum sequence length
        embed_dim: Embedding dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        mlp_ratio: MLP expansion ratio
    """
    
    def __init__(
        self,
        vocab_size: int = 49408,
        max_seq_len: int = 77,
        embed_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, max_seq_len, embed_dim)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.layernorm = nn.LayerNorm(embed_dim)
        
    def forward(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        B, L = input_ids.shape
        
        x = self.token_embedding(input_ids)
        x = x + self.position_embedding[:, :L, :]
        
        if attention_mask is not None:
            attention_mask = attention_mask.float()
            attention_mask = (1.0 - attention_mask) * -10000.0
            
        x = self.transformer(x, src_key_padding_mask=attention_mask)
        
        x = self.layernorm(x)
        
        return x


class CLIPImageEncoder(nn.Module):
    """Image encoder for CLIP-style models.
    
    Args:
        image_size: Input image size
        patch_size: Patch size
        embed_dim: Embedding dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        mlp_ratio: MLP expansion ratio
    """
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        num_patches = (image_size // patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.layernorm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        x = x + self.position_embedding
        
        x = self.transformer(x)
        
        x = self.layernorm(x)
        
        return x[:, 0]


class CLIPLoss(nn.Module):
    """CLIP loss function for image-text contrastive learning.
    
    Args:
        temperature: Temperature for softmax
        symmetric: Whether to use symmetric loss
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        symmetric: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.symmetric = symmetric
        
    def forward(
        self,
        image_features: Tensor,
        text_features: Tensor,
    ) -> Tensor:
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        logits = (image_features @ text_features.T) / self.temperature
        
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=image_features.device)
        
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        
        if self.symmetric:
            return (loss_i + loss_t) / 2
        return loss_i


class CLIPModel(nn.Module):
    """CLIP: Contrastive Language-Image Pre-training.
    
    Args:
        image_encoder: Image encoder network
        text_encoder: Text encoder network
        embed_dim: Embedding dimension
        temperature: Temperature for contrastive loss
    """
    
    def __init__(
        self,
        image_encoder: Optional[nn.Module] = None,
        text_encoder: Optional[nn.Module] = None,
        embed_dim: int = 512,
        temperature: float = 0.07,
    ):
        super().__init__()
        
        if image_encoder is None:
            image_encoder = CLIPImageEncoder(embed_dim=embed_dim)
        if text_encoder is None:
            text_encoder = CLIPTextEncoder(embed_dim=embed_dim)
            
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        
        self.image_projection = nn.Linear(embed_dim, embed_dim)
        self.text_projection = nn.Linear(embed_dim, embed_dim)
        
        self.loss_fn = CLIPLoss(temperature=temperature)
        
    def forward(
        self,
        images: Tensor,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(input_ids, attention_mask)
        
        image_features = self.image_projection(image_features)
        text_features = self.text_projection(text_features)
        
        loss = self.loss_fn(image_features, text_features)
        
        return loss, {
            "image_features": image_features,
            "text_features": text_features,
        }
        
    def get_image_embeddings(self, images: Tensor) -> Tensor:
        features = self.image_encoder(images)
        features = self.image_projection(features)
        return F.normalize(features, dim=-1)
        
    def get_text_embeddings(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        features = self.text_encoder(input_ids, attention_mask)
        features = self.text_projection(features)
        return F.normalize(features, dim=-1)


class AudioEncoder(nn.Module):
    """Audio encoder for multimodal learning.
    
    Args:
        sample_rate: Audio sample rate
        n_fft: FFT size
        hop_length: Hop length
        n_mels: Number of mel bins
        embed_dim: Embedding dimension
        num_layers: Number of transformer layers
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        embed_dim: int = 768,
        num_layers: int = 12,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.embed_dim = embed_dim
        
        self.mel_transform = nn.Sequential(
            nn.Linear(n_mels, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=int(embed_dim * 4),
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.layernorm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        
        x = self._get_mel_spectrogram(x)
        
        x = self.mel_transform(x)
        
        x = self.transformer(x)
        
        x = self.layernorm(x)
        
        return x.mean(dim=1)
        
    def _get_mel_spectrogram(self, x: Tensor) -> Tensor:
        mel_spec = torch.stft(
            x.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            return_complex=True,
        )
        mel_spec = mel_spec.abs()
        
        mel_filterbank = torch.linspace(0, 1, self.n_mels)
        mel_spec = mel_spec[:, :self.n_mels, :] * mel_filterboard.unsqueeze(-1)
        
        mel_spec = torch.log(mel_spec + 1e-9)
        
        return mel_spec.transpose(1, 2)


class AudioVisualSSL(nn.Module):
    """Audio-Visual Self-Supervised Learning.
    
    Args:
        audio_encoder: Audio encoder network
        visual_encoder: Visual encoder network
        embed_dim: Embedding dimension
        temperature: Temperature for contrastive loss
    """
    
    def __init__(
        self,
        audio_encoder: Optional[nn.Module] = None,
        visual_encoder: Optional[nn.Module] = None,
        embed_dim: int = 256,
        temperature: float = 0.1,
    ):
        super().__init__()
        
        if visual_encoder is None:
            visual_encoder = CLIPImageEncoder(embed_dim=embed_dim)
            
        self.visual_encoder = visual_encoder
        self.audio_encoder = audio_encoder or AudioEncoder(embed_dim=embed_dim)
        
        self.visual_projection = nn.Linear(embed_dim, embed_dim)
        self.audio_projection = nn.Linear(embed_dim, embed_dim)
        
        self.temperature = temperature
        
    def forward(
        self,
        videos: Tensor,
        audio: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        video_features = self.visual_encoder(videos)
        audio_features = self.audio_encoder(audio)
        
        video_proj = self.visual_projection(video_features)
        audio_proj = self.audio_projection(audio_features)
        
        loss = self._contrastive_loss(video_proj, audio_proj)
        
        return loss, {
            "video_features": video_features,
            "audio_features": audio_features,
        }
        
    def _contrastive_loss(
        self,
        video_features: Tensor,
        audio_features: Tensor,
    ) -> Tensor:
        video_features = F.normalize(video_features, dim=-1)
        audio_features = F.normalize(audio_features, dim=-1)
        
        logits = (video_features @ audio_features.T) / self.temperature
        
        batch_size = video_features.shape[0]
        labels = torch.arange(batch_size, device=video_features.device)
        
        loss_v = F.cross_entropy(logits, labels)
        loss_a = F.cross_entropy(logits.T, labels)
        
        return (loss_v + loss_a) / 2
        
    def get_video_embeddings(self, videos: Tensor) -> Tensor:
        features = self.visual_encoder(videos)
        features = self.visual_projection(features)
        return F.normalize(features, dim=-1)
        
    def get_audio_embeddings(self, audio: Tensor) -> Tensor:
        features = self.audio_encoder(audio)
        features = self.audio_projection(features)
        return F.normalize(features, dim=-1)


class CrossModalRetriever(nn.Module):
    """Cross-modal retrieval with CLIP-style embeddings.
    
    Args:
        image_encoder: Image encoder
        text_encoder: Text encoder
        embed_dim: Embedding dimension
    """
    
    def __init__(
        self,
        image_encoder: Optional[nn.Module] = None,
        text_encoder: Optional[nn.Module] = None,
        embed_dim: int = 512,
    ):
        super().__init__()
        
        self.image_encoder = image_encoder or CLIPImageEncoder(embed_dim=embed_dim)
        self.text_encoder = text_encoder or CLIPTextEncoder(embed_dim=embed_dim)
        
        self.image_projection = nn.Linear(embed_dim, embed_dim)
        self.text_projection = nn.Linear(embed_dim, embed_dim)
        
    def forward(
        self,
        images: Optional[Tensor] = None,
        text_ids: Optional[Tensor] = None,
        text_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        results = {}
        
        if images is not None:
            img_features = self.image_encoder(images)
            img_features = self.image_projection(img_features)
            results['image_embeddings'] = F.normalize(img_features, dim=-1)
            
        if text_ids is not None:
            txt_features = self.text_encoder(text_ids, text_mask)
            txt_features = self.text_projection(txt_features)
            results['text_embeddings'] = F.normalize(txt_features, dim=-1)
            
        if 'image_embeddings' in results and 'text_embeddings' in results:
            results['similarity'] = (
                results['image_embeddings'] @ results['text_embeddings'].T
            )
            
        return results
        
    def retrieve_images(
        self,
        text_embeddings: Tensor,
        image_embeddings: Tensor,
        top_k: int = 5,
    ) -> Tuple[Tensor, Tensor]:
        similarities = text_embeddings @ image_embeddings.T
        
        scores, indices = similarities.topk(top_k, dim=-1)
        
        return scores, indices
        
    def retrieve_texts(
        self,
        image_embeddings: Tensor,
        text_embeddings: Tensor,
        top_k: int = 5,
    ) -> Tuple[Tensor, Tensor]:
        return self.retrieve_images(text_embeddings, image_embeddings, top_k)


class MultiModalProjector(nn.Module):
    """Multi-modal projector for combining different modalities.
    
    Args:
        modality_dims: Dictionary of modality names to dimensions
        output_dim: Output dimension
        hidden_dim: Hidden dimension
        num_layers: Number of projection layers
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        output_dim: int = 256,
        hidden_dim: int = 1024,
        num_layers: int = 3,
    ):
        super().__init__()
        self.modality_dims = modality_dims
        self.output_dim = output_dim
        
        self.modality_encoders = nn.ModuleDict()
        
        for modality, dim in modality_dims.items():
            if dim != output_dim:
                layers = []
                in_dim = dim
                for i in range(num_layers - 1):
                    layers.extend([
                        nn.Linear(in_dim, hidden_dim),
                        nn.GELU(),
                        nn.BatchNorm1d(hidden_dim),
                    ])
                    in_dim = hidden_dim
                layers.append(nn.Linear(in_dim, output_dim))
                self.modality_encoders[modality] = nn.Sequential(*layers)
            else:
                self.modality_encoders[modality] = nn.Identity()
                
    def forward(
        self,
        features: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        outputs = {}
        
        for modality, feat in features.items():
            if modality in self.modality_encoders:
                outputs[modality] = self.modality_encoders[modality](feat)
                
        return outputs


class ALIGNModel(nn.Module):
    """ALIGN: A Large-scale ImaGe and Noisy Text Embedding.
    
    Args:
        image_encoder: Image encoder
        text_encoder: Text encoder
        embed_dim: Embedding dimension
        temperature: Temperature
    """
    
    def __init__(
        self,
        image_encoder: Optional[nn.Module] = None,
        text_encoder: Optional[nn.Module] = None,
        embed_dim: int = 640,
        temperature: float = 0.1,
    ):
        super().__init__()
        
        self.image_encoder = image_encoder or CLIPImageEncoder(embed_dim=embed_dim)
        self.text_encoder = text_encoder or CLIPTextEncoder(embed_dim=embed_dim)
        
        self.image_projection = nn.Identity()
        self.text_projection = nn.Identity()
        
        self.temperature = temperature
        
    def forward(
        self,
        images: Tensor,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(input_ids, attention_mask)
        
        image_proj = self.image_projection(image_features)
        text_proj = self.text_projection(text_features)
        
        loss = self._align_loss(image_proj, text_proj)
        
        return loss, {
            "image_features": image_proj,
            "text_features": text_proj,
        }
        
    def _align_loss(
        self,
        image_features: Tensor,
        text_features: Tensor,
    ) -> Tensor:
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        logits = (image_features @ text_features.T) / self.temperature
        
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=image_features.device)
        
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        
        return (loss_i + loss_t) / 2
