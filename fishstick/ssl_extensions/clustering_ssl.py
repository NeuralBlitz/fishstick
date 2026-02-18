"""
Clustering-Based Self-Supervised Learning

Extended implementations:
- DeepCluster
- SwAV (online clustering)
- PCL (Prototypical Contrastive Learning)
- SCAN (Semantic Clustering by Adaptive Neighbors)
"""

from typing import Optional, Tuple, Dict, Any, List, Callable
import copy
import random

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.cluster import kmeans
import numpy as np

from fishstick.ssl_extensions.base import (
    MemoryBank,
    stop_gradient,
    gather_from_all,
    L2Normalize,
)


class DeepCluster(nn.Module):
    """DeepCluster: Deep Clustering for Unsupervised Learning.
    
    Args:
        encoder: Backbone encoder
        num_clusters: Number of clusters
        projection_dim: Projection dimension
        hidden_dim: Hidden dimension
        memory_bank_size: Size of feature memory bank
        warmup_epochs: Number of warmup epochs
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        num_clusters: int = 10000,
        projection_dim: int = 256,
        hidden_dim: int = 2048,
        memory_bank_size: int = 65536,
        warmup_epochs: int = 1,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_clusters = num_clusters
        self.warmup_epochs = warmup_epochs
        
        encoder_out_dim = self._get_encoder_dim(encoder)
        
        self.projector = nn.Sequential(
            nn.Linear(encoder_out_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )
        
        self.cluster_centers = None
        self.memory_bank = MemoryBank(
            size=memory_bank_size,
            dim=projection_dim,
        )
        
    def _get_encoder_dim(self, encoder: nn.Module) -> int:
        if hasattr(encoder, "output_dim"):
            return encoder.output_dim
        elif hasattr(encoder, "embed_dim"):
            return encoder.embed_dim
        else:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                return encoder(dummy).flatten(1).shape[1]
                
    def forward(
        self, 
        x: Tensor, 
        cluster_labels: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        features = self.encoder(x)
        projections = self.projector(features)
        
        projections = F.normalize(projections, dim=-1)
        
        self.memory_bank.update(projections.detach())
        
        if cluster_labels is not None and self.cluster_centers is not None:
            loss = self._cluster_loss(projections, cluster_labels)
            return loss, cluster_labels
            
        return projections, None
        
    def _cluster_loss(self, projections: Tensor, cluster_labels: Tensor) -> Tensor:
        normalized_centers = F.normalize(self.cluster_centers, dim=-1)
        
        similarities = projections @ normalized_centers.T
        
        loss = F.cross_entropy(similarities, cluster_labels)
        
        return loss
    
    def update_clusters(self, features: Tensor, batch_size: int = 256):
        """Update cluster assignments using k-means.
        
        Args:
            features: Features to cluster
            batch_size: Batch size for clustering
        """
        all_features = []
        
        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch = features[i:i + batch_size]
                feats = self.projector(self.encoder(batch))
                feats = F.normalize(feats, dim=-1)
                all_features.append(feats)
                
        all_features = torch.cat(all_features, dim=0)
        
        if all_features.device.type != 'cpu':
            all_features = all_features.cpu()
            
        cluster_ids, cluster_centers = kmeans(
            all_features.numpy(), 
            self.num_clusters, 
            distance='cosine',
            iter=100,
        )
        
        self.cluster_centers = torch.from_numpy(cluster_centers).float()
        
        return torch.from_numpy(cluster_ids).long()
    

class SwAV(nn.Module):
    """SwAV: Unsupervised Learning of Visual Features by Contrasting Cluster Assignments.
    
    Args:
        encoder: Backbone encoder
        projection_dim: Projection dimension
        hidden_dim: Hidden dimension
        num_prototypes: Number of prototypes
        queue_size: Size of memory queue
        temperature: Temperature for softmax
        epsilon: Epsilon for sinkhorn
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        projection_dim: int = 256,
        hidden_dim: int = 2048,
        num_prototypes: int = 3000,
        queue_size: int = 4096,
        temperature: float = 0.1,
        epsilon: float = 0.05,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_prototypes = num_prototypes
        self.temperature = temperature
        self.epsilon = epsilon
        
        encoder_out_dim = self._get_encoder_dim(encoder)
        
        self.projector = nn.Sequential(
            nn.Linear(encoder_out_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )
        
        self.prototypes = nn.Linear(projection_dim, num_prototypes, bias=False)
        
        self.queue = torch.zeros(queue_size, projection_dim)
        self.queue_ptr = 0
        
    def _get_encoder_dim(self, encoder: nn.Module) -> int:
        if hasattr(encoder, "output_dim"):
            return encoder.output_dim
        elif hasattr(encoder, "embed_dim"):
            return encoder.embed_dim
        else:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                return encoder(dummy).flatten(1).shape[1]
                
    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))
        
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        
        p1 = self.prototypes(z1)
        p2 = self.prototypes(z2)
        
        with torch.no_grad():
            c1 = self._sinkhorn(p1)
            c2 = self._sinkhorn(p2)
            
        loss = self._swav_loss(p1, p2, c2, c1) + self._swav_loss(p2, p1, c1, c2)
        
        return loss / 2
        
    def _sinkhorn(self, Q: Tensor) -> Tensor:
        K = Q.shape[-1]
        Q = Q / self.epsilon
        
        for _ in range(3):
            sum_Q = Q.sum(dim=-1, keepdim=True)
            Q = Q / sum_Q
            
            sum_K = Q.sum(dim=-2, keepdim=True)
            Q = (Q / sum_K) * K
            
        return Q
        
    def _swav_loss(
        self, 
        p: Tensor, 
        z: Tensor, 
        codes: Tensor, 
        target_codes: Tensor
    ) -> Tensor:
        loss = -(target_codes * torch.log_softmax(p / self.temperature, dim=-1)).sum(dim=-1)
        
        return loss.mean()
        
    def get_embeddings(self, x: Tensor) -> Tensor:
        return F.normalize(self.projector(self.encoder(x)), dim=-1)


class PrototypicalContrastive(nn.Module):
    """PCL: Prototypical Contrastive Learning.
    
    Args:
        encoder: Backbone encoder
        num_protos: Number of prototypes
        projection_dim: Projection dimension
        hidden_dim: Hidden dimension
        temperature: Temperature for contrastive loss
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        num_protos: int = 4096,
        projection_dim: int = 128,
        hidden_dim: int = 2048,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_protos = num_protos
        self.temperature = temperature
        
        encoder_out_dim = self._get_encoder_dim(encoder)
        
        self.projector = nn.Sequential(
            nn.Linear(encoder_out_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )
        
        self.prototype_vectors = nn.Parameter(torch.zeros(num_protos, projection_dim))
        nn.init.uniform_(self.prototype_vectors, -1, 1)
        
    def _get_encoder_dim(self, encoder: nn.Module) -> int:
        if hasattr(encoder, "output_dim"):
            return encoder.output_dim
        elif hasattr(encoder, "embed_dim"):
            return encoder.embed_dim
        else:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                return encoder(dummy).flatten(1).shape[1]
                
    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))
        
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        
        proto_dist1 = self._compute_proto_distances(z1)
        proto_dist2 = self._compute_proto_distances(z2)
        
        loss = self._pcl_loss(proto_dist1, z2) + self._pcl_loss(proto_dist2, z1)
        
        return loss / 2
        
    def _compute_proto_distances(self, z: Tensor) -> Tensor:
        protos = F.normalize(self.prototype_vectors, dim=-1)
        
        dists = 2 - 2 * (z @ protos.T)
        
        return torch.exp(-dists / self.temperature)
        
    def _pcl_loss(self, proto_dists: Tensor, target_z: Tensor) -> Tensor:
        loss = -torch.log(proto_dists.diag() / proto_dists.sum(dim=-1))
        
        return loss.mean()
        
    def update_prototypes(self, features: Tensor, batch_size: int = 256):
        """Update prototype vectors using clustering.
        
        Args:
            features: Features to use for clustering
            batch_size: Batch size
        """
        all_features = []
        
        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch = features[i:i + batch_size]
                feats = self.projector(self.encoder(batch))
                feats = F.normalize(feats, dim=-1)
                all_features.append(feats)
                
        all_features = torch.cat(all_features, dim=0)
        
        for i in range(self.num_protos):
            mask = (torch.rand(len(all_features)) < 0.1)
            if mask.sum() > 0:
                self.prototype_vectors.data[i] = all_features[mask].mean(dim=0)
        
    def get_embeddings(self, x: Tensor) -> Tensor:
        return F.normalize(self.projector(self.encoder(x)), dim=-1)


class SCANLoss(nn.Module):
    """SCAN: Semantic Clustering by Adaptive Neighbors loss.
    
    Args:
        entropy_weight: Weight for entropy term
        confidence_threshold: Threshold for confidence
    """
    
    def __init__(
        self,
        entropy_weight: float = 0.1,
        confidence_threshold: float = 0.0,
    ):
        super().__init__()
        self.entropy_weight = entropy_weight
        self.confidence_threshold = confidence_threshold
        
    def forward(
        self, 
        features: Tensor, 
        cluster_probs: Tensor
    ) -> Tensor:
        cluster_probs = F.softmax(cluster_probs, dim=-1)
        
        entropy = -(cluster_probs * torch.log(cluster_probs + 1e-10)).sum(dim=-1)
        
        max_probs, _ = cluster_probs.max(dim=-1)
        mask = max_probs > self.confidence_threshold
        
        if mask.sum() == 0:
            return torch.tensor(0.0, device=features.device)
            
        loss = (entropy[mask] * self.entropy_weight).mean()
        
        return loss


class SCAN(nn.Module):
    """SCAN: Semantic Clustering by Adaptive Neighbors.
    
    Args:
        encoder: Backbone encoder
        num_clusters: Number of clusters
        projection_dim: Projection dimension
        hidden_dim: Hidden dimension
        temperature: Temperature
        entropy_weight: Weight for entropy term
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        num_clusters: int = 1000,
        projection_dim: int = 256,
        hidden_dim: int = 2048,
        temperature: float = 0.1,
        entropy_weight: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_clusters = num_clusters
        
        encoder_out_dim = self._get_encoder_dim(encoder)
        
        self.projector = nn.Sequential(
            nn.Linear(encoder_out_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )
        
        self.cluster_head = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_clusters),
        )
        
        self.loss_fn = SCANLoss(entropy_weight=entropy_weight)
        self.temperature = temperature
        
    def _get_encoder_dim(self, encoder: nn.Module) -> int:
        if hasattr(encoder, "output_dim"):
            return encoder.output_dim
        elif hasattr(encoder, "embed_dim"):
            return encoder.embed_dim
        else:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                return encoder(dummy).flatten(1).shape[1]
                
    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))
        
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        
        c1 = self.cluster_head(z1)
        c2 = self.cluster_head(z2)
        
        loss = self._scan_loss(z1, z2, c1, c2)
        
        return loss
        
    def _scan_loss(
        self,
        z1: Tensor,
        z2: Tensor,
        c1: Tensor,
        c2: Tensor,
    ) -> Tensor:
        c1 = F.softmax(c1 / self.temperature, dim=-1)
        c2 = F.softmax(c2 / self.temperature, dim=-1)
        
        cluster_consistency = -(
            torch.log((c1 * c2).sum(dim=-1) + 1e-10)
        ).mean()
        
        entropy_loss = (
            self.loss_fn(z1, c1) + self.loss_fn(z2, c2)
        ) / 2
        
        return cluster_consistency + entropy_loss
        
    def get_embeddings(self, x: Tensor) -> Tensor:
        return F.normalize(self.projector(self.encoder(x)), dim=-1)
    
    def get_cluster_probs(self, x: Tensor) -> Tensor:
        z = self.projector(self.encoder(x))
        z = F.normalize(z, dim=-1)
        c = self.cluster_head(z)
        return F.softmax(c / self.temperature, dim=-1)


class OnlineKMeans(nn.Module):
    """Online K-Means clustering for continuous prototype updates.
    
    Args:
        num_prototypes: Number of prototypes
        feature_dim: Feature dimension
        momentum: Momentum for prototype updates
        epsilon: Epsilon for numerical stability
    """
    
    def __init__(
        self,
        num_prototypes: int,
        feature_dim: int,
        momentum: float = 0.1,
        epsilon: float = 1e-6,
    ):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.feature_dim = feature_dim
        self.momentum = momentum
        self.epsilon = epsilon
        
        self.register_buffer(
            'prototypes', 
            torch.randn(num_prototypes, feature_dim) * 0.02
        )
        
    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        features = F.normalize(features, dim=-1)
        
        prototypes = F.normalize(self.prototypes, dim=-1)
        
        similarities = features @ prototypes.T
        
        cluster_ids = similarities.argmax(dim=-1)
        
        return cluster_ids, similarities
        
    def update_prototypes(self, features: Tensor, cluster_ids: Tensor):
        """Update prototypes based on cluster assignments.
        
        Args:
            features: Feature embeddings
            cluster_ids: Cluster assignments
        """
        features = F.normalize(features, dim=-1)
        
        for i in range(self.num_prototypes):
            mask = cluster_ids == i
            if mask.sum() > 0:
                prototype_mean = features[mask].mean(dim=0)
                self.prototypes.data[i] = (
                    self.momentum * prototype_mean + 
                    (1 - self.momentum) * self.prototypes.data[i]
                )
