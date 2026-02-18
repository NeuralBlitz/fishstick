"""
PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation

Implementation of PointNet architecture for point cloud processing including:
- TNet: Spatial transformer network
- PointNet encoder for feature extraction
- Classification and segmentation heads
"""

from typing import Tuple, Optional, List
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math


class TNet(nn.Module):
    """
    TNet: Spatial Transformer Network.

    Learns a transformation matrix to align point clouds or features.
    Can operate on 3D (xyz) or 64D feature space.
    """

    def __init__(self, k: int = 3):
        super().__init__()
        self.k = k

        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, k, N] or [B, k]

        Returns:
            Transformation matrix [B, k, k]
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        batch_size = x.size(0)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        identity = torch.eye(self.k, dtype=torch.float32, device=x.device)
        identity = identity.view(1, self.k, self.k).expand(batch_size, -1, -1)

        return x.view(batch_size, self.k, self.k) + identity


class FeatureTransformer(nn.Module):
    """
    Feature Transformer for attention-based feature alignment.
    """

    def __init__(self, in_channels: int, dim: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.dim = dim

        self.query = nn.Linear(in_channels, dim)
        self.key = nn.Linear(in_channels, dim)
        self.value = nn.Linear(in_channels, dim)

        self.out_proj = nn.Linear(dim, in_channels)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Features [B, N, C]

        Returns:
            Transformed features [B, N, C]
        """
        B, N, C = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attn = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.dim)
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(attn, v)
        out = self.out_proj(out)

        return out + x


class PointNetEncoder(nn.Module):
    """
    PointNet Encoder.

    Extracts global and local features from point clouds.
    """

    def __init__(
        self,
        global_feat: bool = True,
        feature_transform: bool = False,
        channel: int = 3,
    ):
        super().__init__()
        self.global_feat = global_feat
        self.feature_transform = feature_transform

        self.stn = TNet(k=channel)

        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        if feature_transform:
            self.fstn = TNet(k=64)

    def forward(
        self,
        x: Tensor,
        return_trans: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass.

        Args:
            x: Point cloud [B, C, N]
            return_trans: Whether to return transformation matrix

        Returns:
            global_feat: Global features [B, 1024]
            trans: Transformation matrix (if return_trans=True)
        """
        B, D, N = x.size()

        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)

        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        point_feat = x

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        x = torch.max(x, 2)[0]
        global_feat = x

        if self.global_feat:
            if return_trans:
                return global_feat, trans
            return global_feat

        x = global_feat.view(B, 1024, 1).repeat(1, 1, N)
        return torch.cat([point_feat, x], 1)


class PointNetCls(nn.Module):
    """
    PointNet Classification Network.
    """

    def __init__(
        self,
        num_classes: int = 40,
        input_transform: bool = True,
        feature_transform: bool = True,
    ):
        super().__init__()
        self.feat = PointNetEncoder(
            global_feat=False,
            feature_transform=feature_transform,
            channel=3 if input_transform else 3,
        )

        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, num_classes, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Point cloud [B, 3, N]

        Returns:
            Class logits [B, num_classes]
        """
        x, trans = self.feat(x, return_trans=True)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.conv3(x)

        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(x.size(0), -1), dim=-1)
        x = x.view(x.size(0), -1)

        return x


class PointNetSeg(nn.Module):
    """
    PointNet Segmentation Network.
    """

    def __init__(
        self,
        num_classes: int = 50,
        input_transform: bool = True,
        feature_transform: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.feat = PointNetEncoder(
            global_feat=False,
            feature_transform=feature_transform,
        )

        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, num_classes, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Point cloud [B, 3, N]

        Returns:
            Per-point class logits [B, num_classes, N]
        """
        x, _ = self.feat(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)

        return x


class PointNetDenseCls(nn.Module):
    """
    PointNet for dense prediction (part segmentation).
    """

    def __init__(
        self,
        num_classes: int = 50,
        feature_transform: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.feat = PointNetEncoder(
            global_feat=False,
            feature_transform=feature_transform,
        )

        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, num_classes, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: Point cloud [B, 3, N]

        Returns:
            pred: Per-point logits [B, num_classes, N]
            trans: Feature transform matrix (for regularization)
        """
        x, trans = self.feat(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)

        return x, trans


def feature_transform_regularizer(trans: Tensor) -> Tensor:
    """
    Regularization loss for feature transformation matrix.

    Encourages orthogonality of the transformation matrix.

    Args:
        trans: Feature transform [B, K, K]

    Returns:
        Loss value
    """
    d = trans.size()[1]
    I = torch.eye(d, device=trans.device)[None, :, :]
    loss = torch.mean(
        torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2))
    )
    return loss
