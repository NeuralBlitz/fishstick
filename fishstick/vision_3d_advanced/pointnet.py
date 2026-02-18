import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TNet(nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        identity = (
            torch.eye(self.k, device=x.device)
            .view(1, self.k * self.k)
            .repeat(batch_size, 1)
        )
        transform = x + identity
        return transform.view(batch_size, self.k, self.k)


class PointNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 40,
        input_transform: bool = True,
        feature_transform: bool = True,
    ):
        super().__init__()
        self.input_transform = input_transform
        self.feature_transform = feature_transform

        if input_transform:
            self.stn = TNet(k=3)

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        if feature_transform:
            self.fstn = TNet(k=64)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(256, num_classes)

        self.bn_fc1 = nn.BatchNorm1d(512)
        self.bn_fc2 = nn.BatchNorm1d(256)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        num_points = x.size(2)

        if self.input_transform:
            transform = self.stn(x)
            x = torch.bmm(transform, x)
        else:
            transform = None

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        if self.feature_transform:
            transform_feat = self.fstn(x)
            x = torch.bmm(transform_feat, x)
        else:
            transform_feat = None

        point_feat = x
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=-1)

        return x, transform, transform_feat


class PointNetPlusPlusSampler(nn.Module):
    def __init__(self, num_groups: int, num_points: int):
        super().__init__()
        self.num_groups = num_groups
        self.num_points = num_points
        self.conv1 = nn.Conv2d(3, 64, 1)
        self.conv2 = nn.Conv2d(64, 128, 1)
        self.conv3 = nn.Conv2d(128, 256, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)

    def forward(
        self, xyz: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_points, _ = xyz.shape
        xyz = xyz.permute(0, 2, 1).unsqueeze(-1)
        x = F.relu(self.bn1(self.conv1(xyz)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, idx = torch.max(x, 2, keepdim=True)
        x = x.squeeze(2)
        return x, idx, None


class PointNetPlusPlusSetAbstraction(nn.Module):
    def __init__(
        self,
        npoint: int,
        radius: float,
        nsample: int,
        in_channel: int,
        mlp: list,
        group_all: bool = False,
    ):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(
        self, xyz: torch.Tensor, points: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        new_xyz, new_points = self.sample_and_group(
            self.npoint, self.radius, self.nsample, xyz, points
        )

        new_points = new_points.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

    def sample_and_group(
        self,
        npoint: int,
        radius: float,
        nsample: int,
        xyz: torch.Tensor,
        points: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_points, _ = xyz.shape
        if npoint is None:
            new_xyz = xyz
            new_points = points.permute(0, 2, 1).unsqueeze(2)
            new_points = new_points.repeat(1, 1, num_points, 1)
        else:
            new_xyz, idx = self.furthest_point_sample(xyz, npoint)
            new_points = self.query_group(npoint, radius, nsample, xyz, new_xyz, points)
        return new_xyz, new_points

    def furthest_point_sample(
        self, xyz: torch.Tensor, npoint: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = xyz.device
        batch_size, num_points, _ = xyz.shape
        centroids = torch.zeros(batch_size, npoint, dtype=torch.long, device=device)
        distance = torch.ones(batch_size, num_points, device=device) * 1e10
        farthest = torch.randint(
            0, num_points, (batch_size,), dtype=torch.long, device=device
        )
        batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)

        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(batch_size, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            distance = torch.min(distance, dist)
            farthest = torch.max(distance, -1)[1]
        return centroids, None

    def query_group(
        self,
        npoint: int,
        radius: float,
        nsample: int,
        xyz: torch.Tensor,
        new_xyz: torch.Tensor,
        points: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, num_points, _ = xyz.shape
        idx = self.ball_query(radius, nsample, xyz, new_xyz)
        device = xyz.device
        batch_indices = (
            torch.arange(batch_size, device=device)
            .view(-1, 1, 1)
            .repeat(1, npoint, nsample)
        )
        point_indices = idx

        if points is not None:
            grouped_xyz = xyz[batch_indices, point_indices, :]
            grouped_points = points[batch_indices, point_indices, :]
            grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
        else:
            grouped_xyz = xyz[batch_indices, point_indices, :]
            grouped_points = grouped_xyz

        return grouped_points

    def ball_query(
        self, radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor
    ) -> torch.Tensor:
        batch_size, num_points, _ = xyz.shape
        _, npoint, _ = new_xyz.shape
        idx = torch.randint(
            0, num_points, (batch_size, npoint, nsample), device=xyz.device
        )
        return idx


class PointNetPlusPlus(nn.Module):
    def __init__(
        self,
        num_classes: int = 40,
        num_points: int = 1024,
        use_normals: bool = False,
        use_xyz: bool = True,
    ):
        super().__init__()
        self.num_points = num_points
        in_channel = 3 if use_xyz else 0

        self.sa1 = PointNetPlusPlusSetAbstraction(
            npoint=512,
            radius=0.2,
            nsample=32,
            in_channel=in_channel + 3,
            mlp=[64, 64, 128],
        )
        self.sa2 = PointNetPlusPlusSetAbstraction(
            npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256]
        )
        self.sa3 = PointNetPlusPlusSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=256 + 3,
            mlp=[256, 512, 1024],
        )

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        l1_xyz, l1_points = self.sa1(x)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        x = l3_points.view(l3_points.size(0), -1)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=-1)


class DGCNNConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k: int = 20):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_points, _ = x.shape
        x = x.view(batch_size, num_points, -1)
        pairwise_distance = self.get_pairwise_distance(x)
        idx = pairwise_distance.topk(k=self.k, dim=-1)[1]
        idx = idx.view(batch_size, num_points, self.k)

        neighbors = self.get_neighbors(x, idx)
        x = x.view(batch_size, num_points, 1, -1).repeat(1, 1, self.k, 1)
        x = torch.cat([x, neighbors], dim=-1)
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.max(dim=-1)[0]
        return x

    def get_pairwise_distance(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_points, _ = x.shape
        xx = torch.sum(x**2, dim=2, keepdim=True)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2 * torch.bmm(x, x.permute(0, 2, 1))
        return dist

    def get_neighbors(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        batch_size, num_points, _ = x.shape
        k = idx.shape[-1]
        device = x.device
        idx_base = (
            torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        )
        idx = idx + idx_base
        idx = idx.view(-1)
        x = x.view(batch_size * num_points, -1)
        neighbors = x[idx, :].view(batch_size, num_points, k, -1)
        return neighbors


class DGCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 40,
        k: int = 20,
        emb_dims: int = 1024,
    ):
        super().__init__()
        self.k = k
        self.emb_dims = emb_dims

        self.conv1 = DGCNNConv(3, 64, k)
        self.conv2 = DGCNNConv(64, 64, k)
        self.conv3 = DGCNNConv(64, 128, k)
        self.conv4 = DGCNNConv(128, 256, k)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, emb_dims, 1),
            nn.BatchNorm2d(emb_dims),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.linear1 = nn.Linear(emb_dims, 512)
        self.bn6 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = x.permute(0, 2, 1).unsqueeze(-1)

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.conv5(x)

        x = x.view(batch_size, -1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.drop1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.drop2(x)
        x = self.linear3(x)
        return F.log_softmax(x, dim=-1)


class PointConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        nsample: int = 32,
    ):
        super().__init__()
        self.nsample = nsample
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, xyz: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.nsample

        new_xyz, grouped_points = self.sample_and_group(xyz, points, S)
        new_xyz = new_xyz.permute(0, 3, 2, 1)
        grouped_points = grouped_points.permute(0, 3, 2, 1)

        grouped_points_norm = grouped_points - new_xyz
        grouped_points = torch.cat([grouped_points_norm, new_xyz], dim=1)

        grouped_points = self.conv(grouped_points)
        grouped_points = torch.max(grouped_points, 2)[0]
        return grouped_points.permute(0, 2, 1)

    def sample_and_group(self, xyz: torch.Tensor, points: torch.Tensor, nsample: int):
        batch_size, num_points, _ = xyz.shape
        idx = torch.randint(
            0, num_points, (batch_size, num_points, nsample), device=xyz.device
        )
        neighbors = self.get_neighbors(xyz, idx)

        new_xyz = xyz.unsqueeze(2).repeat(1, 1, nsample, 1)
        if points is not None:
            grouped_points = points.unsqueeze(2).repeat(1, 1, nsample, 1)
            grouped_points = torch.cat([grouped_points, neighbors - new_xyz], dim=-1)
        else:
            grouped_points = neighbors - new_xyz

        return new_xyz, grouped_points

    def get_neighbors(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        batch_size, num_points, _ = x.shape
        k = idx.shape[-1]
        device = x.device
        idx_base = (
            torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        )
        idx = idx + idx_base
        idx = idx.view(-1)
        x = x.view(batch_size * num_points, -1)
        neighbors = x[idx, :].view(batch_size, num_points, k, -1)
        return neighbors


class PointConvDensity(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, nsample: int = 32):
        super().__init__()
        self.nsample = nsample
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2 + 1, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, xyz: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.nsample

        new_xyz, grouped_points, density = self.sample_and_group_density(xyz, points, S)

        new_xyz = new_xyz.permute(0, 3, 2, 1)
        grouped_points = grouped_points.permute(0, 3, 2, 1)
        density = density.permute(0, 3, 2, 1)

        grouped_points_norm = grouped_points - new_xyz
        grouped_points = torch.cat([grouped_points_norm, new_xyz, density], dim=1)

        grouped_points = self.conv(grouped_points)
        grouped_points = torch.max(grouped_points, 2)[0]
        return grouped_points.permute(0, 2, 1)

    def sample_and_group_density(
        self, xyz: torch.Tensor, points: torch.Tensor, nsample: int
    ):
        batch_size, num_points, _ = xyz.shape
        idx = torch.randint(
            0, num_points, (batch_size, num_points, nsample), device=xyz.device
        )
        neighbors = self.get_neighbors(xyz, idx)

        new_xyz = xyz.unsqueeze(2).repeat(1, 1, nsample, 1)
        grouped_points = points.unsqueeze(2).repeat(1, 1, nsample, 1)
        grouped_points = torch.cat([grouped_points, neighbors - new_xyz], dim=-1)

        dist = torch.norm(neighbors - new_xyz, dim=-1, keepdim=True)
        density = 1.0 / (dist + 1e-8)

        return new_xyz, grouped_points, density

    def get_neighbors(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        batch_size, num_points, _ = x.shape
        k = idx.shape[-1]
        device = x.device
        idx_base = (
            torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        )
        idx = idx + idx_base
        idx = idx.view(-1)
        x = x.view(batch_size * num_points, -1)
        neighbors = x[idx, :].view(batch_size, num_points, k, -1)
        return neighbors


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel: int, mlp: list):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(
        self,
        xyz1: torch.Tensor,
        xyz2: torch.Tensor,
        points1: torch.Tensor,
        points2: torch.Tensor,
    ) -> torch.Tensor:
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points1.repeat(1, N, 1)
        else:
            dists = self.square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_points = torch.sum(
                index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2
            )

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        return new_points

    def square_distance(self, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        B, N, _ = src.shape
        _, M, _ = dst.shape
        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
        dist += torch.sum(src**2, -1).view(B, N, 1)
        dist += torch.sum(dst**2, -1).view(B, 1, M)
        return dist


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long, device=device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    new_points = points[batch_indices, idx, :]
    return new_points


class PointConv(nn.Module):
    def __init__(
        self,
        num_classes: int = 40,
        num_points: int = 1024,
    ):
        super().__init__()
        self.num_points = num_points

        self.sa1 = PointConvDensity(3, 64, 32)
        self.sa2 = PointConvDensity(64, 128, 32)
        self.sa3 = PointConvDensity(128, 256, 32)

        self.su1 = PointNetFeaturePropagation(128 + 256, [256, 256])
        self.su2 = PointNetFeaturePropagation(64 + 256, [256, 128])
        self.su3 = PointNetFeaturePropagation(128 + 3, [128, 128, 128])

        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        l1_xyz = x
        l1_points = self.sa1(l1_xyz, None)
        l2_xyz = l1_points.permute(0, 2, 1)
        l2_points = self.sa2(l2_xyz, l1_points)
        l3_xyz = l2_points.permute(0, 2, 1)
        l3_points = self.sa3(l3_xyz, l2_points)

        l2_points_interp = self.su1(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points_interp = self.su2(l1_xyz, l2_xyz, l1_points, l2_points_interp)
        x = self.su3(l1_xyz, l1_xyz, None, l1_points_interp)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop1(x)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        return x
