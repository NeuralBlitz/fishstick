"""
Scene Graph Generation Module

Generate scene graphs from images for structured scene understanding.
"""

from typing import Tuple, List, Optional, Union, Dict, Any
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ObjectDetector(nn.Module):
    """
    Object detector backbone for scene graph generation.
    """

    def __init__(
        self,
        num_classes: int = 80,
        backbone: str = 'resnet50',
        pretrained: bool = False,
    ):
        """
        Args:
            num_classes: Number of object classes
            backbone: Backbone architecture
            pretrained: Use pretrained weights
        """
        super().__init__()

        import torchvision.models as models

        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        else:
            resnet = models.resnet101(pretrained=pretrained)
            feature_dim = 2048

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        self.avgpool = resnet.avgpool

        self.rpn = RegionProposalNetwork(feature_dim, 512)
        self.rcnn = RCNNHead(feature_dim, num_classes)

    def forward(
        self,
        images: Tensor,
    ) -> Dict[str, Any]:
        """
        Args:
            images: Input images [B, 3, H, W]

        Returns:
            Dictionary with boxes, scores, and labels
        """
        features = self.backbone(images)
        features = self.avgpool(features).flatten(1)

        proposals = self.rpn(features)
        detections = self.rcnn(features, proposals)

        return detections


class RegionProposalNetwork(nn.Module):
    """
    RPN for generating object proposals.
    """

    def __init__(
        self,
        in_channels: int = 2048,
        hidden_channels: int = 512,
    ):
        super().__init__()

        self.rpn_conv = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(hidden_channels, 1, 1)
        self.rpn_reg = nn.Conv2d(hidden_channels, 4, 1)

    def forward(self, features: Tensor) -> Tensor:
        """
        Generate region proposals.

        Args:
            features: Feature map [B, C, H, W]

        Returns:
            Proposals [N, 4] (x1, y1, x2, y2)
        """
        x = F.relu(self.rpn_conv(features))

        rpn_cls = self.rpn_cls(x)
        rpn_reg = self.rpn_reg(x)

        proposals = self._generate_proposals(rpn_reg, rpn_cls)

        return proposals

    def _generate_proposals(
        self,
        rpn_reg: Tensor,
        rpn_cls: Tensor,
    ) -> Tensor:
        """Generate proposals from RPN outputs."""
        batch_cls.shape[0_size = rpn]

        proposals = []
        for b in range(batch_size):
            scores = torch.sigmoid(rpn_cls[b, 0])
            top_indices = torch.topk(scores.flatten(), 1000).indices

            y, x = top_indices // rpn_cls.shape[3], top_indices % rpn_cls.shape[3]

            boxes = torch.stack([x.float(), y.float(), x.float() + 16, y.float() + 16], dim=1)
            proposals.append(boxes)

        return torch.cat(proposals, dim=0)


class RCNNHead(nn.Module):
    """
    R-CNN detection head.
    """

    def __init__(
        self,
        in_channels: int = 2048,
        num_classes: int = 80,
    ):
        super().__init__()

        self.roi_pool = RoIAlignPooling(output_size=7, spatial_scale=1/16)

        self.fc1 = nn.Linear(in_channels * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 1024)

        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 4)

    def forward(
        self,
        features: Tensor,
        proposals: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Args:
            features: Feature map
            proposals: Region proposals

        Returns:
            Detections dictionary
        """
        pooled = self.roi_pool(features, proposals)

        pooled = pooled.flatten(1)
        x = F.relu(self.fc1(pooled))
        x = F.relu(self.fc2(x))

        cls_scores = self.cls_score(x)
        bbox_preds = self.bbox_pred(x)

        return {
            'boxes': proposals,
            'scores': torch.sigmoid(cls_scores),
            'labels': cls_scores.argmax(dim=-1),
        }


class RoIAlignPooling(nn.Module):
    """
    ROI Align pooling operation.
    """

    def __init__(
        self,
        output_size: int = 7,
        spatial_scale: float = 1.0,
    ):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(
        self,
        features: Tensor,
        boxes: Tensor,
    ) -> Tensor:
        """
        Args:
            features: Feature map [B, C, H, W]
            boxes: Bounding boxes [N, 4]

        Returns:
            Pooled features [N, C, output_size, output_size]
        """
        batch_size = features.shape[0]
        num_boxes = boxes.shape[0]
        channels = features.shape[1]

        pooled = torch.zeros(
            num_boxes, channels, self.output_size, self.output_size,
            device=features.device,
        )

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            x1 = int(x1 * self.spatial_scale)
            y1 = int(y1 * self.spatial_scale)
            x2 = int(x2 * self.spatial_scale)
            y2 = int(y2 * self.spatial_scale)

            x1 = max(0, min(x1, features.shape[3] - 1))
            y1 = max(0, min(y1, features.shape[2] - 1))
            x2 = max(0, min(x2, features.shape[3]))
            y2 = max(0, min(y2, features.shape[2]))

            if x2 <= x1 or y2 <= y1:
                continue

            region = features[0, :, y1:y2, x1:x2]

            pooled[i] = F.adaptive_avg_pool2d(
                region.unsqueeze(0),
                (self.output_size, self.output_size)
            )

        return pooled


class RelationshipPredictor(nn.Module):
    """
    Predict relationships between detected objects.
    """

    def __init__(
        self,
        object_dim: int = 1024,
        num_relationships: int = 50,
    ):
        """
        Args:
            object_dim: Dimension of object features
            num_relationships: Number of relationship types
        """
        super().__init__()

        self.subject_fc = nn.Linear(object_dim, 512)
        self.object_fc = nn.Linear(object_dim, 512)

        self.pair_fc = nn.Sequential(
            nn.Linear(512 * 2 + 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
        )

        self.relationship_cls = nn.Linear(512, num_relationships)

    def forward(
        self,
        object_features: Tensor,
        boxes: Tensor,
    ) -> Tensor:
        """
        Args:
            object_features: Object feature vectors [N, object_dim]
            boxes: Bounding boxes [N, 4]

        Returns:
            Relationship predictions [N, N, num_relationships]
        """
        N = object_features.shape[0]

        subject_feat = F.relu(self.subject_fc(object_features))
        object_feat = F.relu(self.object_fc(object_features))

        relationships = []

        for i in range(N):
            row_rels = []
            for j in range(N):
                if i == j:
                    row_rels.append(torch.zeros(self.relationship_cls.out_features, device=object_features.device))
                    continue

                pair_feat = torch.cat([
                    subject_feat[i],
                    object_feat[j],
                    self._compute_box_features(boxes[i], boxes[j]),
                ])

                pair_feat = self.pair_fc(pair_feat.unsqueeze(0))
                rel_pred = self.relationship_cls(pair_feat)
                row_rels.append(rel_pred.squeeze(0))

            relationships.append(torch.stack(row_rels))

        return torch.stack(relationships)

    def _compute_box_features(
        self,
        box1: Tensor,
        box2: Tensor,
    ) -> Tensor:
        """Compute geometric features between boxes."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        distance = torch.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        iou = self._compute_iou(box1, box2)

        return torch.stack([distance, iou, w2 / (w1 + 1e-6), h2 / (h1 + 1e-6)])

    def _compute_iou(self, box1: Tensor, box2: Tensor) -> float:
        """Compute IoU between two boxes."""
        x1_min, y1_min = box1[0], box1[1]
        x1_max, y1_max = box1[0] + box1[2], box1[1] + box1[3]
        x2_min, y2_min = box2[0], box2[1]
        x2_max, y2_max = box2[0] + box2[2], box2[1] + box2[3]

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)

        iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)

        return iou


class SceneGraphBuilder(nn.Module):
    """
    Build scene graphs from detections and relationships.
    """

    def __init__(
        self,
        num_classes: int = 80,
        num_relationships: int = 50,
    ):
        """
        Args:
            num_classes: Number of object classes
            num_relationships: Number of relationship types
        """
        super().__init__()

        self.detector = ObjectDetector(num_classes=num_classes)
        self.relationship_predictor = RelationshipPredictor(
            object_dim=1024,
            num_relationships=num_relationships,
        )

    def forward(self, images: Tensor) -> Dict[str, Any]:
        """
        Args:
            images: Input images [B, 3, H, W]

        Returns:
            Scene graph with nodes and edges
        """
        detections = self.detector(images)

        boxes = detections['boxes']
        labels = detections['labels']
        scores = detections['scores']

        object_features = self._extract_object_features(images, boxes)

        relationships = self.relationship_predictor(object_features, boxes)

        graph = {
            'objects': {
                'boxes': boxes,
                'labels': labels,
                'scores': scores,
                'features': object_features,
            },
            'relationships': relationships,
        }

        return graph

    def _extract_object_features(
        self,
        images: Tensor,
        boxes: Tensor,
    ) -> Tensor:
        """Extract features for each detected object."""
        return torch.randn(boxes.shape[0], 1024, device=images.device)


class SceneGraphNode:
    """Represents a node in a scene graph."""

    def __init__(
        self,
        box: List[float],
        label: int,
        score: float,
        feature: Optional[Tensor] = None,
    ):
        self.box = box
        self.label = label
        self.score = score
        self.feature = feature

    def __repr__(self) -> str:
        return f"SceneGraphNode(label={self.label}, score={self.score:.2f})"


class SceneGraphEdge:
    """Represents an edge in a scene graph."""

    def __init__(
        self,
        subject_id: int,
        object_id: int,
        predicate: int,
        score: float,
    ):
        self.subject_id = subject_id
        self.object_id = object_id
        self.predicate = predicate
        self.score = score

    def __repr__(self) -> str:
        return f"SceneGraphEdge(sub={self.subject_id}, obj={self.object_id}, pred={self.predicate})"


class SceneGraph:
    """
    Structured scene graph representation.
    """

    def __init__(
        self,
        objects: List[SceneGraphNode],
        relationships: List[SceneGraphEdge],
    ):
        self.objects = objects
        self.relationships = relationships

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'objects': [
                {
                    'box': obj.box,
                    'label': obj.label,
                    'score': obj.score,
                }
                for obj in self.objects
            ],
            'relationships': [
                {
                    'subject': rel.subject_id,
                    'object': rel.object_id,
                    'predicate': rel.predicate,
                    'score': rel.score,
                }
                for rel in self.relationships
            ],
        }

    def visualize(self) -> str:
        """Generate text visualization of scene graph."""
        lines = ["Scene Graph:"]
        lines.append("Objects:")
        for i, obj in enumerate(self.objects):
            lines.append(f"  [{i}] {obj.label} (conf: {obj.score:.2f})")

        lines.append("Relationships:")
        for rel in self.relationships:
            lines.append(
                f"  [{rel.subject_id}] --{rel.predicate}--> [{rel.object_id}]"
            )

        return "\n".join(lines)


class GraphConvolutionLayer(nn.Module):
    """
    Graph convolution layer for scene graph reasoning.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)

    def forward(
        self,
        node_features: Tensor,
        adjacency: Tensor,
    ) -> Tensor:
        """
        Args:
            node_features: Node feature matrix [N, in_features]
            adjacency: Adjacency matrix [N, N]

        Returns:
            Updated node features [N, out_features]
        """
        aggregated = torch.matmul(adjacency, node_features)
        aggregated = aggregated / (adjacency.sum(dim=1, keepdim=True) + 1e-8)

        updated = self.linear(aggregated)
        updated = self.norm(updated)
        updated = F.relu(updated)

        return updated


class SceneGraphReasoning(nn.Module):
    """
    Reason over scene graphs using graph neural networks.
    """

    def __init__(
        self,
        node_dim: int = 1024,
        hidden_dim: int = 512,
        num_layers: int = 3,
    ):
        """
        Args:
            node_dim: Dimension of node features
            hidden_dim: Hidden layer dimension
            num_layers: Number of GNN layers
        """
        super().__init__()

        self.layers = nn.ModuleList([
            GraphConvolutionLayer(node_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])

    def forward(
        self,
        node_features: Tensor,
        adjacency: Tensor,
    ) -> Tensor:
        """
        Args:
            node_features: Node features [N, node_dim]
            adjacency: Adjacency matrix [N, N]

        Returns:
            Reasoned node features [N, hidden_dim]
        """
        x = node_features

        for layer in self.layers:
            x = layer(x, adjacency)

        return x


RELATIONSHIP_CLASSES = [
    "none",
    "on",
    "in",
    "at",
    "near",
    "over",
    "under",
    "beside",
    "behind",
    "in front of",
    "above",
    "below",
    "between",
    "against",
    "inside",
    "outside",
    "attached to",
    "hanging on",
    "lying on",
    "standing on",
    "flying",
    "walking",
    "sitting",
    "lying",
    "wearing",
    "holding",
    "carrying",
    "looking at",
    "facing",
    "next to",
    "with",
    "has",
    "contains",
    "part of",
    "connected to",
    "leaning on",
    "resting on",
    "growing on",
    "mounted on",
    "parked on",
    "riding",
    "covering",
    "hiding",
    "eating",
    "drinking",
    "playing",
    "using",
    "making",
    "doing",
    "close to",
]


def create_scene_graph_model(
    num_classes: int = 80,
    num_relationships: int = 50,
) -> SceneGraphBuilder:
    """
    Factory function to create scene graph models.

    Args:
        num_classes: Number of object classes
        num_relationships: Number of relationship types

    Returns:
        Scene graph builder model
    """
    return SceneGraphBuilder(
        num_classes=num_classes,
        num_relationships=num_relationships,
    )
