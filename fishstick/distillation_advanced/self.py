import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Callable
import copy


class SelfDistillation(nn.Module):
    def __init__(
        self,
        student: nn.Module,
        num_stages: int = 3,
        temperature: float = 4.0,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.num_stages = num_stages
        self.temperature = temperature
        self.alpha = alpha

        self.stages = nn.ModuleList([copy.deepcopy(student) for _ in range(num_stages)])
        self.best_stage = None

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        stage_idx: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = {}

        if stage_idx is not None:
            stage = self.stages[stage_idx]
            logits = stage(x)
            outputs[f"stage_{stage_idx}_logits"] = logits

            if stage_idx > 0:
                prev_logits = self.stages[stage_idx - 1](x).detach()
                T = self.temperature
                kd_loss = F.kl_div(
                    F.log_softmax(logits / T, dim=-1),
                    F.softmax(prev_logits / T, dim=-1),
                    reduction="batchmean",
                ) * (T * T)
                outputs[f"stage_{stage_idx}_kd_loss"] = kd_loss
            else:
                outputs[f"stage_{stage_idx}_kd_loss"] = torch.tensor(
                    0.0, device=logits.device
                )

            outputs[f"stage_{stage_idx}_ce_loss"] = F.cross_entropy(logits, labels)
        else:
            for idx in range(self.num_stages):
                logits = self.stages[idx](x)
                outputs[f"stage_{idx}_logits"] = logits

                if idx > 0:
                    prev_logits = self.stages[idx - 1](x).detach()
                    T = self.temperature
                    kd_loss = F.kl_div(
                        F.log_softmax(logits / T, dim=-1),
                        F.softmax(prev_logits / T, dim=-1),
                        reduction="batchmean",
                    ) * (T * T)
                    outputs[f"stage_{idx}_kd_loss"] = kd_loss
                else:
                    outputs[f"stage_{idx}_kd_loss"] = torch.tensor(
                        0.0, device=logits.device
                    )

                outputs[f"stage_{idx}_ce_loss"] = F.cross_entropy(logits, labels)

        return outputs


class BeYourOwnTeacher(nn.Module):
    def __init__(
        self,
        student: nn.Module,
        num_ensembles: int = 3,
        temperature: float = 4.0,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.num_ensembles = num_ensembles
        self.temperature = temperature
        self.alpha = alpha

        self.ensembles = nn.ModuleList(
            [copy.deepcopy(student) for _ in range(num_ensembles)]
        )

        for i in range(1, num_ensembles):
            self.ensembles[i].load_state_dict(student.state_dict())

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        ensemble_idx: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = {}

        if ensemble_idx is not None:
            logits = self.ensembles[ensemble_idx](x)
            outputs["logits"] = logits

            if ensemble_idx > 0:
                T = self.temperature
                teacher_logits = self.ensembles[0](x).detach()

                kd_loss = F.kl_div(
                    F.log_softmax(logits / T, dim=-1),
                    F.softmax(teacher_logits / T, dim=-1),
                    reduction="batchmean",
                ) * (T * T)
                outputs["kd_loss"] = kd_loss
            else:
                outputs["kd_loss"] = torch.tensor(0.0, device=logits.device)

            outputs["ce_loss"] = F.cross_entropy(logits, labels)
        else:
            all_logits = []
            for ensemble in self.ensembles:
                all_logits.append(ensemble(x))

            outputs["ensemble_logits"] = torch.stack(all_logits, dim=0)
            outputs["avg_logits"] = outputs["ensemble_logits"].mean(dim=0)

            T = self.temperature
            teacher_logits = self.ensembles[0](x).detach()

            for idx, logits in enumerate(all_logits):
                kd_loss = F.kl_div(
                    F.log_softmax(logits / T, dim=-1),
                    F.softmax(teacher_logits / T, dim=-1),
                    reduction="batchmean",
                ) * (T * T)
                outputs[f"kd_loss_{idx}"] = kd_loss

            outputs["ce_loss"] = F.cross_entropy(outputs["avg_logits"], labels)

        return outputs


class DataFreeKnowledgeDistillation(nn.Module):
    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        num_synthetic: int = 1000,
        img_size: int = 32,
        channels: int = 3,
        lr: float = 0.01,
        momentum: float = 0.9,
        ema_decay: float = 0.999,
    ):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.num_synthetic = num_synthetic
        self.img_size = img_size
        self.channels = channels
        self.lr = lr
        self.momentum = momentum
        self.ema_decay = ema_decay

        self.synthetic_data = nn.Parameter(
            torch.randn(num_synthetic, channels, img_size, img_size) * 0.1
        )

        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(
        self,
        labels: Optional[torch.Tensor] = None,
        alpha: float = 0.5,
        temperature: float = 4.0,
    ) -> Dict[str, torch.Tensor]:
        self.teacher.eval()

        bs = self.synthetic_data.size(0)

        synthetic_batch = self.synthetic_data[:bs]

        with torch.no_grad():
            teacher_logits = self.teacher(synthetic_batch)

        student_logits = self.student(synthetic_batch)

        T = temperature
        kd_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction="batchmean",
        ) * (T * T)

        ce_loss = torch.tensor(0.0, device=student_logits.device)
        if labels is not None:
            ce_loss = F.cross_entropy(student_logits, labels[:bs])

        loss = alpha * kd_loss + (1 - alpha) * ce_loss

        return {
            "loss": loss,
            "kd_loss": kd_loss,
            "ce_loss": ce_loss,
        }

    def update_synthetic_data(self, labels: torch.Tensor, num_iterations: int = 10):
        self.teacher.eval()

        optimizer = torch.optim.SGD(
            [self.synthetic_data], lr=self.lr, momentum=self.momentum
        )

        for _ in range(num_iterations):
            optimizer.zero_grad()

            bs = min(labels.size(0), self.synthetic_data.size(0))
            synthetic_batch = self.synthetic_data[:bs]

            with torch.no_grad():
                teacher_logits = self.teacher(synthetic_batch)

            student_logits = self.student(synthetic_batch)

            T = 4.0
            loss = F.kl_div(
                F.log_softmax(student_logits / T, dim=-1),
                F.softmax(teacher_logits / T, dim=-1),
                reduction="batchmean",
            ) * (T * T)

            loss.backward()
            optimizer.step()


class DeepSelfDistillation(nn.Module):
    def __init__(
        self,
        student: nn.Module,
        depth: int = 3,
        temperature: float = 4.0,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.depth = depth
        self.temperature = temperature
        self.alpha = alpha

        self.backbone = copy.deepcopy(student.backbone)
        self.heads = nn.ModuleList([copy.deepcopy(student.head) for _ in range(depth)])

        for head in self.heads:
            for param in head.parameters():
                param.requires_grad = True

    def forward(
        self, x: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        outputs = {}

        feat = self.backbone(x)

        for depth_idx in range(self.depth):
            logits = self.heads[depth_idx](feat)
            outputs[f"logits_d{depth_idx}"] = logits

            if labels is not None:
                ce_loss = F.cross_entropy(logits, labels)
                outputs[f"ce_loss_d{depth_idx}"] = ce_loss

            if depth_idx > 0:
                prev_logits = self.heads[depth_idx - 1](feat).detach()
                T = self.temperature
                kd_loss = F.kl_div(
                    F.log_softmax(logits / T, dim=-1),
                    F.softmax(prev_logits / T, dim=-1),
                    reduction="batchmean",
                ) * (T * T)
                outputs[f"kd_loss_d{depth_idx}"] = kd_loss

        return outputs


class SnapshotDistillation(nn.Module):
    def __init__(
        self,
        student: nn.Module,
        num_checkpoints: int = 3,
        temperature: float = 4.0,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.num_checkpoints = num_checkpoints
        self.temperature = temperature
        self.alpha = alpha

        self.student = student
        self.checkpoints = nn.ModuleList(
            [copy.deepcopy(student) for _ in range(num_checkpoints)]
        )

        self.current_checkpoint_idx = 0

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        logits = self.student(x)

        kd_loss = 0.0
        for idx, checkpoint in enumerate(self.checkpoints):
            with torch.no_grad():
                checkpoint_logits = checkpoint(x)

            T = self.temperature
            kd_loss += F.kl_div(
                F.log_softmax(logits / T, dim=-1),
                F.softmax(checkpoint_logits / T, dim=-1),
                reduction="batchmean",
            ) * (T * T)

        kd_loss = kd_loss / self.num_checkpoints
        ce_loss = F.cross_entropy(logits, labels)

        loss = self.alpha * ce_loss + (1 - self.alpha) * kd_loss

        return {
            "loss": loss,
            "kd_loss": kd_loss,
            "ce_loss": ce_loss,
        }

    def save_checkpoint(self):
        if self.current_checkpoint_idx < self.num_checkpoints:
            self.checkpoints[self.current_checkpoint_idx].load_state_dict(
                self.student.state_dict()
            )
            self.current_checkpoint_idx += 1
