import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import Optional, Callable, List, Dict, Any
from copy import deepcopy


class MAML:
    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5,
        grad_clip: Optional[float] = None,
        first_order: bool = False,
    ):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.grad_clip = grad_clip
        self.first_order = first_order
        self.meta_params = list(model.parameters())

    def inner_update(
        self, model: nn.Module, support_x: torch.Tensor, support_y: torch.Tensor
    ) -> nn.Module:
        inner_model = deepcopy(model)
        inner_model.train()

        for _ in range(self.inner_steps):
            logits = inner_model(support_x)
            loss = nn.functional.cross_entropy(logits, support_y)
            grads = torch.autograd.grad(
                loss, inner_model.parameters(), create_graph=not self.first_order
            )

            with torch.no_grad():
                for param, grad in zip(inner_model.parameters(), grads):
                    if self.grad_clip is not None:
                        grad = torch.clamp(grad, -self.grad_clip, self.grad_clip)
                    param -= self.inner_lr * grad

        return inner_model

    def outer_step(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
    ) -> torch.Tensor:
        inner_model = self.inner_update(self.model, support_x, support_y)

        query_logits = inner_model(query_x)
        loss = nn.functional.cross_entropy(query_logits, query_y)
        return loss

    def meta_train_step(
        self,
        tasks: List[Dict[str, torch.Tensor]],
        optimizer: Optimizer,
    ) -> float:
        meta_loss = 0.0

        for task in tasks:
            support_x = task["support_x"]
            support_y = task["support_y"]
            query_x = task["query_x"]
            query_y = task["query_y"]

            loss = self.outer_step(support_x, support_y, query_x, query_y)
            meta_loss += loss

        meta_loss /= len(tasks)

        optimizer.zero_grad()
        meta_loss.backward()

        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.meta_params, self.grad_clip)

        optimizer.step()

        return meta_loss.item()

    def adapt(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
    ) -> nn.Module:
        return self.inner_update(self.model, support_x, support_y)


class FOMAML(MAML):
    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5,
        grad_clip: Optional[float] = None,
    ):
        super().__init__(
            model, inner_lr, outer_lr, inner_steps, grad_clip, first_order=True
        )


class Reptile:
    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5,
        grad_clip: Optional[float] = None,
    ):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.grad_clip = grad_clip

    def inner_update(
        self, model: nn.Module, support_x: torch.Tensor, support_y: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        inner_model = deepcopy(model)
        inner_model.train()

        for _ in range(self.inner_steps):
            logits = inner_model(support_x)
            loss = nn.functional.cross_entropy(logits, support_y)
            grads = torch.autograd.grad(loss, inner_model.parameters())

            with torch.no_grad():
                for param, grad in zip(inner_model.parameters(), grads):
                    if self.grad_clip is not None:
                        grad = torch.clamp(grad, -self.grad_clip, self.grad_clip)
                    param -= self.inner_lr * grad

        return {n: p.data.clone() for n, p in inner_model.named_parameters()}

    def meta_train_step(
        self,
        tasks: List[Dict[str, torch.Tensor]],
        optimizer: Optimizer,
    ) -> float:
        meta_grads = [torch.zeros_like(p) for p in self.model.parameters()]

        for task in tasks:
            support_x = task["support_x"]
            support_y = task["support_y"]

            adapted_params = self.inner_update(self.model, support_x, support_y)

            for (name, param), grad in zip(self.model.named_parameters(), meta_grads):
                grad += (adapted_params[name] - param.data) / len(tasks)

        optimizer.zero_grad()
        with torch.no_grad():
            for param, grad in zip(self.model.parameters(), meta_grads):
                if self.grad_clip is not None:
                    grad = torch.clamp(grad, -self.grad_clip, self.grad_clip)
                param.add_(self.outer_lr * grad)

        return 0.0

    def adapt(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
    ) -> nn.Module:
        adapted_params = self.inner_update(self.model, support_x, support_y)

        inner_model = deepcopy(self.model)
        with torch.no_grad():
            for name, param in inner_model.named_parameters():
                param.copy_(adapted_params[name])

        return inner_model


class MetaSGD:
    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5,
        grad_clip: Optional[float] = None,
    ):
        self.model = model
        self.inner_lr_base = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.grad_clip = grad_clip

        self.meta_params = list(model.parameters())
        self.meta_lrs = nn.ParameterList(
            [nn.Parameter(torch.tensor(inner_lr)) for _ in self.meta_params]
        )

    def inner_update(
        self,
        model: nn.Module,
        meta_lrs: List[torch.Tensor],
        support_x: torch.Tensor,
        support_y: torch.Tensor,
    ) -> nn.Module:
        inner_model = deepcopy(model)
        inner_model.train()

        for step in range(self.inner_steps):
            logits = inner_model(support_x)
            loss = nn.functional.cross_entropy(logits, support_y)
            grads = torch.autograd.grad(loss, inner_model.parameters())

            with torch.no_grad():
                for param, grad, lr in zip(inner_model.parameters(), grads, meta_lrs):
                    if self.grad_clip is not None:
                        grad = torch.clamp(grad, -self.grad_clip, self.grad_clip)
                    param -= lr * grad

        return inner_model

    def outer_step(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
    ) -> torch.Tensor:
        inner_model = self.inner_update(self.model, self.meta_lrs, support_x, support_y)

        query_logits = inner_model(query_x)
        loss = nn.functional.cross_entropy(query_logits, query_y)
        return loss

    def meta_train_step(
        self,
        tasks: List[Dict[str, torch.Tensor]],
        optimizer: Optimizer,
    ) -> float:
        meta_loss = 0.0

        for task in tasks:
            support_x = task["support_x"]
            support_y = task["support_y"]
            query_x = task["query_x"]
            query_y = task["query_y"]

            loss = self.outer_step(support_x, support_y, query_x, query_y)
            meta_loss += loss

        meta_loss /= len(tasks)

        optimizer.zero_grad()
        meta_loss.backward()

        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.meta_lrs), self.grad_clip
            )

        optimizer.step()

        return meta_loss.item()

    def adapt(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
    ) -> nn.Module:
        learned_lrs = [lr.data.clone() for lr in self.meta_lrs]
        return self.inner_update(self.model, learned_lrs, support_x, support_y)


class MetaLSTM(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        hidden_size: int = 20,
        outer_lr: float = 0.001,
        grad_clip: Optional[float] = None,
    ):
        super().__init__()
        self.model = model
        self.hidden_size = hidden_size
        self.outer_lr = outer_lr
        self.grad_clip = grad_clip

        self.num_params = sum(p.numel() for p in model.parameters())

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2 * self.num_params),
        )

        self.meta_optimizer = torch.optim.Adam(
            list(self.lstm.parameters()) + list(self.mlp.parameters()),
            lr=outer_lr,
        )

    def get_init_state(self, batch_size: int, device: torch.device) -> tuple:
        h0 = torch.zeros(2, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(2, batch_size, self.hidden_size, device=device)
        return (h0, c0)

    def forward(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
    ) -> torch.Tensor:
        params = list(self.model.parameters())

        for step in range(support_x.size(0)):
            logits = self.model(support_x[step : step + 1], params)
            loss = nn.functional.cross_entropy(
                logits, support_y[step : step + 1], reduction="sum"
            )

            grads = torch.autograd.grad(loss, params, retain_graph=True)

            grad_flat = (
                torch.cat([g.flatten() for g in grads]).unsqueeze(0).unsqueeze(0)
            )

            h, c = self.get_init_state(1, support_x.device)
            lstm_out, _ = self.lstm(grad_flat, (h, c))

            delta = self.mlp(lstm_out.squeeze(0))
            delta = delta.view(2, -1)

            lr_update = torch.sigmoid(delta[0]).view(-1, 1)
            param_update = delta[1]

            with torch.no_grad():
                for i, (p, lr, du) in enumerate(zip(params, lr_update, param_update)):
                    p.data = p.data + self.inner_lr * lr * du

        return self.model(query_x, params)

    def meta_train_step(
        self,
        tasks: List[Dict[str, torch.Tensor]],
        base_optimizer: Optimizer,
    ) -> float:
        meta_loss = 0.0

        for task in tasks:
            support_x = task["support_x"]
            support_y = task["support_y"]
            query_x = task["query_x"]
            query_y = task["query_y"]

            inner_model = deepcopy(self.model)
            inner_optimizer = torch.optim.SGD(
                inner_model.parameters(), lr=self.inner_lr
            )

            for step in range(5):
                logits = inner_model(support_x)
                loss = nn.functional.cross_entropy(logits, support_y)
                inner_optimizer.zero_grad()
                loss.backward()
                inner_optimizer.step()

            with torch.no_grad():
                query_logits = inner_model(query_x)

            loss = nn.functional.cross_entropy(query_logits, query_y)

            self.meta_optimizer.zero_grad()
            loss.backward()
            self.meta_optimizer.step()

            meta_loss += loss.item()

        return meta_loss / len(tasks)

    def adapt(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
    ) -> nn.Module:
        inner_model = deepcopy(self.model)
        inner_optimizer = torch.optim.SGD(inner_model.parameters(), lr=0.1)

        for step in range(5):
            logits = inner_model(support_x)
            loss = nn.functional.cross_entropy(logits, support_y)
            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()

        return inner_model


class MetaLearner:
    def __init__(
        self,
        model: nn.Module,
        method: str = "maml",
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5,
        grad_clip: Optional[float] = None,
        **kwargs,
    ):
        self.model = model
        self.method = method

        if method == "maml":
            self.meta_learner = MAML(model, inner_lr, outer_lr, inner_steps, grad_clip)
        elif method == "fomaml":
            self.meta_learner = FOMAML(
                model, inner_lr, outer_lr, inner_steps, grad_clip
            )
        elif method == "reptile":
            self.meta_learner = Reptile(
                model, inner_lr, outer_lr, inner_steps, grad_clip
            )
        elif method == "metasgd":
            self.meta_learner = MetaSGD(
                model, inner_lr, outer_lr, inner_steps, grad_clip
            )
        elif method == "metalstm":
            self.meta_learner = MetaLSTM(
                model, outer_lr=outer_lr, grad_clip=grad_clip, **kwargs
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    def meta_train_step(
        self,
        tasks: List[Dict[str, torch.Tensor]],
        optimizer: Optimizer,
    ) -> float:
        return self.meta_learner.meta_train_step(tasks, optimizer)

    def adapt(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
    ) -> nn.Module:
        return self.meta_learner.adapt(support_x, support_y)


def create_episode(
    x: torch.Tensor,
    y: torch.Tensor,
    n_way: int,
    n_support: int,
    n_query: int,
) -> Dict[str, torch.Tensor]:
    classes = torch.unique(y)
    selected_classes = classes[torch.randperm(len(classes))[:n_way]]

    support_mask = torch.zeros(len(y), dtype=torch.bool)
    query_mask = torch.zeros(len(y), dtype=torch.bool)

    for i, c in enumerate(selected_classes):
        class_mask = y == c
        class_indices = torch.where(class_mask)[0]

        perm = torch.randperm(len(class_indices))
        support_indices = class_indices[perm[:n_support]]
        query_indices = class_indices[perm[n_support : n_support + n_query]]

        support_mask[support_indices] = True
        query_mask[query_indices] = True

    relabel_map = {c.item(): i for i, c in enumerate(selected_classes)}

    support_x = x[support_mask]
    support_y = torch.tensor(
        [relabel_map[y[i].item()] for i in torch.where(support_mask)[0]],
        dtype=torch.long,
    )

    query_x = x[query_mask]
    query_y = torch.tensor(
        [relabel_map[y[i].item()] for i in torch.where(query_mask)[0]],
        dtype=torch.long,
    )

    return {
        "support_x": support_x,
        "support_y": support_y,
        "query_x": query_x,
        "query_y": query_y,
    }
