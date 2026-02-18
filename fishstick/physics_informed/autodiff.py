"""
Automatic Differentiation Operators for PDEs
==============================================

Provides autodiff utilities for computing derivatives, Jacobians,
Hessians, and other differential operators needed for PINNs.

Based on torch.autograd for efficient gradient computation.
"""

from __future__ import annotations

from typing import Optional, Callable, Tuple, Union, List
import torch
from torch import Tensor
import torch.nn.functional as F


def grad(
    u: Tensor,
    x: Tensor,
    create_graph: bool = False,
    retain_graph: bool = True,
) -> Tensor:
    """
    Compute gradient of u with respect to x.

    Args:
        u: Output tensor [batch, ...]
        x: Input tensor [batch, n_dims]
        create_graph: If True, graph of the derivative will be built
        retain_graph: If True, graph is retained for backward passes

    Returns:
        Gradient [batch, ..., n_dims]
    """
    grad_outputs = torch.ones_like(u)

    grad_u = torch.autograd.grad(
        outputs=u,
        inputs=x,
        grad_outputs=grad_outputs,
        create_graph=create_graph,
        retain_graph=retain_graph,
    )[0]

    return grad_u


def jacobian(
    f: Callable[[Tensor], Tensor],
    x: Tensor,
    create_graph: bool = False,
) -> Tensor:
    """
    Compute Jacobian of f(x) with respect to x.

    Args:
        f: Function mapping x to output [batch, m]
        x: Input tensor [batch, n]
        create_graph: If True, build computational graph

    Returns:
        Jacobian [batch, m, n]
    """
    batch_size = x.size(0)
    n = x.size(-1)

    f_val = f(x)
    m = f_val.size(-1)

    jacobian = []

    for i in range(m):
        grad_outputs = torch.zeros(batch_size, m, device=x.device)
        grad_outputs[:, i] = 1.0

        grad_i = torch.autograd.grad(
            outputs=f_val,
            inputs=x,
            grad_outputs=grad_outputs,
            create_graph=create_graph,
            retain_graph=True,
        )[0]

        jacobian.append(grad_i.unsqueeze(1))

    return torch.cat(jacobian, dim=1)


def batch_jacobian(
    f: Callable[[Tensor], Tensor],
    x: Tensor,
    create_graph: bool = False,
) -> Tensor:
    """
    Efficient batch Jacobian computation.

    Args:
        f: Function mapping x to output [batch, m]
        x: Input tensor [batch, n]
        create_graph: If True, build computational graph

    Returns:
        Jacobian [batch, m, n]
    """
    f_val = f(x)
    batch_size, m = f_val.shape

    jacobian = torch.zeros(batch_size, m, x.size(-1), device=x.device, dtype=x.dtype)

    grad_outputs = torch.eye(m, device=x.device).unsqueeze(0).expand(batch_size, -1, -1)

    grad_inputs = torch.autograd.grad(
        outputs=f_val,
        inputs=x,
        grad_outputs=grad_outputs,
        create_graph=create_graph,
        retain_graph=True,
    )[0]

    return grad_inputs.permute(0, 2, 1)


def hessian(
    f: Callable[[Tensor], Tensor],
    x: Tensor,
    create_graph: bool = False,
) -> Tensor:
    """
    Compute Hessian of scalar function f(x) with respect to x.

    Args:
        f: Scalar function [batch] or [batch, 1]
        x: Input tensor [batch, n]
        create_graph: If True, build computational graph

    Returns:
        Hessian [batch, n, n]
    """
    f_val = f(x)
    if f_val.dim() == 2 and f_val.size(-1) == 1:
        f_val = f_val.squeeze(-1)

    batch_size = x.size(0)
    n = x.size(-1)

    grad_outputs = torch.ones(batch_size, device=x.device)

    first_grad = torch.autograd.grad(
        outputs=f_val,
        inputs=x,
        grad_outputs=grad_outputs,
        create_graph=create_graph,
        retain_graph=True,
    )[0]

    hessian_rows = []
    for i in range(n):
        grad_i = first_grad[:, i]
        hessian_i = torch.autograd.grad(
            outputs=grad_i,
            inputs=x,
            grad_outputs=torch.ones_like(grad_i),
            create_graph=create_graph,
            retain_graph=True,
        )[0]
        hessian_rows.append(hessian_i.unsqueeze(1))

    return torch.cat(hessian_rows, dim=1)


def batch_hessian(
    f: Callable[[Tensor], Tensor],
    x: Tensor,
    create_graph: bool = False,
) -> Tensor:
    """
    Efficient batch Hessian computation for scalar functions.

    Args:
        f: Scalar function [batch]
        x: Input tensor [batch, n]
        create_graph: If True, build computational graph

    Returns:
        Hessian [batch, n, n]
    """
    f_val = f(x)
    if f_val.dim() == 2 and f_val.size(-1) == 1:
        f_val = f_val.squeeze(-1)

    batch_size, n = x.size(0), x.size(-1)

    grad_outputs = torch.ones(batch_size, device=x.device)

    first_grad = torch.autograd.grad(
        outputs=f_val,
        inputs=x,
        grad_outputs=grad_outputs,
        create_graph=create_graph,
        retain_graph=True,
    )[0]

    hessian_list = []
    for i in range(n):
        grad_i = first_grad[:, i]
        hess_i = torch.autograd.grad(
            outputs=grad_i,
            inputs=x,
            grad_outputs=torch.ones_like(grad_i),
            create_graph=create_graph,
            retain_graph=(i < n - 1),
        )[0]
        hessian_list.append(hess_i)

    return torch.stack(hessian_list, dim=1)


def divergence(
    f: Callable[[Tensor], Tensor],
    x: Tensor,
    create_graph: bool = False,
) -> Tensor:
    """
    Compute divergence of vector field f(x): R^n -> R^n.

    div(f) = sum_i df_i/dx_i

    Args:
        f: Vector field [batch, n_dims]
        x: Input tensor [batch, n_dims]
        create_graph: If True, build computational graph

    Returns:
        Divergence [batch]
    """
    jac = batch_jacobian(f, x, create_graph=create_graph)
    div = torch.diagonal(jac, dim1=-2, dim2=-1).sum(-1)
    return div


def laplacian(
    u: Tensor,
    x: Tensor,
    create_graph: bool = False,
) -> Tensor:
    """
    Compute Laplacian of u with respect to x.

    Laplacian = div(grad(u)) = sum_i d²u/dx_i²

    Args:
        u: Scalar function [batch]
        x: Input tensor [batch, n_dims]
        create_graph: If True, build computational graph

    Returns:
        Laplacian [batch]
    """
    grad_u = grad(u, x, create_graph=create_graph)
    div_grad = divergence(
        lambda x: grad(u, x, create_graph=create_graph), x, create_graph=create_graph
    )
    return div_grad


def time_derivative(
    u: Tensor,
    t: Tensor,
    create_graph: bool = False,
) -> Tensor:
    """
    Compute time derivative of u with respect to t.

    Args:
        u: Solution [batch, ...]
        t: Time [batch] or scalar
        create_graph: If True, build computational graph

    Returns:
        Time derivative [batch, ...]
    """
    grad_outputs = torch.ones_like(u)

    return torch.autograd.grad(
        outputs=u,
        inputs=t,
        grad_outputs=grad_outputs,
        create_graph=create_graph,
        retain_graph=True,
    )[0]


def gradient_norm(
    u: Tensor,
    x: Tensor,
    p: int = 2,
    create_graph: bool = False,
) -> Tensor:
    """
    Compute L^p norm of gradient.

    Args:
        u: Solution [batch]
        x: Input tensor [batch, n_dims]
        p: Norm order (1, 2, or inf)
        create_graph: If True, build computational graph

    Returns:
        Gradient norm [batch]
    """
    grad_u = grad(u, x, create_graph=create_graph)

    if p == 1:
        return torch.norm(grad_u, p=1, dim=-1)
    elif p == 2:
        return torch.norm(grad_u, p=2, dim=-1)
    elif p == float("inf"):
        return torch.norm(grad_u, p=float("inf"), dim=-1)
    else:
        raise ValueError(f"Unsupported norm order: {p}")


def directional_derivative(
    u: Tensor,
    x: Tensor,
    direction: Tensor,
    create_graph: bool = False,
) -> Tensor:
    """
    Compute directional derivative in given direction.

    D_v u(x) = grad(u) · v

    Args:
        u: Solution [batch]
        x: Input tensor [batch, n_dims]
        direction: Direction vector [batch, n_dims]
        create_graph: If True, build computational graph

    Returns:
        Directional derivative [batch]
    """
    grad_u = grad(u, x, create_graph=create_graph)
    return (grad_u * direction).sum(dim=-1)


def jacobian_vector_product(
    f: Callable[[Tensor], Tensor],
    x: Tensor,
    v: Tensor,
    create_graph: bool = False,
) -> Tensor:
    """
    Compute Jacobian-vector product: J(x) @ v

    More memory efficient than full Jacobian.

    Args:
        f: Function [batch, m]
        x: Input tensor [batch, n]
        v: Vector [batch, m] or [m]
        create_graph: If True, build computational graph

    Returns:
        J @ v [batch, n]
    """
    f_val = f(x)

    if v.dim() == 1:
        v = v.unsqueeze(0).expand(f_val.size(0), -1)

    return torch.autograd.grad(
        outputs=f_val,
        inputs=x,
        grad_outputs=v,
        create_graph=create_graph,
        retain_graph=True,
    )[0]


def hessian_vector_product(
    f: Callable[[Tensor], Tensor],
    x: Tensor,
    v: Tensor,
    create_graph: bool = False,
) -> Tensor:
    """
    Compute Hessian-vector product: H(x) @ v

    More memory efficient than full Hessian.

    Args:
        f: Scalar function [batch]
        x: Input tensor [batch, n]
        v: Vector [batch, n] or [n]
        create_graph: If True, build computational graph

    Returns:
        H @ v [batch, n]
    """
    f_val = f(x)
    if f_val.dim() == 2 and f_val.size(-1) == 1:
        f_val = f_val.squeeze(-1)

    if v.dim() == 1:
        v = v.unsqueeze(0).expand(x.size(0), -1)

    grad_outputs = torch.ones_like(f_val)

    first_grad = torch.autograd.grad(
        outputs=f_val,
        inputs=x,
        grad_outputs=grad_outputs,
        create_graph=create_graph,
        retain_graph=True,
    )[0]

    return torch.autograd.grad(
        outputs=first_grad,
        inputs=x,
        grad_outputs=v,
        create_graph=create_graph,
        retain_graph=True,
    )[0]


def curl(
    f: Callable[[Tensor], Tensor],
    x: Tensor,
    create_graph: bool = False,
) -> Optional[Tensor]:
    """
    Compute curl of 2D or 3D vector field.

    For 2D: curl(f) = df2/dx1 - df1/dx2
    For 3D: curl(f)_i = eps_ijk df_j/dx_k

    Args:
        f: Vector field [batch, 2] or [batch, 3]
        x: Input tensor [batch, 2] or [batch, 3]
        create_graph: If True, build computational graph

    Returns:
        Curl [batch, 2] or [batch, 3] or None if not 2D/3D
    """
    dim = x.size(-1)

    if dim == 2:
        jac = batch_jacobian(f, x, create_graph=create_graph)
        curl_2d = jac[:, 1, 0] - jac[:, 0, 1]
        return curl_2d.unsqueeze(-1)
    elif dim == 3:
        jac = batch_jacobian(f, x, create_graph=create_graph)
        curl_3d = torch.zeros_like(x)
        curl_3d[:, 0] = jac[:, 2, 1] - jac[:, 1, 2]
        curl_3d[:, 1] = jac[:, 0, 2] - jac[:, 2, 0]
        curl_3d[:, 2] = jac[:, 1, 0] - jac[:, 0, 1]
        return curl_3d
    else:
        return None


def compute_pinn_derivatives(
    u: Tensor,
    net: torch.nn.Module,
    x: Tensor,
    t: Optional[Tensor] = None,
    order: int = 1,
) -> Dict[str, Tensor]:
    """
    Compute common derivatives for PINN training.

    Args:
        u: Network output [batch, 1]
        net: Neural network
        x: Spatial coordinates [batch, n_dims]
        t: Time coordinate [batch] (optional)
        order: Derivative order (1 or 2)

    Returns:
        Dictionary of derivatives
    """
    derivatives = {}

    inputs = {"x": x}
    if t is not None:
        inputs["t"] = t

    grad_u = grad(u, x, create_graph=True)
    derivatives["u_x"] = grad_u[..., 0:1]

    if x.size(-1) > 1:
        derivatives["u_y"] = grad_u[..., 1:2]
    if x.size(-1) > 2:
        derivatives["u_z"] = grad_u[..., 2:3]

    if t is not None:
        grad_t = grad(u, t, create_graph=True)
        derivatives["u_t"] = grad_t

    if order >= 2:
        grad_2 = grad(grad_u, x, create_graph=True)

        if x.size(-1) == 1:
            derivatives["u_xx"] = grad_2[..., 0:1]
        else:
            derivatives["u_xx"] = grad_2[..., 0:1]
            derivatives["u_yy"] = grad_2[..., 1:2]
            if x.size(-1) > 2:
                derivatives["u_zz"] = grad_2[..., 2:3]

            if x.size(-1) >= 2:
                derivatives["u_xy"] = (
                    grad_2[..., 0:1, 1:2] if grad_2.dim() == 3 else grad_2[:, 0, 1]
                )

    return derivatives
