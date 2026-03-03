"""Fused JumpReLU Triton kernels with Moreau envelope STE."""

import torch
import triton
import triton.language as tl
from torch import Tensor

_BLOCK_SIZE: int = 2048


@triton.jit
def _fused_jumprelu_fwd_kernel(
    pre_act_ptr,
    theta_ptr,
    z_ptr,
    gate_ptr,
    l0_ptr,
    disc_ptr,
    lambda_disc,
    n_elements,
    F: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused forward: gate + L0 + discretization in single pass."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(pre_act_ptr + offsets, mask=mask, other=0.0)
    theta_idx = offsets % F
    theta = tl.load(theta_ptr + theta_idx, mask=mask, other=0.0)

    gate = (x > theta).to(tl.float32)
    z = x * gate

    l0 = gate

    below = x * (1.0 - gate)
    disc = lambda_disc * below * below

    tl.store(z_ptr + offsets, z, mask=mask)
    tl.store(gate_ptr + offsets, gate, mask=mask)
    tl.store(l0_ptr + offsets, l0, mask=mask)
    tl.store(disc_ptr + offsets, disc, mask=mask)


@triton.jit
def _moreau_ste_bwd_kernel(
    grad_l0_ptr,
    pre_act_ptr,
    theta_ptr,
    gamma_ptr,
    grad_pre_act_ptr,
    grad_theta_ptr,
    n_elements,
    F: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Backward for L0: Moreau envelope proximal gradient of Heaviside.

    Transition zone: -sqrt(2*gamma) < u <= 0 where u = x - theta.
    Gradient in zone: -u / gamma (linear ramp).
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad_l0 = tl.load(grad_l0_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(pre_act_ptr + offsets, mask=mask, other=0.0)

    feat_idx = offsets % F
    theta = tl.load(theta_ptr + feat_idx, mask=mask, other=0.0)
    gamma = tl.load(gamma_ptr + feat_idx, mask=mask, other=1.0)

    u = x - theta
    bandwidth = tl.sqrt(2.0 * gamma)

    in_zone = (u > -bandwidth) & (u <= 0.0)
    moreau_grad = tl.where(in_zone, -u / gamma, 0.0)

    grad_x = grad_l0 * moreau_grad
    grad_t = -grad_l0 * moreau_grad

    tl.store(grad_pre_act_ptr + offsets, grad_x, mask=mask)
    tl.atomic_add(grad_theta_ptr + feat_idx, grad_t, mask=mask)


@triton.jit
def _recon_ste_bwd_kernel(
    grad_z_ptr,
    pre_act_ptr,
    theta_ptr,
    gamma_ptr,
    grad_theta_ptr,
    n_elements,
    F: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Reconstruction-loss STE gradient to theta (JumpReLU paper Eq. 11).

    Routes grad_z through Moreau envelope kernel to threshold:
    ∂L_recon/∂θ_j = -Σ_i [grad_z · pre_act · moreau_kernel(u)]
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad_z = tl.load(grad_z_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(pre_act_ptr + offsets, mask=mask, other=0.0)

    feat_idx = offsets % F
    theta = tl.load(theta_ptr + feat_idx, mask=mask, other=0.0)
    gamma = tl.load(gamma_ptr + feat_idx, mask=mask, other=1.0)

    u = x - theta
    bandwidth = tl.sqrt(2.0 * gamma)
    in_zone = (u > -bandwidth) & (u <= 0.0)
    moreau_kernel = tl.where(in_zone, -u / gamma, 0.0)

    grad_t = -grad_z * x * moreau_kernel
    tl.atomic_add(grad_theta_ptr + feat_idx, grad_t, mask=mask)


class FusedJumpReLUFunction(torch.autograd.Function):
    """Autograd wrapper for fused Triton JumpReLU kernel."""

    @staticmethod
    def forward(
        ctx,
        pre_act: Tensor,
        log_threshold: Tensor,
        gamma: Tensor,
        lambda_disc: float,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Fused forward: returns (z, gate, l0, disc_correction)."""
        B, F = pre_act.shape
        n_elements = B * F

        theta = log_threshold.exp()

        z = torch.empty_like(pre_act)
        gate = torch.empty_like(pre_act)
        l0 = torch.empty_like(pre_act)
        disc = torch.empty_like(pre_act)

        grid = ((n_elements + _BLOCK_SIZE - 1) // _BLOCK_SIZE,)

        _fused_jumprelu_fwd_kernel[grid](
            pre_act, theta,
            z, gate, l0, disc,
            lambda_disc,
            n_elements,
            F=F,
            BLOCK_SIZE=_BLOCK_SIZE,
        )

        ctx.save_for_backward(pre_act, theta, gamma)
        ctx.F = F
        ctx.n_elements = n_elements
        ctx.lambda_disc = lambda_disc

        return z, gate.detach(), l0, disc

    @staticmethod
    def backward(
        ctx, grad_z: Tensor, _grad_gate: Tensor, grad_l0: Tensor, grad_disc: Tensor
    ) -> tuple[Tensor | None, Tensor | None, None, None]:
        pre_act, theta, gamma = ctx.saved_tensors
        F = ctx.F
        n_elements = ctx.n_elements

        gate = (pre_act > theta).float()
        grad_pre_act_from_z = grad_z * gate

        grad_pre_act_from_l0 = torch.zeros_like(pre_act)
        grad_theta_total = torch.zeros(F, device=pre_act.device, dtype=pre_act.dtype)

        grid = ((n_elements + _BLOCK_SIZE - 1) // _BLOCK_SIZE,)

        _moreau_ste_bwd_kernel[grid](
            grad_l0, pre_act, theta, gamma,
            grad_pre_act_from_l0, grad_theta_total,
            n_elements, F=F, BLOCK_SIZE=_BLOCK_SIZE,
        )

        # Reconstruction-loss STE: route grad_z to theta via Moreau envelope (Eq. 11).
        _recon_ste_bwd_kernel[grid](
            grad_z, pre_act, theta, gamma,
            grad_theta_total,
            n_elements, F=F, BLOCK_SIZE=_BLOCK_SIZE,
        )

        below = pre_act * (1.0 - gate)
        grad_pre_act_from_disc = grad_disc * ctx.lambda_disc * 2.0 * below * (1.0 - gate)

        grad_pre_act_total = grad_pre_act_from_z + grad_pre_act_from_l0 + grad_pre_act_from_disc

        grad_log_threshold = grad_theta_total * theta

        return grad_pre_act_total, grad_log_threshold, None, None
