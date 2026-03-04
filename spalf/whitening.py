"""Soft-ZCA whitening: Frequent Directions sketch + LW2020 nonlinear shrinkage.

Covariance estimation via Frequent Directions (Liberty, KDD 2013; Ghashami et al.,
SIAM J. Computing 2016). Regularization via Ledoit-Wolf analytical nonlinear shrinkage
(Annals of Statistics, Vol 48 No 5, 2020).
"""

import torch
from torch import Tensor


def _lw_nonlinear_shrinkage(eigenvalues: Tensor, n: int) -> Tensor:
    """Ledoit-Wolf 2020 analytical nonlinear shrinkage (Eq. 4.3 + 4.9).

    Applies an individually optimal nonlinear function to each eigenvalue
    via the discrete Hilbert transform of the sample spectral density.
    """
    p = len(eigenvalues)
    c = p / n

    lam = eigenvalues.to(dtype=torch.float64).clone()
    # Break ties with minimal jitter for well-defined Hilbert transform.
    lam += torch.finfo(torch.float64).eps * torch.arange(p, 0, -1, dtype=torch.float64, device="cuda")

    # Discrete Hilbert transform: h_j = (1/p) * sum_{k!=j} 1/(lam_k - lam_j).
    diffs = lam.unsqueeze(0) - lam.unsqueeze(1)  # [p, p]
    diffs.fill_diagonal_(1.0)  # avoid division by zero on diagonal
    h = (1.0 / diffs).sum(dim=0) / p
    # Subtract the diagonal contribution we included (1/1.0 per row).
    h -= 1.0 / p

    # Stieltjes transform.
    m = -(1.0 - c) / lam + c * h

    # Oracle shrinkage formula.
    denominator = (1.0 - c - c * lam * m).abs().pow(2)
    d_shrunk = lam / denominator

    return d_shrunk.clamp(min=torch.finfo(torch.float64).eps).to(eigenvalues.dtype)


class FrequentDirections:
    """Frequent Directions streaming sketch (Liberty 2013).

    Maintains an l x d sketch matrix where l = sqrt(d) (Ghashami et al. 2016,
    Theorem 3). Memory: O(l * d) vs O(d^2) for full covariance.
    """

    def __init__(self, d: int) -> None:
        self.d = d
        # Sketch size l = sqrt(d): standard FD choice (Ghashami 2016 Thm 3).
        # Approximation error = ||A||^2_F / l, so l = sqrt(d) gives rank-sqrt(d) accuracy.
        self._l = min(d, int(d ** 0.5))

        # Sketch matrix: [2l, d] to allow batch insertion before SVD compression.
        self._sketch = torch.zeros(2 * self._l, d, dtype=torch.float64, device="cuda")
        self._next_row = 0

        # Incremental mean (Welford, O(d) memory).
        self._n = 0
        self._mean = torch.zeros(d, dtype=torch.float64, device="cuda")

        # Incremental trace: sum of squared centered values / (n-1).
        self._sum_sq = torch.zeros(1, dtype=torch.float64, device="cuda")

        # Convergence detection via singular value stability.
        self._prev_sv: Tensor | None = None
        self._converged = False

    def update(self, x: Tensor) -> None:
        """Insert centered rows into sketch, SVD-compress when full."""
        x = x.to(dtype=torch.float64)
        batch_size = x.shape[0]

        # Update mean (Welford).
        batch_mean = x.mean(dim=0)
        delta = batch_mean - self._mean
        new_n = self._n + batch_size
        self._mean = self._mean + delta * (batch_size / new_n)

        # Parallel Welford M2: within-batch SS + cross-term (Ghashami 2016 Appendix).
        # M2 += Σ||x_i - x̄||² + ||δ||² · n_old · B / N
        within_ss = (x - batch_mean).pow(2).sum()
        cross = delta.pow(2).sum() * (self._n * batch_size / new_n)
        self._sum_sq += within_ss + cross

        self._n = new_n

        # Batch-insert centered rows into sketch, compressing when full.
        centered = x - self._mean
        pos = 0
        while pos < batch_size:
            space = 2 * self._l - self._next_row
            chunk = min(space, batch_size - pos)
            self._sketch[self._next_row : self._next_row + chunk] = centered[pos : pos + chunk]
            self._next_row += chunk
            pos += chunk
            if self._next_row >= 2 * self._l:
                self._compress()

    def _compress(self) -> None:
        """SVD-compress sketch to l rows (FD shrinkage step)."""
        U, s, Vt = torch.linalg.svd(self._sketch, full_matrices=False)
        # FD shrinkage: subtract squared l-th singular value from all.
        if len(s) > self._l:
            delta = s[self._l].pow(2)
            s_shrunk = (s[:self._l].pow(2) - delta).clamp(min=0.0).sqrt()
        else:
            s_shrunk = s[:self._l]

        self._sketch[:self._l] = s_shrunk.unsqueeze(1) * Vt[:self._l]
        self._sketch[self._l:] = 0.0
        self._next_row = self._l

        # Check convergence: relative singular value change < 1/d.
        if self._n >= self.d and self._prev_sv is not None:
            diff = (s_shrunk - self._prev_sv).norm()
            ref = s_shrunk.norm()
            if diff / ref < 1.0 / self.d:
                self._converged = True

        self._prev_sv = s_shrunk.clone()

    @property
    def trace(self) -> float:
        """Total variance estimate: sum(diag(cov)) = sum_sq / (n-1)."""
        return (self._sum_sq / (self._n - 1)).item()

    def get_eigendecomposition(self) -> tuple[Tensor, Tensor]:
        """Extract eigenvalues and eigenvectors from sketch SVD.

        Returns (eigenvalues: [l], eigenvectors: [d, l]) in descending order.
        """
        # Final compress to ensure sketch is up to date.
        if self._next_row > self._l:
            self._compress()

        active = self._sketch[:self._l]
        U, s, Vt = torch.linalg.svd(active, full_matrices=False)

        # Eigenvalues of covariance ≈ s^2 / (n-1).
        eigenvalues = s.pow(2) / (self._n - 1)
        eigenvectors = Vt.T  # [d, l]

        return eigenvalues, eigenvectors


class SoftZCAWhitener:
    """Frozen Soft-ZCA whitening transform with LW2020-optimal per-eigenvalue regularization."""

    def __init__(
        self,
        mean: Tensor,
        eigenvalues: Tensor,
        eigenvectors: Tensor,
        reg_eigenvalues: Tensor,
        n_samples: int,
        total_trace: float = 0.0,
    ) -> None:
        self.d = mean.shape[0]
        self._n_samples = n_samples
        self._total_trace = total_trace

        self.mean = mean.float()
        self._eigenvalues = eigenvalues.float()
        self._eigenvectors = eigenvectors.float()
        self._reg_eigenvalues = reg_eigenvalues.float()

        l = eigenvectors.shape[1]  # number of eigenvectors (l < d for sketch)

        # Spectral gap: find largest relative drop between consecutive eigenvalues.
        # Signal eigenvalues are separated from noise by the Marcenko-Pastur edge.
        ratios = reg_eigenvalues[:-1] / reg_eigenvalues[1:]
        gap_idx = ratios.argmax().item()
        self._k = gap_idx + 1

        # Low-rank when either the spectral gap is interior to the sketch,
        # or the sketch itself is rank-deficient (l < d).
        self.is_low_rank = l < self.d or self._k < l

        if self.is_low_rank:
            self._U_k = self._eigenvectors[:, :self._k]

            # Tail eigenvalue: average variance of unrepresented dimensions.
            # For sketch (l < d): (total_trace - sum(sketch_eigenvalues)) / (d - l)
            # accounts for the d-l dimensions absent from the sketch.
            # For full-rank (l == d): average of below-gap eigenvalues.
            if l < self.d and total_trace > 0.0:
                sketch_trace = self._reg_eigenvalues.sum().item()
                residual_trace = total_trace - sketch_trace
                n_tail = self.d - self._k
                # Blend: sketch tail eigenvalues (l-k of them) + residual (d-l of them).
                sketch_tail_sum = self._reg_eigenvalues[self._k:].sum().item()
                self._lambda_bar = (sketch_tail_sum + residual_trace) / n_tail
            else:
                tail = self._reg_eigenvalues[self._k:]
                self._lambda_bar = float(tail.mean())

            self._scale_k = self._reg_eigenvalues[:self._k].rsqrt()
            self._scale_tail = 1.0 / self._lambda_bar ** 0.5
            self._diff_scale = self._scale_k - self._scale_tail
            self._diff_scale_sq = self._scale_k.pow(2) - self._scale_tail ** 2
            self._scale_tail_sq = self._scale_tail ** 2
        else:
            scales = self._reg_eigenvalues.rsqrt()
            U = self._eigenvectors
            self._W_white = U @ torch.diag(scales) @ U.T

            self._precision = self._W_white.T @ self._W_white

    @property
    def effective_rank(self) -> int:
        """Shannon effective rank: exp(entropy of normalized eigenvalue distribution).

        Measures intrinsic dimensionality of the activation space.
        """
        p = self._eigenvalues / self._eigenvalues.sum()
        return int(torch.exp(-(p * p.log()).sum()).item())

    @property
    def noise_fraction(self) -> float:
        """Fraction of total variance below the rank-k cutoff.

        For sketch-based construction (l < d), includes the d-l unrepresented
        dimensions via total_trace.
        """
        sketch_total = self._eigenvalues.sum().item()
        sketch_noise = self._eigenvalues[self._k:].sum().item()
        # Add unrepresented dimensions if sketch is rank-deficient.
        if self._total_trace > 0.0:
            residual = self._total_trace - sketch_total
            total = self._total_trace
            noise = sketch_noise + residual
        else:
            total = sketch_total
            noise = sketch_noise
        return max(noise / total, 1.0 / self.d)

    @classmethod
    def from_sketch(cls, sketch: FrequentDirections) -> "SoftZCAWhitener":
        """Build whitener from a converged FD sketch with LW2020 nonlinear shrinkage."""
        mean = sketch._mean
        eigenvalues, eigenvectors = sketch.get_eigendecomposition()

        # Clamp eigenvalues to relative spectral floor.
        eigenvalues = eigenvalues.clamp(
            min=torch.finfo(eigenvalues.dtype).eps * eigenvalues.max()
        )

        reg_eigenvalues = _lw_nonlinear_shrinkage(eigenvalues, sketch._n)

        return cls(
            mean=mean,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            reg_eigenvalues=reg_eigenvalues,
            n_samples=sketch._n,
            total_trace=sketch.trace,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Whiten activations."""
        centered = x - self.mean

        if self.is_low_rank:
            proj = centered @ self._U_k
            return (proj * self._diff_scale) @ self._U_k.T + centered * self._scale_tail
        else:
            return centered @ self._W_white.T

    def compute_mahalanobis_sq(self, diff: Tensor) -> Tensor:
        """Compute ||diff||^2 in the regularized precision metric."""
        if self.is_low_rank:
            proj = diff @ self._U_k
            return (proj.pow(2) * self._diff_scale_sq).sum(dim=1) + self._scale_tail_sq * diff.pow(2).sum(dim=1)
        else:
            return (diff @ self._precision * diff).sum(dim=1)

    def state_dict(self) -> dict:
        return {
            "mean": self.mean,
            "eigenvalues": self._eigenvalues,
            "eigenvectors": self._eigenvectors,
            "reg_eigenvalues": self._reg_eigenvalues,
            "n_samples": torch.tensor(self._n_samples),
            "total_trace": torch.tensor(self._total_trace),
        }

    def load_state_dict(self, sd: dict) -> None:
        self.__init__(
            mean=sd["mean"],
            eigenvalues=sd["eigenvalues"],
            eigenvectors=sd["eigenvectors"],
            reg_eigenvalues=sd["reg_eigenvalues"],
            n_samples=int(sd["n_samples"].item()),
            total_trace=float(sd["total_trace"].item()),
        )
