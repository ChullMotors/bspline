"""Python implementation of tensor-product B-spline interpolation. 

Inpsired by bspline-fortran, ref: https://github.com/jacobwilliams/bspline-fortran

Copyright (c) 2026 Monumo Ltd
All rights reserved.

Licensed under the MIT License. See LICENSE file in the project root.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def _build_knot_vector(*, x_nodes: torch.Tensor, k: int) -> torch.Tensor:
    """Build clamped knot vector for B spline interpolation over `x_nodes`.

    Args:
        x_nodes: 1D strictly increasing grid nodes.
        k: Spline order (yields degree `k - 1`).

    Returns:
        Knot vector `t` of length `n + k`, where `n = len(x_nodes)`.

    - First `k` knots are at `x_nodes[0]`.
    - Last `k` knots are at `rnot = x_nodes[-1] + 0.1 * (x_nodes[-1] - x_nodes[-2])`.
    - Interior knots depend on parity of `k`: for even `k`, knots land on data
      points; for odd `k`, they land between data points.
    """
    if x_nodes.ndim != 1:
        msg = f"x_nodes must be 1D, got shape {tuple(x_nodes.shape)}"
        raise ValueError(msg)
    if x_nodes.numel() < 2:
        msg = "x_nodes must have at least 2 points"
        raise ValueError(msg)
    if k < 2:
        msg = "k must be at least 2"
        raise ValueError(msg)
    if not torch.all(x_nodes[1:] > x_nodes[:-1]).item():
        msg = "x_nodes must be strictly increasing"
        raise ValueError(msg)

    n = int(x_nodes.numel())
    if n < k:
        msg = "Number of nodes (n) must be at least equal to spline order (k)"
        raise ValueError(msg)

    # Rightmost knot shifted slightly to avoid right-boundary degeneracy.
    # Standard B-spline evaluation is left-closed and right-open.
    rnot = x_nodes[-1] + 0.1 * (x_nodes[-1] - x_nodes[-2])

    t = torch.empty((n + k,), dtype=x_nodes.dtype, device=x_nodes.device)
    # Repetition at boundaries forces one basis function to survive at each end,
    # giving interpolation through boundary points.
    t[:k] = x_nodes[0]
    t[-k:] = rnot

    if (k % 2) == 0:
        half = k // 2
        # Fortran equivalent: j = k+1..n (1-based)
        for j in range(k, n):
            t[j] = x_nodes[j - half]
    else:
        half = (k + 1) // 2
        for j in range(k, n):
            i = j - half
            t[j] = 0.5 * (x_nodes[i] + x_nodes[i + 1])

    return t


def _find_span(
    *, t: torch.Tensor, ncoef: int, k: int, xq: torch.Tensor
) -> torch.Tensor:
    """Return knot interval (span) index for each query point.

    Args:
        t: Knot vector of shape `(ncoef + k,)`.
        ncoef: Number of spline coefficients along the axis.
        k: Spline order.
        xq: Query points.

    Returns:
        Span indices clamped to `[k - 1, ncoef - 1]`.
    """
    degree = k - 1
    # Left and right boundaries of spline domain.
    domain_min = t[degree]
    domain_max = t[ncoef]

    x_use = xq.clamp(min=domain_min, max=domain_max)

    # Span index s satisfies: t[s] <= x <= t[s + 1]
    span = torch.searchsorted(t, x_use, right=True) - 1
    return span.clamp(min=degree, max=ncoef - 1)


def _bspline_basis_funs(
    *, t: torch.Tensor, k: int, x: torch.Tensor, span: torch.Tensor, eps: float = 1e-14
) -> torch.Tensor:
    """Cox-de Boor basis values for the `k` nonzero B-splines at each query.

    Args:
        t: Knot vector of shape `(ncoef + k,)`.
        k: Spline order (degree `k - 1`).
        x: Query points of shape `(batch,)`.
        span: Knot span indices for each query point, shape `(batch,)`.
        eps: Threshold for near-zero denominator handling.

    Returns:
        Basis values of shape `(batch, k)` corresponding to basis indices
        `(span - k + 1 .. span)` for each query.
    """
    if x.ndim != 1:
        msg = "x must be 1D"
        raise ValueError(msg)
    if span.shape != x.shape:
        msg = "span shape must match x shape"
        raise ValueError(msg)

    batch = int(x.numel())

    # Keep basis values as a growing list to avoid in-place writes that can
    # invalidate autograd for gradients wrt query points.
    basis_cols: list[torch.Tensor] = [torch.ones_like(x)]
    left_cols: list[torch.Tensor] = [x.new_zeros((batch,))]
    right_cols: list[torch.Tensor] = [x.new_zeros((batch,))]

    # Cox-de Boor recursion:
    # https://en.wikipedia.org/wiki/De_Boor%27s_algorithm
    for j in range(1, k):
        # For each query point q, use span[q] to fetch relevant knot values.
        tj_left = t[(span + 1 - j).to(torch.long)]
        tj_right = t[(span + j).to(torch.long)]

        # Distance from point to local knot interval endpoints.
        left_cols.append(x - tj_left)
        right_cols.append(tj_right - x)

        saved = x.new_zeros((batch,))
        next_cols: list[torch.Tensor] = []
        # j + 1 active basis functions in local span.
        for r in range(j):
            denom = right_cols[r + 1] + left_cols[j - r]
            temp = torch.where(
                denom.abs() > eps, basis_cols[r] / denom, torch.zeros_like(denom)
            )
            next_cols.append(saved + right_cols[r + 1] * temp)
            saved = left_cols[j - r] * temp
        next_cols.append(saved)
        basis_cols = next_cols

    return torch.stack(basis_cols, dim=1)


def _build_interp_matrix_1d(
    *, x_nodes: torch.Tensor, t: torch.Tensor, k: int
) -> torch.Tensor:
    """Build dense interpolation matrix `A` where `A[i, j] = N_j(x_i)`.

    Args:
        x_nodes: Grid nodes of shape `(n,)` where interpolation constraints apply.
        t: Knot vector of shape `(n + k,)`.
        k: Spline order.

    Returns:
        Dense interpolation matrix of shape `(n, n)`.
    """
    n = int(x_nodes.numel())
    span = _find_span(t=t, ncoef=n, k=k, xq=x_nodes)
    basis = _bspline_basis_funs(t=t, k=k, x=x_nodes, span=span)

    # Place each row's k local basis values into the corresponding columns.
    cols = (
        span.unsqueeze(1)
        - (k - 1)
        + torch.arange(k, device=x_nodes.device).unsqueeze(0)
    ).to(torch.long)

    interp_matrix = x_nodes.new_zeros((n, n))
    rows = torch.arange(n, device=x_nodes.device).unsqueeze(1).expand_as(cols)
    interp_matrix[rows, cols] = basis
    return interp_matrix


def _solve_multi_rhs(*, matrix: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Solve `A @ x = rhs` for multiple right-hand sides with one LU factorization.

    The solved coefficients should satisfy interpolation constraints when basis
    functions are evaluated on the original grid nodes.
    """
    lu, piv = torch.linalg.lu_factor(matrix)
    return torch.linalg.lu_solve(lu, piv, rhs)


def _bcoef_from_grid_3d(
    *,
    interp_matrix_x: torch.Tensor,
    interp_matrix_y: torch.Tensor,
    interp_matrix_z: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """Compute coefficient tensor for tensor product interpolation.

    Supports values shaped (nx, ny, nz, ...) with optional trailing channels.
    """
    nx, ny, nz = values.shape[:3]
    trailing = values.shape[3:]
    trailing_size = 1
    for s in trailing:
        trailing_size *= int(s)

    # Solve for spline coefficients axis by axis. This works because tensor-product
    # bases are separable across axes.

    # Solve along x for all (y, z, channel) combinations.
    rhs_x = values.reshape(nx, ny * nz * trailing_size)
    coef_after_x = _solve_multi_rhs(matrix=interp_matrix_x, rhs=rhs_x).reshape(
        (nx, ny, nz, *trailing)
    )

    dims_after_x = coef_after_x.ndim
    perm_y = [1, 0, 2, *list(range(3, dims_after_x))]
    # Solve along y for all (x, z, channel) combinations.
    rhs_y = coef_after_x.permute(*perm_y).reshape(ny, nx * nz * trailing_size)
    coef_after_y = _solve_multi_rhs(matrix=interp_matrix_y, rhs=rhs_y).reshape(
        (ny, nx, nz, *trailing)
    )

    inv_perm_y = [1, 0, 2, *list(range(3, coef_after_y.ndim))]
    coef_after_y = coef_after_y.permute(*inv_perm_y)

    dims_after_y = coef_after_y.ndim
    perm_z = [2, 0, 1, *list(range(3, dims_after_y))]
    # Solve along z for all (x, y, channel) combinations.
    rhs_z = coef_after_y.permute(*perm_z).reshape(nz, nx * ny * trailing_size)
    bcoef = _solve_multi_rhs(matrix=interp_matrix_z, rhs=rhs_z).reshape(
        (nz, nx, ny, *trailing)
    )

    inv_perm_z = [1, 2, 0, *list(range(3, bcoef.ndim))]
    return bcoef.permute(*inv_perm_z)


@dataclass(frozen=True, kw_only=True, slots=True)
class TorchBSpline3D:
    """3D tensor product B spline model.

    This object supports two construction paths:
    1. fit: compute knot vectors and coefficients from gridded data
    2. from_bcoef: construct from already computed knots and coefficients

    Evaluation supports query tensors of any shape, as long as xq yq zq shapes match.
    If the coefficient tensor has trailing channel dimensions, the output will carry
    those channel dimensions.

    Unlike current open source tools, supports differentiable and nonuniform x/y/z grids
    and mirrors the bspline-fortran style spline building.
    """

    tx: torch.Tensor
    ty: torch.Tensor
    tz: torch.Tensor
    kx: int
    ky: int
    kz: int
    bcoef: torch.Tensor

    def __post_init__(self) -> None:
        self._validate_initialized()

    @classmethod
    def fit(
        cls,
        *,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        values: torch.Tensor,
        kx: int,
        ky: int,
        kz: int,
    ) -> TorchBSpline3D:
        """Fit a 3D tensor product B spline interpolant to gridded data.

        At a high level, this function wraps the following steps:
        1. Create knot vectors, with knots defined by (though not necessarily lying on)
           the data values.
        2. Form basis functions with the Cox-de Boor recursion and build interpolation
           matrices.
        3. Solve for spline coefficients that weight basis functions along each axis.

        Args:
            x: x grid nodes, shape (nx,)
            y: y grid nodes, shape (ny,)
            z: z grid nodes, shape (nz,)
            values: grid values, shape (nx, ny, nz, ...) with optional channels
            kx: spline order along x
            ky: spline order along y
            kz: spline order along z

        Returns:
            A constructed TorchBSpline3D instance.
        """
        if x.ndim != 1 or y.ndim != 1 or z.ndim != 1:
            msg = "x y z must be 1D tensors"
            raise ValueError(msg)

        nx, ny, nz = x.numel(), y.numel(), z.numel()
        if values.ndim < 3:
            msg = "values must have at least 3 dimensions"
            raise ValueError(msg)
        if tuple(values.shape[:3]) != (nx, ny, nz):
            msg = (
                f"Expected values leading shape {(nx, ny, nz)}, got "
                f"{tuple(values.shape[:3])}"
            )
            raise ValueError(msg)

        if kx < 2 or ky < 2 or kz < 2:
            msg = "Spline orders must be at least 2"
            raise ValueError(msg)
        if kx > nx or ky > ny or kz > nz:
            msg = (
                "Spline orders must not exceed the number of grid points along each "
                f"axis; got kx={kx} with nx={nx}, ky={ky} with ny={ny}, "
                f"kz={kz} with nz={nz}"
            )
            raise ValueError(msg)

        tx = _build_knot_vector(x_nodes=x, k=kx)
        ty = _build_knot_vector(x_nodes=y, k=ky)
        tz = _build_knot_vector(x_nodes=z, k=kz)

        ax = _build_interp_matrix_1d(x_nodes=x, t=tx, k=kx)
        ay = _build_interp_matrix_1d(x_nodes=y, t=ty, k=ky)
        az = _build_interp_matrix_1d(x_nodes=z, t=tz, k=kz)

        bcoef = _bcoef_from_grid_3d(
            interp_matrix_x=ax, interp_matrix_y=ay, interp_matrix_z=az, values=values
        )

        return cls(tx=tx, ty=ty, tz=tz, kx=kx, ky=ky, kz=kz, bcoef=bcoef)

    @classmethod
    def from_bcoef(
        cls,
        *,
        tx: torch.Tensor,
        ty: torch.Tensor,
        tz: torch.Tensor,
        kx: int,
        ky: int,
        kz: int,
        bcoef: torch.Tensor,
    ) -> TorchBSpline3D:
        """Construct from knot vectors and coefficient tensor.

        This is the intended entry point when coefficients are computed offline.
        """
        return cls(tx=tx, ty=ty, tz=tz, kx=kx, ky=ky, kz=kz, bcoef=bcoef)

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        nx, ny, nz = self.bcoef.shape[:3]
        return int(nx), int(ny), int(nz)

    @property
    def channel_shape(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self.bcoef.shape[3:])

    def evaluate(
        self, *, xq: torch.Tensor, yq: torch.Tensor, zq: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate the spline at query points.

        For each query point this function:
        1. finds the spans (`_find_span`)
        2. computes k nonzero basis functions per axis (`_bspline_basis_funs`)
        3. gathers the local coefficient block `(kx, ky, kz)`
        4. forms tensor-product weights and sums

        Args:
            xq: query x coordinates, any shape
            yq: query y coordinates, same shape as xq
            zq: query z coordinates, same shape as xq

        Returns:
            Tensor with shape xq.shape + channel_shape.
        """
        if xq.shape != yq.shape or xq.shape != zq.shape:
            msg = "xq yq zq must have the same shape"
            raise ValueError(msg)

        xq_flat = xq.reshape(-1)
        yq_flat = yq.reshape(-1)
        zq_flat = zq.reshape(-1)

        nx, ny, nz = self.bcoef.shape[:3]

        span_x = _find_span(t=self.tx, ncoef=nx, k=self.kx, xq=xq_flat)
        span_y = _find_span(t=self.ty, ncoef=ny, k=self.ky, xq=yq_flat)
        span_z = _find_span(t=self.tz, ncoef=nz, k=self.kz, xq=zq_flat)

        basis_x = _bspline_basis_funs(t=self.tx, k=self.kx, x=xq_flat, span=span_x)
        basis_y = _bspline_basis_funs(t=self.ty, k=self.ky, x=yq_flat, span=span_y)
        basis_z = _bspline_basis_funs(t=self.tz, k=self.kz, x=zq_flat, span=span_z)

        # If e.g. span=10 and k=4, span gives the rightmost active index and local
        # active coefficient indices are {7, 8, 9, 10}.
        coef_idx_x = (
            span_x.unsqueeze(1)
            - (self.kx - 1)
            + torch.arange(self.kx, device=self.bcoef.device)
        ).to(torch.long)  # (n, kx)
        coef_idx_y = (
            span_y.unsqueeze(1)
            - (self.ky - 1)
            + torch.arange(self.ky, device=self.bcoef.device)
        ).to(torch.long)  # (n, ky)
        coef_idx_z = (
            span_z.unsqueeze(1)
            - (self.kz - 1)
            + torch.arange(self.kz, device=self.bcoef.device)
        ).to(torch.long)  # (n, kz)

        coef_block = self.bcoef[
            coef_idx_x[:, :, None, None],
            coef_idx_y[:, None, :, None],
            coef_idx_z[:, None, None, :],
        ]  # (n, kx, ky, kz, ...)

        weights = (
            basis_x[:, :, None, None]
            * basis_y[:, None, :, None]
            * basis_z[:, None, None, :]
        )  # (n, kx, ky, kz)

        if coef_block.ndim > 4:
            extra = coef_block.ndim - 4
            weights = weights.view(weights.shape + (1,) * extra)

        out = (coef_block * weights).sum(dim=(1, 2, 3))  # (n, ...)

        out_shape = xq.shape + self.channel_shape
        return out.reshape(out_shape)

    def __call__(
        self, *, xq: torch.Tensor, yq: torch.Tensor, zq: torch.Tensor
    ) -> torch.Tensor:
        return self.evaluate(xq=xq, yq=yq, zq=zq)

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> TorchBSpline3D:
        """Return a new instance with tensors moved to device and dtype."""
        return type(self)(
            tx=self.tx.to(device=device, dtype=dtype),
            ty=self.ty.to(device=device, dtype=dtype),
            tz=self.tz.to(device=device, dtype=dtype),
            kx=self.kx,
            ky=self.ky,
            kz=self.kz,
            bcoef=self.bcoef.to(device=device, dtype=dtype),
        )

    def state_dict(self) -> dict[str, Any]:
        """Return a serializable state dict."""
        return {
            "tx": self.tx,
            "ty": self.ty,
            "tz": self.tz,
            "kx": self.kx,
            "ky": self.ky,
            "kz": self.kz,
            "bcoef": self.bcoef,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> TorchBSpline3D:
        """Construct from a state dict created by state_dict."""
        return cls(
            tx=state["tx"],
            ty=state["ty"],
            tz=state["tz"],
            kx=int(state["kx"]),
            ky=int(state["ky"]),
            kz=int(state["kz"]),
            bcoef=state["bcoef"],
        )

    def _validate_initialized(self) -> None:
        if self.kx < 2 or self.ky < 2 or self.kz < 2:
            msg = "Spline orders must be at least 2"
            raise ValueError(msg)

        if self.tx.ndim != 1 or self.ty.ndim != 1 or self.tz.ndim != 1:
            msg = "tx ty tz must be 1D tensors"
            raise ValueError(msg)

        if self.bcoef.ndim < 3:
            msg = "bcoef must have at least 3 dimensions"
            raise ValueError(msg)

        nx, ny, nz = self.bcoef.shape[:3]
        if self.tx.numel() != nx + self.kx:
            msg = "tx length must equal nx + kx"
            raise ValueError(msg)
        if self.ty.numel() != ny + self.ky:
            msg = "ty length must equal ny + ky"
            raise ValueError(msg)
        if self.tz.numel() != nz + self.kz:
            msg = "tz length must equal nz + kz"
            raise ValueError(msg)

        if not torch.all(self.tx[1:] >= self.tx[:-1]).item():
            msg = "tx must be nondecreasing"
            raise ValueError(msg)
        if not torch.all(self.ty[1:] >= self.ty[:-1]).item():
            msg = "ty must be nondecreasing"
            raise ValueError(msg)
        if not torch.all(self.tz[1:] >= self.tz[:-1]).item():
            msg = "tz must be nondecreasing"
            raise ValueError(msg)
