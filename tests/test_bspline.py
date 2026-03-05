"""Unit tests for ``odense.bspline`` and ``TorchBSpline3D``.

Covers:
  * Knot-vector generation for even/odd spline order and invalid inputs.
  * Span lookup behavior for default clamped behavior.
  * Cox-de Boor basis properties (shape + partition of unity) and validation.
  * Interpolation matrix and linear solve helpers.
  * End-to-end spline fitting and exact reconstruction at training grid points.
  * Autograd pathways needed by the application:
    - gradient wrt fitted coefficients (``bcoef``)
    - gradient wrt query points passed to ``evaluate`` (``xq/yq/zq``)
    - gradient wrt solve RHS for ``_solve_multi_rhs``.
  * Model utility APIs (state dict round-trip, ``to`` conversion, constructor checks).
"""

from __future__ import annotations

import unittest
from typing import TYPE_CHECKING

import torch
from scipy.interpolate import NdBSpline

from odense.bspline import (
    TorchBSpline3D,
    _bspline_basis_funs,
    _build_interp_matrix_1d,
    _dbknot,
    _find_span,
    _solve_multi_rhs,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class TestDBKnot(unittest.TestCase):
    """Tests for knot-vector generation and argument checks."""

    def test_dbknot_even_order(self) -> None:
        """Even-order knot generation repeats boundaries/places interior on nodes."""
        x_nodes = torch.tensor([0.0, 1.0, 3.0, 7.0], dtype=torch.float64)
        k = 4

        knots = _dbknot(x_nodes=x_nodes, k=k)
        rnot = x_nodes[-1] + 0.1 * (x_nodes[-1] - x_nodes[-2])
        expected = torch.tensor([0.0, 0.0, 0.0, 0.0, rnot, rnot, rnot, rnot])

        self.assertEqual(knots.shape, (x_nodes.numel() + k,))
        self.assertTrue(torch.allclose(knots, expected))

    def test_dbknot_odd_order(self) -> None:
        """Odd-order knot generation places interior knots between neighboring nodes."""
        x_nodes = torch.tensor([0.0, 1.0, 2.0, 4.0, 8.0], dtype=torch.float64)
        k = 3

        knots = _dbknot(x_nodes=x_nodes, k=k)
        rnot = x_nodes[-1] + 0.1 * (x_nodes[-1] - x_nodes[-2])
        expected = torch.tensor([0.0, 0.0, 0.0, 1.5, 3.0, rnot, rnot, rnot])

        self.assertEqual(knots.shape, (x_nodes.numel() + k,))
        self.assertTrue(torch.allclose(knots, expected))

    def test_dbknot_validation(self) -> None:
        """Invalid dimensionality/monotonicity/order inputs are rejected."""
        with self.assertRaisesRegex(ValueError, "1D"):
            _dbknot(x_nodes=torch.ones((2, 2)), k=3)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            _dbknot(x_nodes=torch.tensor([0.0, 0.0, 1.0]), k=3)
        with self.assertRaisesRegex(ValueError, "at least 2 points"):
            _dbknot(x_nodes=torch.tensor([0.0]), k=3)
        with self.assertRaisesRegex(ValueError, "at least 2"):
            _dbknot(x_nodes=torch.tensor([0.0, 1.0]), k=1)


class TestBasisAndSpans(unittest.TestCase):
    """Tests for span lookup and local basis-function evaluation."""

    def test_find_span_clamps_outside_domain(self) -> None:
        """Out-of-domain points clamp to boundary spans in ``clamp`` mode."""
        x_nodes = torch.linspace(0.0, 4.0, 5, dtype=torch.float64)
        k = 3
        t = _dbknot(x_nodes=x_nodes, k=k)
        xq = torch.tensor([-10.0, 0.2, 10.0], dtype=torch.float64)

        spans = _find_span(t=t, ncoef=x_nodes.numel(), k=k, xq=xq)

        self.assertTrue(torch.equal(spans, torch.tensor([2, 2, 4])))

    def test_bspline_basis_partition_of_unity(self) -> None:
        """Active basis values sum to one for each query point."""
        x_nodes = torch.linspace(0.0, 4.0, 5, dtype=torch.float64)
        k = 4
        t = _dbknot(x_nodes=x_nodes, k=k)
        xq = torch.tensor([0.0, 0.4, 1.5, 3.8, 4.0], dtype=torch.float64)
        span = _find_span(t=t, ncoef=x_nodes.numel(), k=k, xq=xq)

        basis = _bspline_basis_funs(t=t, k=k, x=xq, span=span)

        self.assertEqual(basis.shape, (xq.numel(), k))
        self.assertTrue(torch.allclose(basis.sum(dim=1), torch.ones_like(xq)))

    def test_bspline_basis_validates_shapes(self) -> None:
        """Basis helper validates 1D query shape and span alignment."""
        x = torch.tensor([0.1, 0.2], dtype=torch.float64)
        span = torch.tensor([2, 2], dtype=torch.long)
        t = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)

        with self.assertRaisesRegex(ValueError, "x must be 1D"):
            _bspline_basis_funs(t=t, k=3, x=x.reshape(1, 2), span=span)
        with self.assertRaisesRegex(ValueError, "span shape"):
            _bspline_basis_funs(t=t, k=3, x=x, span=span[:1])


class TestTorchBSpline3D(unittest.TestCase):
    """Tests for model fitting, evaluation, and utility methods."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build one nontrivial smooth field used across tests."""
        cls.x = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
        cls.y = torch.linspace(-1.0, 1.0, 6, dtype=torch.float64)
        cls.z = torch.linspace(2.0, 3.0, 4, dtype=torch.float64)
        cls.kx = 4
        cls.ky = 4
        cls.kz = 2

        # Include mixed term so interpolation and gradients are not trivially separable.
        xx, yy, zz = torch.meshgrid(cls.x, cls.y, cls.z, indexing="ij")
        cls.values = 2.0 * xx - 3.0 * yy + 0.5 * zz + xx * yy * zz

        cls.model = TorchBSpline3D.fit(
            x=cls.x,
            y=cls.y,
            z=cls.z,
            values=cls.values,
            kx=cls.kx,
            ky=cls.ky,
            kz=cls.kz,
        )

    @staticmethod
    def _autograd_directional_derivative(
        fn: Callable, x0: torch.Tensor, direction: torch.Tensor
    ) -> float:
        x = x0.clone().detach().requires_grad_()
        y = fn(x)
        y = y.sum()
        grad = torch.autograd.grad(y, x)[0]
        return torch.sum(grad * direction).item()

    @staticmethod
    def _finite_difference_directional_derivative(
        fn: Callable, x0: torch.Tensor, direction: torch.Tensor, eps: float = 1e-6
    ) -> float:
        y_plus = fn(x0 + eps * direction).item()
        y_minus = fn(x0 - eps * direction).item()
        return (y_plus - y_minus) / (2.0 * eps)

    @staticmethod
    def _assert_close_relative(
        got: float, expected: float, *, rel_tol: float = 1e-7, abs_tol: float = 1e-8
    ) -> None:
        err = abs(got - expected)
        scale = max(abs(expected), abs(got), 1.0)
        assert err <= max(abs_tol, rel_tol * scale), (
            f"Directional derivative mismatch: {got=}, {expected=}, {err=}, {scale=}"
        )

    def test_fit_reconstructs_grid_values(self) -> None:
        """Fitted spline reconstructs training-grid values exactly."""
        xx, yy, zz = torch.meshgrid(self.x, self.y, self.z, indexing="ij")
        out = self.model.evaluate(xq=xx, yq=yy, zq=zz)
        self.assertTrue(torch.allclose(out, self.values))

    def test_evaluate_matches_scipy_ndbspline_off_grid(self) -> None:
        """Off-grid evaluation matches SciPy's tensor-product spline."""
        # Build scipy spline using our knot vectors
        scipy_spline = NdBSpline(
            (
                self.model.tx.detach().cpu().numpy(),
                self.model.ty.detach().cpu().numpy(),
                self.model.tz.detach().cpu().numpy(),
            ),
            self.model.bcoef.detach().cpu().numpy(),
            (self.model.kx - 1, self.model.ky - 1, self.model.kz - 1),
            extrapolate=False,
        )

        # Keep points strictly interior so the comparison focuses on interpolation.
        xq = torch.tensor(
            [0.11, 0.23, 0.37, 0.52, 0.68, 0.81, 0.94], dtype=torch.float64
        )
        yq = torch.tensor(
            [-0.91, -0.53, -0.22, 0.07, 0.34, 0.61, 0.88], dtype=torch.float64
        )
        zq = torch.tensor(
            [2.04, 2.18, 2.31, 2.49, 2.63, 2.79, 2.93], dtype=torch.float64
        )

        ours = self.model.evaluate(xq=xq, yq=yq, zq=zq)
        pts = torch.stack((xq, yq, zq), dim=1).detach().cpu().numpy()
        ref = torch.from_numpy(scipy_spline(pts))

        self.assertTrue(torch.allclose(ours, ref, atol=1e-12, rtol=1e-11))

    def test_build_interp_matrix_and_solve(self) -> None:
        """Helper solve reconstructs RHS when projected back through matrix A."""
        t = _dbknot(x_nodes=self.x, k=self.kx)
        interp_matrix = _build_interp_matrix_1d(x_nodes=self.x, t=t, k=self.kx)

        rhs = torch.sin(self.x)
        coef = _solve_multi_rhs(matrix=interp_matrix, rhs=rhs.reshape(-1, 1)).reshape(
            -1
        )
        reconstructed = interp_matrix @ coef

        self.assertEqual(interp_matrix.shape, (self.x.numel(), self.x.numel()))
        self.assertTrue(torch.allclose(reconstructed, rhs))

    def test_evaluate_validation_for_query_shapes(self) -> None:
        """Evaluate rejects mismatched query tensor shapes."""
        with self.assertRaisesRegex(ValueError, "same shape"):
            self.model.evaluate(
                xq=torch.tensor([0.1, 0.2], dtype=torch.float64),
                yq=torch.tensor([0.1], dtype=torch.float64),
                zq=torch.tensor([2.1, 2.2], dtype=torch.float64),
            )

    def test_evaluate_grad_wrt_bcoef(self) -> None:
        """Evaluate path is differentiable wrt spline coefficients."""
        model = TorchBSpline3D.from_bcoef(
            tx=self.model.tx,
            ty=self.model.ty,
            tz=self.model.tz,
            kx=self.model.kx,
            ky=self.model.ky,
            kz=self.model.kz,
            bcoef=self.model.bcoef.clone().detach().requires_grad_(),
        )
        xq = torch.tensor([0.12, 0.41, 0.88], dtype=torch.float64)
        yq = torch.tensor([-0.6, 0.0, 0.73], dtype=torch.float64)
        zq = torch.tensor([2.1, 2.7, 2.95], dtype=torch.float64)

        out = model.evaluate(xq=xq, yq=yq, zq=zq)
        out.sum().backward()

        assert model.bcoef.grad is not None
        self.assertTrue(torch.isfinite(model.bcoef.grad).all().item())
        self.assertGreater(model.bcoef.grad.abs().sum().item(), 0.0)

    def test_evaluate_grad_wrt_bcoef_matches_finite_differences(self) -> None:
        """Directional derivative wrt spline coefficients matches finite diff."""
        xq = torch.tensor([0.2, 0.45, 0.7], dtype=torch.float64)
        yq = torch.tensor([-0.7, 0.1, 0.6], dtype=torch.float64)
        zq = torch.tensor([2.2, 2.5, 2.8], dtype=torch.float64)
        b0 = self.model.bcoef.clone().detach()
        direction = torch.randn_like(b0)
        direction = direction / torch.linalg.norm(direction)

        def loss_fn(bcoef: torch.Tensor) -> torch.Tensor:
            model = TorchBSpline3D.from_bcoef(
                tx=self.model.tx,
                ty=self.model.ty,
                tz=self.model.tz,
                kx=self.model.kx,
                ky=self.model.ky,
                kz=self.model.kz,
                bcoef=bcoef,
            )
            out = model.evaluate(xq=xq, yq=yq, zq=zq)
            return (out**2).sum()

        ad = self._autograd_directional_derivative(loss_fn, b0, direction)
        fd = self._finite_difference_directional_derivative(loss_fn, b0, direction)
        self._assert_close_relative(ad, fd, rel_tol=1e-7)

    def test_evaluate_grad_wrt_query_points(self) -> None:
        """Evaluate path is differentiable wrt query coordinates xq/yq/zq."""
        xq = torch.tensor([0.12, 0.41, 0.88], dtype=torch.float64, requires_grad=True)
        yq = torch.tensor([-0.6, 0.0, 0.73], dtype=torch.float64, requires_grad=True)
        zq = torch.tensor([2.1, 2.7, 2.95], dtype=torch.float64, requires_grad=True)

        out = self.model.evaluate(xq=xq, yq=yq, zq=zq)
        out.sum().backward()

        assert xq.grad is not None
        assert yq.grad is not None
        assert zq.grad is not None
        self.assertTrue(torch.isfinite(xq.grad).all().item())
        self.assertTrue(torch.isfinite(yq.grad).all().item())
        self.assertTrue(torch.isfinite(zq.grad).all().item())
        self.assertGreater(xq.grad.abs().sum().item(), 0.0)
        self.assertGreater(yq.grad.abs().sum().item(), 0.0)
        self.assertGreater(zq.grad.abs().sum().item(), 0.0)

    def test_evaluate_grad_wrt_query_points_matches_finite_differences(self) -> None:
        """Directional derivative wrt query points matches finite diff."""
        xq0 = torch.tensor([0.2, 0.45, 0.7], dtype=torch.float64)
        yq0 = torch.tensor([-0.7, 0.1, 0.6], dtype=torch.float64)
        zq0 = torch.tensor([2.2, 2.5, 2.8], dtype=torch.float64)
        q0 = torch.cat([xq0, yq0, zq0])
        direction = torch.randn_like(q0)
        direction = direction / torch.linalg.norm(direction)

        def loss_fn(qvec: torch.Tensor) -> torch.Tensor:
            n = xq0.numel()
            xq = qvec[:n]
            yq = qvec[n : 2 * n]
            zq = qvec[2 * n :]
            out = self.model.evaluate(xq=xq, yq=yq, zq=zq)
            return (out**2).sum()

        ad = self._autograd_directional_derivative(loss_fn, q0, direction)
        fd = self._finite_difference_directional_derivative(loss_fn, q0, direction)
        self._assert_close_relative(ad, fd, rel_tol=1e-6)

    def test_solve_multi_rhs_grad_wrt_rhs(self) -> None:
        """LU-based multi-RHS solve is differentiable wrt RHS inputs."""
        matrix = torch.tensor(
            [[3.0, 1.0, 0.0], [1.0, 4.0, 2.0], [0.0, 2.0, 5.0]], dtype=torch.float64
        )
        rhs = torch.tensor(
            [[1.0, 2.0], [0.0, -1.0], [3.0, 4.0]],
            dtype=torch.float64,
            requires_grad=True,
        )

        out = _solve_multi_rhs(matrix=matrix, rhs=rhs)
        out.sum().backward()

        assert rhs.grad is not None
        self.assertTrue(torch.isfinite(rhs.grad).all().item())
        self.assertGreater(rhs.grad.abs().sum().item(), 0.0)

    def test_solve_multi_rhs_grad_wrt_rhs_matches_finite_differences(self) -> None:
        """Directional derivative wrt RHS inputs matches finite differences."""
        matrix = torch.tensor(
            [[3.0, 1.0, 0.0], [1.0, 4.0, 2.0], [0.0, 2.0, 5.0]], dtype=torch.float64
        )
        rhs0 = torch.tensor([[1.0, 2.0], [0.0, -1.0], [3.0, 4.0]], dtype=torch.float64)
        direction = torch.tensor(
            [[0.7, -0.5], [0.1, 0.2], [-0.3, 0.4]], dtype=torch.float64
        )
        direction = direction / torch.linalg.norm(direction)

        def loss_fn(rhs: torch.Tensor) -> torch.Tensor:
            out = _solve_multi_rhs(matrix=matrix, rhs=rhs)
            return (out**2).sum()

        ad = self._autograd_directional_derivative(loss_fn, rhs0, direction)
        fd = self._finite_difference_directional_derivative(loss_fn, rhs0, direction)
        self._assert_close_relative(ad, fd, rel_tol=1e-7)

    def test_state_dict_round_trip(self) -> None:
        """Serialized/deserialized model remains numerically equivalent."""
        restored = TorchBSpline3D.from_state_dict(self.model.state_dict())
        xq = torch.tensor([0.2, 0.5], dtype=torch.float64)
        yq = torch.tensor([-0.4, 0.9], dtype=torch.float64)
        zq = torch.tensor([2.2, 2.9], dtype=torch.float64)

        self.assertTrue(
            torch.allclose(
                self.model.evaluate(xq=xq, yq=yq, zq=zq),
                restored.evaluate(xq=xq, yq=yq, zq=zq),
            )
        )

    def test_to_changes_dtype(self) -> None:
        """``to(dtype=...)`` propagates dtype conversion to spline tensors."""
        model32 = self.model.to(dtype=torch.float32)
        self.assertEqual(model32.tx.dtype, torch.float32)
        self.assertEqual(model32.bcoef.dtype, torch.float32)

    def test_constructor_validation(self) -> None:
        """Constructor validates knot-vector lengths against coefficient shape/order."""
        with self.assertRaisesRegex(ValueError, "tx length must equal"):
            TorchBSpline3D(
                tx=self.model.tx[:-1],
                ty=self.model.ty,
                tz=self.model.tz,
                kx=self.model.kx,
                ky=self.model.ky,
                kz=self.model.kz,
                bcoef=self.model.bcoef,
            )
