import torch
from torch import Tensor
from typing import Optional, Callable, Dict, Any, List, Tuple

from research.engine.basis import SpectralBasis

class SpectralOperator:
    """
    Affine operator in coefficient space:
        O(α) = A α + b
    """

    def __init__(
        self,
        basis: SpectralBasis,
        A: Tensor,
        b: Tensor
    ):
        self.m_basis = basis
        M      = basis.m_M
        device = basis.m_basisRaw.device
        dtype  = basis.m_basisRaw.dtype

        if A.shape != (M, M):
            raise ValueError(f"Matrix A must be shape [{M}, {M}], got {A.shape}.")
        if b.shape != (M,):
            raise ValueError(f"Vector b must be shape [{M}], got {b.shape}.")

        self.m_A = A.to(device=device, dtype=dtype)
        self.m_b = b.to(device=device, dtype=dtype)

    def apply(self, state):
        """α ← A α + b"""
        state.m_coeffs = torch.addmv(self.m_b, self.m_A, state.m_coeffs)

    def compose(self, other: "SpectralOperator") -> "SpectralOperator":
        """Returns self ∘ other (applies other first, then self)."""
        if self.m_basis is not other.m_basis:
            raise ValueError("Basis mismatch in operator composition.")

        A_new = self.m_A @ other.m_A
        b_new = self.m_A @ other.m_b + self.m_b

        return SpectralOperator(self.m_basis, A_new, b_new)

    @staticmethod
    def identity(basis: SpectralBasis) -> "SpectralOperator":
        M      = basis.m_M
        device = basis.m_basisRaw.device
        dtype  = basis.m_basisRaw.dtype
        return SpectralOperator(
            basis,
            torch.eye(M, device=device, dtype=dtype),
            torch.zeros(M, device=device, dtype=dtype)
        )

    @staticmethod
    def zero(basis: SpectralBasis) -> "SpectralOperator":
        M      = basis.m_M
        device = basis.m_basisRaw.device
        dtype  = basis.m_basisRaw.dtype
        return SpectralOperator(
            basis,
            torch.zeros((M, M), device=device, dtype=dtype),
            torch.zeros(M, device=device, dtype=dtype)
        )


class SpectralOperatorFactory:
    """
    Factory for creating specialized spectral operators.
    Consolidates logic for absorption, reflectance, emission, and localization.
    """

    @staticmethod
    def createAbsorption(
        basis: SpectralBasis,
        sigmaA: Callable[[Tensor], Tensor],
        distance: float
    ) -> SpectralOperator:
        B    = basis.m_basisRaw
        w    = basis.m_domain.m_weights
        L    = basis.m_chol
        lbda = basis.m_domain.m_lambda

        T = torch.exp(-sigmaA(lbda) * distance)
        M_raw = (B * (w * T)) @ B.T

        Y = torch.linalg.solve_triangular(L,   M_raw, upper=False)
        A = torch.linalg.solve_triangular(L.T, Y,     upper=True)
        b = torch.zeros(basis.m_M, device=A.device, dtype=A.dtype)

        return SpectralOperator(basis, A, b)

    @staticmethod
    def createReflectance(
        basis: SpectralBasis,
        reflectance: Tensor   # [L]
    ) -> SpectralOperator:
        B = basis.m_basisRaw
        w = basis.m_domain.m_weights
        L = basis.m_chol

        M_raw = (B * (w * reflectance)) @ B.T

        Y = torch.linalg.solve_triangular(L,   M_raw, upper=False)
        A = torch.linalg.solve_triangular(L.T, Y,     upper=True)
        b = torch.zeros(basis.m_M, device=A.device, dtype=A.dtype)

        return SpectralOperator(basis, A, b)

    @staticmethod
    def createEmission(
        basis: SpectralBasis,
        emission: Callable[[Tensor], Tensor]
    ) -> SpectralOperator:
        lbda = basis.m_domain.m_lambda
        S = emission(lbda)
        b = basis.project(S)
        A = torch.zeros((basis.m_M, basis.m_M), device=b.device, dtype=b.dtype)

        return SpectralOperator(basis, A, b)

    @staticmethod
    def createWhiten(basis: SpectralBasis) -> SpectralOperator:
        L = basis.m_chol
        A = L.T
        b = torch.zeros(basis.m_M, device=L.device, dtype=L.dtype)
        return SpectralOperator(basis, A, b)

    @staticmethod
    def createUnwhiten(basis: SpectralBasis) -> SpectralOperator:
        L      = basis.m_chol
        M      = basis.m_M
        device = L.device
        dtype  = L.dtype
        I = torch.eye(M, device=device, dtype=dtype)
        A = torch.linalg.solve_triangular(L.T, I, upper=True)
        b = torch.zeros(M, device=device, dtype=dtype)
        return SpectralOperator(basis, A, b)

    @staticmethod
    def createLocalization(
        basis: SpectralBasis,
        lambdaQ: float,
        sigma: Optional[float] = None,
        normalized: bool = False
    ) -> SpectralOperator:
        # Ported from engine/localization.py
        lbda = basis.m_domain.m_lambda
        device = lbda.device
        dtype  = lbda.dtype

        if sigma is None:
            # Infer sigma logic
            span = (lbda[-1] - lbda[0]).item()
            if hasattr(basis, "m_wideSigmaMin"):
                centers = basis.m_centers
                dists   = (centers - lambdaQ).abs()
                kNear   = dists.argmin().item()
                sigma = basis.m_wideSigmaMin if kNear < basis.m_numWide else basis.m_narrowSigmaMin
            elif hasattr(basis, "m_sigma"):
                sigma = basis.m_sigma
            else:
                sigma = span / (2.0 * basis.m_K)

        lambdaQT = torch.tensor(lambdaQ, device=device, dtype=dtype)
        sigmaT    = torch.tensor(sigma,    device=device, dtype=dtype)

        T = torch.exp(-0.5 * ((lbda - lambdaQT) / sigmaT) ** 2)

        if normalized:
            centersT = basis.m_centers.to(device=device, dtype=dtype)
            allT = torch.exp(-0.5 * ((lbda.unsqueeze(0) - centersT.unsqueeze(1)) / sigmaT) ** 2)
            denom = allT.sum(dim=0).clamp(min=1e-12)
            T = T / denom

        B = basis.m_basisRaw
        w = basis.m_domain.m_weights
        L = basis.m_chol

        M_raw = (B * (w * T)) @ B.T
        Y = torch.linalg.solve_triangular(L,   M_raw, upper=False)
        A = torch.linalg.solve_triangular(L.T, Y,     upper=True)
        b = torch.zeros(basis.m_M, device=device, dtype=dtype)

        return SpectralOperator(basis, A, b)
