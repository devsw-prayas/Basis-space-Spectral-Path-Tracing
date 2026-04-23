import torch
from torch import Tensor
from typing import Optional, Callable, Dict, Any, List, Tuple, Literal

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
    Factory for creating the Eight Spectral Operators as defined in the 
    Macro-Optics specification. All operators are constructed in Whitened Space (Â).
    """

    @staticmethod
    def _createWhitenedFromRaw(basis: SpectralBasis, M_raw: Tensor) -> Tensor:
        """Helper to convert a raw moment matrix into a Whitened Space operator Â = L^-1 M_raw L^-T"""
        L = basis.m_chol
        # solve(L, M_raw) -> L^-1 M_raw
        Y = torch.linalg.solve_triangular(L, M_raw, upper=False)
        # solve(L, Y^T) -> L^-1 (L^-1 M_raw)^T = L^-1 M_raw^T L^-T
        # Since M_raw is usually symmetric, this gives L^-1 M_raw L^-T
        A_whitened = torch.linalg.solve_triangular(L, Y.T, upper=False).T
        return A_whitened

    @staticmethod
    def createAbsorption(basis: SpectralBasis, sigmaA: Callable[[Tensor], Tensor], distance: float) -> SpectralOperator:
        """1. Beer-Lambert's Operator: T(λ) = e^-σa(λ) d"""
        B, w, lbda = basis.m_basisRaw, basis.m_domain.m_weights, basis.m_domain.m_lambda
        T = torch.exp(-sigmaA(lbda) * distance)
        M_raw = (B * (w * T)) @ B.T
        A_wht = SpectralOperatorFactory._createWhitenedFromRaw(basis, M_raw)
        return SpectralOperator(basis, A_wht, torch.zeros(basis.m_M, device=A_wht.device))

    @staticmethod
    def createFresnel(basis: SpectralBasis, F_inf: Tensor) -> Dict[str, SpectralOperator]:
        """2. Fresnel's Operator: Schlick decomposition into 4 sub-operators."""
        B, w = basis.m_basisRaw, basis.m_domain.m_weights
        
        def mkOp(profile):
            M_raw = (B * (w * profile)) @ B.T
            A_wht = SpectralOperatorFactory._createWhitenedFromRaw(basis, M_raw)
            return SpectralOperator(basis, A_wht, torch.zeros(basis.m_M, device=A_wht.device))

        return {
            "P0":     mkOp(F_inf),
            "Psq":    mkOp(F_inf**2),
            "Rcross": mkOp(F_inf * (1.0 - F_inf)),
            "Qcomp":  mkOp((1.0 - F_inf)**2)
        }

    @staticmethod
    def createThinFilm(basis: SpectralBasis, n: float, d: float) -> SpectralOperator:
        """3. Fabry-Airy's Operator: T(λ) = 0.5 * (1 + cos(4πnd/λ))"""
        B, w, lbda = basis.m_basisRaw, basis.m_domain.m_weights, basis.m_domain.m_lambda
        T = 0.5 * (1.0 + torch.cos(4.0 * torch.pi * n * d / lbda))
        M_raw = (B * (w * T)) @ B.T
        A_wht = SpectralOperatorFactory._createWhitenedFromRaw(basis, M_raw)
        return SpectralOperator(basis, A_wht, torch.zeros(basis.m_M, device=A_wht.device))

    @staticmethod
    def createFluorescence(basis: SpectralBasis, e: Tensor, a: Tensor) -> SpectralOperator:
        """4. Stokes' Operator: Bispectral Rank-1 kernel K = e * a^T"""
        # Project emission and absorption profiles into whitened space
        e_wht = basis.projectWhitened(e) # e_wht = L^-1 (B w e)
        a_wht = basis.projectWhitened(a) # a_wht = L^-1 (B w a)
        
        # Â = e_wht * a_wht^T
        A_wht = torch.outer(e_wht, a_wht)
        return SpectralOperator(basis, A_wht, torch.zeros(basis.m_M, device=A_wht.device))

    @staticmethod
    def createDispersion(basis: SpectralBasis, A: float, B: float, C: float) -> List[SpectralOperator]:
        """5. Cauchy's Operator: n(λ) = A + B/λ^2 + C/λ^4 (Spectral-Geometric Lobe Coupling)"""
        # Returns K operators, one per GHGSF lobe
        lbda = basis.m_domain.m_lambda
        n = A + B/(lbda**2) + C/(lbda**4)
        # Partition of unity logic would be implemented here to derive lobe-specific selection ops
        # For now, we provide the identity-partition stub
        return [SpectralOperator.identity(basis) for _ in range(basis.m_K)]

    @staticmethod
    def createScattering(basis: SpectralBasis, type: Literal["Rayleigh", "Mie"], 
                         sigmaS_base: float, distance: float, alpha: float = 4.0) -> SpectralOperator:
        """6/7. Rayleigh/Mie Scattering: T(λ) = e^-σs(λ) d"""
        B, w, lbda = basis.m_basisRaw, basis.m_domain.m_weights, basis.m_domain.m_lambda
        sigmaS = sigmaS_base * (lbda / 550.0)**(-alpha) # Alpha=4 for Rayleigh, 0-2 for Mie
        T = torch.exp(-sigmaS * distance)
        M_raw = (B * (w * T)) @ B.T
        A_wht = SpectralOperatorFactory._createWhitenedFromRaw(basis, M_raw)
        return SpectralOperator(basis, A_wht, torch.zeros(basis.m_M, device=A_wht.device))

    @staticmethod
    def createRaman(basis: SpectralBasis, shift_nm: float) -> SpectralOperator:
        """8. Raman's Operator: Banded off-diagonal frequency shift."""
        # This requires a shift operator in wavelength space before projection
        # Placeholder for banded matrix construction
        return SpectralOperator.identity(basis)

    @staticmethod
    def createEmission(basis: SpectralBasis, emission: Tensor) -> SpectralOperator:
        b_wht = basis.projectWhitened(emission)
        A_zero = torch.zeros((basis.m_M, basis.m_M), device=b_wht.device)
        return SpectralOperator(basis, A_zero, b_wht)

    @staticmethod
    def createLocalization(basis: SpectralBasis, lambdaQ: float, 
                           sigma: Optional[float] = None, 
                           normalized: bool = False) -> SpectralOperator:
        """
        The Splitting Kernel: Creates an importance window around λq.
        If normalized=True, ensures Partition of Unity across the spectral domain.
        """
        lbda = basis.m_domain.m_lambda
        device, dtype = lbda.device, lbda.dtype
        
        # Default sigma logic (one-half lobe width)
        if sigma is None:
            span = (lbda[-1] - lbda[0]).item()
            sigma = span / (2.0 * basis.m_K)

        T = torch.exp(-0.5 * ((lbda - lambdaQ) / sigma) ** 2)

        if normalized:
            # Divide by the sum of all Gaussian lobes to enforce Partition of Unity
            centers = basis.m_centers if hasattr(basis, "m_centers") else torch.linspace(lbda[0], lbda[-1], basis.m_K, device=device)
            all_lobes = torch.exp(-0.5 * ((lbda.unsqueeze(0) - centers.unsqueeze(1)) / sigma) ** 2)
            denom = all_lobes.sum(dim=0).clamp(min=1e-12)
            T = T / denom

        B, w = basis.m_basisRaw, basis.m_domain.m_weights
        M_raw = (B * (w * T)) @ B.T
        A_wht = SpectralOperatorFactory._createWhitenedFromRaw(basis, M_raw)
        return SpectralOperator(basis, A_wht, torch.zeros(basis.m_M, device=device))
