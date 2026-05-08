import torch
from torch import Tensor
from typing import Optional, Callable, Dict, Any, List, Tuple, Literal, NamedTuple

from research.engine.basis import SpectralBasis


class FresnelOps(NamedTuple):
    P0: "SpectralOperator"
    Psq: "SpectralOperator"
    Rcross: "SpectralOperator"
    Qcomp: "SpectralOperator"


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
        M = basis.m_M
        device = basis.m_basisRaw.device
        dtype = basis.m_basisRaw.dtype

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
        M = basis.m_M
        device = basis.m_basisRaw.device
        dtype = basis.m_basisRaw.dtype
        return SpectralOperator(
            basis,
            torch.eye(M, device=device, dtype=dtype),
            torch.zeros(M, device=device, dtype=dtype)
        )

    @staticmethod
    def zero(basis: SpectralBasis) -> "SpectralOperator":
        M = basis.m_M
        device = basis.m_basisRaw.device
        dtype = basis.m_basisRaw.dtype
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
    def createFresnel(basis: SpectralBasis, F_inf: Tensor) -> FresnelOps:
        """2. Fresnel's Operator: Schlick decomposition into 4 sub-operators."""
        B, w = basis.m_basisRaw, basis.m_domain.m_weights

        def mkOp(profile):
            M_raw = (B * (w * profile)) @ B.T
            A_wht = SpectralOperatorFactory._createWhitenedFromRaw(basis, M_raw)
            return SpectralOperator(basis, A_wht, torch.zeros(basis.m_M, device=A_wht.device, dtype=A_wht.dtype))

        return FresnelOps(
            P0=mkOp(F_inf),
            Psq=mkOp(F_inf ** 2),
            Rcross=mkOp(F_inf * (1.0 - F_inf)),
            Qcomp=mkOp((1.0 - F_inf) ** 2)
        )

    @staticmethod
    def assembleFresnel(
            fresnel_ops: FresnelOps,
            cos_theta: float,
            basis: SpectralBasis
    ) -> "SpectralOperator":
        """
        Render-time Fresnel assembly. The only dynamic operator in the hot loop.
        Â_F(θ) = P0 + (1 - cosθ)^5 * (I_M - P0)
               = (1 - t) * P0 + t * I_M    where t = (1 - cosθ)^5

        Takes the 4 baked operators from createFresnel and a per-ray cos_theta.
        Cost: two scalar-matrix multiplies + one matrix add. O(M^2) but cheap.
        """
        t = (1.0 - cos_theta) ** 5
        M = basis.m_M
        device = basis.m_basisRaw.device
        dtype = basis.m_basisRaw.dtype

        I_M = torch.eye(M, device=device, dtype=dtype)
        A_assembled = (1.0 - t) * fresnel_ops.P0.m_A + t * I_M
        return SpectralOperator(
            basis,
            A_assembled,
            torch.zeros(M, device=device, dtype=dtype)
        )

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
        e_wht = basis.projectWhitened(e)  # e_wht = L^-1 (B w e)
        a_wht = basis.projectWhitened(a)  # a_wht = L^-1 (B w a)

        # Â = e_wht * a_wht^T
        A_wht = torch.outer(e_wht, a_wht)
        return SpectralOperator(basis, A_wht, torch.zeros(basis.m_M, device=A_wht.device))

    @staticmethod
    def createDispersion(basis: SpectralBasis, A: float, B: float, C: float) -> List[SpectralOperator]:
        """5. Cauchy's Operator: K rank-1 lobes with softmax partition of unity Σ Âk = IM.
        A, B, C define n(λ) = A + B/λ² + C/λ⁴ — used by the renderer for per-lobe refraction
        angles at geometric routing time. The spectral operators are the partition decomposition.
        """
        B_mat, w, lbda = basis.m_basisRaw, basis.m_domain.m_weights, basis.m_domain.m_lambda
        centers = basis.m_centers
        K = basis.m_K
        device, dtype = lbda.device, lbda.dtype

        # Lobe window width estimated from center spacing
        if K > 1:
            sigmaLobe = (centers[-1] - centers[0]) / (K - 1) * 0.6
        else:
            sigmaLobe = torch.tensor(50.0, device=device, dtype=dtype)

        # Unnormalized Gaussian windows per lobe: (K, L)
        windows = torch.exp(-0.5 * ((lbda.unsqueeze(0) - centers.unsqueeze(1)) / sigmaLobe) ** 2)

        # Softmax normalize → Σ_k T_k(λ) = 1 → guarantees Σ_k Âk = IM in whitened space
        T_lobes = windows / windows.sum(dim=0, keepdim=True).clamp(min=1e-12)

        ops = []
        for k in range(K):
            M_raw = (B_mat * (w * T_lobes[k])) @ B_mat.T
            A_wht = SpectralOperatorFactory._createWhitenedFromRaw(basis, M_raw)
            ops.append(SpectralOperator(basis, A_wht, torch.zeros(basis.m_M, device=device, dtype=dtype)))

        return ops

    @staticmethod
    def createScattering(basis: SpectralBasis, type: Literal["Rayleigh", "Mie"],
                         sigmaS_base: float, distance: float, alpha: float = 4.0) -> SpectralOperator:
        """6/7. Rayleigh/Mie Scattering: T(λ) = e^-σs(λ) d"""
        B, w, lbda = basis.m_basisRaw, basis.m_domain.m_weights, basis.m_domain.m_lambda
        sigmaS = sigmaS_base * (lbda / 550.0) ** (-alpha)  # Alpha=4 for Rayleigh, 0-2 for Mie
        T = torch.exp(-sigmaS * distance)
        M_raw = (B * (w * T)) @ B.T
        A_wht = SpectralOperatorFactory._createWhitenedFromRaw(basis, M_raw)
        return SpectralOperator(basis, A_wht, torch.zeros(basis.m_M, device=A_wht.device))

    @staticmethod
    def createRaman(basis: SpectralBasis, shift_nm: float, sigmaRaman: float = 10.0) -> SpectralOperator:
        """8. Raman's Operator: Banded off-diagonal via 2D Galerkin projection of shift kernel.
        K(λo, λi) is a Gaussian concentrated near λo = λi + shift_nm.
        sigmaRaman controls Raman linewidth (~10 nm typical).
        Note: allocates a transient (L×L) kernel — ~134 MB at L=4096 float64.
        """
        print("weight sum =", basis.m_domain.m_weights.sum().item())
        print("w[0] =", basis.m_domain.m_weights[0].item())
        print("h =", (830 - 380) / (4096 - 1))
        B_mat, w, lbda = basis.m_basisRaw, basis.m_domain.m_weights, basis.m_domain.m_lambda
        device, dtype = lbda.device, lbda.dtype

        # K(λo, λi) = Gaussian(λo - λi - shift_nm): (L, L)
        lbda_o = lbda.unsqueeze(1)
        lbda_i = lbda.unsqueeze(0)
        K_mat = torch.exp(-0.5 * ((lbda_o - lbda_i - shift_nm) / sigmaRaman) ** 2)

        # Column-normalize: unit quantum yield per input wavelength
        K_mat = K_mat / K_mat.sum(dim=0, keepdim=True).clamp(min=1e-12)

        # 2D Galerkin: M_raw[i,j] = Σ_{a,b} B[i,a] w_a K[a,b] w_b B[j,b]
        Bw = B_mat * w
        M_raw = Bw @ K_mat @ Bw.T

        Bw = basis.m_basisRaw * basis.m_domain.m_weights
        print("Bw[0,0] =", Bw[0, 0].item())
        print("Bw[0,1] =", Bw[0, 1].item())
        print("w[0] =", basis.m_domain.m_weights[0].item())
        print("B[0,0] =", basis.m_basisRaw[0, 0].item())

        A_wht = SpectralOperatorFactory._createWhitenedFromRaw(basis, M_raw)
        return SpectralOperator(basis, A_wht, torch.zeros(basis.m_M, device=device, dtype=dtype))

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
            centers = basis.m_centers if hasattr(basis, "m_centers") else torch.linspace(lbda[0], lbda[-1], basis.m_K,
                                                                                         device=device)
            all_lobes = torch.exp(-0.5 * ((lbda.unsqueeze(0) - centers.unsqueeze(1)) / sigma) ** 2)
            denom = all_lobes.sum(dim=0).clamp(min=1e-12)
            T = T / denom

        B, w = basis.m_basisRaw, basis.m_domain.m_weights
        M_raw = (B * (w * T)) @ B.T
        A_wht = SpectralOperatorFactory._createWhitenedFromRaw(basis, M_raw)
        return SpectralOperator(basis, A_wht, torch.zeros(basis.m_M, device=device))