import torch
from torch import Tensor
from abc import ABC, abstractmethod
from typing import List, Literal, Optional

from research.engine.domain import SpectralDomain
from research.engine.utils import hermiteBasis

class SpectralBasis(ABC):
    """
    Abstract Base Class for all spectral bases.
    Defines the interface for projection, reconstruction, and Gram matrix storage.
    """
    def __init__(self, domain: SpectralDomain):
        self.m_domain = domain
        self.m_basisRaw: Optional[Tensor] = None
        self.m_gram: Optional[Tensor] = None
        self.m_chol: Optional[Tensor] = None
        self.m_M: int = 0

    @abstractmethod
    def buildBasis(self):
        pass

    def buildGram(self):
        B = self.m_basisRaw
        w = self.m_domain.m_weights
        self.m_gram = (B * w) @ B.T
        self.m_gram = 0.5 * (self.m_gram + self.m_gram.T)

    def buildCholesky(self):
        self.m_chol = torch.linalg.cholesky(self.m_gram)
        # Precompute Whitened Basis for fast reconstruction: B_wht = L^-1 B_raw
        self.m_basisWhitened = torch.linalg.solve_triangular(self.m_chol, self.m_basisRaw, upper=False)

    def toWhitened(self, alpha_raw: Tensor) -> Tensor:
        """Converts raw coefficients to whitened space: alpha_w = L^T alpha_raw"""
        return (self.m_chol.T @ alpha_raw.unsqueeze(-1)).squeeze(-1)

    def toRaw(self, alpha_w: Tensor) -> Tensor:
        """Converts whitened coefficients to raw space: alpha_raw = L^-T alpha_w"""
        return torch.linalg.solve_triangular(self.m_chol.T, alpha_w.unsqueeze(-1), upper=True).squeeze(-1)

    def project(self, spectrum: Tensor) -> Tensor:
        """Returns standard raw coefficients (alpha_raw)."""
        B, w = self.m_basisRaw, self.m_domain.m_weights
        if spectrum.device != B.device: spectrum = spectrum.to(B.device)
        if spectrum.dtype != B.dtype: spectrum = spectrum.to(B.dtype)
        
        b = (B * w) @ spectrum
        y = torch.linalg.solve_triangular(self.m_chol, b.unsqueeze(-1), upper=False)
        alpha = torch.linalg.solve_triangular(self.m_chol.T, y, upper=True)
        return alpha.squeeze(-1)

    def projectWhitened(self, spectrum: Tensor) -> Tensor:
        """Projects directly into whitened space (alpha_w)."""
        B, w = self.m_basisRaw, self.m_domain.m_weights
        if spectrum.device != B.device: spectrum = spectrum.to(B.device)
        if spectrum.dtype != B.dtype: spectrum = spectrum.to(B.dtype)
        
        b = (B * w) @ spectrum
        y = torch.linalg.solve_triangular(self.m_chol, b.unsqueeze(-1), upper=False)
        return y.squeeze(-1)

    def reconstruct(self, alpha_raw: Tensor) -> Tensor:
        """Reconstructs from raw coefficients: s = alpha_raw @ B_raw"""
        return alpha_raw @ self.m_basisRaw

    def reconstructWhitened(self, alpha_w: Tensor) -> Tensor:
        """Reconstructs from whitened coefficients: s = alpha_w @ B_wht"""
        return alpha_w @ self.m_basisWhitened


class GHGSFBasis(SpectralBasis):
    """
    Gaussian-Hermite Global Spectral Function (Multi-Lobe)
    Standard implementation with uniform sigma.
    """
    def __init__(
        self,
        domain: SpectralDomain,
        centers: List[float],
        sigma: float,
        order: int
    ):
        super().__init__(domain)
        self.m_centers = torch.tensor(
            centers,
            device=domain.m_device,
            dtype=domain.m_dtype
        )
        self.m_sigma = sigma
        self.m_order = order
        self.m_K = len(centers)
        self.m_N = order
        self.m_M = self.m_K * self.m_N

        self.buildBasis()
        self.buildGram()

    def buildBasis(self):
        lbda    = self.m_domain.m_lambda
        sigma   = self.m_sigma
        centers = self.m_centers
        device  = lbda.device
        dtype   = lbda.dtype
        K       = self.m_K
        N       = self.m_N
        L       = lbda.shape[0]

        n_idx      = torch.arange(N, device=device, dtype=dtype)
        factorials = torch.exp(torch.lgamma(n_idx + 1))
        sqrt_pi    = torch.tensor(torch.pi, device=device, dtype=dtype).sqrt()
        norms      = torch.sqrt((2.0 ** n_idx) * factorials * sqrt_pi)

        x = (lbda.unsqueeze(0) - centers.unsqueeze(1)) / sigma
        x_rep  = x.unsqueeze(1).expand(K, N, L)
        x_flat = x_rep.reshape(K * N, L)

        H_full = hermiteBasis(N, x_flat)
        row_idx = torch.arange(K * N, device=device)
        ord_idx = row_idx % N
        H_diag  = H_full[row_idx, ord_idx, :]

        x_tiled  = x.unsqueeze(1).expand(K, N, L).reshape(K * N, L)
        gaussian = torch.exp(-0.5 * x_tiled ** 2)
        norms_tiled = norms.repeat(K)

        self.m_basisRaw = (H_diag * gaussian) / norms_tiled.unsqueeze(1)


ScaleType = Literal["constant", "linear", "sqrt", "power"]

class GHGSFDualDomainBasis(SpectralBasis):
    """
    Gaussian-Hermite Multi-Lobe Basis (Frontier Implementation)
    Supports two independent sigma domains (Wide and Narrow lobes)
    and various scaling schedules.
    """
    def __init__(
        self,
        domain: SpectralDomain,
        centers: List[float],
        wideIndices: List[int],
        wideSigmaMin: float,
        wideSigmaMax: Optional[float],
        wideScaleType: ScaleType = "sqrt",
        wideGamma: float = 0.5,
        narrowSigmaMin: float = 4.0,
        narrowSigmaMax: Optional[float] = None,
        narrowScaleType: ScaleType = "sqrt",
        narrowGamma: float = 0.5,
        order: int = 6
    ):
        super().__init__(domain)
        self.m_centers = torch.tensor(
            centers, device=domain.m_device, dtype=domain.m_dtype
        )
        self.m_K = len(centers)
        self.m_N = order
        self.m_M = self.m_K * self.m_N

        wideSet = set(wideIndices)
        if any(i < 0 or i >= self.m_K for i in wideSet):
            raise ValueError("wideIndices contains out-of-range index.")

        self.m_wideIndices   = list(wideIndices)
        self.m_narrowIndices = [i for i in range(self.m_K) if i not in wideSet]

        self.m_wideSigmaMin  = wideSigmaMin
        self.m_wideSigmaMax  = wideSigmaMax if wideSigmaMax is not None else wideSigmaMin
        self.m_wideScaleType = wideScaleType
        self.m_wideGamma      = wideGamma

        self.m_narrowSigmaMin  = narrowSigmaMin
        self.m_narrowSigmaMax  = narrowSigmaMax if narrowSigmaMax is not None else narrowSigmaMin
        self.m_narrowScaleType = narrowScaleType
        self.m_narrowGamma      = narrowGamma

        self.buildBasis()
        self.buildGram()

    def _sigmaSchedule(
        self,
        sigmaMin: float,
        sigmaMax: float,
        scaleType: ScaleType,
        gamma: float,
        device,
        dtype
    ) -> Tensor:
        N = self.m_N
        if N <= 1 or scaleType == "constant":
            return torch.full((N,), sigmaMin, device=device, dtype=dtype)

        t     = torch.linspace(0.0, 1.0, N, device=device, dtype=dtype)
        delta = sigmaMax - sigmaMin

        if scaleType == "linear":
            return sigmaMin + delta * t
        elif scaleType == "sqrt":
            return sigmaMin + delta * torch.sqrt(t)
        elif scaleType == "power":
            return sigmaMin + delta * torch.pow(t, gamma)
        else:
            raise ValueError(f"Unknown scaleType: {scaleType}")

    def buildBasis(self):
        lbda    = self.m_domain.m_lambda
        device  = lbda.device
        dtype   = lbda.dtype
        K       = self.m_K
        N       = self.m_N
        L       = lbda.shape[0]

        wideSigmas   = self._sigmaSchedule(
            self.m_wideSigmaMin, self.m_wideSigmaMax,
            self.m_wideScaleType, self.m_wideGamma, device, dtype
        )
        narrowSigmas = self._sigmaSchedule(
            self.m_narrowSigmaMin, self.m_narrowSigmaMax,
            self.m_narrowScaleType, self.m_narrowGamma, device, dtype
        )

        sigmaMatrix  = torch.empty(K, N, device=device, dtype=dtype)
        wideTensor   = torch.tensor(self.m_wideIndices,   device=device, dtype=torch.long)
        narrowTensor = torch.tensor(self.m_narrowIndices, device=device, dtype=torch.long)
        if wideTensor.numel() > 0:
            sigmaMatrix[wideTensor, :]   = wideSigmas.unsqueeze(0)
        if narrowTensor.numel() > 0:
            sigmaMatrix[narrowTensor, :] = narrowSigmas.unsqueeze(0)

        lbdaExp    = lbda.unsqueeze(0).unsqueeze(0)
        centersExp = self.m_centers.unsqueeze(1).unsqueeze(2)
        sigmaExp   = sigmaMatrix.unsqueeze(2)

        xFull = (lbdaExp - centersExp) / sigmaExp
        xFlat = xFull.reshape(K * N, L)

        HFull = hermiteBasis(N, xFlat)
        rowIdx = torch.arange(K * N, device=device)
        ordIdx = rowIdx % N
        HDiag  = HFull[rowIdx, ordIdx, :]

        nIdx      = torch.arange(N, device=device, dtype=dtype)
        factorials = torch.exp(torch.lgamma(nIdx + 1))
        sqrtPi    = torch.tensor(torch.pi, device=device, dtype=dtype).sqrt()
        norms      = torch.sqrt((2.0 ** nIdx) * factorials * sqrtPi)
        normsTiled = norms.repeat(K)

        gaussian = torch.exp(-0.5 * xFlat ** 2)
        sigma_flat = sigmaMatrix.reshape(K * N, 1)

        self.m_basisRaw = ((HDiag * gaussian) / normsTiled.unsqueeze(1)) / torch.sqrt(sigma_flat)