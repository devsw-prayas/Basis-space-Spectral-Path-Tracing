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

    def buildCholesky(self):
        self.m_chol = torch.linalg.cholesky(self.m_gram)

    def project(self, spectrum: Tensor) -> Tensor:
        B = self.m_basisRaw
        w = self.m_domain.m_weights

        if spectrum.device != B.device:
            spectrum = spectrum.to(B.device)
        if spectrum.dtype != B.dtype:
            spectrum = spectrum.to(B.dtype)

        b = ((B * w) @ spectrum).unsqueeze(1)   # [M, 1]

        y     = torch.linalg.solve_triangular(self.m_chol,   b, upper=False)
        alpha = torch.linalg.solve_triangular(self.m_chol.T, y, upper=True)

        return alpha.squeeze(1)

    def projectBatch(self, spectra: Tensor) -> Tensor:
        B = self.m_basisRaw
        w = self.m_domain.m_weights

        if spectra.device != B.device:
            spectra = spectra.to(B.device)
        if spectra.dtype != B.dtype:
            spectra = spectra.to(B.dtype)

        Bw    = B * w
        rhs   = (Bw @ spectra.T).T
        y     = torch.linalg.solve_triangular(self.m_chol,   rhs.T, upper=False)
        alpha = torch.linalg.solve_triangular(self.m_chol.T, y,     upper=True)

        return alpha.T

    def reconstruct(self, coeffs: Tensor) -> Tensor:
        B = self.m_basisRaw
        if coeffs.device != B.device:
            coeffs = coeffs.to(B.device)
        if coeffs.dtype != B.dtype:
            coeffs = coeffs.to(B.dtype)
        return coeffs @ B


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
        self.buildCholesky()

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
        numWide: int,
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

        if numWide > self.m_K:
            raise ValueError("numWide cannot exceed number of centers.")

        self.m_numWide   = numWide
        self.m_numNarrow = self.m_K - numWide

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
        self.buildCholesky()

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

        sigmaMatrix = torch.empty(K, N, device=device, dtype=dtype)
        sigmaMatrix[:self.m_numWide, :]  = wideSigmas.unsqueeze(0)
        sigmaMatrix[self.m_numWide:, :]  = narrowSigmas.unsqueeze(0)

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
        self.m_basisRaw = (HDiag * gaussian) / normsTiled.unsqueeze(1)
