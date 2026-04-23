import torch
from torch import Tensor
from research.engine.basis import SpectralBasis

class SpectralState:
    """
    Coefficient-space state:
        α ∈ R^M
    Mutable.
    """

    def __init__(self, basis: SpectralBasis, coeffs: Tensor):
        self.m_basis = basis

        if coeffs.device != basis.m_basisRaw.device:
            coeffs = coeffs.to(basis.m_basisRaw.device)
        if coeffs.dtype != basis.m_basisRaw.dtype:
            coeffs = coeffs.to(basis.m_basisRaw.dtype)
        if coeffs.shape[0] != basis.m_M:
            raise ValueError(
                f"Coefficient dimension mismatch: got {coeffs.shape[0]}, "
                f"expected {basis.m_M}."
            )

        self.m_coeffs = coeffs.clone()

    def norm(self) -> Tensor:
        return torch.linalg.norm(self.m_coeffs)

    def zeroInplace(self):
        self.m_coeffs.zero_()

    def clone(self) -> "SpectralState":
        return SpectralState(self.m_basis, self.m_coeffs.clone())
