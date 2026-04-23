import torch
from torch import Tensor

class SpectralDomain:
    """
    Discretized spectral interval [λ_min, λ_max]
    with trapezoidal quadrature weights.

    Owns:
        - sample points
        - integration weights
        - dtype
        - device
    """

    def __init__(
        self,
        lambdaMin: float,
        lambdaMax: float,
        numSamples: int,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64
    ):
        self.m_device = device
        self.m_dtype  = dtype
        self.m_count  = numSamples

        self.m_lambda = torch.linspace(
            lambdaMin,
            lambdaMax,
            numSamples,
            device=device,
            dtype=dtype
        )

        self.m_delta = self.m_lambda[1] - self.m_lambda[0]

        # Trapezoidal weights
        w = torch.ones(numSamples, device=device, dtype=dtype)
        w[0] *= 0.5
        w[-1] *= 0.5
        self.m_weights = w * self.m_delta

    def integrate(self, f: Tensor) -> Tensor:
        """
        ∫ f(λ) dλ
        """
        return torch.sum(f * self.m_weights)

    def innerProduct(self, f: Tensor, g: Tensor) -> Tensor:
        """
        ⟨f, g⟩ = ∫ f(λ) g(λ) dλ
        """
        return self.integrate(f * g)
