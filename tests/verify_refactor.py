import torch
import sys
import os

# Add the current directory to path so we can import the 'research' package
sys.path.append(os.getcwd())

from research.engine.domain import SpectralDomain
from research.engine.basis import GHGSFDualDomainBasis
from research.engine.operator import SpectralOperatorFactory
from research.engine.state import SpectralState

def testRefactor():
    print("Initializing Spectral Domain...")
    domain = SpectralDomain(
        lambdaMin=380.0,
        lambdaMax=780.0,
        numSamples=401,
        device=torch.device("cpu"),
        dtype=torch.float64
    )

    print("Initializing Dual Domain Basis...")
    centers = torch.linspace(400, 700, 8).tolist()
    basis = GHGSFDualDomainBasis(
        domain=domain,
        centers=centers,
        wideIndices=list(range(4)),
        wideSigmaMin=10.0,
        wideSigmaMax=15.0,
        narrowSigmaMin=5.0,
        narrowSigmaMax=8.0,
        order=6
    )
    basis.buildCholesky()

    print(f"Basis size M: {basis.m_M}")

    # Create a test spectrum (Gaussian)
    lbda = domain.m_lambda
    testSpectrum = torch.exp(-0.5 * ((lbda - 550.0) / 20.0)**2)

    print("Projecting spectrum...")
    coeffs = basis.project(testSpectrum)
    state = SpectralState(basis, coeffs)

    print(f"Initial state norm: {state.norm().item():.6f}")

    # Create an absorption operator
    def sigmaAFn(l): return torch.full_like(l, 0.01)
    absOp = SpectralOperatorFactory.createAbsorption(basis, sigmaAFn, distance=10.0)

    print("Applying absorption...")
    absOp.apply(state)
    print(f"Post-absorption state norm: {state.norm().item():.6f}")

    # Reconstruct
    reconstructed = basis.reconstruct(state.m_coeffs)
    peak = reconstructed.max().item()
    print(f"Reconstructed peak value: {peak:.6f}")

    # Success check
    if peak < 1.0 and state.norm() < coeffs.norm():
        print("\nRefactor Verification: SUCCESS ✓")
    else:
        print("\nRefactor Verification: FAILED ✗")

if __name__ == "__main__":
    testRefactor()
