import torch
import numpy as np
from research.engine.domain import SpectralDomain
from research.engine.basis import GHGSFDualDomainBasis
from research.engine.operator import SpectralOperatorFactory
from research.engine.topology import generateTopology

def testOperators():
    print("=" * 63)
    print(" SPECTRAL ENGINE - OPERATOR VALIDATION SUITE")
    print("=" * 63)
    
    # Golden Config: Family 0, K=8, N=11, Scaling 1 (Linear)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domain = SpectralDomain(380, 830, 4096, device=device, dtype=torch.float64)
    
    K, N = 8, 11
    centers = generateTopology(0, K, margin=0.0)
    basis = GHGSFDualDomainBasis(
        domain=domain,
        centers=centers,
        numWide=K//2,
        wideSigmaMin=9.5, wideSigmaMax=11.5, wideScaleType="linear",
        narrowSigmaMin=7.0, narrowSigmaMax=9.0, narrowScaleType="linear",
        order=N
    )
    
    print(f"Basis Initialized: K={K}, N={N} (Size: {basis.m_M})")
    print(f"Condition Number: {torch.linalg.cond(basis.m_gram):.2e}")

    # Test Spectrum: Flat 1.0
    s_in = torch.ones(domain.m_count, device=device, dtype=torch.float64)
    alpha_raw = basis.project(s_in)
    alpha_wht = basis.projectWhitened(s_in)

    # 1. TEST: Dirac-like Stress Test (1nm peak at 500nm)
    print("\n[HARD TEST 1] Narrow Dirac Spike (1nm width at 500nm)")
    s_spike = torch.exp(-0.5 * ((domain.m_lambda - 500) / 1.0)**2)
    alpha_spike = basis.projectWhitened(s_spike)
    s_recon_spike = basis.reconstructWhitened(alpha_spike)
    
    # Measure RMS error and peak preservation
    rms_error = torch.sqrt(torch.mean((s_spike - s_recon_spike)**2)).item()
    peak_recon = s_recon_spike.max().item()
    print(f"  RMS Error: {rms_error:.2e}")
    print(f"  Peak Preservation: {peak_recon:.6f} (Original: 1.0) | {'OK' if rms_error < 5e-2 else 'FAIL'}")

    # 2. TEST: Beer-Lambert Absorption (50% uniform)
    print("\n[TEST 2] Beer-Lambert Absorption (50% uniform)")
    sigma_a = lambda lbda: torch.ones_like(lbda) * (-np.log(0.5) / 1.0)
    op_abs = SpectralOperatorFactory.createAbsorption(basis, sigma_a, distance=1.0)
    alpha_wht_out = op_abs.m_A @ alpha_wht
    s_out = basis.reconstructWhitened(alpha_wht_out)
    mean_val = s_out.mean().item()
    print(f"  Actual: {mean_val:.10f} | Error: {abs(mean_val - 0.5):.2e}")

    # 3. TEST: Stokes Fluorescence (Multi-Channel)
    print("\n[TEST 3] Multi-Channel Stokes (Complex Emission)")
    # Absorbs at 400nm, emits at 500nm AND 600nm
    a_prof = torch.exp(-0.5 * ((domain.m_lambda - 400) / 10.0)**2)
    e_prof = torch.exp(-0.5 * ((domain.m_lambda - 500) / 10.0)**2) + 0.5 * torch.exp(-0.5 * ((domain.m_lambda - 600) / 10.0)**2)
    
    op_stokes = SpectralOperatorFactory.createFluorescence(basis, e_prof, a_prof)
    alpha_in = basis.projectWhitened(a_prof)
    alpha_out = op_stokes.m_A @ alpha_in
    s_out_fluo = basis.reconstructWhitened(alpha_out)
    
    peak1 = s_out_fluo[torch.abs(domain.m_lambda - 500).argmin()].item()
    peak2 = s_out_fluo[torch.abs(domain.m_lambda - 600).argmin()].item()
    print(f"  Peak 1 (500nm): {peak1:.4f} | Peak 2 (600nm): {peak2:.4f}")

    # 4. TEST: Fresnel Normal Incidence (P0)
    print("\n[TEST 4] Fresnel Normal Incidence (P0)")
    f_inf = torch.ones_like(domain.m_lambda) * 0.04
    ops_fresnel = SpectralOperatorFactory.createFresnel(basis, f_inf)
    op_p0 = ops_fresnel["P0"]
    alpha_out_fres = op_p0.m_A @ alpha_wht
    s_out_fres = basis.reconstructWhitened(alpha_out_fres)
    reflectance = s_out_fres.mean().item()
    print(f"  Actual: {reflectance:.10f} | Error: {abs(reflectance - 0.04):.2e}")

    print("\n" + "=" * 63)
    print(" HARD VALIDATION COMPLETE")
    print("=" * 63)

    print("\n" + "=" * 63)
    print(" VALIDATION COMPLETE")
    print("=" * 63)

if __name__ == "__main__":
    testOperators()
