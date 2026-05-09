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
    # SUB-NANOMETER RESOLUTION: 16,384 samples (~0.027nm delta)
    domain = SpectralDomain(380, 830, 4096, device=device, dtype=torch.float64)
    
    K, N = 8, 11
    centers, _ = generateTopology(0, K, margin=0.0)
    basis = GHGSFDualDomainBasis(
        domain=domain,
        centers=centers,
        wideIndices=list(range(K // 2)),
        wideSigmaMin=9.5, wideSigmaMax=11.5, wideScaleType="linear",
        narrowSigmaMin=7.0, narrowSigmaMax=9.0, narrowScaleType="linear",
        order=N
    )
    basis.buildCholesky()

    print(f"Basis Initialized: K={K}, N={N} (Size: {basis.m_M})")
    print(f"Domain Resolution: {domain.m_count} samples ({domain.m_delta.item():.4f}nm delta)")
    print(f"Condition Number: {torch.linalg.cond(basis.m_gram):.2e}")

    # Test Spectrum: Flat 1.0
    s_in = torch.ones(domain.m_count, device=device, dtype=torch.float64)
    alpha_raw = basis.project(s_in)
    alpha_wht = basis.projectWhitened(s_in)

    # 1. TEST: Dirac-like Stress Test (0.1nm peak at 500nm)
    print("\n[HARD TEST 1] Ultra-Narrow Spike (0.1nm width at 500nm)")
    s_spike = torch.exp(-0.5 * ((domain.m_lambda - 500) / 0.1)**2)
    alpha_spike = basis.projectWhitened(s_spike)
    s_recon_spike = basis.reconstructWhitened(alpha_spike)
    
    # ENERGY PRESERVATION (True Error)
    energy_in = domain.integrate(s_spike).item()
    energy_out = domain.integrate(s_recon_spike).item()
    energy_error = abs(energy_in - energy_out)

    # Measure RMS error and peak preservation
    rms_error = torch.sqrt(torch.mean((s_spike - s_recon_spike)**2)).item()
    peak_recon = s_recon_spike.max().item()
    
    print(f"  Energy In: {energy_in:.10f}")
    print(f"  Energy Out: {energy_out:.10f}")
    print(f"  True Energy Error: {energy_error:.2e} | {'OK' if energy_error < 1e-3 else 'FAIL'}")
    print(f"  RMS Error: {rms_error:.2e}")
    print(f"  Peak Preservation: {peak_recon:.6f} (Original: 1.0)")

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
    op_p0 = ops_fresnel.P0
    alpha_out_fres = op_p0.m_A @ alpha_wht
    s_out_fres = basis.reconstructWhitened(alpha_out_fres)
    reflectance = s_out_fres.mean().item()
    print(f"  Actual: {reflectance:.10f} | Error: {abs(reflectance - 0.04):.2e}")

    # 5. TEST: Multibounce Absorption (10 bounces at 50%)
    print("\n[MULTIBOUNCE 1] Deep Absorption (10 bounces at 50%)")
    sigma_a = lambda lbda: torch.ones_like(lbda) * (-np.log(0.5) / 1.0)
    op_abs = SpectralOperatorFactory.createAbsorption(basis, sigma_a, distance=1.0)
    
    alpha_multi = alpha_wht.clone()
    for _ in range(10):
        alpha_multi = op_abs.m_A @ alpha_multi
    
    s_multi = basis.reconstructWhitened(alpha_multi)
    expected = 0.5**10
    actual = s_multi.mean().item()
    print(f"  Expected: {expected:.10f} | Actual: {actual:.10f}")
    print(f"  Error: {abs(actual - expected):.2e} | {'OK' if abs(actual - expected) < 1e-6 else 'FAIL'}")

    # 6. TEST: Operator Composition (Absorption * Fresnel)
    print("\n[MULTIBOUNCE 2] Operator Composition (Abs 50% * Fresnel 4%)")
    alpha_iter = op_p0.m_A @ (op_abs.m_A @ alpha_wht)
    A_composed = op_p0.m_A @ op_abs.m_A
    alpha_comp = A_composed @ alpha_wht
    diff = torch.norm(alpha_iter - alpha_comp).item()
    print(f"  Composition Diff: {diff:.2e} | {'OK' if diff < 1e-12 else 'FAIL'}")

    print("\n" + "=" * 63)
    print(" VALIDATION COMPLETE")
    print("=" * 63)

    # sharp absorption — e.g. narrow band absorber
    sigmaA = lambda lbda: 10.0 * torch.exp(-torch.pow((lbda - 550.0) / 20.0, 2))
    op = SpectralOperatorFactory.createAbsorption(basis, sigmaA, distance=1.0)

    A = op.m_A
    diag_energy = torch.diag(A).pow(2).sum()
    total_energy = A.pow(2).sum()
    off_diag_fraction = 1.0 - (diag_energy / total_energy)
    print(f"off-diagonal energy fraction: {off_diag_fraction:.6f}")

    sigmaS_base = 0.01
    alpha = 4.0
    sigmaS = lambda lbda: sigmaS_base * (lbda / 550.0) ** (-alpha)
    op = SpectralOperatorFactory.createScattering(basis, "Rayleigh", sigmaS_base, 1.0, alpha)
    print(op.m_A.diag()[:5])
    print(op.m_A[0, :5])

    # simple test — gaussian emission at 650nm, gaussian absorption at 450nm
    lbda = basis.m_domain.m_lambda
    e = torch.exp(-((lbda - 650.0) / 30.0).pow(2))
    a = torch.exp(-((lbda - 450.0) / 30.0).pow(2))
    op = SpectralOperatorFactory.createFluorescence(basis, e, a)

    # check rank
    U, S, V = torch.linalg.svd(op.m_A)
    print("singular values[:5]:", S[:5])

    # check first row and diagonal
    print("diag[:5]:", op.m_A.diag()[:5])
    print("row0[:5]:", op.m_A[0, :5])
    U, S, V = torch.linalg.svd(op.m_A)
    print("singular values[:5]:", S[:5])

    op = SpectralOperatorFactory.createThinFilm(basis, n=1.33, d=300.0)
    print(op.m_A.diag()[:5])
    print(op.m_A[0, :5])

    lbda = basis.m_domain.m_lambda
    F_inf = 0.1 + 0.8 * ((lbda - 380.0) / 450.0) ** 2
    ops = SpectralOperatorFactory.createFresnel(basis, F_inf)

    A_normal = SpectralOperatorFactory.assembleFresnel(ops, 1.0, basis)
    print("normal diag[:5]:", A_normal.m_A.diag()[:5])

    A_grazing = SpectralOperatorFactory.assembleFresnel(ops, 0.0, basis)
    print("grazing diag[:5]:", A_grazing.m_A.diag()[:5])

    ops = SpectralOperatorFactory.createDispersion(basis, 0.0, 0.0, 0.0)

    # partition of unity
    total = sum(op.m_A for op in ops)
    print("partition diag[:5]:", total.diag()[:5])
    print("partition row0[1:5]:", total[0, 1:5])

    # lobe dominance
    print("lobe 0 diag[:5]:", ops[0].m_A.diag()[:5])
    print("lobe 7 diag[83:]:", ops[7].m_A.diag()[83:])

    op = SpectralOperatorFactory.createRaman(basis, shift_nm=70.0, sigmaRaman=10.0)
    print("diagonal[:5]:", op.m_A.diag()[:5])
    print("row0[:10]:", op.m_A[0, :10])
    max_idx = op.m_A.abs().argmax()
    i, j = max_idx // 88, max_idx % 88
    print(f"max element: [{i}, {j}] = {op.m_A[i, j]:.6f}")
    print(f"offset from diagonal: {i - j}")



if __name__ == "__main__":
    testOperators()

