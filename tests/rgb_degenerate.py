"""
Operator Leakage Test — RGB Hypothesis
K=3, N=1, M=3 | Tristimulus Topology

Leakage definition:
    For each operator O with physical kernel T(λ):
        GT(λ)        = T(λ) · s(λ)                      (ground truth in λ-space)
        BsSPT(λ)     = reconstruct(Â · project(s))       (operator in coefficient space)
        leakage      = ||GT - BsSPT||² / ||GT||²         (relative L2)

If RGB is a special case of BsSPT, leakage on broadband spectra → 0.
Leakage on spectrally complex inputs (fluorescence, thin film, Raman) → exposes M=3 poverty.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from research.engine.config   import TorchConfig
from research.engine.domain   import SpectralDomain
from research.engine.topology import topologyTristimulus
from research.engine.basis    import GHGSFBasis
from research.engine.state    import SpectralState
from research.engine.operator import SpectralOperatorFactory

# ── Config ────────────────────────────────────────────────────────────────────

cfg    = TorchConfig.setMode("reference")
device = cfg["device"]
dtype  = cfg["dtype"]

L_MIN, L_MAX, N_SAMPLES = 380.0, 830.0, 1024

domain  = SpectralDomain(L_MIN, L_MAX, N_SAMPLES, device=device, dtype=dtype)
lbda    = domain.m_lambda
w       = domain.m_weights

centers = topologyTristimulus(K=3, lMin=L_MIN, lMax=L_MAX)  # [450, 550, 650]
sigma   = (L_MAX - L_MIN) / 3.0                              # ~150 nm — wide lobes

basis   = GHGSFBasis(domain=domain, centers=centers, sigma=sigma, order=1)

print(f"=== RGB Hypothesis Leakage Test ===")
print(f"K={basis.m_K}  N={basis.m_N}  M={basis.m_M}")
print(f"Centers (nm): {[f'{c:.1f}' for c in centers]}")
print(f"Sigma   (nm): {sigma:.1f}")
print(f"Basis shape : {basis.m_basisRaw.shape}")
print()

# ── Leakage metric ────────────────────────────────────────────────────────────

def leakage(s: torch.Tensor, T: torch.Tensor) -> dict:
    """
    s : (L,) input spectrum
    T : (L,) multiplicative kernel (pointwise in λ-space)

    Returns relative L2 leakage and absolute residual norm.
    """
    gt       = T * s                                      # ground truth in λ-space
    alpha    = basis.project(s)                           # project s → raw coeffs
    state    = SpectralState(basis, alpha)
    # Build operator from T, apply in coefficient space
    M_raw    = (basis.m_basisRaw * (w * T)) @ basis.m_basisRaw.T
    L_chol   = basis.m_chol
    Y        = torch.linalg.solve_triangular(L_chol, M_raw, upper=False)
    A_wht    = torch.linalg.solve_triangular(L_chol, Y.T, upper=False).T
    b_wht    = torch.zeros(basis.m_M, device=device, dtype=dtype)

    from research.engine.operator import SpectralOperator
    op = SpectralOperator(basis, A_wht, b_wht)
    op.apply(state)

    recon    = basis.reconstructWhitened(state.m_coeffs)  # back to λ-space
    residual = gt - recon
    gt_norm  = torch.sqrt(domain.integrate(gt * gt))
    res_norm = torch.sqrt(domain.integrate(residual * residual))
    rel      = (res_norm / gt_norm).item() if gt_norm > 1e-12 else float('nan')

    return {
        "leakage_rel": rel,
        "leakage_abs": res_norm.item(),
        "gt_norm":     gt_norm.item(),
    }

def report(name: str, results: dict):
    pct = results["leakage_rel"] * 100.0
    bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
    print(f"  {name:<28} leakage = {pct:6.2f}%  [{bar}]")

# ── Test spectra ──────────────────────────────────────────────────────────────

# Broadband flat — should be near-zero leakage if RGB hypothesis holds
s_flat       = torch.ones(N_SAMPLES, device=device, dtype=dtype)

# Smooth reflectance (gold-like) — broadband but shaped
s_gold       = torch.exp(-((lbda - 580.0) / 80.0) ** 2) * 0.8 + 0.2

# Narrow emission line (laser @ 532nm) — spectrally complex
s_laser      = torch.exp(-((lbda - 532.0) / 5.0) ** 2)

# Fluorescence-like: absorbs at 450, emits at 520 — two narrow peaks
s_fluoro     = (torch.exp(-((lbda - 450.0) / 15.0) ** 2) +
                0.6 * torch.exp(-((lbda - 520.0) / 20.0) ** 2))

# Thin-film fringe — high frequency oscillation
s_thinfilm   = 0.5 * (1.0 + torch.cos(2.0 * torch.pi * lbda / 30.0))

# Raman-like: sharp peak offset from excitation
s_raman      = torch.exp(-((lbda - 600.0) / 8.0) ** 2)

spectra = {
    "flat":         s_flat,
    "gold":         s_gold,
    "laser_532nm":  s_laser,
    "fluorescence": s_fluoro,
    "thin_film":    s_thinfilm,
    "raman":        s_raman,
}

# ── Operator kernels ──────────────────────────────────────────────────────────

# 1. Absorption — Beer-Lambert: T(λ) = exp(-σa(λ)·d)
#    σa = Gaussian absorption peak at 500nm
sigmaA_fn  = lambda l: 0.05 * torch.exp(-((l - 500.0) / 30.0) ** 2)
T_absorb   = lambda l: torch.exp(-sigmaA_fn(l) * 10.0)

# 2. Thin Film — Fabry-Airy: T(λ) = 0.5(1 + cos(4πnd/λ))
n_tf, d_tf = 1.5, 150.0
T_thinfilm = lambda l: 0.5 * (1.0 + torch.cos(4.0 * torch.pi * n_tf * d_tf / l))

# 3. Rayleigh scattering: T(λ) = exp(-σs(λ/550)^-4 · d)
sigmaS_base, dist = 0.01, 5.0
T_rayleigh  = lambda l: torch.exp(-sigmaS_base * (l / 550.0) ** (-4.0) * dist)

# 4. Mie scattering: T(λ) = exp(-σs(λ/550)^-1 · d)
T_mie       = lambda l: torch.exp(-sigmaS_base * (l / 550.0) ** (-1.0) * dist)

# 5. Raman shift kernel — row-wise: T(λ) = Gaussian(λ - shift)
#    Approximated as a pointwise weight (marginal of the 2D kernel)
shift_nm   = 20.0
T_raman    = lambda l: torch.exp(-0.5 * ((l - (l + shift_nm)) / 10.0) ** 2)
# Raman is banded off-diagonal — for leakage we use the diagonal marginal
# True leakage test uses the full Galerkin path (see note below)

operators = {
    "Absorption (Beer-Lambert)":   T_absorb,
    "Thin Film  (Fabry-Airy)":     T_thinfilm,
    "Rayleigh Scattering":         T_rayleigh,
    "Mie Scattering":              T_mie,
}

# ── Run ───────────────────────────────────────────────────────────────────────

print("─" * 80)
for spec_name, s in spectra.items():
    print(f"\nSpectrum: {spec_name}")
    for op_name, T_fn in operators.items():
        T = T_fn(lbda)
        r = leakage(s, T)
        report(op_name, r)

# ── Dispersion partition-of-unity check ───────────────────────────────────────
print()
print("─" * 80)
print("\nDispersion — Partition of Unity check (Σ Âk = I_M)")
disp_ops = SpectralOperatorFactory.createDispersion(basis, A=1.5, B=5000.0, C=0.0)
A_sum = sum(op.m_A for op in disp_ops)
I_M   = torch.eye(basis.m_M, device=device, dtype=dtype)
pou_err = torch.linalg.norm(A_sum - I_M).item()
print(f"  ||Σ Âk - I_M||_F = {pou_err:.6e}  {'✓ PoU holds' if pou_err < 1e-6 else '✗ PoU VIOLATED'}")

# ── Basis reconstruction self-test ────────────────────────────────────────────
print()
print("─" * 80)
print("\nBasis self-reconstruction (project → reconstruct) per spectrum")
for spec_name, s in spectra.items():
    alpha = basis.project(s)
    recon = basis.reconstruct(alpha)
    res   = s - recon
    gt_n  = torch.sqrt(domain.integrate(s * s))
    re_n  = torch.sqrt(domain.integrate(res * res))
    rel   = (re_n / gt_n).item() * 100.0
    bar   = "█" * int(rel / 2) + "░" * (50 - int(rel / 2))
    print(f"  {spec_name:<28} basis error = {rel:6.2f}%  [{bar}]")

# ── Gram / condition summary ──────────────────────────────────────────────────
print()
print("─" * 80)
print("\nBasis numerical health")
G      = basis.m_gram
eigs   = torch.linalg.eigvalsh(G)
cond   = (eigs.max() / eigs.min().clamp(min=1e-30)).item()
print(f"  Gram eigenvalues : {eigs.cpu().numpy()}")
print(f"  Condition number : {cond:.4e}")
print(f"  M=3 — this IS RGB-grade: {'YES' if cond < 1e6 else 'NO (degenerate)'}")

print()
print("─" * 80)
print("Done.")