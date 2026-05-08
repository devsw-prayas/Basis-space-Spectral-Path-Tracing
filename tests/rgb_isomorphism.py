"""
RGB Isomorphism Test — M=3 BsSPT vs CIE XYZ Tristimulus
=========================================================

Topology: UNIFORM restricted to CMF support [400, 700nm], sigma=100nm

Subspace test: does span(GHGSF M=3) == span(CIE CMFs)?
"""

import torch
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from research.engine.config import TorchConfig
from research.engine.domain import SpectralDomain
from research.engine.basis  import GHGSFBasis

cfg    = TorchConfig.setMode("reference")
device = cfg["device"]
dtype  = cfg["dtype"]

L_MIN, L_MAX, N_SAMPLES = 380.0, 830.0, 1024

domain = SpectralDomain(L_MIN, L_MAX, N_SAMPLES, device=device, dtype=dtype)
lbda   = domain.m_lambda
w      = domain.m_weights

# Uniform inside CMF support, tight enough to not bleed into IR dead zone
centers = [400.0, 550.0, 700.0]
sigma   = 100.0
order   = 3

basis = GHGSFBasis(domain=domain, centers=centers, sigma=sigma, order=order)

print("=== RGB Isomorphism Test ===")
print(f"Centers : {centers} nm")
print(f"Sigma   : {sigma} nm  |  N=3  M={3*3}")
print()

# ── CIE 1931 CMFs — Wyman et al. 2013 ────────────────────────────────────────

def cie_xbar(l):
    t1 = l - 442.0
    g1 = 0.362  * torch.exp(-0.5 * (t1 / torch.where(t1 < 0, torch.full_like(t1, 16.0), torch.full_like(t1, 26.7))) ** 2)
    t2 = l - 599.8
    g2 = 1.056  * torch.exp(-0.5 * (t2 / torch.where(t2 < 0, torch.full_like(t2, 37.9), torch.full_like(t2, 31.0))) ** 2)
    t3 = l - 501.1
    g3 = -0.065 * torch.exp(-0.5 * (t3 / torch.where(t3 < 0, torch.full_like(t3, 20.4), torch.full_like(t3, 13.0))) ** 2)
    return g1 + g2 + g3

def cie_ybar(l):
    t1 = l - 568.8
    g1 = 0.821 * torch.exp(-0.5 * (t1 / torch.where(t1 < 0, torch.full_like(t1, 46.9), torch.full_like(t1, 40.5))) ** 2)
    t2 = l - 530.9
    g2 = 0.286 * torch.exp(-0.5 * (t2 / torch.where(t2 < 0, torch.full_like(t2, 16.3), torch.full_like(t2, 31.1))) ** 2)
    return g1 + g2

def cie_zbar(l):
    t1 = l - 437.0
    g1 = 1.217 * torch.exp(-0.5 * (t1 / torch.where(t1 < 0, torch.full_like(t1, 11.8), torch.full_like(t1, 36.0))) ** 2)
    t2 = l - 459.0
    g2 = 0.681 * torch.exp(-0.5 * (t2 / torch.where(t2 < 0, torch.full_like(t2, 26.0), torch.full_like(t2, 13.8))) ** 2)
    return g1 + g2

xbar = cie_xbar(lbda)
ybar = cie_ybar(lbda)
zbar = cie_zbar(lbda)
cmf_basis = torch.stack([xbar, ybar, zbar], dim=0)  # (3, L)

G_cmf = (cmf_basis * w) @ cmf_basis.T
L_cmf = torch.linalg.cholesky(G_cmf)

# ── Gram health ───────────────────────────────────────────────────────────────
eigs_g = torch.linalg.eigvalsh(basis.m_gram)
eigs_c = torch.linalg.eigvalsh(G_cmf)
print(f"GHGSF Gram eigs : {eigs_g.cpu().numpy()}  cond={( eigs_g.max()/eigs_g.min()).item():.3e}")
print(f"CMF   Gram eigs : {eigs_c.cpu().numpy()}  cond={(eigs_c.max()/eigs_c.min()).item():.3e}")
print()

# ── Subspace alignment ────────────────────────────────────────────────────────
def project_onto_ghgsf(f):
    alpha = basis.project(f)
    recon = basis.reconstruct(alpha)
    n_f   = torch.sqrt(domain.integrate(f * f))
    n_res = torch.sqrt(domain.integrate((f - recon) ** 2))
    return (n_res / n_f).item() * 100.0

def project_onto_cmf(f):
    b     = (cmf_basis * w) @ f
    y     = torch.linalg.solve_triangular(L_cmf, b.unsqueeze(-1), upper=False)
    c     = torch.linalg.solve_triangular(L_cmf.T, y, upper=True).squeeze(-1)
    recon = c @ cmf_basis
    n_f   = torch.sqrt(domain.integrate(f * f))
    n_res = torch.sqrt(domain.integrate((f - recon) ** 2))
    return (n_res / n_f).item() * 100.0

print("CMF → GHGSF span (energy outside GHGSF subspace):")
for cmf, name in zip([xbar, ybar, zbar], ["xbar", "ybar", "zbar"]):
    print(f"  {name} : {project_onto_ghgsf(cmf):.4f}%")

print()
print("GHGSF → CMF span (energy outside CMF subspace):")
B = basis.m_basisRaw
for i in range(3):
    print(f"  phi_{i}({centers[i]:.0f}nm) : {project_onto_cmf(B[i]):.4f}%")

print()
print("-" * 70)
print()

# ── Per-spectrum Path A vs Path B ─────────────────────────────────────────────
def path_A(s):
    return basis.reconstruct(basis.project(s))

def path_B(s):
    b = (cmf_basis * w) @ s
    y = torch.linalg.solve_triangular(L_cmf, b.unsqueeze(-1), upper=False)
    c = torch.linalg.solve_triangular(L_cmf.T, y, upper=True).squeeze(-1)
    return c @ cmf_basis

spectra = {
    "flat":         torch.ones(N_SAMPLES, device=device, dtype=dtype),
    "gold":         torch.exp(-((lbda - 580.0) / 80.0) ** 2) * 0.8 + 0.2,
    "sky_blue":     torch.exp(-((lbda - 460.0) / 40.0) ** 2),
    "grass_green":  torch.exp(-((lbda - 540.0) / 35.0) ** 2),
    "laser_532nm":  torch.exp(-((lbda - 532.0) /  5.0) ** 2),
    "fluorescence": (torch.exp(-((lbda - 450.0) / 15.0) ** 2) +
                     0.6 * torch.exp(-((lbda - 520.0) / 20.0) ** 2)),
    "thin_film":    0.5 * (1.0 + torch.cos(2.0 * torch.pi * lbda / 30.0)),
    "raman":        torch.exp(-((lbda - 600.0) /  8.0) ** 2),
}

print(f"  {'Spectrum':<20}  {'|sA-sB|/|s|':>12}  {'cos(sA,sB)':>12}  Verdict")
print("-" * 70)
for name, s in spectra.items():
    sA     = path_A(s)
    sB     = path_B(s)
    diff   = sA - sB
    norm_s = torch.sqrt(domain.integrate(s * s))
    norm_d = torch.sqrt(domain.integrate(diff * diff))
    norm_A = torch.sqrt(domain.integrate(sA * sA))
    norm_B = torch.sqrt(domain.integrate(sB * sB))
    rel    = (norm_d / norm_s).item() * 100.0
    cos    = (domain.integrate(sA * sB) / (norm_A * norm_B).clamp(min=1e-12)).item()
    if rel < 1.0:   verdict = "ISOMORPHIC"
    elif rel < 10.: verdict = "CLOSE"
    elif cos > 0.99:verdict = "SHAPE MATCH"
    else:           verdict = "DIVERGE"
    print(f"  {name:<20}  {rel:10.2f}%  {cos:12.6f}  {verdict}")

print()
print("-" * 70)
print("subspace leakage < 1% both ways -> isomorphic")
print("Done.")

# ── Path C: unwhitened GHGSF M=3 ─────────────────────────────────────────────
# Raw Gram solve — no Cholesky, just G^-1 (B w s), reconstruct via B_raw
# This is the RGB analog: store raw coefficients, no whitening transform

def path_C(s):
    b     = (basis.m_basisRaw * w) @ s          # (3,) raw inner products
    alpha = torch.linalg.solve(basis.m_gram, b) # (3,) raw coeffs via Gram solve
    return alpha @ basis.m_basisRaw             # (L,) reconstruction

# subspace alignment — unwhitened GHGSF vs CMF
print("Unwhitened GHGSF → CMF span:")
for i in range(3):
    phi = basis.m_basisRaw[i]
    b   = (cmf_basis * w) @ phi
    y   = torch.linalg.solve_triangular(L_cmf, b.unsqueeze(-1), upper=False)
    c   = torch.linalg.solve_triangular(L_cmf.T, y, upper=True).squeeze(-1)
    rec = c @ cmf_basis
    n_f = torch.sqrt(domain.integrate(phi * phi))
    n_r = torch.sqrt(domain.integrate((phi - rec) ** 2))
    print(f"  phi_{i}({centers[i]:.0f}nm) : {(n_r/n_f).item()*100:.4f}%")

print()
print("CMF → Unwhitened GHGSF span:")
for cmf, name in zip([xbar, ybar, zbar], ["xbar", "ybar", "zbar"]):
    b     = (basis.m_basisRaw * w) @ cmf
    alpha = torch.linalg.solve(basis.m_gram, b)
    rec   = alpha @ basis.m_basisRaw
    n_f   = torch.sqrt(domain.integrate(cmf * cmf))
    n_r   = torch.sqrt(domain.integrate((cmf - rec) ** 2))
    print(f"  {name} : {(n_r/n_f).item()*100:.4f}%")

print()
print("-" * 70)
print()

# per-spectrum Path C vs Path B
print(f"  {'Spectrum':<20}  {'|C-B|/|s|':>12}  {'cos(C,B)':>12}  Verdict")
print("-" * 70)
for name, s in spectra.items():
    sC     = path_C(s)
    sB     = path_B(s)
    diff   = sC - sB
    norm_s = torch.sqrt(domain.integrate(s    * s))
    norm_d = torch.sqrt(domain.integrate(diff * diff))
    norm_C = torch.sqrt(domain.integrate(sC   * sC))
    norm_B = torch.sqrt(domain.integrate(sB   * sB))
    rel    = (norm_d / norm_s).item() * 100.0
    cos    = (domain.integrate(sC * sB) / (norm_C * norm_B).clamp(min=1e-12)).item()
    if rel < 1.0:    verdict = "ISOMORPHIC"
    elif rel < 10.:  verdict = "CLOSE"
    elif cos > 0.99: verdict = "SHAPE MATCH"
    else:            verdict = "DIVERGE"
    print(f"  {name:<20}  {rel:10.2f}%  {cos:12.6f}  {verdict}")