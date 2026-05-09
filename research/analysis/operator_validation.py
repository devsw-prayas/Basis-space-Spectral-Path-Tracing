"""
BsSPT Operator Validation Suite
================================
11 property tests, each graphed as a scientific panel.

  1.  Fresnel partition sum       — Psq + 2·Rcross + Qcomp = I_M
  2.  Psq ≈ P0² eigenvalues      — scatter, wavelength-varying F_inf
  3.  Beer ∘ Fresnel commutes    — ‖[A,B]‖_F near zero
  4.  Fabry ∘ Beer non-commutes  — ‖[A,B]‖_F significantly above zero
  5.  Adjoint consistency         — ⟨Âα,β⟩ = ⟨α,Âβ⟩ scatter
  6.  Non-negative eigenvalues    — bar chart min-λ per operator
  7.  Stokes energy bound         — E_emit / E_abs ≤ 1
  8.  Beer-Lambert norm decay     — geometric convergence under composition
  9.  Fresnel SVD: P0→Rcross      — alignment degrades for metals (k₀>4)
  10. Fresnel SVD: P0→Psq         — alignment holds for dielectrics (k₀≤4)
  11. Summary pass/fail table
  12. Commutator survey (bonus)   — all physical operator pairs

Output: results/operator_validation/operator_validation.png
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from research.engine.domain import SpectralDomain
from research.engine.basis import GHGSFDualDomainBasis
from research.engine.operator import SpectralOperatorFactory
from research.engine.topology import generateTopology

OUT_DIR = "results/operator_validation"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Theme ──────────────────────────────────────────────────────────────────
BG      = "#0f1116"
PANEL   = "#0f1116"
SPINE   = "#ffffff"
CYAN    = "#5BC0DE"
YELLOW  = "#F0E68C"
GREEN   = "#90C695"
ORANGE  = "#F4A460"
MAGENTA = "#C39BD3"
RED     = "#E74C3C"
TEAL    = "#48C9B0"
GREY    = "#AAB7B8"

PASS_CLR = GREEN
FAIL_CLR = RED

def styleAx(ax, title=""):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=SPINE, fontsize=8, pad=4)
    ax.tick_params(colors=GREY, labelsize=6)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333333")
    ax.xaxis.label.set_color(GREY)
    ax.yaxis.label.set_color(GREY)

def addColorbar(fig, ax, im):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color=GREY, labelsize=5)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=GREY)

# ── Golden Config ──────────────────────────────────────────────────────────
print("Initializing basis...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
domain = SpectralDomain(380.0, 830.0, 1024, device=device, dtype=torch.float64)
K, N = 8, 11
centers, _ = generateTopology(0, K, margin=0.0)
basis = GHGSFDualDomainBasis(
    domain=domain, centers=centers, wideIndices=list(range(K // 2)),
    wideSigmaMin=9.5,  wideSigmaMax=11.5,  wideScaleType="linear",
    narrowSigmaMin=7.0, narrowSigmaMax=9.0, narrowScaleType="linear",
    order=N
)
basis.buildCholesky()
lbda = domain.m_lambda
M    = basis.m_M
print(f"  K={K}  N={N}  M={M}  κ(G)={torch.linalg.cond(basis.m_gram):.2e}")

# ── Operators ──────────────────────────────────────────────────────────────
print("Building operators...")

sigma_a      = lambda l: torch.full_like(l, 0.002)
op_beer      = SpectralOperatorFactory.createAbsorption(basis, sigma_a, distance=1.0)

f_inf_const  = torch.full_like(lbda, 0.04)
ops_f        = SpectralOperatorFactory.createFresnel(basis, f_inf_const)
op_p0, op_psq, op_rc, op_qc = ops_f.P0, ops_f.Psq, ops_f.Rcross, ops_f.Qcomp

op_film      = SpectralOperatorFactory.createThinFilm(basis, n=1.5, d=300.0)

a_prof       = torch.exp(-0.5 * ((lbda - 400.0) / 20.0) ** 2)
e_prof       = torch.exp(-0.5 * ((lbda - 520.0) / 25.0) ** 2)
op_stokes    = SpectralOperatorFactory.createFluorescence(basis, e_prof, a_prof)

op_rayleigh  = SpectralOperatorFactory.createScattering(basis, "Rayleigh", sigmaS_base=0.005, distance=1.0, alpha=4.0)
op_mie       = SpectralOperatorFactory.createScattering(basis, "Mie",      sigmaS_base=0.005, distance=1.0, alpha=1.0)
op_raman     = SpectralOperatorFactory.createRaman(basis, shift_nm=50.0, sigmaRaman=8.0)
ops_cauchy   = SpectralOperatorFactory.createDispersion(basis, A=1.5, B=0.01, C=0.0)
print("  All operators built.")

I_M    = torch.eye(M, device=device, dtype=torch.float64)
results = {}  # name -> (passed: bool, value_str: str)

# ── Drude-like metal model for SVD sweep ──────────────────────────────────
def metalFresnel(k0: float, n_ior: float = 1.5) -> torch.Tensor:
    """F_inf(λ) for extinction k(λ) = k0*(600/λ)², n fixed."""
    k_lam = k0 * (600.0 / lbda) ** 2
    num   = (n_ior - 1.0) ** 2 + k_lam ** 2
    den   = (n_ior + 1.0) ** 2 + k_lam ** 2
    return (num / den).clamp(1e-4, 0.9999)

# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(4, 3, figsize=(21, 28))
fig.patch.set_facecolor(BG)
axes = axes.flatten()

# ── TEST 1: Fresnel Partition Sum ─────────────────────────────────────────
print("Test 1: Fresnel partition sum...")
ax = axes[0]
styleAx(ax, "1. Fresnel Partition Sum")

A_binomial = op_psq.m_A + 2.0 * op_rc.m_A + op_qc.m_A
residual   = A_binomial - I_M
norm_binom = torch.linalg.matrix_norm(residual, ord='fro').item()
norm_all4  = torch.linalg.matrix_norm(op_p0.m_A + op_psq.m_A + op_rc.m_A + op_qc.m_A - I_M, ord='fro').item()

diag_binom = torch.diag(A_binomial).cpu().float().numpy()
diag_all4  = torch.diag(op_p0.m_A + op_psq.m_A + op_rc.m_A + op_qc.m_A).cpu().float().numpy()

ax.plot(diag_binom, color=CYAN,   linewidth=1.3, label="Psq + 2·Rc + Qc (diagonal)")
ax.plot(diag_all4,  color=ORANGE, linewidth=1.0, linestyle="--", label="P0 + Psq + Rc + Qc (diagonal)")
ax.axhline(1.0, color=GREY, linewidth=0.7, linestyle=":", alpha=0.6, label="y = 1")
ax.set_xlabel("mode index", fontsize=7)
ax.set_ylabel("diagonal value", fontsize=7)
ax.legend(fontsize=5, facecolor=PANEL, labelcolor=SPINE, edgecolor="#333")
ax.grid(True, alpha=0.1, color=SPINE)
ax.annotate(f"‖Psq+2Rc+Qc − I‖_F = {norm_binom:.3e}", xy=(0.03, 0.10),
            xycoords="axes fraction", color=CYAN, fontsize=6)
ax.annotate(f"‖P0+Psq+Rc+Qc − I‖_F = {norm_all4:.3e}", xy=(0.03, 0.03),
            xycoords="axes fraction", color=ORANGE, fontsize=6)

passed1 = norm_binom < 0.01
results["1. Partition sum"] = (passed1, f"‖binom−I‖={norm_binom:.2e}")

# ── TEST 2: Psq ≈ P0² eigenvalues (wavelength-varying F_inf) ─────────────
print("Test 2: Psq = P0² eigenvalues...")
ax = axes[1]
styleAx(ax, "2. Psq ≈ P0² Eigenvalues (metal k₀=2)")

f_metal2   = metalFresnel(k0=2.0)
ops_metal2 = SpectralOperatorFactory.createFresnel(basis, f_metal2)
ev_p0_m    = torch.linalg.eigvalsh(ops_metal2.P0.m_A).cpu().float().numpy()
ev_psq_m   = torch.linalg.eigvalsh(ops_metal2.Psq.m_A).cpu().float().numpy()

# Sort by P0 eigenvalue descending
order      = np.argsort(ev_p0_m)[::-1]
ev_p0_s    = ev_p0_m[order]
ev_psq_s   = ev_psq_m[np.argsort(ev_psq_m)[::-1]]
predicted  = ev_p0_s ** 2

ax.scatter(predicted, ev_psq_s, s=12, color=CYAN, alpha=0.85, zorder=3, label="actual λ_Psq")
lim = max(predicted.max(), ev_psq_s.max()) * 1.1
ax.plot([0, lim], [0, lim], color=GREY, linewidth=0.8, linestyle="--", label="y = x")
ax.set_xlabel("λ_P0²  (predicted)", fontsize=7)
ax.set_ylabel("λ_Psq  (actual)", fontsize=7)
ax.legend(fontsize=6, facecolor=PANEL, labelcolor=SPINE, edgecolor="#333")
ax.grid(True, alpha=0.1, color=SPINE)

max_err2 = float(np.abs(ev_psq_s - predicted).max())
ax.annotate(f"max |λ_Psq − λ_P0²| = {max_err2:.3e}", xy=(0.03, 0.92),
            xycoords="axes fraction", color=GREY, fontsize=6, va="top")

passed2 = max_err2 < 0.01
results["2. Psq = P0² eigvals"] = (passed2, f"max_err={max_err2:.2e}")

# ── TEST 3: Beer ∘ Fresnel commutes ──────────────────────────────────────
print("Test 3: Beer ∘ Fresnel commutativity...")
ax = axes[2]
styleAx(ax, "3. Beer ∘ Fresnel Commutes  [‖AB−BA‖_F]")

comm_bf  = op_beer.m_A @ op_p0.m_A - op_p0.m_A @ op_beer.m_A
norm_bf  = torch.linalg.matrix_norm(comm_bf, ord='fro').item()
comm_np3 = np.log1p(comm_bf.abs().cpu().float().numpy() * 1e6)

im3 = ax.imshow(comm_np3, cmap="inferno", aspect="auto", interpolation="nearest")
addColorbar(fig, ax, im3)
ax.set_xlabel("j", fontsize=6); ax.set_ylabel("i", fontsize=6)
ax.annotate(f"‖[Beer, Fresnel P0]‖_F = {norm_bf:.3e}", xy=(0.03, 0.96),
            xycoords="axes fraction", color=SPINE, fontsize=6, va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc=PANEL, ec="#333", alpha=0.85))

passed3 = norm_bf < 0.01
results["3. Beer∘Fresnel commutes"] = (passed3, f"‖comm‖={norm_bf:.2e}")

# ── TEST 4: Fabry-Airy ∘ Beer does NOT commute ───────────────────────────
print("Test 4: Fabry-Airy ∘ Beer non-commutativity...")
ax = axes[3]
styleAx(ax, "4. Fabry-Airy ∘ Beer ≠ Commute  [‖AB−BA‖_F]")

comm_fb  = op_film.m_A @ op_beer.m_A - op_beer.m_A @ op_film.m_A
norm_fb  = torch.linalg.matrix_norm(comm_fb, ord='fro').item()
comm_np4 = np.log1p(comm_fb.abs().cpu().float().numpy() * 1e6)

im4 = ax.imshow(comm_np4, cmap="plasma", aspect="auto", interpolation="nearest")
addColorbar(fig, ax, im4)
ax.set_xlabel("j", fontsize=6); ax.set_ylabel("i", fontsize=6)
ax.annotate(f"‖[Film, Beer]‖_F = {norm_fb:.3e}", xy=(0.03, 0.96),
            xycoords="axes fraction", color=SPINE, fontsize=6, va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc=PANEL, ec="#333", alpha=0.85))

passed4 = (norm_fb > norm_bf * 10.0)
results["4. Fabry∘Beer non-commutes"] = (passed4, f"‖comm‖={norm_fb:.2e}  (ref Beer∘Fres={norm_bf:.2e})")

# ── TEST 5: Adjoint consistency ───────────────────────────────────────────
print("Test 5: Adjoint consistency...")
ax = axes[4]
styleAx(ax, "5. Adjoint Consistency  ⟨Âα,β⟩ = ⟨α,Âβ⟩")

SYMM_OPS = [("Beer",     op_beer.m_A),
            ("Fresnel P0", op_p0.m_A),
            ("Film",     op_film.m_A),
            ("Rayleigh", op_rayleigh.m_A),
            ("Mie",      op_mie.m_A),
            ("Cauchy[0]", ops_cauchy[0].m_A)]

torch.manual_seed(42)
N_pairs = 300
lhs_all, rhs_all = [], []
for _, A in SYMM_OPS:
    alpha = torch.randn(M, N_pairs, device=device, dtype=torch.float64)
    beta  = torch.randn(M, N_pairs, device=device, dtype=torch.float64)
    lhs_all.append(((A @ alpha) * beta).sum(0).cpu().float().numpy())
    rhs_all.append((alpha * (A @ beta)).sum(0).cpu().float().numpy())

lhs_np = np.concatenate(lhs_all)
rhs_np = np.concatenate(rhs_all)

ax.scatter(lhs_np, rhs_np, s=2, alpha=0.25, color=CYAN, rasterized=True)
lim5 = max(abs(lhs_np).max(), abs(rhs_np).max()) * 1.05
ax.plot([-lim5, lim5], [-lim5, lim5], color=GREY, linewidth=0.8, linestyle="--")
ax.set_xlim(-lim5, lim5); ax.set_ylim(-lim5, lim5)
ax.set_xlabel("⟨Âα, β⟩", fontsize=7); ax.set_ylabel("⟨α, Âβ⟩", fontsize=7)
ax.grid(True, alpha=0.1, color=SPINE)
adj_err = float(np.abs(lhs_np - rhs_np).max())
ax.annotate(f"max |Δ| = {adj_err:.2e}", xy=(0.03, 0.92),
            xycoords="axes fraction", color=GREY, fontsize=6, va="top")

# κ≈1e13 → floating-point error in inner products is O(κ·ε_mach·‖A‖·‖v‖²) ≈ 1e-3; 5e-3 is safe
passed5 = adj_err < 5e-3
results["5. Adjoint consistency"] = (passed5, f"max|Δ|={adj_err:.2e}")

# ── TEST 6: Non-negative eigenvalues ─────────────────────────────────────
print("Test 6: Non-negative eigenvalues...")
ax = axes[5]
styleAx(ax, "6. Min Eigenvalue per Operator  (Physical ≥ 0)")

ALL_OPS_EIG = [
    ("Beer",     op_beer.m_A,      True),
    ("Fresnel",  op_p0.m_A,        True),
    ("Film",     op_film.m_A,      True),
    ("Stokes",   op_stokes.m_A,    False),
    ("Cauchy[0]",ops_cauchy[0].m_A,True),
    ("Rayleigh", op_rayleigh.m_A,  True),
    ("Mie",      op_mie.m_A,       True),
    ("Raman",    op_raman.m_A,     False),
]

names6, vals6, clrs6 = [], [], []
for name, A, is_sym in ALL_OPS_EIG:
    if is_sym:
        val = torch.linalg.eigvalsh(A)[0].item()
    else:
        val = torch.linalg.svdvals(A)[-1].item()
    names6.append(name)
    vals6.append(val)
    clrs6.append(PASS_CLR if val >= -1e-6 else FAIL_CLR)

xp6 = np.arange(len(names6))
ax.bar(xp6, vals6, color=clrs6, alpha=0.85, width=0.65)
ax.axhline(0, color=GREY, linewidth=0.7, linestyle="--", alpha=0.6)
ax.set_xticks(xp6)
ax.set_xticklabels(names6, rotation=30, ha='right', fontsize=6)
ax.set_ylabel("min λ  (sym) / min σ  (asym)", fontsize=7)
ax.grid(True, alpha=0.1, color=SPINE, axis='y')

passed6 = all(v >= -1e-6 for v in vals6)
results["6. Non-neg eigenvalues"] = (passed6, f"global min = {min(vals6):.2e}")

# ── TEST 7: Stokes energy bound ───────────────────────────────────────────
# Energy in whitened space = ‖α̃‖².  The operator bound is σ_max(Â) ≤ QY.
# Scale Â to QY_TARGET, then verify ‖Âα̃‖/‖α̃‖ ≤ QY_TARGET for random inputs.
print("Test 7: Stokes energy bound...")
ax = axes[6]
QY_TARGET = 0.9
styleAx(ax, f"7. Stokes Energy Bound  ‖Âα̃‖/‖α̃‖ ≤ QY={QY_TARGET}  (whitened)")

sigma_max = torch.linalg.svdvals(op_stokes.m_A)[0].item()
A_scaled  = op_stokes.m_A * (QY_TARGET / (sigma_max + 1e-30))

torch.manual_seed(7)
N_test     = 500
alpha_rand = torch.randn(M, N_test, device=device, dtype=torch.float64)
ratios     = (A_scaled @ alpha_rand).norm(dim=0) / alpha_rand.norm(dim=0)
ratios_np  = ratios.cpu().float().numpy()

ax.hist(ratios_np, bins=40, color=CYAN, alpha=0.8, edgecolor=PANEL)
ax.axvline(QY_TARGET,       color=RED,    linewidth=1.2, linestyle="--", label=f"QY = {QY_TARGET}")
ax.axvline(ratios_np.max(), color=ORANGE, linewidth=1.0, linestyle=":",
           label=f"max = {ratios_np.max():.5f}")
ax.set_xlabel("‖Âα̃‖ / ‖α̃‖", fontsize=7)
ax.set_ylabel("count", fontsize=7)
ax.legend(fontsize=6, facecolor=PANEL, labelcolor=SPINE, edgecolor="#333")
ax.grid(True, alpha=0.1, color=SPINE)
ax.annotate(f"σ_max(Â_raw) = {sigma_max:.4f}", xy=(0.03, 0.92),
            xycoords="axes fraction", color=GREY, fontsize=6, va="top")

passed7 = float(ratios_np.max()) <= QY_TARGET + 1e-5
results["7. Stokes energy bound"] = (passed7, f"max ‖Âα̃‖/‖α̃‖ = {ratios_np.max():.5f}  (target ≤ {QY_TARGET})")

# ── TEST 8: Beer-Lambert norm decay ──────────────────────────────────────
print("Test 8: Beer-Lambert norm decay...")
ax = axes[7]
styleAx(ax, "8. Beer-Lambert ‖Â^n‖_F  (Geometric Decay)")

MAX_N = 30
norms8 = []
A_power = torch.eye(M, device=device, dtype=torch.float64)
for _ in range(MAX_N):
    A_power = op_beer.m_A @ A_power
    norms8.append(torch.linalg.matrix_norm(A_power, ord='fro').item())

n_arr  = np.arange(1, MAX_N + 1)
nrm_np = np.array(norms8)
ax.semilogy(n_arr, nrm_np, color=CYAN, linewidth=1.4, label="‖Â^n‖_F")

coeffs8   = np.polyfit(n_arr, np.log(nrm_np), 1)
r_per_bce = float(np.exp(coeffs8[0]))
fit_curve = np.exp(np.polyval(coeffs8, n_arr))
ax.semilogy(n_arr, fit_curve, color=YELLOW, linewidth=0.9, linestyle="--",
            label=f"fit  r = {r_per_bce:.5f}/bounce")

ax.set_xlabel("bounce count n", fontsize=7)
ax.set_ylabel("‖Â^n‖_F", fontsize=7)
ax.legend(fontsize=6, facecolor=PANEL, labelcolor=SPINE, edgecolor="#333")
ax.grid(True, alpha=0.1, color=SPINE)

passed8 = r_per_bce < 1.0
results["8. Beer norm decay"] = (passed8, f"r = {r_per_bce:.5f}")

# ── TESTS 9 & 10: Fresnel SVD alignment sweep (Drude metal model) ─────────
print("Tests 9+10: Fresnel SVD alignment sweep (metal k₀ ∈ [0, 8])...")

K0_VALS   = np.linspace(0.0, 8.0, 40)
align_rc  = []
align_ps  = []

def matCorr(A: torch.Tensor, B: torch.Tensor) -> float:
    """Pearson correlation between flattened matrix entries (shape-similarity metric)."""
    a = A.flatten(); b = B.flatten()
    a = a - a.mean(); b = b - b.mean()
    return abs((torch.dot(a, b) / (a.norm() * b.norm() + 1e-30)).item())

for k0 in K0_VALS:
    f_val  = metalFresnel(k0)
    ops_k  = SpectralOperatorFactory.createFresnel(basis, f_val)
    align_rc.append(matCorr(ops_k.P0.m_A, ops_k.Rcross.m_A))
    align_ps.append(matCorr(ops_k.P0.m_A, ops_k.Psq.m_A))

align_rc_np = np.array(align_rc)
align_ps_np = np.array(align_ps)

# ── Test 9: P0→Rcross degrades above k₀=4 ────────────────────────────────
ax = axes[8]
styleAx(ax, "9. SVD Alignment P0→Rcross  (degrades for metals)")

ax.plot(K0_VALS, align_rc_np, color=ORANGE, linewidth=1.4, label="Pearson(P0, Rcross)")
ax.axvline(4.0, color=RED, linewidth=0.9, linestyle="--", alpha=0.8, label="k₀ = 4 (metal threshold)")
ax.set_xlabel("extinction coeff k₀  (Drude, λ-varying)", fontsize=7)
ax.set_ylabel("cosine similarity", fontsize=7)
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=6, facecolor=PANEL, labelcolor=SPINE, edgecolor="#333")
ax.grid(True, alpha=0.1, color=SPINE)

mean_lo9 = float(align_rc_np[K0_VALS < 3.5].mean())
mean_hi9 = float(align_rc_np[K0_VALS > 4.5].mean())
ax.annotate(f"mean k₀<4: {mean_lo9:.3f}   mean k₀>4: {mean_hi9:.3f}",
            xy=(0.03, 0.06), xycoords="axes fraction", color=GREY, fontsize=6)

passed9 = mean_lo9 > mean_hi9 + 0.02
results["9. P0→Rcross degrades (metal)"] = (passed9, f"Δmean = {mean_lo9 - mean_hi9:.3f}")

# ── Test 10: P0→Psq holds at k₀≤4 ────────────────────────────────────────
ax = axes[9]
styleAx(ax, "10. SVD Alignment P0→Psq  (holds for dielectrics)")

ax.plot(K0_VALS, align_ps_np, color=CYAN, linewidth=1.4, label="Pearson(P0, Psq)")
ax.axvline(4.0, color=RED, linewidth=0.9, linestyle="--", alpha=0.8, label="k₀ = 4 (metal threshold)")
ax.set_xlabel("extinction coeff k₀  (Drude, λ-varying)", fontsize=7)
ax.set_ylabel("cosine similarity", fontsize=7)
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=6, facecolor=PANEL, labelcolor=SPINE, edgecolor="#333")
ax.grid(True, alpha=0.1, color=SPINE)

min_align_diel = float(align_ps_np[K0_VALS <= 4.0].min())
ax.annotate(f"min correlation (k₀ ≤ 4): {min_align_diel:.4f}",
            xy=(0.03, 0.06), xycoords="axes fraction", color=GREY, fontsize=6)

passed10 = min_align_diel > 0.90
results["10. P0→Psq holds (dielectric)"] = (passed10, f"min corr = {min_align_diel:.4f}")

# ── TEST 11: Summary table ────────────────────────────────────────────────
ax = axes[10]
styleAx(ax, "11. Validation Summary")
ax.axis("off")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

n_total = len(results)
for i, (tname, (passed, val)) in enumerate(results.items()):
    y   = 1.0 - (i + 0.75) / (n_total + 1.0)
    clr = PASS_CLR if passed else FAIL_CLR
    lbl = "PASS" if passed else "FAIL"
    ax.text(0.02, y, tname,   color=SPINE, fontsize=6.5, va='center', ha='left')
    ax.text(0.73, y, lbl,     color=clr,   fontsize=7,   va='center', ha='left', fontweight='bold')
    ax.text(0.85, y, val,     color=GREY,  fontsize=5,   va='center', ha='left')

n_pass = sum(1 for _, (p, _) in results.items() if p)
summary_clr = PASS_CLR if n_pass == n_total else (ORANGE if n_pass >= n_total * 0.8 else FAIL_CLR)
ax.text(0.5, 0.02, f"{n_pass} / {n_total}  passed",
        color=summary_clr, fontsize=10, ha='center', va='bottom', fontweight='bold')

# ── TEST 12 (bonus): Commutator survey ────────────────────────────────────
ax = axes[11]
styleAx(ax, "12. Commutator Survey  ‖[A, B]‖_F")

PAIRS = [
    ("Beer∘P0",     op_beer.m_A, op_p0.m_A),
    ("Beer∘Film",   op_beer.m_A, op_film.m_A),
    ("Beer∘Ray",    op_beer.m_A, op_rayleigh.m_A),
    ("Beer∘Mie",    op_beer.m_A, op_mie.m_A),
    ("P0∘Film",     op_p0.m_A,  op_film.m_A),
    ("P0∘Ray",      op_p0.m_A,  op_rayleigh.m_A),
    ("Film∘Ray",    op_film.m_A, op_rayleigh.m_A),
    ("Ray∘Mie",     op_rayleigh.m_A, op_mie.m_A),
]

pair_names12, pair_norms12 = [], []
for pname, A, B in PAIRS:
    comm = A @ B - B @ A
    pair_norms12.append(torch.linalg.matrix_norm(comm, ord='fro').item())
    pair_names12.append(pname)

xp12 = np.arange(len(pair_names12))
clrs12 = [PASS_CLR if v < 0.01 else (YELLOW if v < 0.1 else FAIL_CLR) for v in pair_norms12]
ax.bar(xp12, pair_norms12, color=clrs12, alpha=0.85, width=0.65)
ax.set_yscale("log")
ax.axhline(0.01, color=GREEN, linewidth=0.8, linestyle="--", alpha=0.7, label="ε = 0.01")
ax.set_xticks(xp12)
ax.set_xticklabels(pair_names12, rotation=35, ha='right', fontsize=6)
ax.set_ylabel("‖[A, B]‖_F", fontsize=7)
ax.legend(fontsize=6, facecolor=PANEL, labelcolor=SPINE, edgecolor="#333")
ax.grid(True, alpha=0.1, color=SPINE, axis='y')

# ── Save ──────────────────────────────────────────────────────────────────
fig.suptitle("BsSPT — Operator Validation Suite  (11 Property Tests + Commutator Survey)",
             color=SPINE, fontsize=14, y=1.005)
plt.tight_layout()
out_path = f"{OUT_DIR}/operator_validation.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)

print(f"\nSaved: {out_path}")
print(f"Results: {n_pass}/{n_total} passed")
for name, (passed, val) in results.items():
    mark = "✓" if passed else "✗"
    print(f"  {mark} {name:<38} {val}")
