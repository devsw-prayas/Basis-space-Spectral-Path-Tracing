"""
BsSPT Operator Analysis Suite
==============================
Generates three figures using the golden basis configuration:

  1. operator_heatmaps.png     — log|Â| structure for all 8 operators
  2. operator_eigenspectra.png — eigenvalue / singular value spectra
  3. d65_response.png          — D65 illuminant input vs output per operator

Output: results/operator_analysis/
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from research.engine.domain import SpectralDomain
from research.engine.basis import GHGSFDualDomainBasis
from research.engine.operator import SpectralOperatorFactory
from research.engine.topology import generateTopology

# ── Output ─────────────────────────────────────────────────────
OUT_DIR = "results/operator_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Theme ──────────────────────────────────────────────────────
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

LOBE_COLORS = [CYAN, YELLOW, GREEN, ORANGE, MAGENTA, RED, TEAL, GREY]

def styleAx(ax, title=""):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=SPINE, fontsize=8, pad=4)
    ax.tick_params(colors=GREY, labelsize=6)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333333")
    ax.xaxis.label.set_color(GREY)
    ax.yaxis.label.set_color(GREY)

# ── Golden Config ───────────────────────────────────────────────
print("Initializing basis...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
domain = SpectralDomain(380.0, 830.0, 4096, device=device, dtype=torch.float64)

K, N = 8, 11
centers = generateTopology(0, K, margin=0.0)
basis = GHGSFDualDomainBasis(
    domain=domain, centers=centers, numWide=K // 2,
    wideSigmaMin=9.5,  wideSigmaMax=11.5,  wideScaleType="linear",
    narrowSigmaMin=7.0, narrowSigmaMax=9.0, narrowScaleType="linear",
    order=N
)

lbda = domain.m_lambda.cpu().numpy()
M    = basis.m_M
print(f"  K={K}  N={N}  M={M}  κ(G)={torch.linalg.cond(basis.m_gram):.2e}")

# ── CIE D65 Illuminant (5 nm, 380–830 nm) ──────────────────────
D65_NM = np.arange(380, 835, 5, dtype=float)
D65_SPD = np.array([
    49.98,  52.31,  54.65,  68.70,  82.75,  87.12,  91.49,  92.46,  93.43,  90.06,
    86.68,  95.77, 104.86, 110.94, 117.01, 117.41, 117.81, 116.34, 114.86, 115.39,
   115.92, 112.37, 108.81, 109.08, 109.35, 108.58, 107.80, 106.30, 104.79, 106.24,
   107.69, 106.05, 104.41, 104.23, 104.05, 102.02, 100.00,  98.17,  96.33,  96.06,
    95.79,  92.24,  88.69,  89.35,  90.01,  89.80,  89.60,  88.65,  87.70,  85.49,
    83.29,  83.49,  83.70,  81.86,  80.03,  80.12,  80.21,  81.25,  82.28,  80.28,
    78.28,  74.00,  69.72,  70.67,  71.61,  72.98,  74.35,  67.98,  61.60,  65.74,
    69.89,  72.49,  75.09,  69.34,  63.59,  55.01,  46.42,  56.61,  66.81,  65.09,
    63.38,  63.84,  64.30,  61.88,  59.46,  55.71,  51.96,  54.03,  56.10,  56.02,
    55.95,
], dtype=float)

d65_np   = np.interp(lbda, D65_NM, D65_SPD)
d65_norm = d65_np / d65_np.max()
d65      = torch.tensor(d65_norm, device=device, dtype=torch.float64)
alpha_d65 = basis.projectWhitened(d65)

# ── Build All 8 Operators ───────────────────────────────────────
print("Building operators...")

sigma_a    = lambda l: torch.full_like(l, 0.002)
op_beer    = SpectralOperatorFactory.createAbsorption(basis, sigma_a, distance=1.0)

f_inf      = torch.full_like(basis.m_domain.m_lambda, 0.04)
op_p0      = SpectralOperatorFactory.createFresnel(basis, f_inf).P0

op_film    = SpectralOperatorFactory.createThinFilm(basis, n=1.5, d=300.0)

a_prof     = torch.exp(-0.5 * ((basis.m_domain.m_lambda - 400.0) / 20.0) ** 2)
e_prof     = torch.exp(-0.5 * ((basis.m_domain.m_lambda - 520.0) / 25.0) ** 2)
op_stokes  = SpectralOperatorFactory.createFluorescence(basis, e_prof, a_prof)

ops_cauchy = SpectralOperatorFactory.createDispersion(basis, A=1.5, B=0.01, C=0.0)

op_rayleigh = SpectralOperatorFactory.createScattering(basis, "Rayleigh", sigmaS_base=0.005, distance=1.0, alpha=4.0)
op_mie      = SpectralOperatorFactory.createScattering(basis, "Mie",      sigmaS_base=0.005, distance=1.0, alpha=1.0)
op_raman    = SpectralOperatorFactory.createRaman(basis, shift_nm=50.0, sigmaRaman=8.0)

print("  All operators built.")

# Ordered registry — Cauchy represented by lobe 0 for matrix figures
OPERATORS = [
    ("Beer-Lambert",  op_beer.m_A),
    ("Fresnel P0",    op_p0.m_A),
    ("Fabry-Airy",    op_film.m_A),
    ("Stokes",        op_stokes.m_A),
    ("Cauchy[k=0]",   ops_cauchy[0].m_A),
    ("Rayleigh",      op_rayleigh.m_A),
    ("Mie",           op_mie.m_A),
    ("Raman",         op_raman.m_A),
]

# Symmetric operators use eigvalsh; Stokes/Raman use svd singular values
SYMMETRIC = {"Beer-Lambert", "Fresnel P0", "Fabry-Airy", "Cauchy[k=0]", "Rayleigh", "Mie"}

# ═══════════════════════════════════════════════════════════════
# FIGURE 1 — Operator Heatmaps
# ═══════════════════════════════════════════════════════════════
print("Rendering: operator heatmaps...")

fig, axes = plt.subplots(2, 4, figsize=(28, 14))
fig.patch.set_facecolor(BG)
axes = axes.flatten()

for ax, (name, A) in zip(axes, OPERATORS):
    A_np  = A.cpu().float().numpy()
    img   = np.log1p(np.abs(A_np) * 1000)
    im    = ax.imshow(img, cmap="inferno", aspect="auto", interpolation="nearest")
    styleAx(ax, title=name)
    ax.set_xlabel("j", fontsize=6)
    ax.set_ylabel("i", fontsize=6)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="log(1 + 1000|Â|)")
    cb = ax.images[0].colorbar
    if cb:
        cb.ax.yaxis.set_tick_params(color=GREY, labelsize=5)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=GREY)

fig.suptitle("BsSPT — Operator Matrix Structure (Whitened Space, log|Â|)",
             color=SPINE, fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/operator_heatmaps.png", dpi=150,
            bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("  Saved: operator_heatmaps.png")

# ═══════════════════════════════════════════════════════════════
# FIGURE 2 — Eigenspectra
# ═══════════════════════════════════════════════════════════════
print("Rendering: eigenspectra...")

fig, axes = plt.subplots(2, 4, figsize=(28, 10))
fig.patch.set_facecolor(BG)
axes = axes.flatten()

for ax, (name, A) in zip(axes, OPERATORS):
    styleAx(ax, title=name)

    if name in SYMMETRIC:
        ev = torch.linalg.eigvalsh(A).cpu().float().numpy()
        ev_sorted = np.sort(ev)[::-1]
        ax.plot(ev_sorted, color=CYAN, linewidth=1.4, label="eigenvalues")
        ax.axhline(0, color=GREY, linewidth=0.6, linestyle="--", alpha=0.5)
        ax.set_ylabel("λᵢ", fontsize=7)
    else:
        sv = torch.linalg.svdvals(A).cpu().float().numpy()
        ax.plot(sv, color=ORANGE, linewidth=1.4, label="singular values")
        ax.set_ylabel("σᵢ", fontsize=7)
        ax.set_title(f"{name}  (SVD — asymmetric)", color=SPINE, fontsize=8, pad=4)

    ax.set_xlabel("index", fontsize=7)
    ax.grid(True, alpha=0.1, color=SPINE)

    # Annotate max and min
    vals = torch.linalg.eigvalsh(A).cpu().float().numpy() if name in SYMMETRIC \
           else torch.linalg.svdvals(A).cpu().float().numpy()
    ax.annotate(f"max={vals.max():.3f}", xy=(0.03, 0.95),
                xycoords="axes fraction", color=GREY, fontsize=6, va="top")
    ax.annotate(f"min={vals.min():.3f}", xy=(0.03, 0.85),
                xycoords="axes fraction", color=GREY, fontsize=6, va="top")

fig.suptitle("BsSPT — Operator Eigenspectra (Symmetric) / Singular Values (Asymmetric)",
             color=SPINE, fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/operator_eigenspectra.png", dpi=150,
            bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("  Saved: operator_eigenspectra.png")

# ═══════════════════════════════════════════════════════════════
# FIGURE 3 — D65 Response
# ═══════════════════════════════════════════════════════════════
print("Rendering: D65 response...")

fig, axes = plt.subplots(2, 4, figsize=(28, 12))
fig.patch.set_facecolor(BG)
axes = axes.flatten()

for ax, (name, A) in zip(axes, OPERATORS):
    styleAx(ax, title=f"{name} — D65 response")

    # Input D65
    ax.plot(lbda, d65_norm, color=GREY, linewidth=1.0,
            linestyle="--", alpha=0.6, label="D65 input")

    # Apply operator
    alpha_out = A @ alpha_d65
    s_out     = basis.reconstructWhitened(alpha_out).cpu().float().numpy()
    ax.plot(lbda, s_out, color=CYAN, linewidth=1.4, label="output")

    ax.set_xlabel("λ (nm)", fontsize=7)
    ax.set_ylabel("relative radiance", fontsize=7)
    ax.grid(True, alpha=0.1, color=SPINE)
    ax.legend(fontsize=6, facecolor=PANEL, labelcolor=SPINE,
              edgecolor="#333", loc="upper right")

# Special panel for Cauchy: replace axes[4] with per-lobe D65 decomposition
ax = axes[4]
for sp in ax.spines.values(): sp.set_edgecolor("#333333")
ax.set_facecolor(PANEL)
ax.set_title("Cauchy — per-lobe D65 decomposition", color=SPINE, fontsize=8, pad=4)
ax.tick_params(colors=GREY, labelsize=6)
ax.plot(lbda, d65_norm, color=GREY, linewidth=1.0, linestyle="--", alpha=0.5, label="D65")
lobe_centers = generateTopology(0, K, margin=0.0)
for k, op_k in enumerate(ops_cauchy):
    alpha_k = op_k.m_A @ alpha_d65
    s_k     = basis.reconstructWhitened(alpha_k).cpu().float().numpy()
    ax.plot(lbda, s_k, color=LOBE_COLORS[k], linewidth=1.1,
            label=f"k={k} ({lobe_centers[k]:.0f}nm)")
ax.set_xlabel("λ (nm)", fontsize=7)
ax.set_ylabel("relative radiance", fontsize=7)
ax.grid(True, alpha=0.1, color=SPINE)
ax.legend(fontsize=5, facecolor=PANEL, labelcolor=SPINE,
          edgecolor="#333", loc="upper right", ncol=2)
ax.xaxis.label.set_color(GREY)
ax.yaxis.label.set_color(GREY)

fig.suptitle("BsSPT — D65 Illuminant Response per Operator (Normalized Input)",
             color=SPINE, fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/d65_response.png", dpi=150,
            bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("  Saved: d65_response.png")

print(f"\nDone — all figures in {OUT_DIR}/")
