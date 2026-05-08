"""
Fabry-Airy Angle Interpolation Validation
FlashPath / BsSPT — StormWeaver Studios
Prayas Bharadwaj — 2026

Part A — theta-space (baseline, expected to fail):
  Tests 1-5: bake at degree intervals, lerp in theta-space.
  Shows why theta-space interpolation is wrong — oscillation couples lambda and theta.

Part B — p-space (correct parameterization):
  Tests 6-9: bake at OPD intervals p = d*cosθ (nm), lerp in p-space.
  p is the optical path difference — the natural physical parameter.
  Per-ray: cosθ from hit record. d is material constant.
  Theoretical step size from error bound: Δp ≤ λ_min/(4πn) * sqrt(2ε)
  For ε=0.005, λ_min=380nm, n=1.33: Δp ≤ ~22.7nm → ~14 ops for d=300nm.

Usage:
    python fabry_airy_interp_test.py

Dependencies:
    torch (matches project engine)
    Place this file alongside the research/engine/ package, or
    run from the project root so imports resolve.
"""

import sys
import math
import torch
import numpy as np

# ── Project imports ───────────────────────────────────────────────────────────
# Assumes run from project root with research/ on path
sys.path.insert(0, ".")
from research.engine.config  import TorchConfig
from research.engine.domain  import SpectralDomain
from research.engine.basis   import GHGSFDualDomainBasis
from research.engine.operator import SpectralOperator, SpectralOperatorFactory
from research.engine.topology import topologyUniform

# ── Config ────────────────────────────────────────────────────────────────────
cfg    = TorchConfig.setMode("reference")          # float64, grad off
device = cfg["device"]
dtype  = cfg["dtype"]

# ── Basis parameters (locked — K=8 N=11 M=88) ────────────────────────────────
K          = 8
N_ORDER    = 11
L_SAMPLES  = 4096
LAM_MIN    = 380.0
LAM_MAX    = 830.0

# GHGSFDualDomainBasis hyperparams (from build_plan.md)
NUM_WIDE         = 4
WIDE_SIGMA_MIN   = 9.5
WIDE_SIGMA_MAX   = 11.5
WIDE_SCALE       = "linear"
NARROW_SIGMA_MIN = 7.0
NARROW_SIGMA_MAX = 9.0
NARROW_SCALE     = "linear"

# Thin film parameters
N_IOR = 1.33        # water / soap film
D_NM  = 300.0       # film thickness nm — fixed for angle sweep

# Test parameters
STEP_DEG   = 5                   # bake interval
N_TEST     = 500                 # random test angles
SEED       = 42

# ── Build domain and basis ────────────────────────────────────────────────────
print("=" * 60)
print("Fabry-Airy Angle Interpolation Validation")
print("=" * 60)

print(f"\nBuilding SpectralDomain  L={L_SAMPLES}  [{LAM_MIN},{LAM_MAX}] nm")
domain = SpectralDomain(LAM_MIN, LAM_MAX, L_SAMPLES, device=device, dtype=dtype)

centers = topologyUniform(K, LAM_MIN, LAM_MAX)
print(f"Building GHGSFDualDomainBasis  K={K}  N={N_ORDER}  M={K*N_ORDER}")
basis = GHGSFDualDomainBasis(
    domain        = domain,
    centers       = centers,
    numWide       = NUM_WIDE,
    wideSigmaMin  = WIDE_SIGMA_MIN,
    wideSigmaMax  = WIDE_SIGMA_MAX,
    wideScaleType = WIDE_SCALE,
    narrowSigmaMin  = NARROW_SIGMA_MIN,
    narrowSigmaMax  = NARROW_SIGMA_MAX,
    narrowScaleType = NARROW_SCALE,
    order         = N_ORDER
)
M = basis.m_M
print(f"  M={M}  Gram deviation from I: ", end="")
G_wht = (basis.m_basisWhitened * domain.m_weights) @ basis.m_basisWhitened.T
print(f"{(G_wht - torch.eye(M, device=device, dtype=dtype)).abs().max().item():.2e}")

# ── Precompute CIE Y sensor vector in whitened space ─────────────────────────
# Approximate CIE Y via Gaussian centred at 555nm for pixel error metric
lam   = domain.m_lambda
cie_y = torch.exp(-0.5 * ((lam - 555.0) / 40.0) ** 2)
cie_y = cie_y / domain.integrate(cie_y)          # normalise
y_wht = basis.projectWhitened(cie_y)             # [M] sensor vector

# ── createThinFilmAngle — angle-aware factory extension ──────────────────────
def createThinFilmAngle(
    basis: GHGSFDualDomainBasis,
    n: float,
    d: float,
    theta_deg: float
) -> SpectralOperator:
    """
    Angle-aware Fabry-Airy operator.
    T(λ, θ) = 0.5 * (1 + cos(4π n d cosθ / λ))

    Extension of SpectralOperatorFactory.createThinFilm that adds
    the cosθ term for angle-dependent thin film interference.

    Args:
        basis:     GHGSFDualDomainBasis instance
        n:         refractive index (scalar)
        d:         film thickness in nm
        theta_deg: angle of incidence in degrees
    """
    B, w, lbda = basis.m_basisRaw, basis.m_domain.m_weights, basis.m_domain.m_lambda
    cos_theta  = math.cos(math.radians(theta_deg))
    T          = 0.5 * (1.0 + torch.cos(4.0 * torch.pi * n * d * cos_theta / lbda))
    M_raw      = (B * (w * T)) @ B.T
    A_wht      = SpectralOperatorFactory._createWhitenedFromRaw(basis, M_raw)
    return SpectralOperator(
        basis, A_wht,
        torch.zeros(basis.m_M, device=A_wht.device, dtype=A_wht.dtype)
    )

# ── Bake operator table at STEP_DEG intervals ─────────────────────────────────
angles_deg = list(range(0, 90 + STEP_DEG, STEP_DEG))   # [0, 5, 10, ..., 90]
print(f"\nBaking operator table at {STEP_DEG}° intervals  ({len(angles_deg)} operators)...")

op_table = {}
for theta in angles_deg:
    op_table[float(theta)] = createThinFilmAngle(basis, N_IOR, D_NM, theta)

table_kb = len(angles_deg) * M * M * 8 / 1024
print(f"  Table size: {len(angles_deg)} × {M}×{M} float64 = {table_kb:.1f} KB")
print(f"  BF16 equivalent (render): {table_kb / 4:.1f} KB")

# ── Interpolation helpers ─────────────────────────────────────────────────────
def lerp_operator(theta_deg: float, op_table: dict, step_deg: float) -> SpectralOperator:
    """
    Elementwise lerp in whitened operator space.
    Materialized path — produces interpolated A matrix explicitly.
    """
    lo = float(int(theta_deg / step_deg) * step_deg)
    hi = min(lo + step_deg, 90.0)
    lo = max(lo, 0.0)

    if lo == hi:
        return op_table[lo]

    t    = (theta_deg - lo) / step_deg
    A_lo = op_table[lo].m_A
    A_hi = op_table[hi].m_A
    A_interp = (1.0 - t) * A_lo + t * A_hi
    return SpectralOperator(
        basis, A_interp,
        torch.zeros(M, device=device, dtype=dtype)
    )

def tc_native_apply(
    theta_deg: float,
    alpha: torch.Tensor,
    op_table: dict,
    step_deg: float
) -> torch.Tensor:
    """
    TC-native interpolation path.
    Two matmuls on shared alpha, scalar combine in registers.
    Never materializes A_interp — simulates the GPU hot path.

        tmp_lo = A_lo @ alpha
        tmp_hi = A_hi @ alpha
        result = (1-t)*tmp_lo + t*tmp_hi
    """
    lo = float(int(theta_deg / step_deg) * step_deg)
    hi = min(lo + step_deg, 90.0)
    lo = max(lo, 0.0)

    if lo == hi:
        return op_table[lo].m_A @ alpha

    t      = (theta_deg - lo) / step_deg
    tmp_lo = op_table[lo].m_A @ alpha
    tmp_hi = op_table[hi].m_A @ alpha
    return (1.0 - t) * tmp_lo + t * tmp_hi

# ── Test 1: Interpolated vs ground truth ─────────────────────────────────────
print(f"\n{'─'*60}")
print(f"Test 1 — Lerped operator vs directly constructed (N={N_TEST} random angles)")
print(f"{'─'*60}")

rng = np.random.default_rng(SEED)

# Random test angles — avoid exact grid points to stress the lerp
test_angles = rng.uniform(0.1, 89.9, N_TEST)

# Random alpha vectors — unit norm in whitened space
alphas_np  = rng.standard_normal((N_TEST, M))
alphas_np /= np.linalg.norm(alphas_np, axis=1, keepdims=True)
alphas     = torch.tensor(alphas_np, device=device, dtype=dtype)

frob_errors  = []
pixel_errors = []

for i, (theta, alpha) in enumerate(zip(test_angles, alphas)):
    # Ground truth: directly baked at exact angle
    op_exact   = createThinFilmAngle(basis, N_IOR, D_NM, float(theta))
    alpha_exact = op_exact.m_A @ alpha

    # Interpolated path
    op_interp       = lerp_operator(float(theta), op_table, STEP_DEG)
    alpha_interp    = op_interp.m_A @ alpha

    # Frobenius error on output vector (relative)
    norm_exact  = torch.linalg.norm(alpha_exact).item()
    frob_err    = (torch.linalg.norm(alpha_exact - alpha_interp) / (norm_exact + 1e-12)).item()
    frob_errors.append(frob_err)

    # Pixel error: |y^T alpha_exact - y^T alpha_interp| / |y^T alpha_exact|
    lum_exact  = (y_wht @ alpha_exact).item()
    lum_interp = (y_wht @ alpha_interp).item()
    pix_err    = abs(lum_exact - lum_interp) / (abs(lum_exact) + 1e-12)
    pixel_errors.append(pix_err)

frob_errors  = np.array(frob_errors)
pixel_errors = np.array(pixel_errors)

print(f"  Frobenius error (relative output vector):")
print(f"    max  = {frob_errors.max():.4e}")
print(f"    mean = {frob_errors.mean():.4e}")
print(f"    p95  = {np.percentile(frob_errors, 95):.4e}")
print(f"  Pixel error (|Δy^T α| / |y^T α|):")
print(f"    max  = {pixel_errors.max():.4e}  ({pixel_errors.max()*100:.3f}%)")
print(f"    mean = {pixel_errors.mean():.4e}  ({pixel_errors.mean()*100:.3f}%)")
print(f"    p95  = {np.percentile(pixel_errors, 95):.4e}  ({np.percentile(pixel_errors, 95)*100:.3f}%)")

# ── Test 2: TC-native path == materialized lerp ───────────────────────────────
print(f"\n{'─'*60}")
print(f"Test 2 — TC-native path vs materialized lerp (same {N_TEST} angles)")
print(f"{'─'*60}")

tc_errors = []
for theta, alpha in zip(test_angles, alphas):
    alpha_mat = lerp_operator(float(theta), op_table, STEP_DEG).m_A @ alpha
    alpha_tc  = tc_native_apply(float(theta), alpha, op_table, STEP_DEG)

    err = (torch.linalg.norm(alpha_mat - alpha_tc) /
           (torch.linalg.norm(alpha_mat) + 1e-12)).item()
    tc_errors.append(err)

tc_errors = np.array(tc_errors)
print(f"  Max deviation materialized vs TC-native: {tc_errors.max():.2e}")
print(f"  Mean deviation:                          {tc_errors.mean():.2e}")
print(f"  (expect ~0 — pure floating point associativity difference)")

# ── Test 3: Error vs step size sweep ─────────────────────────────────────────
print(f"\n{'─'*60}")
print(f"Test 3 — Frobenius & pixel error vs step size")
print(f"{'─'*60}")
print(f"  {'Step':>6}  {'#ops':>5}  {'Frob max':>12}  {'Frob mean':>12}  "
      f"{'Pix max %':>10}  {'Pix mean %':>10}  {'KB (BF16)':>10}")
print(f"  {'─'*6}  {'─'*5}  {'─'*12}  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*10}")

for step in [1, 2, 5, 10, 15, 30]:
    step_angles = list(range(0, 90 + step, step))
    step_table  = {}
    for theta in step_angles:
        step_table[float(theta)] = createThinFilmAngle(basis, N_IOR, D_NM, theta)

    f_errs = []
    p_errs = []
    for theta, alpha in zip(test_angles, alphas):
        op_exact     = createThinFilmAngle(basis, N_IOR, D_NM, float(theta))
        alpha_exact  = op_exact.m_A @ alpha
        alpha_interp = lerp_operator(float(theta), step_table, step).m_A @ alpha

        norm_e  = torch.linalg.norm(alpha_exact).item()
        f_errs.append((torch.linalg.norm(alpha_exact - alpha_interp) / (norm_e + 1e-12)).item())

        lum_e = (y_wht @ alpha_exact).item()
        lum_i = (y_wht @ alpha_interp).item()
        p_errs.append(abs(lum_e - lum_i) / (abs(lum_e) + 1e-12))

    f_errs = np.array(f_errs)
    p_errs = np.array(p_errs)
    n_ops  = len(step_angles)
    kb     = n_ops * M * M * 2 / 1024   # BF16

    print(f"  {step:>5}°  {n_ops:>5}  {f_errs.max():>12.4e}  {f_errs.mean():>12.4e}  "
          f"{p_errs.max()*100:>10.4f}  {p_errs.mean()*100:>10.4f}  {kb:>10.1f}")

# ── Test 4: Worst-case angles (grazing) ──────────────────────────────────────
print(f"\n{'─'*60}")
print(f"Test 4 — Error at specific angles of interest")
print(f"{'─'*60}")

interest_angles = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 87.5, 89.0, 89.9]
alpha_test = alphas[0]   # single representative alpha

print(f"  {'Angle':>8}  {'Frob err':>12}  {'Pix err %':>12}  Notes")
print(f"  {'─'*8}  {'─'*12}  {'─'*12}  {'─'*20}")

for theta in interest_angles:
    op_exact     = createThinFilmAngle(basis, N_IOR, D_NM, theta)
    alpha_exact  = op_exact.m_A @ alpha_test
    alpha_interp = lerp_operator(theta, op_table, STEP_DEG).m_A @ alpha_test

    norm_e = torch.linalg.norm(alpha_exact).item()
    f_err  = (torch.linalg.norm(alpha_exact - alpha_interp) / (norm_e + 1e-12)).item()
    lum_e  = (y_wht @ alpha_exact).item()
    lum_i  = (y_wht @ alpha_interp).item()
    p_err  = abs(lum_e - lum_i) / (abs(lum_e) + 1e-12) * 100

    # Flag if this angle is on the grid (should be ~0)
    on_grid = abs(theta % STEP_DEG) < 1e-9
    note    = "ON GRID (expect ~0)" if on_grid else ""
    print(f"  {theta:>7.1f}°  {f_err:>12.4e}  {p_err:>11.4f}%  {note}")

# ── Precision path helpers ────────────────────────────────────────────────────
# Simulates the actual GPU pipeline:
#   Operator table  → BF16
#   Alpha vectors   → FP32
#   HMMA            → FP32 accumulate

def cast_op_bf16_fp32(op: SpectralOperator) -> torch.Tensor:
    """Cast operator matrix to BF16 then back to FP32 — matches operator table on GPU."""
    return op.m_A.to(torch.bfloat16).to(torch.float32)

def apply_bf16_fp32(A_f32: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Apply BF16-quantized operator to FP32 alpha. Returns FP32."""
    return A_f32 @ alpha.to(torch.float32)

def lerp_operator_bf16_fp32(
    theta_deg: float,
    op_table: dict,
    step_deg: float
) -> torch.Tensor:
    """
    TC-native interpolation in BF16+FP32:
      A_lo, A_hi cast to BF16 → FP32
      tmp_lo = A_lo_f32 @ alpha_f32
      tmp_hi = A_hi_f32 @ alpha_f32
      result = (1-t)*tmp_lo + t*tmp_hi   (FP32)
    Returns the assembled FP32 interpolated matrix (for reuse across alphas).
    """
    lo = float(int(theta_deg / step_deg) * step_deg)
    hi = min(lo + step_deg, 90.0)
    lo = max(lo, 0.0)

    A_lo_f32 = cast_op_bf16_fp32(op_table[lo])
    if lo == hi:
        return A_lo_f32

    t        = (theta_deg - lo) / step_deg
    A_hi_f32 = cast_op_bf16_fp32(op_table[hi])
    # Materialize interpolated matrix in FP32 — for batch alpha application
    # On the GPU this never materializes — stays as two separate HMMA calls
    # The result is numerically identical either way
    return (1.0 - t) * A_lo_f32 + t * A_hi_f32

# ── Test 5: Three precision paths ────────────────────────────────────────────
# 1a  f64 interp    vs f64 exact       — pure interpolation error
# 1b  BF16+FP32     vs f64 exact       — combined interp + quantization error
# 1c  BF16+FP32     vs BF16+FP32 exact — apples-to-apples, what renderer sees

print(f"\n{'─'*60}")
print(f"Test 5 — Three precision paths  (N={N_TEST} random angles)")
print(f"{'─'*60}")
print(f"  1a  f64   interp vs f64   exact   — pure interpolation error")
print(f"  1b  BF16+FP32   vs f64   exact   — combined interp + quantization")
print(f"  1c  BF16+FP32   vs BF16+FP32 exact — renderer apples-to-apples")
print()

# Precompute BF16+FP32 versions of the baked table
op_table_f32 = {
    theta: cast_op_bf16_fp32(op)
    for theta, op in op_table.items()
}

p1a_frob, p1a_pix = [], []
p1b_frob, p1b_pix = [], []
p1c_frob, p1c_pix = [], []

for theta, alpha in zip(test_angles, alphas):
    theta = float(theta)

    # ── ground truths ──
    op_exact_f64  = createThinFilmAngle(basis, N_IOR, D_NM, theta)
    A_exact_f64   = op_exact_f64.m_A                              # float64
    A_exact_f32   = cast_op_bf16_fp32(op_exact_f64)               # BF16 → FP32

    alpha_f64 = alpha                                              # float64
    alpha_f32 = alpha.to(torch.float32)

    out_exact_f64 = A_exact_f64 @ alpha_f64                       # f64 reference
    out_exact_f32 = A_exact_f32 @ alpha_f32                       # BF16+FP32 reference

    # ── interpolated outputs ──
    # 1a: f64 interp → f64 alpha
    A_interp_f64  = lerp_operator(theta, op_table, STEP_DEG).m_A
    out_interp_f64 = A_interp_f64 @ alpha_f64

    # 1b / 1c: BF16+FP32 interp → FP32 alpha
    A_interp_f32  = lerp_operator_bf16_fp32(theta, op_table, STEP_DEG)
    out_interp_f32 = A_interp_f32 @ alpha_f32

    # ── errors ──
    def vec_errors(out_test, out_ref, y_vec):
        out_ref_cast = out_ref.to(out_test.dtype)
        y_cast       = y_vec.to(out_test.dtype)
        norm_ref = torch.linalg.norm(out_ref_cast).item()
        frob     = (torch.linalg.norm(out_test - out_ref_cast) / (norm_ref + 1e-12)).item()
        lum_ref  = (y_cast @ out_ref_cast).item()
        lum_tst  = (y_cast @ out_test).item()
        pix      = abs(lum_ref - lum_tst) / (abs(lum_ref) + 1e-12)
        return frob, pix

    f, p = vec_errors(out_interp_f64, out_exact_f64, y_wht)
    p1a_frob.append(f); p1a_pix.append(p)

    f, p = vec_errors(out_interp_f32, out_exact_f64, y_wht)
    p1b_frob.append(f); p1b_pix.append(p)

    f, p = vec_errors(out_interp_f32, out_exact_f32, y_wht)
    p1c_frob.append(f); p1c_pix.append(p)

def report(label, frob, pix):
    frob, pix = np.array(frob), np.array(pix)
    print(f"  {label}")
    print(f"    Frob  max={frob.max():.4e}  mean={frob.mean():.4e}  p95={np.percentile(frob,95):.4e}")
    print(f"    Pixel max={pix.max()*100:.4f}%  mean={pix.mean()*100:.4f}%  p95={np.percentile(pix,95)*100:.4f}%")
    print()
    return frob, pix

f1a, p1a = report("1a  f64 interp vs f64 exact     (pure interpolation error)",   p1a_frob, p1a_pix)
f1b, p1b = report("1b  BF16+FP32  vs f64 exact     (interp + quantization)",      p1b_frob, p1b_pix)
f1c, p1c = report("1c  BF16+FP32  vs BF16+FP32 exact (renderer apples-to-apples)",p1c_frob, p1c_pix)

# Decompose error budget
print(f"  Error budget decomposition:")
print(f"    Interpolation only    (1a pixel max):          {p1a.max()*100:.4f}%")
print(f"    Quantization only     (1b - 1a pixel max):     {(p1b.max()-p1a.max())*100:.4f}%  (approx)")
print(f"    Combined render error (1b pixel max):          {p1b.max()*100:.4f}%")
print(f"    Renderer sees (1c):                            {p1c.max()*100:.4f}%")
print(f"    BF16 cast fidelity reference (from bf16 analysis): ~1.52e-3 Frobenius")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Summary")
print(f"{'='*60}")
print(f"  Basis:           K={K} N={N_ORDER} M={M}  D={D_NM}nm  n={N_IOR}")
print(f"  Step size:       {STEP_DEG}°  ({len(angles_deg)} baked operators)")
print(f"  Test angles:     {N_TEST} random in (0.1°, 89.9°)")
print()
print(f"  f64 interpolation vs ground truth (Test 1 / 5-1a):")
print(f"    Max pixel error:  {pixel_errors.max()*100:.4f}%")
print(f"    Mean pixel error: {pixel_errors.mean()*100:.4f}%")
print()
print(f"  TC-native vs materialized lerp (Test 2):")
print(f"    Max deviation:    {tc_errors.max():.2e}  (floating point only)")
print()
print(f"  Precision paths (Test 5):")
print(f"    1a pure interp error:      {p1a.max()*100:.4f}% max pixel")
print(f"    1b interp + quant error:   {p1b.max()*100:.4f}% max pixel")
print(f"    1c renderer actual error:  {p1c.max()*100:.4f}% max pixel")
print()
print(f"  Conclusion:")
THRESHOLD = 0.005
if p1b.max() < THRESHOLD:
    print(f"    PASS — combined error {p1b.max()*100:.4f}% < 0.5% threshold")
    print(f"    5-degree table with BF16 operators is sufficient for SC26")
else:
    print(f"    FAIL — combined error {p1b.max()*100:.4f}% exceeds 0.5% threshold")
    print(f"    Check Test 3 sweep — reduce step size or upgrade to FP32 operators")

# ═══════════════════════════════════════════════════════════
# PART B — p-space interpolation (correct parameterization)
# ═══════════════════════════════════════════════════════════
#
# p = d * cosθ   [nm]  — optical path difference
#
# T(λ, p) = 0.5 * (1 + cos(4π n p / λ))
#
# p is the natural parameter — no arbitrary reference wavelength.
# d is material constant. cosθ is the only per-ray quantity.
#
# Theoretical step bound for lerp error ε:
#   Δp ≤ λ_min / (4π n) * sqrt(2ε)
#   For ε=0.005, λ_min=380nm, n=1.33: Δp ≤ 22.7nm
#   For d=300nm: N_ops = ceil(300/22.7) ≈ 14
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"PART B — p-space interpolation  (p = d·cosθ, units: nm)")
print(f"{'='*60}")

# ── Theoretical step size ─────────────────────────────────
EPSILON_TARGET = 0.005    # 0.5% pixel error threshold
delta_p_theory = (LAM_MIN / (4.0 * math.pi * N_IOR)) * math.sqrt(2.0 * EPSILON_TARGET)
n_ops_theory   = math.ceil(D_NM / delta_p_theory)
print(f"\nTheoretical step size for ε={EPSILON_TARGET*100:.1f}%:")
print(f"  Δp ≤ {delta_p_theory:.2f} nm")
print(f"  N_ops for d={D_NM}nm: {n_ops_theory}")

# ── p-space factory ───────────────────────────────────────
def createThinFilmP(
    basis: GHGSFDualDomainBasis,
    n: float,
    p: float
) -> SpectralOperator:
    """
    Bake Fabry-Airy operator at optical path difference p = d*cosθ (nm).
    T(λ, p) = 0.5 * (1 + cos(4π n p / λ))
    p is the only parameter — d and cosθ don't appear separately.
    """
    B, w, lbda = basis.m_basisRaw, basis.m_domain.m_weights, basis.m_domain.m_lambda
    T          = 0.5 * (1.0 + torch.cos(4.0 * torch.pi * n * p / lbda))
    M_raw      = (B * (w * T)) @ B.T
    A_wht      = SpectralOperatorFactory._createWhitenedFromRaw(basis, M_raw)
    return SpectralOperator(
        basis, A_wht,
        torch.zeros(basis.m_M, device=A_wht.device, dtype=A_wht.dtype)
    )

# ── Bake p-space table at theoretical step size ───────────
P_STEP = delta_p_theory
p_samples = np.arange(0.0, D_NM + P_STEP, P_STEP)
p_samples = np.clip(p_samples, 0.0, D_NM)

print(f"\nBaking p-space table  Δp={P_STEP:.2f}nm  ({len(p_samples)} operators)...")
p_op_table = {}
for p_val in p_samples:
    p_op_table[float(p_val)] = createThinFilmP(basis, N_IOR, float(p_val))

table_kb_p = len(p_samples) * M * M * 8 / 1024
print(f"  Table size: {len(p_samples)} × {M}×{M} float64 = {table_kb_p:.1f} KB")
print(f"  BF16 equivalent (render): {table_kb_p/4:.1f} KB")

# ── p-space lerp helpers ──────────────────────────────────
def lerp_p(p_val: float, p_op_table: dict, p_step: float) -> SpectralOperator:
    """Lerp in p-space. p_val = d * cosθ from hit record."""
    p_arr  = np.array(sorted(p_op_table.keys()))
    lo_val = float(p_arr[np.searchsorted(p_arr, p_val, side='right') - 1])
    hi_idx = min(np.searchsorted(p_arr, p_val, side='right'), len(p_arr) - 1)
    hi_val = float(p_arr[hi_idx])

    if lo_val == hi_val:
        return p_op_table[lo_val]

    t        = (p_val - lo_val) / (hi_val - lo_val)
    A_lo     = p_op_table[lo_val].m_A
    A_hi     = p_op_table[hi_val].m_A
    A_interp = (1.0 - t) * A_lo + t * A_hi
    return SpectralOperator(basis, A_interp, torch.zeros(M, device=device, dtype=dtype))

def tc_native_apply_p(p_val: float, alpha: torch.Tensor,
                       p_op_table: dict, p_step: float) -> torch.Tensor:
    """TC-native path in p-space. Two matmuls + scalar combine."""
    p_arr  = np.array(sorted(p_op_table.keys()))
    lo_val = float(p_arr[np.searchsorted(p_arr, p_val, side='right') - 1])
    hi_idx = min(np.searchsorted(p_arr, p_val, side='right'), len(p_arr) - 1)
    hi_val = float(p_arr[hi_idx])

    if lo_val == hi_val:
        return p_op_table[lo_val].m_A @ alpha

    t      = (p_val - lo_val) / (hi_val - lo_val)
    tmp_lo = p_op_table[lo_val].m_A @ alpha
    tmp_hi = p_op_table[hi_val].m_A @ alpha
    return (1.0 - t) * tmp_lo + t * tmp_hi

# ── Test 6: p-space lerp vs ground truth (f64) ───────────
print(f"\n{'─'*60}")
print(f"Test 6 — p-space lerp vs ground truth  (N={N_TEST}, f64)")
print(f"{'─'*60}")

# Random p values — uniformly in [0, D_NM]
# p = d * cosθ, so this covers all physical angles
rng2        = np.random.default_rng(SEED + 1)
test_p_vals = rng2.uniform(0.0, D_NM, N_TEST)

# Random alpha vectors
alphas2_np  = rng2.standard_normal((N_TEST, M))
alphas2_np /= np.linalg.norm(alphas2_np, axis=1, keepdims=True)
alphas2     = torch.tensor(alphas2_np, device=device, dtype=dtype)

t6_frob, t6_pix = [], []

for p_val, alpha in zip(test_p_vals, alphas2):
    op_exact      = createThinFilmP(basis, N_IOR, float(p_val))
    alpha_exact   = op_exact.m_A @ alpha
    alpha_interp  = lerp_p(float(p_val), p_op_table, P_STEP).m_A @ alpha

    norm_e = torch.linalg.norm(alpha_exact).item()
    t6_frob.append((torch.linalg.norm(alpha_exact - alpha_interp) / (norm_e + 1e-12)).item())
    lum_e  = (y_wht @ alpha_exact).item()
    lum_i  = (y_wht @ alpha_interp).item()
    t6_pix.append(abs(lum_e - lum_i) / (abs(lum_e) + 1e-12))

t6_frob = np.array(t6_frob)
t6_pix  = np.array(t6_pix)
print(f"  Frobenius  max={t6_frob.max():.4e}  mean={t6_frob.mean():.4e}  p95={np.percentile(t6_frob,95):.4e}")
print(f"  Pixel      max={t6_pix.max()*100:.4f}%  mean={t6_pix.mean()*100:.4f}%  p95={np.percentile(t6_pix,95)*100:.4f}%")

# ── Test 7: TC-native path in p-space ────────────────────
print(f"\n{'─'*60}")
print(f"Test 7 — TC-native path in p-space  (two matmuls + scalar combine)")
print(f"{'─'*60}")

t7_errs = []
for p_val, alpha in zip(test_p_vals, alphas2):
    alpha_mat = lerp_p(float(p_val), p_op_table, P_STEP).m_A @ alpha
    alpha_tc  = tc_native_apply_p(float(p_val), alpha, p_op_table, P_STEP)
    t7_errs.append((torch.linalg.norm(alpha_mat - alpha_tc) /
                    (torch.linalg.norm(alpha_mat) + 1e-12)).item())

t7_errs = np.array(t7_errs)
print(f"  Max deviation materialized vs TC-native: {t7_errs.max():.2e}")
print(f"  Mean deviation:                          {t7_errs.mean():.2e}")
print(f"  (expect ~0 — floating point associativity only)")

# ── Test 8: Precision paths in p-space ───────────────────
print(f"\n{'─'*60}")
print(f"Test 8 — p-space precision paths  (f64 / BF16+FP32)")
print(f"{'─'*60}")

p_op_table_f32 = {
    p_val: cast_op_bf16_fp32(op)
    for p_val, op in p_op_table.items()
}

def lerp_p_bf16_fp32(p_val: float, p_op_table_f32: dict) -> torch.Tensor:
    p_arr  = np.array(sorted(p_op_table_f32.keys()))
    lo_val = float(p_arr[np.searchsorted(p_arr, p_val, side='right') - 1])
    hi_idx = min(np.searchsorted(p_arr, p_val, side='right'), len(p_arr) - 1)
    hi_val = float(p_arr[hi_idx])

    A_lo_f32 = p_op_table_f32[lo_val]
    if lo_val == hi_val:
        return A_lo_f32

    t        = (p_val - lo_val) / (hi_val - lo_val)
    A_hi_f32 = p_op_table_f32[hi_val]
    return (1.0 - t) * A_lo_f32 + t * A_hi_f32

p8a_frob, p8a_pix = [], []
p8b_frob, p8b_pix = [], []
p8c_frob, p8c_pix = [], []

for p_val, alpha in zip(test_p_vals, alphas2):
    p_val = float(p_val)

    op_exact_f64  = createThinFilmP(basis, N_IOR, p_val)
    A_exact_f64   = op_exact_f64.m_A
    A_exact_f32   = cast_op_bf16_fp32(op_exact_f64)

    alpha_f64 = alpha
    alpha_f32 = alpha.to(torch.float32)

    out_exact_f64 = A_exact_f64 @ alpha_f64
    out_exact_f32 = A_exact_f32 @ alpha_f32

    # 8a: f64 interp vs f64 exact
    A_interp_f64   = lerp_p(p_val, p_op_table, P_STEP).m_A
    out_interp_f64 = A_interp_f64 @ alpha_f64

    # 8b / 8c: BF16+FP32 interp vs f64 / BF16+FP32 exact
    A_interp_f32   = lerp_p_bf16_fp32(p_val, p_op_table_f32)
    out_interp_f32 = A_interp_f32 @ alpha_f32

    f, p = vec_errors(out_interp_f64, out_exact_f64, y_wht)
    p8a_frob.append(f); p8a_pix.append(p)

    f, p = vec_errors(out_interp_f32, out_exact_f64, y_wht)
    p8b_frob.append(f); p8b_pix.append(p)

    f, p = vec_errors(out_interp_f32, out_exact_f32, y_wht)
    p8c_frob.append(f); p8c_pix.append(p)

f8a, p8a = report("8a  f64 interp vs f64 exact     (pure interpolation)",    p8a_frob, p8a_pix)
f8b, p8b = report("8b  BF16+FP32  vs f64 exact     (interp + quantization)", p8b_frob, p8b_pix)
f8c, p8c = report("8c  BF16+FP32  vs BF16+FP32 exact (renderer actual)",     p8c_frob, p8c_pix)

# ── Test 9: Step size sweep in p-space ───────────────────
print(f"\n{'─'*60}")
print(f"Test 9 — p-space step size sweep  (theory predicts Δp≤{delta_p_theory:.1f}nm)")
print(f"{'─'*60}")
print(f"  {'Δp (nm)':>8}  {'#ops':>5}  {'Frob max':>12}  {'Frob mean':>12}  "
      f"{'Pix max %':>10}  {'Pix mean %':>10}  {'KB (BF16)':>10}  {'vs theory':>10}")
print(f"  {'─'*8}  {'─'*5}  {'─'*12}  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")

for dp in [5.0, 10.0, 15.0, 22.7, 30.0, 50.0, 100.0]:
    sweep_p  = np.arange(0.0, D_NM + dp, dp)
    sweep_p  = np.clip(sweep_p, 0.0, D_NM)
    sweep_tbl = {float(pv): createThinFilmP(basis, N_IOR, float(pv)) for pv in sweep_p}

    sf, sp = [], []
    for p_val, alpha in zip(test_p_vals, alphas2):
        op_ex      = createThinFilmP(basis, N_IOR, float(p_val))
        a_ex       = op_ex.m_A @ alpha
        a_int      = lerp_p(float(p_val), sweep_tbl, dp).m_A @ alpha
        norm_e     = torch.linalg.norm(a_ex).item()
        sf.append((torch.linalg.norm(a_ex - a_int) / (norm_e + 1e-12)).item())
        lum_e = (y_wht @ a_ex).item()
        lum_i = (y_wht @ (lerp_p(float(p_val), sweep_tbl, dp).m_A @ alpha)).item()
        sp.append(abs(lum_e - lum_i) / (abs(lum_e) + 1e-12))

    sf, sp   = np.array(sf), np.array(sp)
    n_ops    = len(sweep_p)
    kb       = n_ops * M * M * 2 / 1024
    vs_theory = "≤ bound" if dp <= delta_p_theory else "> bound"
    print(f"  {dp:>7.1f}  {n_ops:>5}  {sf.max():>12.4e}  {sf.mean():>12.4e}  "
          f"{sp.max()*100:>10.4f}  {sp.mean()*100:>10.4f}  {kb:>10.1f}  {vs_theory:>10}")

# ── Final summary ─────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Final Summary — p-space vs theta-space")
print(f"{'='*60}")
print(f"  Parameterization    Max pixel error   N_ops   KB BF16")
print(f"  {'─'*55}")
print(f"  theta-space 5°      {p1b.max()*100:>10.4f}%      19    287.4")
print(f"  p-space Δp={delta_p_theory:.1f}nm   {p8b.max()*100:>10.4f}%      {len(p_samples):>2}    {len(p_samples)*M*M*2/1024:.1f}")
print()
print(f"  p-space error budget:")
print(f"    Pure interpolation (8a):     {f8a.max()*100:.4f}% frob  {p8a.max()*100:.4f}% pixel")
print(f"    + BF16 quantization (8b):    {f8b.max()*100:.4f}% frob  {p8b.max()*100:.4f}% pixel")
print(f"    Renderer apples-to-apples (8c): {p8c.max()*100:.4f}% pixel")
print()
THRESHOLD = 0.005
verdict = "PASS" if p8b.max() < THRESHOLD else "FAIL"
print(f"  Verdict: {verdict} — combined BF16+FP32 error {p8b.max()*100:.4f}% "
      f"({'<' if p8b.max() < THRESHOLD else '>'} {THRESHOLD*100:.1f}% threshold)")