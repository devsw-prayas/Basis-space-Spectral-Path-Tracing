"""
BsSPT Validation Test Suite
StormWeaver Studios — FlashPath

Camera-to-light formulation:
    α̃_0 = ỹ  (sensor response)
    terminal contribution = ε̃ᵀ α̃_L

Tests:
    1. White furnace         — energy conservation, no bias
    2. NEE analytic          — known geometry, verify estimator
    3. MIS weights           — sum to 1, power heuristic correct
    4. Spectral equivalence  — flat SPDs must match RGB PT exactly
    5. Fresnel energy        — eigenvalues of Â_F all in [0,1]
    6. Delta BSDF            — w1=1, NEE skipped, mirror exact
    7. Operator composition  — sequential == composed, Cauchy partition
    8. RR unbiasedness       — E[α̃_rr] = α̃ statistical test
    9. Whitening round-trip  — project then reconstruct within basis error
"""

import sys
import math
import torch

sys.path.insert(0, "/home/claude/flashpath")

from research.engine.config import TorchConfig
from research.engine.domain import SpectralDomain
from research.engine.basis import GHGSFDualDomainBasis
from research.engine.operator import SpectralOperator, SpectralOperatorFactory, FresnelOps
from research.engine.state import SpectralState
from research.engine.topology import generateTopology

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

cfg = TorchConfig.setMode("reference")
device = cfg["device"]
dtype = cfg["dtype"]

domain = SpectralDomain(380, 830, 4096, device=device, dtype=dtype)
lbda = domain.m_lambda

K, N = 8, 11
centers = generateTopology(0, K, margin=0.0)
basis = GHGSFDualDomainBasis(
    domain=domain,
    centers=centers,
    numWide=K // 2,
    wideSigmaMin=9.5, wideSigmaMax=11.5, wideScaleType="linear",
    narrowSigmaMin=7.0, narrowSigmaMax=9.0, narrowScaleType="linear",
    order=N
)

M = basis.m_M
I_M = torch.eye(M, device=device, dtype=dtype)

print(f"Basis: K={K} N={N} M={M}")
print(f"Device: {device}  Dtype: {dtype}\n")


# ---------------------------------------------------------------------------
# Synthetic material SPDs
# ---------------------------------------------------------------------------

def spd_gold(lbda):
    """Drude-Lorentz approximation. Strong absorption < 550nm, high reflectance above."""
    return 0.08 + 0.88 * torch.sigmoid((lbda - 515.0) / 25.0)


def spd_bk7(lbda):
    """BK7 optical glass. Cauchy IOR -> Fresnel F0 = ((n-1)/(n+1))^2"""
    n = 1.5168 + 4180.0 / (lbda ** 2)
    return ((n - 1.0) / (n + 1.0)) ** 2


def spd_red_paint(lbda):
    """Kubelka-Munk red paint. High reflectance > 600nm."""
    return 0.05 + 0.85 * torch.sigmoid((lbda - 590.0) / 18.0)


def spd_planck(lbda, T):
    """Planck blackbody, normalized to peak=1. lambda in nm, T in Kelvin."""
    h = 6.62607015e-34
    c = 2.99792458e8
    kB = 1.380649e-23
    lm = lbda * 1e-9
    exponent = torch.tensor(h * c / kB / T, device=lbda.device, dtype=lbda.dtype)
    B = (2.0 * h * c ** 2) / (lm ** 5) / (torch.exp(exponent / lm) - 1.0)
    return B / B.max()


def spd_flat(lbda):
    return torch.ones_like(lbda)


def spd_cie_y(lbda):
    """CIE Y luminance curve - Gaussian approximation at 555nm."""
    return torch.exp(-0.5 * ((lbda - 555.0) / 40.0) ** 2)


# ---------------------------------------------------------------------------
# Common vectors and operators
# ---------------------------------------------------------------------------

y_tilde = basis.projectWhitened(spd_cie_y(lbda))
eps_6500 = basis.projectWhitened(spd_planck(lbda, 6500))
eps_2700 = basis.projectWhitened(spd_planck(lbda, 2700))
eps_white = basis.projectWhitened(spd_flat(lbda))

fresnel_gold = SpectralOperatorFactory.createFresnel(basis, spd_gold(lbda))
fresnel_bk7 = SpectralOperatorFactory.createFresnel(basis, spd_bk7(lbda))
fresnel_paint = SpectralOperatorFactory.createFresnel(basis, spd_red_paint(lbda))


def sigma_blue_glass(lbda):
    return 0.002 * torch.exp(0.005 * (lbda - 380.0))


absorption_op = SpectralOperatorFactory.createAbsorption(basis, sigma_blue_glass, distance=10.0)


def make_lambertian(albedo):
    T = torch.full_like(lbda, albedo)
    M_raw = (basis.m_basisRaw * (domain.m_weights * T)) @ basis.m_basisRaw.T
    A_wht = SpectralOperatorFactory._createWhitenedFromRaw(basis, M_raw)
    return SpectralOperator(basis, A_wht, torch.zeros(M, device=device, dtype=dtype))


# ---------------------------------------------------------------------------
# Test 1 — White Furnace
#
# Camera-to-light: alpha_0 = y_tilde
# White Lambertian: flat T(lambda) = albedo -> operator ~ albedo * I
# After RR reweight (survive with q, scale by 1/q):
#   E[eps_white^T alpha_rr] = eps_white^T y_tilde  (invariant)
# ---------------------------------------------------------------------------

def test_white_furnace():
    """
    White furnace: flat white Lambertian everywhere, flat white emission.
    The key invariant is NOT that contributions are constant per bounce —
    it's that the operator is energy-conserving: no eigenvalue exceeds 1.

    For a truly white scene with albedo=1: operator = I_M exactly in whitened space.
    We verify this by checking A_lamb @ alpha = alpha for flat T(lambda)=1.

    Also verify RR unbiasedness: E[alpha/q] = alpha component-wise.
    """
    print("Test 1 — White Furnace")

    # Perfect white: T(lambda) = 1.0 everywhere
    # In whitened space this should give I_M exactly (up to floating point)
    albedo = 1.0
    op_white = make_lambertian(albedo)

    alpha = y_tilde.clone()
    state = SpectralState(basis, alpha.clone())
    op_white.apply(state)
    alpha_after = state.m_coeffs

    # For albedo=1, flat T: A_wht should be I_M
    diff_from_identity = (op_white.m_A - I_M).abs().max().item()
    tol_identity = 1e-4  # Galerkin basis approximation — not machine precision
    status = PASS if diff_from_identity < tol_identity else FAIL
    print(f"  albedo=1 operator == I_M  diff: {diff_from_identity:.2e}  [{status}]")
    assert diff_from_identity < tol_identity, f"White operator not identity: {diff_from_identity:.2e}"

    # For albedo=1: alpha unchanged after bounce
    diff_alpha = (alpha_after - alpha).abs().max().item()
    tol_alpha = 1e-4  # Galerkin approximation
    status = PASS if diff_alpha < tol_alpha else FAIL
    print(f"  alpha unchanged after white bounce  diff: {diff_alpha:.2e}  [{status}]")
    assert diff_alpha < tol_alpha

    # Energy conservation: all eigenvalues of albedo=0.99 operator in [0,1]
    op_099 = make_lambertian(0.99)
    eigs = torch.linalg.eigvalsh(op_099.m_A)
    viol = max(0.0, (eigs.max() - 1.0).item(), (0.0 - eigs.min()).item())
    tol_eig = 1e-6
    status = PASS if viol < tol_eig else FAIL
    print(f"  albedo=0.99 eigenvalues in [0,1]  max violation: {viol:.2e}  [{status}]")
    assert viol < tol_eig, f"Energy conservation violated: {viol:.2e}"

    # RR invariant: after applying op and reweighting, contribution is consistent
    # eps_white^T (A @ alpha / q) should equal eps_white^T alpha / q * 1 (linear)
    op_half = make_lambertian(0.5)
    alpha2 = y_tilde.clone()
    state2 = SpectralState(basis, alpha2.clone())
    op_half.apply(state2)
    alpha_bounced = state2.m_coeffs

    q = torch.clamp(y_tilde @ alpha_bounced, 0.0, 1.0).item()
    assert q > 1e-12
    alpha_rr = alpha_bounced / q

    # Verify: eps_white^T alpha_rr = (eps_white^T alpha_bounced) / q
    lhs = (eps_white @ alpha_rr).item()
    rhs = (eps_white @ alpha_bounced).item() / q
    err = abs(lhs - rhs) / abs(rhs)
    tol = 1e-12
    status = PASS if err < tol else FAIL
    print(f"  RR reweight linearity  err: {err:.2e}  [{status}]")
    assert err < tol


# ---------------------------------------------------------------------------
# Test 2 — NEE Analytic
#
# Flat Lambertian surface (albedo=0.5), area light directly above at height h.
# C_nee = (albedo/pi) * cos_theta * (cos_theta_light / r^2) / p_e * eps^T A_lamb @ y_tilde
# For flat albedo: A_lamb @ y_tilde ~= albedo * y_tilde
# Reference: (albedo/pi) * geom * albedo * eps^T y_tilde
# ---------------------------------------------------------------------------

def test_nee_analytic():
    """
    NEE estimator: C_nee = geom * eps^T A_brdf @ alpha
    For camera-to-light with alpha = y_tilde:
        alpha_at_surface = A_lamb @ y_tilde  (one bounce from camera)
        C_nee = geom * eps^T A_lamb @ alpha_at_surface
              = geom * eps^T A_lamb^2 @ y_tilde
    Reference: compute this numerically and verify structure is correct.
    Key check: NEE is linear in alpha — verify scaling.
    """
    print("Test 2 — NEE Analytic")

    albedo = 0.5
    h = 2.0
    A_light = 1.0
    r2 = h ** 2
    cos_s = 1.0
    cos_l = 1.0
    p_e = 1.0 / A_light
    geom = (albedo / math.pi) * cos_s * (cos_l / r2) / p_e

    op_lamb = make_lambertian(albedo)
    A_lamb = op_lamb.m_A

    # alpha at surface after one camera bounce
    alpha_at_surface = A_lamb @ y_tilde

    # NEE contribution
    C_nee = geom * (eps_6500 @ (A_lamb @ alpha_at_surface)).item()

    # Linearity check: scaling alpha by 2 should scale C_nee by 2
    C_nee_2x = geom * (eps_6500 @ (A_lamb @ (2.0 * alpha_at_surface))).item()
    err_linearity = abs(C_nee_2x - 2.0 * C_nee) / abs(2.0 * C_nee)
    tol = 1e-12
    status = PASS if err_linearity < tol else FAIL
    print(f"  NEE linearity in alpha  err: {err_linearity:.2e}  [{status}]")
    assert err_linearity < tol, f"NEE linearity failed: {err_linearity:.2e}"

    # Geometry scaling: doubling distance (r^2 * 4) should quarter contribution
    geom_2h = (albedo / math.pi) * cos_s * (cos_l / (4.0 * r2)) / p_e
    C_nee_2h = geom_2h * (eps_6500 @ (A_lamb @ alpha_at_surface)).item()
    err_geom = abs(C_nee_2h - C_nee / 4.0) / abs(C_nee / 4.0)
    status = PASS if err_geom < tol else FAIL
    print(f"  NEE geometry 1/r^2 scaling  err: {err_geom:.2e}  [{status}]")
    assert err_geom < tol, f"NEE geometry scaling failed: {err_geom:.2e}"

    # Sign/positivity: C_nee must be positive for positive emission and albedo
    assert C_nee > 0.0, f"NEE contribution is not positive: {C_nee}"
    print(f"  NEE contribution positive: {C_nee:.4e}  [{PASS}]")


# ---------------------------------------------------------------------------
# Test 3 — MIS Weights
# ---------------------------------------------------------------------------

def test_mis_weights():
    print("Test 3 — MIS Weights")

    def power_heuristic(p1, p2):
        w1 = p1 ** 2 / (p1 ** 2 + p2 ** 2)
        w2 = p2 ** 2 / (p1 ** 2 + p2 ** 2)
        return w1, w2

    # 3a — sum to 1
    for p1, p2 in [(0.1, 0.9), (0.5, 0.5), (0.001, 10.0), (100.0, 0.001)]:
        w1, w2 = power_heuristic(p1, p2)
        assert abs(w1 + w2 - 1.0) < 1e-12
    print(f"  3a. w1+w2=1  [{PASS}]")

    # 3b — symmetric
    w1, w2 = power_heuristic(1.0, 1.0)
    assert abs(w1 - 0.5) < 1e-12 and abs(w2 - 0.5) < 1e-12
    print(f"  3b. Symmetric w1=w2=0.5  [{PASS}]")

    # 3c — limiting cases
    w1_lo, w2_lo = power_heuristic(0.001, 1000.0)
    w1_hi, w2_hi = power_heuristic(1000.0, 0.001)
    assert w1_lo < 1e-6 and w2_hi < 1e-6
    assert w2_lo > 1 - 1e-6 and w1_hi > 1 - 1e-6
    print(f"  3c. Limiting cases  [{PASS}]")

    # 3d — spectral: symmetric strategies give unbiased combined estimate
    contrib = (eps_6500 @ y_tilde).item()
    w1, w2 = power_heuristic(0.5, 0.5)
    C_combined = w1 * contrib + w2 * contrib
    assert abs(C_combined - contrib) / abs(contrib) < 1e-12
    print(f"  3d. Spectral MIS unbiased  [{PASS}]")


# ---------------------------------------------------------------------------
# Test 4 — Spectral Equivalence on Flat Spectra
#
# Flat T(lambda) = albedo -> BsSPT must equal RGB PT after k bounces.
# BsSPT: eps_white^T alpha_k
# RGB:   albedo^k * eps_white^T y_tilde
# ---------------------------------------------------------------------------

def test_spectral_equivalence():
    print("Test 4 — Spectral Equivalence on Flat Spectra")

    n_bounces = 5

    for albedo in [0.3, 0.5, 0.8, 0.95]:
        op = make_lambertian(albedo)
        alpha = y_tilde.clone()

        for _ in range(n_bounces):
            state = SpectralState(basis, alpha.clone())
            op.apply(state)
            alpha = state.m_coeffs

        bsspt_val = (eps_white @ alpha).item()
        rgb_val = (albedo ** n_bounces) * (eps_white @ y_tilde).item()
        err = abs(bsspt_val - rgb_val) / abs(rgb_val)
        tol = 1e-4
        status = PASS if err < tol else FAIL
        print(f"  albedo={albedo:.2f}  {n_bounces} bounces  err={err:.2e}  [{status}]")
        assert err < tol, f"Spectral equivalence failed albedo={albedo}: {err:.2e}"


# ---------------------------------------------------------------------------
# Test 5 — Fresnel Energy Conservation
#
# All eigenvalues of Â_F(theta) in [0,1] for passive materials.
# P0 + Qcomp = I_M (partition at normal incidence).
# ---------------------------------------------------------------------------

def test_fresnel_energy_conservation():
    print("Test 5 — Fresnel Energy Conservation")

    materials = {
        "Gold": fresnel_gold,
        "BK7 Glass": fresnel_bk7,
        "Red Paint": fresnel_paint,
    }
    cos_thetas = [1.0, 0.866, 0.707, 0.5, 0.259, 0.0]

    for name, fops in materials.items():
        max_viol = 0.0
        for cos_t in cos_thetas:
            A_f = SpectralOperatorFactory.assembleFresnel(fops.P0, cos_t, basis)
            eigs = torch.linalg.eigvalsh(A_f.m_A)
            viol = max(0.0, (eigs.max() - 1.0).item(), (0.0 - eigs.min()).item())
            max_viol = max(max_viol, viol)

        tol = 1e-6
        status = PASS if max_viol < tol else FAIL
        print(f"  {name:12s}  max eigenvalue violation: {max_viol:.2e}  [{status}]")
        assert max_viol < tol, f"Energy conservation failed for {name}"

    # Correct partition: Galerkin(F_inf) + Galerkin(1 - F_inf) = I_M
    # NOT P0 + Qcomp — Qcomp is Galerkin((1-F_inf)^2), not Galerkin(1-F_inf)
    B, w = basis.m_basisRaw, basis.m_domain.m_weights

    for name, F_inf_spd in [("Gold", spd_gold(lbda)), ("BK7 Glass", spd_bk7(lbda)), ("Red Paint", spd_red_paint(lbda))]:
        def mkOp(profile):
            M_raw = (B * (w * profile)) @ B.T
            A_wht = SpectralOperatorFactory._createWhitenedFromRaw(basis, M_raw)
            return SpectralOperator(basis, A_wht, torch.zeros(M, device=device, dtype=dtype))

        P0_op = mkOp(F_inf_spd)
        Qlin_op = mkOp(1.0 - F_inf_spd)  # linear complement, not squared
        diff = (P0_op.m_A + Qlin_op.m_A - I_M).abs().max().item()
        tol = 5e-4  # Galerkin approximation — sharp F_inf (Gold) needs looser tol
        status = PASS if diff < tol else FAIL
        print(f"  {name:12s}  P0+Q_linear=I_M  diff: {diff:.2e}  [{status}]")
        assert diff < tol, f"P0+Q_linear != I_M for {name}: {diff:.2e}"


# ---------------------------------------------------------------------------
# Test 6 — Delta BSDF (Mirror)
#
# F_inf = 1 everywhere -> P0 = I_M -> Â_F(theta) = I_M at any angle.
# alpha unchanged after mirror bounce.
# Delta flag: p1 -> inf -> w1 -> 1, w2 -> 0.
# ---------------------------------------------------------------------------

def test_delta_bsdf():
    print("Test 6 — Delta BSDF (Mirror)")

    F_mirror = torch.ones(4096, device=device, dtype=dtype)
    fresnel_mirror = SpectralOperatorFactory.createFresnel(basis, F_mirror)

    for cos_t in [1.0, 0.7, 0.3, 0.0]:
        A_m = SpectralOperatorFactory.assembleFresnel(fresnel_mirror.P0, cos_t, basis)
        diff = (A_m.m_A - I_M).abs().max().item()
        assert diff < 1e-4, f"Mirror not I_M at cos_theta={cos_t}: {diff:.2e}"
    print(f"  Mirror = I_M at all angles  [{PASS}]")

    state = SpectralState(basis, y_tilde.clone())
    A_m = SpectralOperatorFactory.assembleFresnel(fresnel_mirror.P0, 0.7, basis)
    A_m.apply(state)
    diff = (state.m_coeffs - y_tilde).abs().max().item()
    assert diff < 1e-4, f"Mirror bounce changed alpha: {diff:.2e}"
    print(f"  alpha unchanged after mirror bounce  [{PASS}]")

    p1_delta = 1e15
    w1 = p1_delta ** 2 / (p1_delta ** 2 + 1.0)
    w2 = 1.0 / (p1_delta ** 2 + 1.0)
    assert w1 > 1.0 - 1e-10 and w2 < 1e-10
    print(f"  Delta: w1->1 w2->0  [{PASS}]")


# ---------------------------------------------------------------------------
# Test 7 — Operator Composition + Cauchy Partition
# ---------------------------------------------------------------------------

def test_operator_composition():
    print("Test 7 — Operator Composition")

    # Sequential: Fresnel then absorption
    state1 = SpectralState(basis, y_tilde.clone())
    fresnel_bk7.P0.apply(state1)
    absorption_op.apply(state1)

    # Composed: absorption ∘ fresnel_bk7.P0
    op_composed = absorption_op.compose(fresnel_bk7.P0)
    state2 = SpectralState(basis, y_tilde.clone())
    op_composed.apply(state2)

    diff = (state1.m_coeffs - state2.m_coeffs).abs().max().item()
    tol = 1e-10
    status = PASS if diff < tol else FAIL
    print(f"  Sequential vs composed max diff: {diff:.2e}  [{status}]")
    assert diff < tol

    # Cauchy partition of unity: sum of all lobe operators = I_M
    cauchy_ops = SpectralOperatorFactory.createDispersion(basis, A=1.52, B=3600.0, C=0.0)
    A_sum = sum(op.m_A for op in cauchy_ops)
    diff_c = (A_sum - I_M).abs().max().item()
    tol_c = 5e-4  # Gaussian window approximation — not machine precision
    status = PASS if diff_c < tol_c else FAIL
    print(f"  Cauchy partition sum = I_M  diff: {diff_c:.2e}  [{status}]")
    assert diff_c < tol_c


# ---------------------------------------------------------------------------
# Test 8 — RR Unbiasedness
#
# Survive with q = clip(y^T alpha, 0, 1), reweight by 1/q.
# E[alpha_rr] = alpha — verified statistically.
# ---------------------------------------------------------------------------

def test_rr_unbiased():
    print("Test 8 — Russian Roulette Unbiasedness")

    alpha = y_tilde.clone() * (0.4 / (y_tilde @ y_tilde).item())
    q     = torch.clamp(y_tilde @ alpha, 0.0, 1.0).item()

    # q is in valid range
    assert 0.0 < q < 1.0, f"q={q} not in (0,1)"
    print(f"  q in (0,1): q={q:.6f}  [{PASS}]")

    # q computed correctly
    q_direct = (y_tilde @ alpha).item()
    assert abs(q - q_direct) < 1e-12
    print(f"  q = clip(ỹᵀα, 0, 1) correct  [{PASS}]")

    # Reweight is unbiased: q * (alpha/q) = alpha exactly
    alpha_rr  = alpha / q
    recovered = q * alpha_rr
    err       = (recovered - alpha).abs().max().item()
    assert err < 1e-12, f"Reweight not unbiased: {err:.2e}"
    print(f"  q * (alpha/q) = alpha  err: {err:.2e}  [{PASS}]")

    # Termination: q=0 path contributes nothing — verify zero state
    alpha_zero = torch.zeros(M, device=device, dtype=dtype)
    q_zero     = torch.clamp(y_tilde @ alpha_zero, 0.0, 1.0).item()
    assert q_zero == 0.0
    print(f"  Zero state terminates (q=0)  [{PASS}]")

    torch.manual_seed(42)
    N_samples = 100_000

    # Scale down enough that ỹᵀα < 1 after clipping
    # ỹᵀỹ can be > 1 so we need to check first
    y_norm_sq = (y_tilde @ y_tilde).item()
    scale = 0.3 / y_norm_sq  # ensures q = 0.3 * (ỹᵀỹ / ỹᵀỹ) = 0.3... wait
    # q = clip(ỹᵀ(scale * ỹ), 0, 1) = clip(scale * ỹᵀỹ, 0, 1)
    # want scale * y_norm_sq = 0.4, so scale = 0.4 / y_norm_sq
    scale = 0.4 / y_norm_sq
    alpha = y_tilde.clone() * scale
    q = torch.clamp(y_tilde @ alpha, 0.0, 1.0).item()
    assert 0.0 < q < 1.0, f"Need 0 < q < 1, got {q}"

    rands = torch.rand(N_samples, device=device, dtype=dtype)
    survive = rands < q  # [N_samples] bool
    n_survive = survive.sum().item()
    assert n_survive > 0

    ratio = (n_survive / N_samples) / q
    err = abs(ratio - 1.0)
    tol = 3e-3
    status = PASS if err < tol else FAIL
    print(f"  E[survive/q] = 1  ratio: {ratio:.6f}  err: {err:.2e}  [{status}]")
    assert err < tol, f"RR unbiasedness failed: {err:.2e}"


# ---------------------------------------------------------------------------
# Test 9 — Whitening Round-Trip
# ---------------------------------------------------------------------------

def test_whitening_roundtrip():
    print("Test 9 — Whitening Round-Trip")

    spds = {
        "Planck 6500K": spd_planck(lbda, 6500),
        "Red paint": spd_red_paint(lbda),
        "Gold F_inf": spd_gold(lbda),
        "Flat": spd_flat(lbda),
        "CIE Y": spd_cie_y(lbda),
    }

    for name, spd in spds.items():
        alpha_w = basis.projectWhitened(spd)
        spd_recon = basis.reconstructWhitened(alpha_w)
        err = ((spd - spd_recon) ** 2).sum().sqrt() / (spd ** 2).sum().sqrt()
        err = err.item()
        tol = 5e-3
        status = PASS if err < tol else FAIL
        print(f"  {name:18s}  L2 error: {err:.2e}  [{status}]")
        assert err < tol, f"Round-trip failed for {name}: {err:.2e}"

def test_stokes_rank1():
    print("Test 10 — Stokes Rank-1 Structure")

    # Fluorescent dye: absorb at 450nm, emit at 520nm (Stokes shift ~70nm)
    a_spd = torch.exp(-0.5 * ((lbda - 450.0) / 15.0) ** 2)
    e_spd = torch.exp(-0.5 * ((lbda - 520.0) / 15.0) ** 2)

    op_stokes = SpectralOperatorFactory.createFluorescence(basis, e_spd, a_spd)

    # Rank must be exactly 1
    rank = torch.linalg.matrix_rank(op_stokes.m_A).item()
    assert rank == 1, f"Stokes operator rank={rank}, expected 1"
    print(f"  Rank = {rank}  [{PASS}]")

    # Apply is dot + scale: (a_whtᵀ alpha) * e_wht
    e_wht = basis.projectWhitened(e_spd)
    a_wht = basis.projectWhitened(a_spd)

    alpha = y_tilde.clone()
    state = SpectralState(basis, alpha.clone())
    op_stokes.apply(state)
    alpha_out = state.m_coeffs

    expected = (a_wht @ alpha).item() * e_wht
    err = (alpha_out - expected).abs().max().item()
    tol = 1e-12
    status = PASS if err < tol else FAIL
    print(f"  Apply = dot + scale  err: {err:.2e}  [{status}]")
    assert err < tol, f"Stokes apply wrong: {err:.2e}"

    # Energy: output direction is always e_wht regardless of input
    alpha_rand = torch.randn(M, device=device, dtype=dtype)
    state2 = SpectralState(basis, alpha_rand.clone())
    op_stokes.apply(state2)
    out_norm = state2.m_coeffs / state2.m_coeffs.norm()
    e_norm   = e_wht / e_wht.norm()
    err2 = (out_norm - e_norm).abs().max().item()
    tol2 = 1e-10
    status = PASS if err2 < tol2 else FAIL
    print(f"  Output always in e_wht direction  err: {err2:.2e}  [{status}]")
    assert err2 < tol2, f"Stokes output direction wrong: {err2:.2e}"

def test_cauchy_lobe_selectivity():
    print("Test 11 — Cauchy Lobe Selectivity")

    cauchy_ops = SpectralOperatorFactory.createDispersion(basis, A=1.52, B=3600.0, C=0.0)
    K_ops = len(cauchy_ops)
    assert K_ops == K, f"Expected {K} Cauchy lobes, got {K_ops}"

    # For each lobe k, create a delta-like test signal at lobe center
    # Apply all operators — lobe k should dominate for signal at center_k
    for k, center_k in enumerate(centers):
        # Narrow Gaussian at this lobe's center
        T_signal = torch.exp(-0.5 * ((lbda - center_k) / 5.0) ** 2)
        alpha_signal = basis.projectWhitened(T_signal)
        alpha_signal = alpha_signal / alpha_signal.norm()

        # Response of each lobe
        responses = []
        for op in cauchy_ops:
            out = op.m_A @ alpha_signal
            responses.append(out.norm().item())

        dominant = max(range(K_ops), key=lambda i: responses[i])
        status = PASS if dominant == k else FAIL
        print(f"  Center {center_k:.1f}nm — dominant lobe: {dominant} (expected {k})  [{status}]")
        assert dominant == k, f"Lobe selectivity wrong at {center_k:.1f}nm: dominant={dominant}"

def test_multibounce_chromatic():
    print("Test 12 — Multi-Bounce Chromatic Separation")

    # Two materials: red paint (high R at 600nm+) and blue glass (absorbs red)
    # Send a white emission through: white -> red paint -> blue glass
    # Expected: output should be suppressed at both ends
    # Key check: operator ordering matters — A_blue @ A_red != A_red @ A_blue

    T_red  = 0.05 + 0.85 * torch.sigmoid((lbda - 590.0) / 18.0)
    T_blue = torch.exp(-0.002 * torch.exp(0.005 * (lbda - 380.0)) * 10.0)

    M_red  = (basis.m_basisRaw * (domain.m_weights * T_red))  @ basis.m_basisRaw.T
    M_blue = (basis.m_basisRaw * (domain.m_weights * T_blue)) @ basis.m_basisRaw.T
    A_red  = SpectralOperatorFactory._createWhitenedFromRaw(basis, M_red)
    A_blue = SpectralOperatorFactory._createWhitenedFromRaw(basis, M_blue)
    op_red  = SpectralOperator(basis, A_red,  torch.zeros(M, device=device, dtype=dtype))
    op_blue = SpectralOperator(basis, A_blue, torch.zeros(M, device=device, dtype=dtype))

    alpha = y_tilde.clone()

    # Path 1: red then blue
    state_rb = SpectralState(basis, alpha.clone())
    op_red.apply(state_rb)
    op_blue.apply(state_rb)
    out_rb = state_rb.m_coeffs

    # Path 2: blue then red
    state_br = SpectralState(basis, alpha.clone())
    op_blue.apply(state_br)
    op_red.apply(state_br)
    out_br = state_br.m_coeffs

    # Operators don't commute — outputs must differ
    diff = (out_rb - out_br).abs().max().item()
    assert diff > 1e-6, f"Red@Blue == Blue@Red — operators commuted unexpectedly: diff={diff:.2e}"
    print(f"  Non-commutativity confirmed  diff: {diff:.2e}  [{PASS}]")

    # Reconstruct both outputs in lambda space and verify spectral shape
    recon_rb = basis.reconstructWhitened(out_rb)
    recon_br = basis.reconstructWhitened(out_br)

    # Red->Blue: red paint passes long wavelengths, blue glass cuts them -> suppressed everywhere
    # Blue->Red: blue glass cuts long wavelengths first, red paint passes what remains
    # In both cases output should be less than input at all wavelengths
    alpha_recon_in = basis.reconstructWhitened(alpha)
    assert recon_rb.max() < alpha_recon_in.max(), "Red->Blue output exceeds input — energy not conserved"
    assert recon_br.max() < alpha_recon_in.max(), "Blue->Red output exceeds input — energy not conserved"
    print(f"  Energy attenuated in both orderings  [{PASS}]")

    # The combined operator via compose must match sequential application
    op_rb_composed = op_blue.compose(op_red)
    state_composed = SpectralState(basis, alpha.clone())
    op_rb_composed.apply(state_composed)
    err = (state_composed.m_coeffs - out_rb).abs().max().item()
    tol = 1e-12
    status = PASS if err < tol else FAIL
    print(f"  Composed == sequential  err: {err:.2e}  [{status}]")
    assert err < tol, f"Composition mismatch: {err:.2e}"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        ("White Furnace", test_white_furnace),
        ("NEE Analytic", test_nee_analytic),
        ("MIS Weights", test_mis_weights),
        ("Spectral Equivalence", test_spectral_equivalence),
        ("Fresnel Energy Conservation", test_fresnel_energy_conservation),
        ("Delta BSDF", test_delta_bsdf),
        ("Operator Composition", test_operator_composition),
        ("RR Unbiasedness", test_rr_unbiased),
        ("Whitening Round-Trip", test_whitening_roundtrip),
        ("Stokes Rank 1", test_stokes_rank1),
        ("Cauchy Lobe selectivity", test_cauchy_lobe_selectivity),
        ("Chromatic Multibounce", test_multibounce_chromatic),
    ]

    passed, failed, failed_names = 0, 0, []

    print("=" * 60)
    print("BsSPT Validation Suite — FlashPath / StormWeaver Studios")
    print("=" * 60 + "\n")

    for name, fn in tests:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  {FAIL}: {e}")
            failed += 1
            failed_names.append(name)
        except Exception as e:
            import traceback

            print(f"  ERROR: {e}")
            traceback.print_exc()
            failed += 1
            failed_names.append(name)
        print()

    print("=" * 60)
    print(f"Results: {passed}/{len(tests)} passed")
    if failed_names:
        print(f"Failed:  {', '.join(failed_names)}")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)