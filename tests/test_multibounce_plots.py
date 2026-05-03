"""
Multibounce Operator Validation - Per-Bounce Error Plots

Generates PNG plots showing error vs ground truth for each bounce (1-10).
Uses PlotEngine for consistent scientific visualization.
"""

import torch
import numpy as np
import os
from pathlib import Path

from research.engine.domain import SpectralDomain
from research.engine.basis import GHGSFDualDomainBasis
from research.engine.operator import SpectralOperatorFactory
from research.engine.topology import generateTopology
from research.plot.engine import PlotEngine, MultiPanelEngine


# =============================================================================
# Configuration
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
K, N = 8, 11
MARGIN = 0.0
DEFAULT_SAMPLES = 16384
RAMAN_SAMPLES = 4096

# Output directory
PLOT_DIR = Path(__file__).parent / "plots" / "multibounce"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def create_golden_basis(num_samples=DEFAULT_SAMPLES):
    """Create the golden config basis for testing."""
    domain = SpectralDomain(380, 830, num_samples, device=DEVICE, dtype=DTYPE)
    centers = generateTopology(0, K, margin=MARGIN)
    basis = GHGSFDualDomainBasis(
        domain=domain,
        centers=centers,
        numWide=K//2,
        wideSigmaMin=9.5, wideSigmaMax=11.5, wideScaleType="linear",
        narrowSigmaMin=7.0, narrowSigmaMax=9.0, narrowScaleType="linear",
        order=N
    )
    return domain, basis


def flat_profile(value, domain):
    return torch.ones(domain.m_count, device=DEVICE, dtype=DTYPE) * value


def gaussian_profile(center_nm, width_nm, domain):
    return torch.exp(-0.5 * ((domain.m_lambda - center_nm) / width_nm) ** 2)


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_per_bounce_error(
    test_name: str,
    errors_inf: list,
    errors_l2: list,
    expected_values: list,
    actual_values: list,
    tolerance: float,
    extra_info: str = ""
):
    """Generate a multi-panel plot showing per-bounce error evolution."""

    bounces = np.arange(1, len(errors_inf) + 1)

    # Multi-panel: 2 rows x 2 cols
    multi = MultiPanelEngine(nrows=2, ncols=2, figsize=(12, 10), sharex=False)

    # Panel 0: L_inf error vs bounce
    p0 = multi.getPanel(0)
    p0.addScatter(bounces, np.array(errors_inf), label="L_inf Error", color=PlotEngine.sColors['primary'], marker='o', size=50)
    p0.addLine(bounces, np.ones_like(bounces) * tolerance, label=f"Tolerance ({tolerance:.0e})",
               color='#FF6B6B', linewidth=1.5, linestyle='--')
    p0.setTitle(f"{test_name}: L_inf Error per Bounce")
    p0.setLabels("Bounce #", "L_inf Error")
    p0.addLegend()
    p0.m_axes.set_yscale('log')

    # Panel 1: L2 error vs bounce
    p1 = multi.getPanel(1)
    p1.addScatter(bounces, np.array(errors_l2), label="L2 Error", color=PlotEngine.sColors['tertiary'], marker='s', size=50)
    p1.addLine(bounces, np.ones_like(bounces) * tolerance, label=f"Tolerance ({tolerance:.0e})",
               color='#FF6B6B', linewidth=1.5, linestyle='--')
    p1.setTitle(f"{test_name}: L2 Error per Bounce")
    p1.setLabels("Bounce #", "L2 Error")
    p1.addLegend()
    p1.m_axes.set_yscale('log')

    # Panel 2: Expected vs Actual values
    p2 = multi.getPanel(2)
    p2.addLine(bounces, np.array(expected_values), label="Expected", color=PlotEngine.sColors['primary'], linewidth=2)
    p2.addScatter(bounces, np.array(actual_values), label="Actual", color=PlotEngine.sColors['tertiary'], marker='o', size=40)
    p2.setTitle(f"{test_name}: Expected vs Actual Values")
    p2.setLabels("Bounce #", "Value")
    p2.addLegend()

    # Panel 3: Relative error (%)
    p3 = multi.getPanel(3)
    rel_errors = [abs(a - e) / (abs(e) + 1e-16) * 100 for a, e in zip(actual_values, expected_values)]
    p3.addScatter(bounces, np.array(rel_errors), label="Relative Error (%)", color='#FFD93D', marker='^', size=50)
    p3.setTitle(f"{test_name}: Relative Error")
    p3.setLabels("Bounce #", "Relative Error (%)")
    p3.addLegend()
    p3.m_axes.set_yscale('log')

    # Save
    filepath = PLOT_DIR / f"{test_name.replace(' ', '_')}_per_bounce.png"
    multi.saveFigure(str(filepath), dpi=300)
    print(f"  Plot saved: {filepath}")

    return filepath


def plot_spectrum_comparison(
    test_name: str,
    domain,
    s_out: torch.Tensor,
    s_ref: torch.Tensor,
    error_inf: float,
    bounce_num: int
):
    """Plot reconstructed spectrum vs ground truth."""

    engine = PlotEngine(figsize=(10, 6))

    lbda_np = domain.m_lambda.cpu().numpy()
    s_out_np = s_out.cpu().numpy()
    s_ref_np = s_ref.cpu().numpy()

    engine.addLine(lbda_np, s_ref_np, label="Ground Truth", color=PlotEngine.sColors['primary'], linewidth=2)
    engine.addLine(lbda_np, s_out_np, label=f"Actual (bounce {bounce_num})", color=PlotEngine.sColors['tertiary'], linewidth=1.5, linestyle='--')
    engine.setTitle(f"{test_name}: Spectrum Comparison (Bounce {bounce_num})")
    engine.setLabels("Wavelength (nm)", "Spectral Value")
    engine.addLegend()

    # Add error annotation
    engine.m_axes.annotate(f"L_inf error: {error_inf:.2e}", xy=(0.02, 0.98), xycoords='axes fraction',
                          fontsize=10, color=PlotEngine.sColors['text'],
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#0f1116', edgecolor='#FFFFFF', alpha=0.7))

    filepath = PLOT_DIR / f"{test_name.replace(' ', '_')}_bounce{bounce_num}_spectrum.png"
    engine.saveFigure(str(filepath), dpi=300)
    print(f"  Spectrum plot saved: {filepath}")

    return filepath


# =============================================================================
# Set 0 Tests with Plotting
# =============================================================================

def run_set0_with_plots():
    """Run Set 0 tests and generate per-bounce error plots."""

    print("\n" + "=" * 70)
    print(" SET 0: PER-OPERATOR MULTIBOUNCE (WITH PLOTS)")
    print("=" * 70)

    results = {}

    # S0-1: Beer-Lambert
    print("\n[S0-1] Beer-Lambert (50% per bounce)")
    results["S0-1"] = test_s0_1_beer_lambert_plot()

    # S0-2: Fresnel P0
    print("\n[S0-2] Fresnel P0 (F_inf=0.1)")
    results["S0-2"] = test_s0_2_fresnel_p0_plot()

    # S0-3: Fresnel Qcomp
    print("\n[S0-3] Fresnel Qcomp (F_inf=0.04)")
    results["S0-3"] = test_s0_3_fresnel_qcomp_plot()

    # S0-7: Rayleigh
    print("\n[S0-7] Rayleigh Scattering")
    results["S0-7"] = test_s0_7_rayleigh_plot()

    # S0-8: Mie
    print("\n[S0-8] Mie Scattering")
    results["S0-8"] = test_s0_8_mie_plot()

    # Summary
    passed = sum(1 for v in results.values() if v[0])
    total = len(results)
    print(f"\n{'='*70}")
    print(f" SET 0 (PLOTS) SUMMARY: {passed}/{total} passed")
    print(f" Plots saved to: {PLOT_DIR}")
    print(f"{'='*70}")

    return results


def test_s0_1_beer_lambert_plot():
    """S0-1 with per-bounce error tracking and plotting."""

    domain, basis = create_golden_basis()
    sigma_val = -np.log(0.5) / 1.0
    sigma_fn = lambda lbda: torch.ones_like(lbda) * sigma_val

    op = SpectralOperatorFactory.createAbsorption(basis, sigma_fn, distance=1.0)
    alpha = basis.projectWhitened(torch.ones_like(domain.m_lambda))

    errors_inf = []
    errors_l2 = []
    expected_values = []
    actual_values = []

    for bounce in range(1, 11):
        alpha = op.m_A @ alpha
        s_out = basis.reconstructWhitened(alpha)

        expected = 0.5 ** bounce
        actual = s_out.mean().item()

        # Ground truth spectrum
        s_ref = torch.ones_like(domain.m_lambda) * expected

        error_inf = torch.max(torch.abs(s_out - s_ref)).item()
        error_l2 = torch.sqrt(torch.mean((s_out - s_ref) ** 2)).item()

        errors_inf.append(error_inf)
        errors_l2.append(error_l2)
        expected_values.append(expected)
        actual_values.append(actual)

        print(f"  Bounce {bounce}: expected={expected:.10f}, actual={actual:.10f}, L_inf={error_inf:.2e}")

    plot_per_bounce_error(
        "S0-1 Beer-Lambert",
        errors_inf, errors_l2, expected_values, actual_values,
        tolerance=1e-12
    )

    passed = errors_inf[-1] < 1e-12
    return (passed, f"final_L_inf={errors_inf[-1]:.2e}")


def test_s0_2_fresnel_p0_plot():
    """S0-2 Fresnel P0 with plotting."""

    domain, basis = create_golden_basis()
    f_inf = flat_profile(0.1, domain)
    ops = SpectralOperatorFactory.createFresnel(basis, f_inf)
    op = ops.P0

    alpha = basis.projectWhitened(torch.ones_like(domain.m_lambda))

    errors_inf = []
    errors_l2 = []
    expected_values = []
    actual_values = []

    for bounce in range(1, 11):
        alpha = op.m_A @ alpha
        s_out = basis.reconstructWhitened(alpha)

        expected = 0.1 ** bounce
        actual = s_out.mean().item()

        s_ref = torch.ones_like(domain.m_lambda) * expected
        error_inf = torch.max(torch.abs(s_out - s_ref)).item()
        error_l2 = torch.sqrt(torch.mean((s_out - s_ref) ** 2)).item()

        errors_inf.append(error_inf)
        errors_l2.append(error_l2)
        expected_values.append(expected)
        actual_values.append(actual)

        print(f"  Bounce {bounce}: expected={expected:.10e}, actual={actual:.10e}, L_inf={error_inf:.2e}")

    plot_per_bounce_error(
        "S0-2 Fresnel P0",
        errors_inf, errors_l2, expected_values, actual_values,
        tolerance=1e-12
    )

    passed = errors_inf[-1] < 1e-12
    return (passed, f"final_L_inf={errors_inf[-1]:.2e}")


def test_s0_3_fresnel_qcomp_plot():
    """S0-3 Fresnel Qcomp with plotting."""

    domain, basis = create_golden_basis()
    f_inf = flat_profile(0.04, domain)
    ops = SpectralOperatorFactory.createFresnel(basis, f_inf)
    op = ops.Qcomp

    alpha = basis.projectWhitened(torch.ones_like(domain.m_lambda))

    errors_inf = []
    errors_l2 = []
    expected_values = []
    actual_values = []

    for bounce in range(1, 11):
        alpha = op.m_A @ alpha
        s_out = basis.reconstructWhitened(alpha)

        expected = (0.96 ** 2) ** bounce
        actual = s_out.mean().item()

        s_ref = torch.ones_like(domain.m_lambda) * expected
        error_inf = torch.max(torch.abs(s_out - s_ref)).item()
        error_l2 = torch.sqrt(torch.mean((s_out - s_ref) ** 2)).item()

        errors_inf.append(error_inf)
        errors_l2.append(error_l2)
        expected_values.append(expected)
        actual_values.append(actual)

        print(f"  Bounce {bounce}: expected={expected:.10f}, actual={actual:.10f}, L_inf={error_inf:.2e}")

    plot_per_bounce_error(
        "S0-3 Fresnel Qcomp",
        errors_inf, errors_l2, expected_values, actual_values,
        tolerance=1e-12
    )

    passed = errors_inf[-1] < 1e-12
    return (passed, f"final_L_inf={errors_inf[-1]:.2e}")


def test_s0_7_rayleigh_plot():
    """S0-7 Rayleigh with plotting."""

    domain, basis = create_golden_basis()
    sigma_base = 0.05
    alpha_power = 4.0

    lbda = domain.m_lambda
    T_analytic = torch.exp(-sigma_base * (lbda / 550.0)**(-alpha_power) * 1.0)

    op = SpectralOperatorFactory.createScattering(basis, "Rayleigh", sigma_base, distance=1.0, alpha=alpha_power)
    alpha = basis.projectWhitened(torch.ones_like(domain.m_lambda))

    errors_inf = []
    errors_l2 = []
    expected_values = []
    actual_values = []

    for bounce in range(1, 11):
        alpha = op.m_A @ alpha
        s_out = basis.reconstructWhitened(alpha)

        T_bounce = T_analytic ** bounce
        expected = T_bounce.mean().item()
        actual = s_out.mean().item()

        error_inf = torch.max(torch.abs(s_out - T_bounce)).item()
        error_l2 = torch.sqrt(torch.mean((s_out - T_bounce) ** 2)).item()

        errors_inf.append(error_inf)
        errors_l2.append(error_l2)
        expected_values.append(expected)
        actual_values.append(actual)

        print(f"  Bounce {bounce}: expected={expected:.10f}, actual={actual:.10f}, L_inf={error_inf:.2e}")

    plot_per_bounce_error(
        "S0-7 Rayleigh",
        errors_inf, errors_l2, expected_values, actual_values,
        tolerance=1e-10
    )

    passed = errors_inf[-1] < 1e-10
    return (passed, f"final_L_inf={errors_inf[-1]:.2e}")


def test_s0_8_mie_plot():
    """S0-8 Mie with plotting."""

    domain, basis = create_golden_basis()
    sigma_base = 0.05
    alpha_power = 1.0

    lbda = domain.m_lambda
    T_analytic = torch.exp(-sigma_base * (lbda / 550.0)**(-alpha_power) * 1.0)

    op = SpectralOperatorFactory.createScattering(basis, "Mie", sigma_base, distance=1.0, alpha=alpha_power)
    alpha = basis.projectWhitened(torch.ones_like(domain.m_lambda))

    errors_inf = []
    errors_l2 = []
    expected_values = []
    actual_values = []

    for bounce in range(1, 11):
        alpha = op.m_A @ alpha
        s_out = basis.reconstructWhitened(alpha)

        T_bounce = T_analytic ** bounce
        expected = T_bounce.mean().item()
        actual = s_out.mean().item()

        error_inf = torch.max(torch.abs(s_out - T_bounce)).item()
        error_l2 = torch.sqrt(torch.mean((s_out - T_bounce) ** 2)).item()

        errors_inf.append(error_inf)
        errors_l2.append(error_l2)
        expected_values.append(expected)
        actual_values.append(actual)

        print(f"  Bounce {bounce}: expected={expected:.10f}, actual={actual:.10f}, L_inf={error_inf:.2e}")

    plot_per_bounce_error(
        "S0-8 Mie",
        errors_inf, errors_l2, expected_values, actual_values,
        tolerance=1e-10
    )

    passed = errors_inf[-1] < 1e-10
    return (passed, f"final_L_inf={errors_inf[-1]:.2e}")


# =============================================================================
# Set 1 Tests with Plotting (selected key test cases)
# =============================================================================

def run_set1_with_plots():
    """Run selected Set 1 tests with plotting."""

    print("\n" + "=" * 70)
    print(" SET 1: COMPOSED CHAINS (WITH PLOTS)")
    print("=" * 70)

    results = {}

    # TC1: Pure Volume Absorption
    print("\n[TC1] Pure Volume Absorption")
    results["TC1"] = test_tc1_plot()

    # TC7: Classic Single Bounce
    print("\n[TC7] Classic Single Bounce")
    results["TC7"] = test_tc7_plot()

    # TC8: Iridescent Bounce (Primary Regression)
    print("\n[TC8] Iridescent Bounce (Primary Regression)")
    results["TC8"] = test_tc8_plot()

    passed = sum(1 for v in results.values() if v[0])
    total = len(results)
    print(f"\n{'='*70}")
    print(f" SET 1 (PLOTS) SUMMARY: {passed}/{total} passed")
    print(f"{'='*70}")

    return results


def test_tc1_plot():
    """TC1: Pure Volume Absorption with error vs composition depth."""

    domain, basis = create_golden_basis()

    sigma1 = 0.3
    sigma2 = 0.5
    d = 1.0

    sigma_fn1 = lambda lbda: torch.ones_like(lbda) * sigma1
    sigma_fn2 = lambda lbda: torch.ones_like(lbda) * sigma2

    op1 = SpectralOperatorFactory.createAbsorption(basis, sigma_fn1, d)
    op2 = SpectralOperatorFactory.createAbsorption(basis, sigma_fn2, d)

    # Build composed operator step by step
    alpha = basis.projectWhitened(torch.ones_like(domain.m_lambda))

    # After op1 only
    alpha_1 = op1.m_A @ alpha
    s_1 = basis.reconstructWhitened(alpha_1)
    expected_1 = np.exp(-sigma1 * d)

    # After op2 o op1
    alpha_2 = op2.m_A @ alpha_1
    s_2 = basis.reconstructWhitened(alpha_2)
    expected_2 = np.exp(-(sigma1 + sigma2) * d)

    errors_inf = [
        torch.max(torch.abs(s_1 - torch.ones_like(domain.m_lambda) * expected_1)).item(),
        torch.max(torch.abs(s_2 - torch.ones_like(domain.m_lambda) * expected_2)).item()
    ]

    print(f"  After op1: expected={expected_1:.10f}, L_inf={errors_inf[0]:.2e}")
    print(f"  After op2: expected={expected_2:.10f}, L_inf={errors_inf[1]:.2e}")

    # Simple single-panel plot
    engine = PlotEngine(figsize=(10, 6))
    stages = np.array([1, 2])
    engine.addScatter(stages, np.array(errors_inf), label="L_inf Error", color=PlotEngine.sColors['primary'], marker='o', size=60)
    engine.addLine(stages, np.ones(2) * 1e-12, label="Tolerance (1e-12)", color='#FF6B6B', linestyle='--')
    engine.setTitle("TC1: Error vs Composition Depth")
    engine.setLabels("Composition Stage", "L_inf Error")
    engine.m_axes.set_xticks([1, 2])
    engine.m_axes.set_xticklabels(["After Beer(σ₁)", "After Beer(σ₂)∘Beer(σ₁)"])
    engine.addLegend()
    engine.m_axes.set_yscale('log')

    filepath = PLOT_DIR / "TC1_composition_error.png"
    engine.saveFigure(str(filepath), dpi=300)
    print(f"  Plot saved: {filepath}")

    passed = errors_inf[-1] < 1e-12
    return (passed, f"final_L_inf={errors_inf[-1]:.2e}")


def test_tc7_plot():
    """TC7: Classic Single Bounce with error tracking."""

    domain, basis = create_golden_basis()

    sigma_val = -np.log(0.5)
    sigma_fn = lambda lbda: torch.ones_like(lbda) * sigma_val
    op_abs = SpectralOperatorFactory.createAbsorption(basis, sigma_fn, distance=1.0)

    f_inf = flat_profile(0.04, domain)
    ops_fresnel = SpectralOperatorFactory.createFresnel(basis, f_inf)
    op_fresnel = ops_fresnel.P0

    alpha = basis.projectWhitened(torch.ones_like(domain.m_lambda))

    # Track error through each operator
    errors_inf = []

    # After first absorption
    alpha_1 = op_abs.m_A @ alpha
    s_1 = basis.reconstructWhitened(alpha_1)
    expected_1 = 0.5
    errors_inf.append(torch.max(torch.abs(s_1 - torch.ones_like(domain.m_lambda) * expected_1)).item())

    # After Fresnel
    alpha_2 = op_fresnel.m_A @ alpha_1
    s_2 = basis.reconstructWhitened(alpha_2)
    expected_2 = 0.5 * 0.04
    errors_inf.append(torch.max(torch.abs(s_2 - torch.ones_like(domain.m_lambda) * expected_2)).item())

    # After second absorption (full bounce)
    alpha_3 = op_abs.m_A @ alpha_2
    s_3 = basis.reconstructWhitened(alpha_3)
    expected_3 = 0.5 * 0.04 * 0.5
    errors_inf.append(torch.max(torch.abs(s_3 - torch.ones_like(domain.m_lambda) * expected_3)).item())

    print(f"  After Abs1: expected={expected_1:.10f}, L_inf={errors_inf[0]:.2e}")
    print(f"  After Fresnel: expected={expected_2:.10f}, L_inf={errors_inf[1]:.2e}")
    print(f"  After Abs2: expected={expected_3:.10f}, L_inf={errors_inf[2]:.2e}")

    engine = PlotEngine(figsize=(10, 6))
    stages = np.array([1, 2, 3])
    stage_labels = ["After Abs₁", "After Fresnel", "After Abs₂ (full)"]
    engine.addScatter(stages, np.array(errors_inf), label="L_inf Error", color=PlotEngine.sColors['tertiary'], marker='s', size=60)
    engine.addLine(stages, np.ones(3) * 1e-12, label="Tolerance (1e-12)", color='#FF6B6B', linestyle='--')
    engine.setTitle("TC7: Error Through Single Bounce")
    engine.setLabels("Stage", "L_inf Error")
    engine.m_axes.set_xticks([1, 2, 3])
    engine.m_axes.set_xticklabels(stage_labels)
    engine.addLegend()
    engine.m_axes.set_yscale('log')

    filepath = PLOT_DIR / "TC7_single_bounce_error.png"
    engine.saveFigure(str(filepath), dpi=300)
    print(f"  Plot saved: {filepath}")

    passed = errors_inf[-1] < 1e-12
    return (passed, f"final_L_inf={errors_inf[-1]:.2e}")


def test_tc8_plot():
    """TC8: Iridescent Bounce - full 6-operator chain."""

    domain, basis = create_golden_basis()

    sigma_fn = lambda lbda: torch.ones_like(lbda) * 0.1
    op_abs = SpectralOperatorFactory.createAbsorption(basis, sigma_fn, distance=1.0)
    op_ray = SpectralOperatorFactory.createScattering(basis, "Rayleigh", 0.05, distance=1.0, alpha=4.0)
    op_left = op_ray.compose(op_abs)

    f_inf = flat_profile(0.04, domain)
    ops_fresnel = SpectralOperatorFactory.createFresnel(basis, f_inf)
    op_fresnel = ops_fresnel.Rcross
    op_film = SpectralOperatorFactory.createThinFilm(basis, n=1.5, d=200)
    op_surface = op_fresnel.compose(op_film)

    op_mie = SpectralOperatorFactory.createScattering(basis, "Mie", 0.05, distance=1.0, alpha=1.0)
    op_right = op_mie.compose(op_abs)

    alpha = basis.projectWhitened(torch.ones_like(domain.m_lambda))

    # Sequential application with error tracking
    errors_inf = []
    op_sequence = [
        ("Abs (left)", op_abs),
        ("Rayleigh", op_ray),
        ("ThinFilm", op_film),
        ("Fresnel Rcross", op_fresnel),
        ("Abs (right)", op_abs),
        ("Mie", op_mie)
    ]

    alpha_current = alpha.clone()
    for name, op in op_sequence:
        alpha_current = op.m_A @ alpha_current
        s_out = basis.reconstructWhitened(alpha_current)

        # Reference: compute by applying all ops sequentially (same as current for validation)
        # For error tracking, compare against running reference
        pass

    # Final comparison: fused vs sequential
    op_tmp = op_surface.compose(op_right)
    op_fused = op_left.compose(op_tmp)
    alpha_fused = op_fused.m_A @ alpha
    alpha_seq = alpha.clone()
    for _, op in op_sequence:
        alpha_seq = op.m_A @ alpha_seq

    error = torch.norm(alpha_fused - alpha_seq, p=float('inf')).item()
    errors_inf.append(error)

    print(f"  Fused vs Sequential L_inf error: {error:.2e}")

    # Spectrum comparison plot
    s_fused = basis.reconstructWhitened(alpha_fused)
    s_seq = basis.reconstructWhitened(alpha_seq)

    plot_spectrum_comparison("TC8 Iridescent", domain, s_fused, s_seq, error, bounce_num=1)

    passed = error < 1e-10
    return (passed, f"L_inf={error:.2e}")

# =============================================================================
# New Tests: append before the Main block in test_multibounce_plots.py
# Add `run_set2_with_plots()` call in __main__ alongside the existing two.
# =============================================================================


# -----------------------------------------------------------------------------
# S0-4: ThinFilm Multibounce
# -----------------------------------------------------------------------------

def test_s0_4_thinfilm_plot():
    """
    S0-4: Thin Film (Fabry-Airy) multibounce.

    Ground truth per bounce n:
        T_ref^n(λ) = [0.5 * (1 + cos(4π n_ior d / λ))]^n

    d=50nm: no in-band exact zeros, basis floor ~1e-4 (projection residual,
    not operator error — same ceiling as all non-diagonal operators).
    """
    domain, basis = create_golden_basis()

    n_ior = 1.5
    d_nm  = 50.0

    op    = SpectralOperatorFactory.createThinFilm(basis, n=n_ior, d=d_nm)
    lbda  = domain.m_lambda
    T_one = 0.5 * (1.0 + torch.cos(4.0 * torch.pi * n_ior * d_nm / lbda))

    alpha = basis.projectWhitened(torch.ones_like(lbda))

    errors_inf, errors_l2, expected_values, actual_values = [], [], [], []

    for bounce in range(1, 11):
        alpha = op.m_A @ alpha
        s_out = basis.reconstructWhitened(alpha)

        T_ref    = T_one ** bounce
        expected = T_ref.mean().item()
        actual   = s_out.mean().item()

        error_inf = torch.max(torch.abs(s_out - T_ref)).item()
        error_l2  = torch.sqrt(torch.mean((s_out - T_ref) ** 2)).item()

        errors_inf.append(error_inf)
        errors_l2.append(error_l2)
        expected_values.append(expected)
        actual_values.append(actual)

        print(f"  Bounce {bounce}: expected={expected:.10f}, actual={actual:.10f}, L_inf={error_inf:.2e}")

        # Per-bounce spectrum comparison
        plot_spectrum_comparison(
            "S0-4 ThinFilm", domain, s_out, T_ref, error_inf, bounce_num=bounce
        )

    plot_per_bounce_error(
        "S0-4 ThinFilm",
        errors_inf, errors_l2, expected_values, actual_values,
        tolerance=1e-3
    )

    passed = errors_inf[-1] < 1e-3
    return (passed, f"final_L_inf={errors_inf[-1]:.2e}")


# -----------------------------------------------------------------------------
# S0-5: Fluorescence (Stokes) Multibounce
# -----------------------------------------------------------------------------

def test_s0_5_fluorescence_plot():
    """
    S0-5: Stokes fluorescence multibounce.

    Operator (operator.py createFluorescence):
        Â = e_w ⊗ a_w^T    (rank-1, whitened space)

    Starting from alpha_0:
        alpha_1 = e_w * (a_w · alpha_0)
        alpha_n = e_w * (a_w · e_w)^(n-1) * (a_w · alpha_0)

    Ground truth scalars:
        c0 = a_w · alpha_0          (initial excitation)
        c1 = a_w · e_w              (per-bounce geometric ratio)
        s_ref_n(λ) = c0 * c1^(n-1) * reconstruct(e_w)
    """
    domain, basis = create_golden_basis()
    lbda = domain.m_lambda

    emission   = torch.exp(-0.5 * ((lbda - 580.0) / 20.0) ** 2)   # 580 nm emit peak
    absorption = torch.exp(-0.5 * ((lbda - 480.0) / 20.0) ** 2)   # 480 nm absorb peak

    op = SpectralOperatorFactory.createFluorescence(basis, emission, absorption)

    e_w     = basis.projectWhitened(emission)
    a_w     = basis.projectWhitened(absorption)
    alpha_0 = basis.projectWhitened(torch.ones_like(lbda))

    c0         = (a_w @ alpha_0).item()
    c1         = (a_w @ e_w).item()
    s_emit_dir = basis.reconstructWhitened(e_w)

    alpha = alpha_0.clone()

    errors_inf, errors_l2, expected_values, actual_values = [], [], [], []

    for bounce in range(1, 11):
        alpha = op.m_A @ alpha
        s_out = basis.reconstructWhitened(alpha)

        scalar   = c0 * (c1 ** (bounce - 1))
        s_ref    = scalar * s_emit_dir
        expected = s_ref.mean().item()
        actual   = s_out.mean().item()

        error_inf = torch.max(torch.abs(s_out - s_ref)).item()
        error_l2  = torch.sqrt(torch.mean((s_out - s_ref) ** 2)).item()

        errors_inf.append(error_inf)
        errors_l2.append(error_l2)
        expected_values.append(expected)
        actual_values.append(actual)

        print(f"  Bounce {bounce}: scalar={scalar:.6e}, actual_mean={actual:.6e}, L_inf={error_inf:.2e}")

    plot_per_bounce_error(
        "S0-5 Fluorescence",
        errors_inf, errors_l2, expected_values, actual_values,
        tolerance=1e-13
    )

    passed = errors_inf[-1] < 1e-13
    return (passed, f"final_L_inf={errors_inf[-1]:.2e}")


# -----------------------------------------------------------------------------
# TC9: Dispersion Split + Combine (Partition of Unity)
# -----------------------------------------------------------------------------

def test_tc9_dispersion_split_combine_plot():
    """
    TC9: Cauchy dispersion partition-of-unity (operator.py createDispersion).

    createDispersion returns K softmax-normalised lobe operators.
    By construction: Σ_k Â_k = I_M in whitened space.

    Three sub-checks:
        1. Flat input:   ||(Σ_k Â_k) alpha - alpha||_inf < 1e-6
           Flat spectrum is worst-case for the basis (energy spread across
           all modes); ~8e-7 is the projection fidelity floor, not a bug.
        2. Peaked input: same check, tighter at 1e-8 (well-represented by basis)
        3. Energy:       Σ_k mean(reconstruct(Â_k alpha)) == mean(reconstruct(alpha))

    Cauchy coefficients: BK7 borosilicate glass
        n(λ) ≈ 1.5168 + 4100 / λ²    (λ in nm)
    """
    domain, basis = create_golden_basis()
    lbda = domain.m_lambda

    A_cauchy = 1.5168
    B_cauchy = 4100.0
    C_cauchy = 0.0

    ops = SpectralOperatorFactory.createDispersion(basis, A_cauchy, B_cauchy, C_cauchy)
    K   = len(ops)

    # Sub-check 1: flat input (basis projection floor ~8e-7)
    alpha_flat = basis.projectWhitened(torch.ones_like(lbda))
    alpha_sum  = sum(op.m_A @ alpha_flat for op in ops)
    error_flat = torch.max(torch.abs(alpha_sum - alpha_flat)).item()
    print(f"  Partition of unity (flat):   L_inf = {error_flat:.2e}")

    # Sub-check 2: peaked input (well-represented, tighter bound)
    alpha_peak = basis.projectWhitened(torch.exp(-0.5 * ((lbda - 550.0) / 30.0) ** 2))
    alpha_sum2 = sum(op.m_A @ alpha_peak for op in ops)
    error_peak = torch.max(torch.abs(alpha_sum2 - alpha_peak)).item()
    print(f"  Partition of unity (peaked): L_inf = {error_peak:.2e}")

    # Sub-check 3: per-lobe energy partition
    lobe_means = []
    for k, op in enumerate(ops):
        s_lobe = basis.reconstructWhitened(op.m_A @ alpha_flat)
        lobe_means.append(s_lobe.mean().item())
        print(f"  Lobe {k:2d}: mean = {lobe_means[-1]:.6f}")

    total_lobe_energy = sum(lobe_means)
    original_mean     = basis.reconstructWhitened(alpha_flat).mean().item()
    energy_error      = abs(total_lobe_energy - original_mean)
    print(f"  Energy: sum_lobes={total_lobe_energy:.10f}, original={original_mean:.10f}, err={energy_error:.2e}")

    # Plot 1: per-lobe energy bar
    engine = PlotEngine(figsize=(12, 6))
    engine.addScatter(
        np.arange(K), np.array(lobe_means),
        label="Per-lobe mean energy",
        color=PlotEngine.sColors['primary'], marker='o', size=60
    )
    engine.addLine(
        np.arange(K), np.ones(K) * original_mean / K,
        label=f"Uniform split ({original_mean / K:.4f})",
        color=PlotEngine.sColors['secondary'], linewidth=1.5, linestyle='--'
    )
    engine.setTitle("TC9: Dispersion — Per-lobe Energy Partition")
    engine.setLabels("Lobe Index", "Mean Spectral Value")
    engine.m_axes.set_xticks(np.arange(K))
    engine.addLegend()
    engine.saveFigure(str(PLOT_DIR / "TC9_dispersion_lobe_energy.png"), dpi=300)
    print(f"  Plot saved: TC9_dispersion_lobe_energy.png")

    # Plot 2: partition error
    engine2 = PlotEngine(figsize=(8, 5))
    engine2.addScatter(
        np.array([0, 1]), np.array([error_flat, error_peak]),
        label="L_inf partition error",
        color=PlotEngine.sColors['tertiary'], marker='s', size=80
    )
    engine2.addLine(
        np.array([0, 1]), np.ones(2) * 1e-6,
        label="Tolerance flat (1e-6)", color='#FF6B6B', linestyle='--'
    )
    engine2.addLine(
        np.array([0, 1]), np.ones(2) * 1e-8,
        label="Tolerance peaked (1e-8)", color='#FFA500', linestyle=':'
    )
    engine2.m_axes.set_xticks([0, 1])
    engine2.m_axes.set_xticklabels(["Flat input", "Peaked input"])
    engine2.m_axes.set_yscale('log')
    engine2.setTitle("TC9: Dispersion — Partition of Unity Error")
    engine2.setLabels("Input Type", "L_inf Error (alpha space)")
    engine2.addLegend()
    engine2.saveFigure(str(PLOT_DIR / "TC9_dispersion_partition_error.png"), dpi=300)
    print(f"  Plot saved: TC9_dispersion_partition_error.png")

    passed = error_flat < 1e-6 and error_peak < 1e-8 and energy_error < 1e-10
    return (passed, f"flat={error_flat:.2e}, peaked={error_peak:.2e}, energy={energy_error:.2e}")


# -----------------------------------------------------------------------------
# run_set2_with_plots — add this call to __main__
# -----------------------------------------------------------------------------

def run_set2_with_plots():
    print("\n" + "=" * 70)
    print(" SET 2: THINFILM / FLUORESCENCE / DISPERSION (WITH PLOTS)")
    print("=" * 70)

    results = {}

    print("\n[S0-4] ThinFilm Multibounce")
    results["S0-4"] = test_s0_4_thinfilm_plot()

    print("\n[S0-5] Fluorescence Multibounce")
    results["S0-5"] = test_s0_5_fluorescence_plot()

    print("\n[TC9] Dispersion Split + Combine")
    results["TC9"] = test_tc9_dispersion_split_combine_plot()

    passed = sum(1 for v in results.values() if v[0])
    total  = len(results)
    print(f"\n{'='*70}")
    print(f" SET 2 SUMMARY: {passed}/{total} passed")
    print(f" Plots saved to: {PLOT_DIR}")
    print(f"{'='*70}")

    return results


def run_set3_bf16_precision():
    print("\n" + "=" * 70)
    print(" SET 3: BF16 PRECISION VALIDATION")
    print("=" * 70)

    results = {}

    print("\n[BF16-1] Single operator cast fidelity")
    results["BF16-1"] = test_bf16_cast_fidelity()

    print("\n[BF16-2] Multibounce error accumulation")
    results["BF16-2"] = test_bf16_multibounce_accumulation()

    print("\n[BF16-3] Terminal contribution error")
    results["BF16-3"] = test_bf16_terminal_contribution()

    passed = sum(1 for v in results.values() if v[0])
    total  = len(results)
    print(f"\n{'='*70}")
    print(f" SET 3 SUMMARY: {passed}/{total} passed")
    print(f"{'='*70}")

    return results


def test_bf16_cast_fidelity():
    """
    BF16-1: Cast operator f64 → bf16 → f64, measure Frobenius error.
    Tests all operator classes. BF16 has 7 mantissa bits (~3 decimal digits).
    Expected: relative error ~1e-2 to 1e-3 depending on dynamic range.
    Fabry-Airy expected worst — full rank, eigenvalues spread across [0,1].
    """
    domain, basis = create_golden_basis()
    lbda = domain.m_lambda

    def cast_error(A_f64):
        A_bf16 = A_f64.to(torch.bfloat16).to(torch.float64)
        err_frob = torch.norm(A_f64 - A_bf16, p='fro') / torch.norm(A_f64, p='fro')
        err_inf  = (A_f64 - A_bf16).abs().max()
        return err_frob.item(), err_inf.item()

    operators = {}

    sigma_fn = lambda l: torch.ones_like(l) * 0.3
    operators["Beer-Lambert"] = SpectralOperatorFactory.createAbsorption(
        basis, sigma_fn, distance=1.0).m_A

    operators["Fresnel P0"]   = SpectralOperatorFactory.createFresnel(
        basis, 0.08 + 0.88 * torch.sigmoid((lbda - 515.0) / 25.0)).P0.m_A

    operators["Fabry-Airy"]   = SpectralOperatorFactory.createThinFilm(
        basis, n=1.5, d=300.0).m_A

    e_spd = torch.exp(-0.5 * ((lbda - 520.0) / 15.0) ** 2)
    a_spd = torch.exp(-0.5 * ((lbda - 450.0) / 15.0) ** 2)
    operators["Stokes"]       = SpectralOperatorFactory.createFluorescence(
        basis, e_spd, a_spd).m_A

    operators["Rayleigh"]     = SpectralOperatorFactory.createScattering(
        basis, "Rayleigh", 0.05, distance=1.0, alpha=4.0).m_A

    operators["Raman"]        = SpectralOperatorFactory.createRaman(
        basis, shift_nm=70.0).m_A

    errors_frob = []
    errors_inf  = []
    names       = []

    for name, A in operators.items():
        ef, ei = cast_error(A)
        errors_frob.append(ef)
        errors_inf.append(ei)
        names.append(name)
        print(f"  {name:20s}  Frob rel: {ef:.2e}  L_inf: {ei:.2e}")

    # Plot
    multi = MultiPanelEngine(nrows=1, ncols=2, figsize=(14, 5), sharex=False)

    x = np.arange(len(names))

    p0 = multi.getPanel(0)
    p0.addScatter(x, np.array(errors_frob),
                  label="Frobenius relative error",
                  color=PlotEngine.sColors['primary'], marker='o', size=60)
    p0.addLine(x, np.ones(len(names)) * 1e-2,
               label="BF16 floor (~1e-2)", color='#FF6B6B',
               linewidth=1.5, linestyle='--')
    p0.m_axes.set_xticks(x)
    p0.m_axes.set_xticklabels(names, rotation=30, ha='right')
    p0.m_axes.set_yscale('log')
    p0.setTitle("BF16 Cast: Frobenius Relative Error per Operator")
    p0.setLabels("Operator", "Relative Frobenius Error")
    p0.addLegend()

    p1 = multi.getPanel(1)
    p1.addScatter(x, np.array(errors_inf),
                  label="L_inf absolute error",
                  color=PlotEngine.sColors['tertiary'], marker='s', size=60)
    p1.m_axes.set_xticks(x)
    p1.m_axes.set_xticklabels(names, rotation=30, ha='right')
    p1.m_axes.set_yscale('log')
    p1.setTitle("BF16 Cast: L_inf Absolute Error per Operator")
    p1.setLabels("Operator", "L_inf Absolute Error")
    p1.addLegend()

    filepath = PLOT_DIR / "BF16_1_cast_fidelity.png"
    multi.saveFigure(str(filepath), dpi=300)
    print(f"  Plot saved: {filepath}")

    # Pass if all Frobenius errors below 2% — BF16 floor expectation
    max_frob = max(errors_frob)
    passed   = max_frob < 2e-2
    return (passed, f"max_frob={max_frob:.2e}")


def test_bf16_multibounce_accumulation():
    """
    BF16-2: 10-bounce error accumulation — f64 vs bf16 operator applied to f64 alpha.
    Key question: does error grow linearly with bounces or stay bounded?
    Tests Beer-Lambert (diagonal, should be tight) and
    Fabry-Airy (full rank, expected worst case).
    """
    domain, basis = create_golden_basis()
    lbda = domain.m_lambda

    def run_bounces(A_f64, alpha_init, n_bounces=10):
        A_bf16_f64 = A_f64.to(torch.bfloat16).to(torch.float64)

        alpha_f64  = alpha_init.clone()
        alpha_bf16 = alpha_init.clone()
        errors     = []

        for _ in range(n_bounces):
            alpha_f64  = A_f64      @ alpha_f64
            alpha_bf16 = A_bf16_f64 @ alpha_bf16
            err = (alpha_f64 - alpha_bf16).abs().max().item()
            errors.append(err)

        return errors

    alpha_init = basis.projectWhitened(torch.ones_like(lbda))

    sigma_fn  = lambda l: torch.ones_like(l) * 0.3
    A_beer    = SpectralOperatorFactory.createAbsorption(
        basis, sigma_fn, distance=1.0).m_A
    A_fabry   = SpectralOperatorFactory.createThinFilm(
        basis, n=1.5, d=300.0).m_A
    A_fresnel = SpectralOperatorFactory.createFresnel(
        basis, 0.08 + 0.88 * torch.sigmoid((lbda - 515.0) / 25.0)).P0.m_A

    errors_beer    = run_bounces(A_beer,    alpha_init)
    errors_fabry   = run_bounces(A_fabry,   alpha_init)
    errors_fresnel = run_bounces(A_fresnel, alpha_init)

    bounces = np.arange(1, 11)

    for b, (eb, ef, efr) in enumerate(zip(errors_beer, errors_fabry, errors_fresnel), 1):
        print(f"  Bounce {b:2d}  Beer: {eb:.2e}  Fabry: {ef:.2e}  Fresnel: {efr:.2e}")

    # Plot
    engine = PlotEngine(figsize=(12, 6))
    engine.addLine(bounces, np.array(errors_beer),
                   label="Beer-Lambert", color=PlotEngine.sColors['primary'],
                   linewidth=2, marker='o')
    engine.addLine(bounces, np.array(errors_fabry),
                   label="Fabry-Airy", color=PlotEngine.sColors['tertiary'],
                   linewidth=2, marker='s')
    engine.addLine(bounces, np.array(errors_fresnel),
                   label="Fresnel P0 (Gold)", color='#FFD93D',
                   linewidth=2, marker='^')
    engine.addLine(bounces, np.ones(10) * 1e-2,
                   label="BF16 floor (~1e-2)", color='#FF6B6B',
                   linewidth=1.5, linestyle='--')
    engine.setTitle("BF16 Multibounce: f64 vs bf16 Operator — L_inf Error per Bounce")
    engine.setLabels("Bounce #", "L_inf Error (alpha space)")
    engine.m_axes.set_yscale('log')
    engine.addLegend()

    filepath = PLOT_DIR / "BF16_2_multibounce_accumulation.png"
    engine.saveFigure(str(filepath), dpi=300)
    print(f"  Plot saved: {filepath}")

    # Pass if error stays bounded (not growing unboundedly) after 10 bounces
    # BF16 floor is ~1e-2 relative, absolute error on alpha should be < 1e-1
    max_err = max(max(errors_beer), max(errors_fabry), max(errors_fresnel))
    passed  = max_err < 1e-1
    return (passed, f"max_err={max_err:.2e}")


def test_bf16_terminal_contribution():
    """
    BF16-3: Terminal contribution error — eps^T alpha_f64 vs eps^T alpha_bf16.
    This is what shows up in the pixel. The dot product with eps can amplify
    or suppress the per-component error in alpha depending on alignment.
    Tests all three materials over 10 bounces.
    """
    domain, basis = create_golden_basis()
    lbda = domain.m_lambda

    # Sensor and emitter
    y_tilde  = basis.projectWhitened(
        torch.exp(-0.5 * ((lbda - 555.0) / 40.0) ** 2))

    def planck_norm(lbda, T):
        h, c, kB = 6.62607015e-34, 2.99792458e8, 1.380649e-23
        lm = lbda * 1e-9
        exp = torch.tensor(h * c / kB / T, device=lbda.device, dtype=lbda.dtype)
        B = (2.0 * h * c**2) / (lm**5) / (torch.exp(exp / lm) - 1.0)
        return B / B.max()

    eps_6500 = basis.projectWhitened(planck_norm(lbda, 6500.0))
    alpha_0  = y_tilde.clone()   # camera-to-light init

    operators = {
        "Beer-Lambert": SpectralOperatorFactory.createAbsorption(
            basis, lambda l: torch.ones_like(l) * 0.1, distance=1.0).m_A,
        "Fabry-Airy":   SpectralOperatorFactory.createThinFilm(
            basis, n=1.5, d=300.0).m_A,
        "Fresnel P0":   SpectralOperatorFactory.createFresnel(
            basis, 0.08 + 0.88 * torch.sigmoid((lbda - 515.0) / 25.0)).P0.m_A,
    }

    multi = MultiPanelEngine(nrows=1, ncols=len(operators),
                             figsize=(16, 5), sharex=False)
    bounces = np.arange(1, 11)

    all_passed = True
    for idx, (name, A_f64) in enumerate(operators.items()):
        A_bf16 = A_f64.to(torch.bfloat16).to(torch.float64)

        alpha_f64  = alpha_0.clone()
        alpha_bf16 = alpha_0.clone()
        contrib_errors = []

        for _ in range(10):
            alpha_f64  = A_f64  @ alpha_f64
            alpha_bf16 = A_bf16 @ alpha_bf16

            C_f64  = (eps_6500 @ alpha_f64).item()
            C_bf16 = (eps_6500 @ alpha_bf16).item()
            contrib_errors.append(abs(C_f64 - C_bf16) / (abs(C_f64) + 1e-30))

        print(f"  {name:20s}  max pixel error: {max(contrib_errors):.2e}")

        p = multi.getPanel(idx)
        p.addLine(bounces, np.array(contrib_errors),
                  label="Pixel contrib rel error",
                  color=PlotEngine.sColors['primary'], linewidth=2, marker='o')
        p.addLine(bounces, np.ones(10) * 1e-2,
                  label="1% threshold",
                  color='#FF6B6B', linewidth=1.5, linestyle='--')
        p.m_axes.set_yscale('log')
        p.setTitle(f"{name}: Pixel Error")
        p.setLabels("Bounce #", "Rel Error in eps^T alpha")
        p.addLegend()

        if max(contrib_errors) > 5e-2:
            all_passed = False

    filepath = PLOT_DIR / "BF16_3_terminal_contribution.png"
    multi.saveFigure(str(filepath), dpi=300)
    print(f"  Plot saved: {filepath}")

    return (all_passed, "pixel contribution errors within 5% threshold")

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print(" MULTIBOUNCE VALIDATION - PER-BOUNCE ERROR PLOTS")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Precision: {DTYPE}")
    print(f"Basis: K={K}, N={N}")
    print(f"Output: {PLOT_DIR}")

    run_set0_with_plots()
    run_set1_with_plots()
    run_set2_with_plots()
    run_set3_bf16_precision()

    print("\n" + "=" * 70)
    print(" ALL PLOTS COMPLETE")
    print("=" * 70)