# Phase 2 — Basis Stress Testing & Golden Configuration Selection

## Overview

Phase 2 determines the **golden configuration** — the optimal basis parameters for production BsSPT use.
It stress-tests every candidate config from Phase 1 (stable + rescued) across two independent axes:

- **Phase 2A** — Spectral Reconstruction Strength (how faithfully the basis represents real spectra)
- **Phase 2B** — Operator Leakage (how well each operator's energy is captured in the basis subspace)

Together they form the full Phase 2 evaluation. Pareto frontier analysis runs across all columns from both.

---

## Inputs

Phase 1 filtering produces **6 parquets** (3 margins × 2 classification types):

```
research/stability/results/
    stability_margin_0_stable.parquet
    stability_margin_0_rescued.parquet
    stability_margin_10_stable.parquet
    stability_margin_10_rescued.parquet
    stability_margin_20_stable.parquet
    stability_margin_20_rescued.parquet
```

All Phase 2 sweeps run independently over each of these 6 files. No concatenation.

---

## Phase 2A — Spectral Reconstruction Strength

### Goal
For each config, project 600 hard spectra through the basis and measure how accurately they reconstruct.
Deltas are a separate sub-phase with additional metrics.

### Output: 18 Parquets
Each of the 6 input files produces 3 output parquets (one per spectrum phase):

```
research/phase2a/results/
    phase2a_margin_0_stable_A.parquet
    phase2a_margin_0_stable_B.parquet
    phase2a_margin_0_stable_C.parquet
    phase2a_margin_0_rescued_A.parquet
    ...
    phase2a_margin_20_rescued_C.parquet
```

### Spectrum Suite (600 total)

#### Phase A — Normal Spectra (~200)
| Category | Count | Description |
|---|---|---|
| Blackbodies | 18 | T = 1000–25000K, log-spaced |
| Standard Illuminants | 14 | D50, D55, D65, D75, A, F2, F7, F11, F12, LED-B1, LED-B3, LED-B5, LED-RGB1, LED-V1 |
| Macbeth ColorChecker | 24 | All 24 patches (analytically modeled) |
| Narrow Gaussians | 60 | 12 centers × 5 sigmas (10, 15, 20, 30, 50 nm) |
| Multi-peak | 50 | 30 dual-peak + 20 triple-peak combinations |
| LED / Phosphor SPDs | 30 | Phosphor-converted, RGB, warm/cool white variants |
| **Total** | **196** | |

#### Phase B — Delta Spectra (~200)
Near-delta Gaussians — the hardest single-feature test for any basis.

| Parameter | Values |
|---|---|
| Centers | 40 wavelengths, evenly spaced 385–825 nm |
| Widths (σ) | 0.5, 1.0, 2.0, 5.0, 10.0 nm |
| **Total** | **200 spectra** |

Delta spectra get **additional metrics** beyond the base set (see Metrics section).

#### Phase C — Misc Hard Cases (~200)
| Category | Count | Description |
|---|---|---|
| Linear ramps | 10 | Up, down, partial-range variants |
| Step functions | 15 | Thresholds swept across visible range |
| Notch filters | 15 | Narrow absorption bands at varying positions |
| Broad Gaussians | 20 | σ = 60–150 nm |
| Laser lines | 20 | σ = 0.5 nm at 20 specific wavelengths |
| Sine-modulated | 15 | Flat + sinusoidal spectral ripple |
| Exponential | 10 | Decay and rise across domain |
| Near-flat | 5 | Constant with small perturbations |
| Random smooth | 30 | Seeded sum-of-Gaussians, reproducible |
| Hyperspectral-like | 30 | Complex multi-feature synthetic spectra |
| **Total** | **170** | Padded to 200 with additional random smooth |

### Metrics

#### Base Metrics (Phase A, B, C)
| Column | Definition |
|---|---|
| `l2` | `sqrt(Σ w·(f̂ - f)²)` weighted L2 error |
| `nrmse` | `l2 / sqrt(Σ w·f²)` normalized |
| `maxError` | `max |f̂(λ) - f(λ)|` |
| `xyzDelta` | `‖XYZ(f̂) - XYZ(f)‖₂` under D65 |
| `perceptualDeltaE` | Full CIEDE2000 ΔE₀₀ under D65 |

#### Delta Extras (Phase B only)
| Column | Definition |
|---|---|
| `energyRetention` | `Σ w·f̂² / Σ w·f²` |
| `amplitudeAccuracy` | `max(f̂) / max(f)` |
| `peakShiftNm` | `λ(argmax f̂) - λ(argmax f)` in nm |
| `sideLobeEnergy` | Fraction of f̂ energy outside center ± 3σ |

### Storage
- `configId` = row index in source Phase 1 parquet (int32)
- `spectrumId` = spectrum index 0–199 within phase (int16)
- All metric columns: float32
- Compression: **parquet zstd**
- Estimated compressed size: **~41 GB total** across 18 files

### Implementation

#### Files
```
research/phase2a/
    __init__.py
    cmf.py           — CIE 1931 2° CMF, XYZ, Lab, CIEDE2000 (vectorized PyTorch)
    spectrum_gen.py  — Generates all 600 spectra on domain grid
    schema.py        — Column definitions for A/B/C
    metrics.py       — GPU-vectorized metric computation over [S, L] tensors
    sweep.py         — Core sweep: one input parquet → 3 output parquets
    master.py        — Orchestrates all 6 input parquets
```

#### GPU Strategy
- Domain + all 600 spectra `[600, L]` loaded onto GPU once at startup
- CMF matrix `[3, L]` on GPU
- Per config:
  - Build basis on GPU (float64)
  - **One batched triangular solve** for all 600 spectra: `L⁻¹ (B·w·F^T)` → `[M, 600]`
  - **One batched matmul** to reconstruct: `α_w^T · B_wht` → `[600, L]`
  - Vectorized metrics over full `[600, L]` error tensor
  - Copy only scalar metrics tensor to CPU
- Checkpoint every 50K configs, write 3 parquets simultaneously

---

## Phase 2B — Operator Leakage

### Goal
For each config, measure how much of each spectral operator's energy is captured by the basis subspace.
Three complementary leakage definitions applied to 9 operator types.

### Output: 6 Parquets
One per Phase 1 input file, matching the Phase 2A structure:

```
research/phase2b/results/
    phase2b_margin_0_stable.parquet
    phase2b_margin_0_rescued.parquet
    ...
    phase2b_margin_20_rescued.parquet
```

Each row = one config, wide format with all leakage columns.

### Leakage Definitions

For each operator, canonical hard parameter instances are swept. Metrics are aggregated as **mean** and **worst-case** across instances.

#### Setup 1 — Spectral Reconstruction Error
```
leakage₁ = mean over instances of:
    ‖T·f - reconstruct(A·project(f))‖² / ‖T·f‖²
    averaged over a small set of test spectra
```

#### Setup 2 — SVD Subspace Energy Capture
```
leakage₂ = 1 - ‖M_raw‖²_F / ‖T_full_weighted‖²_F

where M_raw_ij = ∫ T(λ) bᵢ(λ) bⱼ(λ) dλ  (Galerkin matrix)
      T_full_weighted = diag(T(λ)) · diag(√w)  (weighted spectral operator)
```

#### Setup 3 — Frobenius Norm Residual
```
leakage₃ = ‖T_exact - T_basis‖_F / ‖T_exact‖_F

where T_basis is the operator reconstructed from basis coefficients
```

### Operators & Canonical Hard Instances

#### 1. Absorption (Beer-Lambert)
`T(λ) = exp(-σ_a(λ) · d)`

| Instance | σ_a profile | Distance |
|---|---|---|
| 1 | Narrow Gaussian peak at 450nm, σ=5nm | 1.0 |
| 2 | Narrow Gaussian peak at 650nm, σ=5nm | 1.0 |
| 3 | Broad band 500–600nm | 2.0 |
| 4 | Double narrow peaks 430+620nm | 1.5 |
| 5 | Steep edge at 500nm | 3.0 |
| 6 | Near-flat weak absorption | 5.0 |

#### 2. Fresnel (Schlick assembled)
`Â_F(θ) = (1-t)·P0 + t·I_M`, `t = (1-cosθ)⁵`

| Instance | F_inf profile | cos_θ values |
|---|---|---|
| 1 | Flat F_inf = 0.04 (glass) | 0.0, 0.3, 0.6, 0.9, 1.0 |
| 2 | Spectrally structured F_inf (metal-like) | 0.0, 0.3, 0.6, 0.9, 1.0 |
| 3 | High reflectance F_inf = 0.9 | 0.0, 0.5, 1.0 |

15 assembled operator instances total.

#### 3. Thin Film (Fabry-Airy)
`T(λ) = 0.5 · (1 + cos(4πnd/λ))`

| Instance | n | d (nm) | Oscillation rate |
|---|---|---|---|
| 1 | 1.5 | 100 | Slow |
| 2 | 1.5 | 300 | Medium |
| 3 | 1.5 | 800 | Fast |
| 4 | 2.5 | 200 | Medium-fast |
| 5 | 2.5 | 600 | Very fast |
| 6 | 4.0 | 150 | Fast |
| 7 | 4.0 | 500 | Extreme |
| 8 | 1.5 | 1500 | Max stress |

#### 4. Fluorescence (Stokes rank-1 kernel)
`Â = e_wht · a_wht^T`

| Instance | Excitation a(λ) | Emission e(λ) | Stokes shift |
|---|---|---|---|
| 1 | Narrow peak 380nm | Narrow peak 450nm | 70nm |
| 2 | Narrow peak 450nm | Broad 530nm | 80nm |
| 3 | Narrow peak 500nm | Narrow peak 580nm | 80nm |
| 4 | Broad 400–450nm | Narrow peak 620nm | large |
| 5 | Narrow 350nm edge | Narrow peak 430nm | 80nm |
| 6 | Narrow peak 600nm | Broad 680nm | 80nm |

#### 5. Dispersion (Cauchy partition)
`n(λ) = A + B/λ² + C/λ⁴`, K Gaussian lobe windows, softmax normalized

Special metric: `dispersion_partition_residual = ‖ΣÂk - I_M‖_F / √M`

| Instance | A | B | C | Character |
|---|---|---|---|---|
| 1 | 1.5 | 0.01 | 0.0 | Weak dispersion |
| 2 | 1.7 | 0.05 | 0.001 | Medium |
| 3 | 2.0 | 0.10 | 0.005 | Strong |
| 4 | 2.5 | 0.20 | 0.010 | Extreme |

#### 6. Rayleigh Scattering
`T(λ) = exp(-σ_s · (λ/550)^{-4} · d)`

| Instance | σ_s_base | d |
|---|---|---|
| 1 | 0.01 | 10 |
| 2 | 0.05 | 20 |
| 3 | 0.10 | 50 |
| 4 | 0.20 | 100 |

#### 7. Mie Scattering
`T(λ) = exp(-σ_s · (λ/550)^{-α} · d)`

| Instance | σ_s_base | d | α |
|---|---|---|---|
| 1 | 0.05 | 10 | 0.5 (near flat) |
| 2 | 0.05 | 10 | 2.0 (moderate) |
| 3 | 0.10 | 30 | 0.5 |
| 4 | 0.10 | 30 | 2.0 |

#### 8. Emission
`Â = 0`, `b̂ = projectWhitened(e)`

Leakage measured on b vector: `‖e - reconstruct(b̂)‖² / ‖e‖²`

| Instance | e(λ) |
|---|---|
| 1 | Blackbody 2800K (warm) |
| 2 | Blackbody 6500K (daylight) |
| 3 | Narrow laser line 532nm |
| 4 | F11 fluorescent lamp (narrow bands) |
| 5 | LED warm white (blue peak + phosphor) |

#### 9. Localization (Splitting Kernel)
`T(λ) = Gaussian(λ - λ_q, σ)`, optionally normalized

| Instance | λ_q (nm) | σ (nm) | Normalized |
|---|---|---|---|
| 1 | 380 (edge) | 5 | No |
| 2 | 450 | 5 | No |
| 3 | 550 | 5 | No |
| 4 | 700 | 5 | No |
| 5 | 830 (edge) | 5 | No |
| 6 | 550 | 2 (very narrow) | Yes |

### Output Schema

Per-config columns (54 leakage + 1 partition residual):

```
configId

# For each operator op in {absorption, fresnel, thinfilm, fluorescence,
#   dispersion, rayleigh, mie, emission, localization}:
{op}_recon_mean       # Setup 1 averaged across instances
{op}_recon_worst      # Setup 1 worst-case instance
{op}_svd_mean         # Setup 2 averaged
{op}_svd_worst        # Setup 2 worst-case
{op}_frob_mean        # Setup 3 averaged
{op}_frob_worst       # Setup 3 worst-case

# Dispersion only:
dispersion_partition_residual
```

Total: 9 × 6 + 1 = **55 columns** per row.

### Implementation

#### Files
```
research/phase2b/
    __init__.py
    operator_params.py   — Canonical instance definitions for all 9 operators
    leakage.py           — 3 leakage computation functions (GPU-vectorized)
    schema.py            — Column definitions
    sweep.py             — Core sweep: one input parquet → 1 output parquet
    master.py            — Orchestrates all 6 input parquets
```

#### GPU Strategy
- Per config: build basis on GPU once
- For each operator: instantiate all instances in batch → compute M_raw batch → compute all 3 leakage setups
- All matrix ops are M×M (M ≤ 144) → very fast, operator-side is much cheaper than Phase 2A
- No per-spectrum loops — leakage is purely basis×operator computation

---

## Combined Phase 2 Output Summary

| Phase | Files | Rows (10M configs) | Compressed size |
|---|---|---|---|
| 2A | 18 parquets | ~6B | ~41 GB |
| 2B | 6 parquets | ~10M | ~2 GB |
| **Total** | **24 parquets** | | **~43 GB** |

---

## Phase 2 → Pareto Frontier

After both phases complete, Pareto analysis runs across:

- `K` (minimize)
- `raw_condition` (minimize, from Phase 1)
- `raw_entropy` (maximize, from Phase 1)
- `rescued` flag (prefer 0)
- Phase 2A: `nrmse` mean/worst per spectrum category (minimize)
- Phase 2A: `perceptualDeltaE` mean/worst (minimize)
- Phase 2A Phase B: `energyRetention` (maximize), `sideLobeEnergy` (minimize)
- Phase 2B: per-operator leakage mean/worst for all 9 operators × 3 setups (minimize)
- Phase 2B: `dispersion_partition_residual` (minimize)

The golden configuration is the Pareto-dominant point across all axes, broken by preference for smaller K.
