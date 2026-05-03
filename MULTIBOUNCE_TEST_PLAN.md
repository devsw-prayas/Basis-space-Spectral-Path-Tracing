# Multibounce Operator Test Plan

## Overview

Three test sets, building in complexity.

- **Set 0** — Each operator standalone, 10 bounces. Verify `A^10 @ α` against analytic ground truth derived from the operator's physics.
- **Set 1** — Single bounce, composed chains (TC1–TC10). Build the fused operator. Verify `A_fused @ α` against sequential per-operator application.
- **Set 2** — 10 bounces of each composed chain. Apply the same fused operator 10 times. Verify against 10× sequential application.

**Golden config throughout**: `K=8, N=11, uniform topology, margin=0, 16384 samples, float64`.

**Ground truth method**: For Set 1/2, compute the reference by applying each operator independently in sequence (`A_n @ ... @ A_2 @ A_1 @ α`). Error metric: `||s_fused - s_ref||_∞` and `||s_fused - s_ref||_2`.

---

## Set 0 — Per-Operator Multibounce (Standalone)

Each operator is applied 10 times to a flat input (`s = 1` everywhere). Ground truth is derived analytically from the operator's spectral transfer function.

### S0-1 — Beer-Lambert (Absorption)
- `σ = ln(2)` (flat), `d = 1.0` → 50% per bounce.
- **Ground truth**: `(0.5)^10 = 1/1024`. Reconstructed mean should match to < 1e-12.
- **Extra check**: `A^10 = (0.5)^10 · I` exactly (flat σ → scalar × I per bounce).

### S0-2 — Fresnel P0
- `F∞ = 0.1` (flat) → `Â_P0 = 0.1·I`.
- **Ground truth**: `(0.1)^10 = 1e-10`. Reconstructed mean to < 1e-12.

### S0-3 — Fresnel Qcomp
- `F∞ = 0.04` (flat) → `Â_Qcomp = (0.96)^2 · I = 0.9216·I`.
- **Ground truth**: `(0.9216)^10`. Reconstructed mean to < 1e-12.

### S0-4 — ThinFilm
- `n=1.5, d=200nm`. Transfer function `T(λ) = 0.5(1 + cos(4πnd/λ))` oscillates in [0,1].
- **Ground truth**: `T(λ)^10` pointwise — compute analytically, project to basis, reconstruct. Check `||s_10 - s_ref||_∞` < 1e-10.
- **Extra check**: Output is still non-negative everywhere.

### S0-5 — Fluorescence (Stokes)
- `e` = Gaussian at 550nm, `a` = Gaussian at 450nm. Rank-1 operator: `Â = e_w aᵀ_w`.
- After first application: `α_1 = (aᵀ_w α_0) · e_w` (scalar × `e_w`).
- After second: `α_2 = (aᵀ_w α_1) · e_w = (aᵀ_w e_w)(aᵀ_w α_0) · e_w`.
- **Ground truth**: `α_10 = (aᵀ_w e_w)^9 · (aᵀ_w α_0) · e_w`. Geometric decay by scalar `c = aᵀ_w e_w` per bounce after the first.
- **Check**: reconstructed spectrum matches `||s_10 - s_ref||_∞` < 1e-10.

### S0-6 — Cauchy Dispersion (all lobes summed)
- `Σ_k Â_k = I` — one application is identity. 10 applications is still identity.
- **Ground truth**: `α_10 = α_0`, i.e., `||s_10 - s_in||_∞` < 1e-10.

### S0-7 — Rayleigh Scattering
- `σ_base = 0.05`, `d = 1.0`, `α_power = 4.0`. Transfer: `T(λ) = exp(-σ_base·(λ/550)^{-4}·d)`.
- **Ground truth**: `T(λ)^10` pointwise. Project and reconstruct. Check `||s_10 - s_ref||_∞` < 1e-10.
- **Extra check**: `s_10[380nm] < s_10[700nm]` (blue end more attenuated).

### S0-8 — Mie Scattering
- Same as S0-7 but `α_power = 1.0` (weaker wavelength dependence).
- **Ground truth**: `T(λ)^10` pointwise. Check `||s_10 - s_ref||_∞` < 1e-10.
- **Extra check**: Attenuation weaker than Rayleigh at 380nm (less blue bias).

### S0-9 — Raman
- `shift = 50nm`, `σ_Raman = 10nm`. Input: Gaussian spike at 500nm.
- After 10 applications the peak should shift toward `500 + 10×50 = 1000nm` — beyond domain, so energy exits. Check that output energy is strictly less than input energy and the remaining peak is red-shifted.
- **No exact analytic ground truth** (convolution accumulates width). Use sequential 10× application as reference. Check `||s_10_fused - s_10_seq||_∞` < 1e-10.
- **Note**: Use 4096-sample domain.

### S0-10 — Emission
- Emission has `A = 0`, so `O(α) = b`. Applying it 10 times always yields `b` — the operator is a constant map regardless of input.
- **Ground truth**: `s_10 = reconstruct(b)` for any input. Check `||s_10 - s_b||_∞` < 1e-12.

---

## Set 1 & 2 — Composed Chain Test Cases

### TC1 — Pure Volume Absorption
```
Beer(σ1, d) ∘ Beer(σ2, d)
```
Two segments of the same medium, flat σ.

- **Set 1 ground truth**: `exp(-(σ1+σ2)·d)` analytically — fused A should be scalar × I.
- **Set 2**: 10 bounces → `exp(-10·(σ1+σ2)·d)` × I.
- **Check**: `||A_fused - exp(-(σ1+σ2)·d)·I||` < 1e-12 (exact for flat σ).

---

### TC2 — Volume with Wavelength-Dependent Scattering
```
Absorption(flat σ, d) ∘ Rayleigh(σ_base, d)
```
Blue-biased attenuation after flat absorption.

- **Set 1 ground truth**: sequential — `A_abs @ (A_ray @ α)`. Fused diagonal should show stronger suppression at short λ.
- **Set 2**: 10× fused vs 10× sequential.
- **Check**: `||s_fused - s_ref||_∞` < 1e-10. Also verify `s_fused[380nm] < s_fused[700nm]` (blue bias).

---

### TC3 — Raman In-Flight then Absorbed
```
Absorption(σ, d) ∘ Raman(shift=50nm)
```
Raman shifts energy to longer λ, Beer then attenuates.

- **Set 1 ground truth**: sequential application. Fused matrix should show off-diagonal structure (from Raman) that is then row-scaled by the absorption profile.
- **Set 2**: 10× fused vs 10× sequential.
- **Check**: `||s_fused - s_ref||_∞` < 1e-10.
- **Note**: Use 4096-sample domain for Raman to stay within memory budget (16384² kernel = ~2 GB).

---

### TC4 — Double Thin Film Pass
```
ThinFilm(n=1.5, d=200nm) ∘ ThinFilm(n=1.5, d=200nm)
```
Double pass through same film.

- **Set 1 ground truth**: sequential. Fused result should be a valid spectral modulation, not identity.
- **Set 2**: 10× fused vs 10× sequential.
- **Check**: `||s_fused - s_ref||_∞` < 1e-10. Verify `A_fused ≠ I` (non-trivial modulation).

---

### TC5 — Fresnel Lobe + Thin Film (Iridescent Reflectance)
```
Fresnel["P0"](F∞) ∘ ThinFilm(n=1.5, d=200nm)
```
Thin-film interference baked into reflectance.

- **Set 1 ground truth**: sequential. Fused op encodes spectrally-varying reflectance.
- **Set 2**: 10× fused vs 10× sequential.
- **Check**: `||s_fused - s_ref||_∞` < 1e-10.

---

### TC6 — Fluorescence After Absorption
```
Fluorescence(e, a) ∘ Absorption(σ, d)
```
Medium absorbs first, then fluoresces. Since Fluorescence is rank-1, fused matrix is also rank-1.

- **Set 1 ground truth**: sequential. Verify `rank(A_fused) == 1` (to numerical tolerance: second singular value < 1e-10 × first).
- **Set 2**: 10× fused vs 10× sequential. Also verify `A_fused^10 ≈ scalar × A_fused^9` (rank-1 idempotency up to scale).
- **Check**: `||s_fused - s_ref||_∞` < 1e-10.

---

### TC7 — Classic Single Bounce
```
Absorption(σ, d) ∘ Fresnel["P0"](F∞) ∘ Absorption(σ, d)
```
Enter medium → reflect at surface → exit medium. Flat σ and F∞.

- **Set 1 ground truth**: `exp(-σd) · r · exp(-σd)` × I analytically (flat inputs).
- **Set 2**: `(exp(-σd) · r · exp(-σd))^10` × I.
- **Check**: `||A_fused - expected·I||` < 1e-12 (exact for flat inputs).

---

### TC8 — Iridescent Bounce with Volume (Primary Regression Case)
```
(Absorption ∘ Rayleigh) ∘ (Fresnel["Rcross"] ∘ ThinFilm) ∘ (Absorption ∘ Mie)
```
Full single bounce, different media on each side, iridescent surface.

- **Set 1 ground truth**: sequential per-operator application (6 operators).
- **Set 2**: 10× fused vs 10× sequential.
- **Check**: `||s_fused - s_ref||_∞` < 1e-10. This is the primary regression case for the CUDA backend — if this fuses correctly, the algebra holds end-to-end.
- **Note**: Use 4096-sample domain (Raman not used here, but Rayleigh/Mie are fine at 16384).

---

### TC9 — Fluorescence Terminal in Multi-Hop Chain
```
Absorption ∘ Raman(50nm) ∘ Absorption ∘ Fluorescence(e, a)
```
Two volume segments with Raman shift, then fluoresces at surface. Fluorescence is terminal (rank-1).

- **Set 1 ground truth**: sequential (4 operators). Verify `rank(A_fused) == 1`.
- **Set 2**: 10× fused vs 10× sequential. Verify `A_fused^2 ≈ scalar × A_fused` (rank-1 idempotency).
- **Check**: `||s_fused - s_ref||_∞` < 1e-10.
- **Note**: Use 4096-sample domain for Raman.

---

### TC10 — Cauchy Dispersion Lobe in Full Path
```
For each lobe k:
    A_bounce_k = Absorption ∘ Cauchy_lobe[k] ∘ ThinFilm ∘ Absorption

Partition check:
    Σ_k A_bounce_k  =?=  Absorption ∘ ThinFilm ∘ Absorption
```
Run all K lobes independently. Partition of unity should survive end-to-end fusion.

- **Set 1 ground truth**: `Σ_k A_bounce_k` vs `A_abs @ A_film @ A_abs` — exact by Cauchy softmax partition.
- **Set 2 (Option B — sum-then-power)**: Build `A_bounce = Σ_k A_bounce_k`. Verify it equals the reference (TC7-style operator without Fresnel). Then apply `A_bounce^10` and check it matches `(A_abs @ A_film @ A_abs)^10`. The partition of unity survives through all 10 bounces because it collapses to identity at each step before powering.
- **Check (Set 1)**: `||Σ_k A_bounce_k - A_ref||` < 1e-10.
- **Check (Set 2)**: `||A_bounce^10 @ α - A_ref^10 @ α||_∞` < 1e-10.

---

## Implementation Notes

| | |
|---|---|
| **File** | `tests/test_multibounce.py` |
| **Domain** | 16384 samples by default; 4096 for any TC involving Raman |
| **Precision** | float64 throughout |
| **Error metrics** | `L∞` and `L2` on reconstructed spectrum; matrix norm for exact cases |
| **Tolerances** | 1e-12 for flat-input analytic cases; 1e-10 for general numerical cases |
| **Output** | Pass/Fail per TC per Set, with error values printed |
