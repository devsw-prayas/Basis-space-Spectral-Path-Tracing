import torch
from torch import Tensor
from typing import Dict, List, Tuple
import math

# ── D65 tabulated at 10 nm, 380–780 nm (41 values) ─────────────────────────
_D65_LAMBDA = [float(x) for x in range(380, 781, 10)]
_D65_SPD = [
    49.9755, 54.6482, 82.7549, 91.4860, 93.4318, 86.6823, 104.8650, 117.0080,
    117.8120, 114.8610, 115.9230, 108.8110, 109.3540, 107.8020, 104.7900, 107.6890,
    104.4050, 104.0460, 100.0000,  96.3342,  95.7880,  88.6856,  90.0062,  89.5991,
     87.6987,  83.2886,  83.6992,  80.0268,  80.2146,  82.2778,  78.2842,  69.7213,
     71.6091,  74.3490,  61.6040,  69.8856,  75.0870,  63.5927,  46.4182,  66.8054,
     63.3828,
]

# ── Illuminant A tabulated at 10 nm, 380–780 nm (41 values) ────────────────
_ILLA_LAMBDA = [float(x) for x in range(380, 781, 10)]
_ILLA_SPD = [
     9.7951,  12.0853,  14.7080,  17.6753,  20.9950,  24.6700,  28.7020,  33.0930,
    37.8117,  42.8693,  48.2424,  53.9132,  59.8611,  66.0635,  72.4939,  79.1088,
    85.8629,  92.7040,  99.5915, 106.4780, 113.3790, 120.3180, 127.3060, 134.3220,
   141.4180, 148.5300, 155.6400, 162.8350, 170.0610, 177.2840, 184.4840, 191.6210,
   198.6610, 205.6250, 212.4960, 219.2700, 225.8560, 232.3420, 238.7370, 245.0960,
   251.2900,
]


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def _interp(lbda: Tensor, x_tab: List[float], y_tab: List[float]) -> Tensor:
    device, dtype = lbda.device, lbda.dtype
    xt = torch.tensor(x_tab, device=device, dtype=dtype)
    yt = torch.tensor(y_tab, device=device, dtype=dtype)
    idx = torch.searchsorted(xt, lbda).clamp(1, len(xt) - 1)
    lo, hi = idx - 1, idx
    t = ((lbda - xt[lo]) / (xt[hi] - xt[lo])).clamp(0.0, 1.0)
    out = yt[lo] + t * (yt[hi] - yt[lo])
    out = torch.where(lbda < xt[0],  yt[0]  * torch.ones_like(out), out)
    out = torch.where(lbda > xt[-1], yt[-1] * torch.ones_like(out), out)
    return out


def _gauss(lbda: Tensor, center: float, sigma: float, amp: float = 1.0) -> Tensor:
    return amp * torch.exp(-0.5 * ((lbda - center) / sigma) ** 2)


def _sigmoid(lbda: Tensor, center: float, width: float) -> Tensor:
    return torch.sigmoid((lbda - center) / width)


def _planck(lbda_nm: Tensor, T: float) -> Tensor:
    h = 6.62607015e-34
    c = 2.99792458e8
    k = 1.380649e-23
    lbda_m = lbda_nm * 1e-9
    exponent = (h * c) / (lbda_m * k * T)
    B = (2.0 * h * c ** 2) / (lbda_m ** 5) / (torch.exp(exponent) - 1.0)
    return B / B.max()


def _normalize(s: Tensor) -> Tensor:
    return s / s.max().clamp(min=1e-12)


# ════════════════════════════════════════════════════════════════════════════
# Phase A — Normal Spectra
# ════════════════════════════════════════════════════════════════════════════

def _blackbodies(lbda: Tensor) -> Tuple[Tensor, List[str]]:
    temps = [1000, 1500, 2000, 2800, 3200, 4000, 5000, 5500,
             6000, 6500, 7000, 8000, 9000, 10000, 12000, 15000, 20000, 25000]
    spectra = [_planck(lbda, T) for T in temps]
    ids     = [f"bb_{T}K" for T in temps]
    return torch.stack(spectra), ids


def _illuminants(lbda: Tensor) -> Tuple[Tensor, List[str]]:
    spectra, ids = [], []

    # D65
    d65 = _interp(lbda, _D65_LAMBDA, _D65_SPD)
    spectra.append(_normalize(d65)); ids.append("D65")

    # D50 — approximated as D65 with shifted correlated colour temp
    d50 = _interp(lbda, _D65_LAMBDA, _D65_SPD) * (1.0 + 0.15 * _sigmoid(lbda, 550, 80))
    spectra.append(_normalize(d50)); ids.append("D50")

    # D55
    d55 = _interp(lbda, _D65_LAMBDA, _D65_SPD) * (1.0 + 0.07 * _sigmoid(lbda, 560, 80))
    spectra.append(_normalize(d55)); ids.append("D55")

    # D75 — bluer than D65
    d75 = _interp(lbda, _D65_LAMBDA, _D65_SPD) * (1.0 - 0.10 * _sigmoid(lbda, 530, 80))
    spectra.append(_normalize(d75)); ids.append("D75")

    # Illuminant A
    illa = _interp(lbda, _ILLA_LAMBDA, _ILLA_SPD)
    spectra.append(_normalize(illa)); ids.append("IllA")

    # F2 — cool white fluorescent (broad + narrow lines)
    f2 = (0.4 * _gauss(lbda, 480, 30) + 0.5 * _gauss(lbda, 545, 25) +
          0.3 * _gauss(lbda, 612, 20) + 0.15 * torch.ones_like(lbda))
    spectra.append(_normalize(f2)); ids.append("F2")

    # F7 — broadband daylight fluorescent
    f7 = (0.3 * _gauss(lbda, 450, 40) + 0.5 * _gauss(lbda, 540, 35) +
          0.35 * _gauss(lbda, 610, 30) + 0.20 * torch.ones_like(lbda))
    spectra.append(_normalize(f7)); ids.append("F7")

    # F11 — narrow-band tri-phosphor (hard: sharp lines)
    f11 = (0.20 * torch.ones_like(lbda)
           + 1.5 * _gauss(lbda, 404,  3) + 1.8 * _gauss(lbda, 436,  3)
           + 2.0 * _gauss(lbda, 546,  4) + 1.2 * _gauss(lbda, 578,  4)
           + 2.5 * _gauss(lbda, 611,  5) + 0.8 * _gauss(lbda, 626,  4)
           + 0.5 * _gauss(lbda, 760, 10))
    spectra.append(_normalize(f11)); ids.append("F11")

    # F12 — narrow-band warm white fluorescent
    f12 = (0.15 * torch.ones_like(lbda)
           + 1.2 * _gauss(lbda, 436,  3) + 1.5 * _gauss(lbda, 546,  4)
           + 3.0 * _gauss(lbda, 610,  5) + 1.0 * _gauss(lbda, 650,  6))
    spectra.append(_normalize(f12)); ids.append("F12")

    # LED-B1 — narrow blue + broad phosphor
    led_b1 = 1.5 * _gauss(lbda, 450, 18) + _gauss(lbda, 570, 70)
    spectra.append(_normalize(led_b1)); ids.append("LED-B1")

    # LED-B3 — broader blue + phosphor
    led_b3 = 1.2 * _gauss(lbda, 455, 25) + _gauss(lbda, 565, 80)
    spectra.append(_normalize(led_b3)); ids.append("LED-B3")

    # LED-B5 — deep blue pump
    led_b5 = 2.0 * _gauss(lbda, 440, 15) + 0.8 * _gauss(lbda, 580, 90)
    spectra.append(_normalize(led_b5)); ids.append("LED-B5")

    # LED-RGB1 — three narrow primaries
    led_rgb = (1.0 * _gauss(lbda, 465, 12) + 1.0 * _gauss(lbda, 535, 12) +
               1.0 * _gauss(lbda, 615, 12))
    spectra.append(_normalize(led_rgb)); ids.append("LED-RGB1")

    # LED-V1 — violet pump + broad phosphor
    led_v = 2.0 * _gauss(lbda, 405, 12) + _gauss(lbda, 555, 90)
    spectra.append(_normalize(led_v)); ids.append("LED-V1")

    return torch.stack(spectra), ids


def _macbeth(lbda: Tensor) -> Tuple[Tensor, List[str]]:
    patches = []

    patches.append(_normalize(0.06 + 0.30 * _sigmoid(lbda, 540, 60)))               # 1 Dark Skin
    patches.append(_normalize(0.15 + 0.37 * _sigmoid(lbda, 520, 80)))               # 2 Light Skin
    patches.append(_normalize(0.35 - 0.25 * _sigmoid(lbda, 490, 60)))               # 3 Blue Sky
    s4 = 0.04 + _gauss(lbda, 550, 30, 0.12) + 0.30 * _sigmoid(lbda, 690, 10)
    patches.append(_normalize(s4.clamp(min=0)))                                       # 4 Foliage
    patches.append(_normalize(_gauss(lbda, 450, 50, 0.35) + 0.05 + 0.10 * _sigmoid(lbda, 600, 40)))  # 5 Blue Flower
    patches.append(_normalize(_gauss(lbda, 490, 60, 0.38) + 0.05))                  # 6 Bluish Green
    patches.append(_normalize(0.05 + 0.70 * _sigmoid(lbda, 570, 25)))               # 7 Orange
    patches.append(_normalize(_gauss(lbda, 440, 40, 0.28) + 0.03 + 0.08 * _sigmoid(lbda, 620, 30)))  # 8 Purplish Blue
    patches.append(_normalize(0.08 + 0.52 * _sigmoid(lbda, 600, 30)))               # 9 Moderate Red
    patches.append(_normalize(_gauss(lbda, 430, 30, 0.18) + _gauss(lbda, 640, 40, 0.15) + 0.03))  # 10 Purple
    patches.append(_normalize(_gauss(lbda, 560, 50, 0.40) + 0.06))                  # 11 Yellow Green
    patches.append(_normalize(0.08 + 0.44 * _sigmoid(lbda, 550, 30)))               # 12 Orange Yellow
    patches.append(_normalize(_gauss(lbda, 450, 35, 0.40) + 0.02))                  # 13 Blue
    patches.append(_normalize(_gauss(lbda, 535, 45, 0.42) + 0.04))                  # 14 Green
    patches.append(_normalize(0.04 + 0.60 * _sigmoid(lbda, 625, 20)))               # 15 Red
    patches.append(_normalize(0.10 + 0.50 * _sigmoid(lbda, 555, 25)))               # 16 Yellow
    s17 = 0.05 + _gauss(lbda, 420, 40, 0.30) + 0.35 * _sigmoid(lbda, 610, 30)
    patches.append(_normalize(s17))                                                    # 17 Magenta
    patches.append(_normalize(0.40 - 0.30 * _sigmoid(lbda, 530, 40)))               # 18 Cyan
    for level in [0.90, 0.59, 0.36, 0.19, 0.09, 0.03]:                              # 19-24 Neutrals
        patches.append(torch.full_like(lbda, level))

    ids = [f"macbeth_{i+1:02d}" for i in range(24)]
    return torch.stack(patches), ids


def _narrowGaussians(lbda: Tensor) -> Tuple[Tensor, List[str]]:
    centers = [400, 430, 460, 490, 520, 550, 580, 610, 640, 670, 700, 760]
    sigmas  = [10, 15, 20, 30, 50]
    spectra, ids = [], []
    for c in centers:
        for s in sigmas:
            spectra.append(_normalize(_gauss(lbda, float(c), float(s))))
            ids.append(f"narrow_c{c}_s{s}")
    return torch.stack(spectra), ids


def _multiPeak(lbda: Tensor) -> Tuple[Tensor, List[str]]:
    rng = torch.Generator()
    rng.manual_seed(42)
    spectra, ids = [], []

    # 30 dual-peak
    dual_configs = [
        (420, 650, 20, 20), (440, 620, 15, 15), (460, 610, 25, 20),
        (400, 700, 10, 15), (450, 550, 20, 25), (480, 640, 15, 20),
        (430, 590, 20, 30), (410, 680, 12, 18), (500, 700, 25, 20),
        (420, 560, 15, 35), (460, 630, 20, 15), (490, 710, 18, 22),
        (440, 580, 25, 20), (415, 655, 10, 25), (470, 600, 30, 20),
        (430, 670, 15, 15), (450, 720, 20, 20), (480, 560, 25, 30),
        (400, 620, 18, 18), (460, 590, 22, 22), (500, 650, 15, 15),
        (420, 720, 12, 20), (440, 600, 20, 25), (470, 640, 18, 18),
        (410, 550, 15, 40), (450, 680, 20, 15), (490, 600, 25, 20),
        (430, 560, 30, 25), (460, 720, 15, 18), (480, 630, 20, 20),
    ]
    for i, (c1, c2, s1, s2) in enumerate(dual_configs):
        s = _gauss(lbda, float(c1), float(s1)) + _gauss(lbda, float(c2), float(s2))
        spectra.append(_normalize(s)); ids.append(f"dual_{i:02d}")

    # 20 triple-peak
    triple_configs = [
        (430, 540, 660, 15, 20, 15), (440, 550, 650, 10, 25, 10),
        (420, 530, 700, 12, 18, 20), (450, 560, 680, 20, 20, 15),
        (410, 520, 640, 15, 25, 20), (460, 570, 700, 18, 15, 18),
        (430, 545, 670, 10, 22, 12), (440, 560, 720, 15, 20, 15),
        (420, 530, 660, 20, 18, 20), (450, 550, 690, 12, 25, 12),
        (400, 510, 630, 10, 20, 15), (460, 570, 680, 15, 15, 20),
        (430, 540, 710, 20, 20, 10), (440, 555, 650, 18, 22, 18),
        (415, 525, 640, 12, 18, 25), (455, 565, 720, 20, 15, 15),
        (435, 545, 665, 15, 20, 20), (445, 555, 680, 18, 18, 18),
        (425, 535, 655, 12, 25, 12), (465, 575, 700, 20, 20, 15),
    ]
    for i, (c1, c2, c3, s1, s2, s3) in enumerate(triple_configs):
        s = (_gauss(lbda, float(c1), float(s1)) + _gauss(lbda, float(c2), float(s2)) +
             _gauss(lbda, float(c3), float(s3)))
        spectra.append(_normalize(s)); ids.append(f"triple_{i:02d}")

    return torch.stack(spectra), ids


def _ledPhosphor(lbda: Tensor) -> Tuple[Tensor, List[str]]:
    configs = [
        # (pump_center, pump_sigma, pump_amp, phosphor_center, phosphor_sigma)
        (450, 18, 2.0, 570, 70), (455, 20, 1.8, 565, 75), (445, 15, 2.5, 580, 65),
        (460, 22, 1.5, 560, 80), (440, 16, 2.2, 575, 72), (450, 19, 1.9, 585, 68),
        (465, 24, 1.4, 555, 85), (448, 17, 2.1, 570, 70), (453, 21, 1.7, 568, 76),
        (442, 15, 2.3, 578, 66), (458, 20, 1.6, 562, 78), (447, 18, 2.0, 572, 71),
        # Warm white (redder phosphor)
        (450, 18, 2.0, 590, 80), (455, 20, 1.8, 595, 85), (445, 15, 2.5, 585, 78),
        (460, 22, 1.5, 600, 82), (440, 16, 2.2, 580, 75), (450, 19, 1.9, 595, 80),
        # PC-amber
        (455, 18, 2.5, 560, 50), (448, 16, 2.8, 555, 45),
        # RGB-like (three peaks)
        (455, 12, 1.0, 0, 0),  # handled below
        (465, 12, 1.0, 0, 0),
        (445, 12, 1.0, 0, 0),
        (450, 15, 1.0, 0, 0),
        (440, 10, 1.0, 0, 0),
        (460, 10, 1.0, 0, 0),
        (448, 14, 1.0, 0, 0),
        (453, 16, 1.0, 0, 0),
        (442, 11, 1.0, 0, 0),
        (462, 13, 1.0, 0, 0),
    ]
    spectra, ids = [], []
    green_centers = [530, 535, 540, 545, 525, 538, 532, 536, 528, 542]
    red_centers   = [610, 615, 620, 625, 605, 618, 612, 616, 608, 622]

    for i, (pc, ps, pa, phc, phs) in enumerate(configs):
        if phc == 0:
            gi = i - 20
            s = (_gauss(lbda, float(pc), float(ps), pa) +
                 _gauss(lbda, float(green_centers[gi % len(green_centers)]), 12.0) +
                 _gauss(lbda, float(red_centers[gi % len(red_centers)]), 12.0))
        else:
            s = pa * _gauss(lbda, float(pc), float(ps)) + _gauss(lbda, float(phc), float(phs))
        spectra.append(_normalize(s)); ids.append(f"led_phosphor_{i:02d}")

    return torch.stack(spectra), ids


def generatePhaseA(lbda: Tensor) -> Dict:
    bb_s,  bb_ids  = _blackbodies(lbda)
    ill_s, ill_ids = _illuminants(lbda)
    mb_s,  mb_ids  = _macbeth(lbda)
    ng_s,  ng_ids  = _narrowGaussians(lbda)
    mp_s,  mp_ids  = _multiPeak(lbda)
    lp_s,  lp_ids  = _ledPhosphor(lbda)

    spectra    = torch.cat([bb_s, ill_s, mb_s, ng_s, mp_s, lp_s], dim=0)
    ids        = bb_ids + ill_ids + mb_ids + ng_ids + mp_ids + lp_ids
    categories = (["blackbody"] * len(bb_ids) + ["illuminant"] * len(ill_ids) +
                  ["macbeth"]   * len(mb_ids) + ["narrow_gauss"] * len(ng_ids) +
                  ["multi_peak"] * len(mp_ids) + ["led_phosphor"] * len(lp_ids))
    return {"spectra": spectra, "ids": ids, "categories": categories, "phase": "A"}


# ════════════════════════════════════════════════════════════════════════════
# Phase B — Delta Spectra
# ════════════════════════════════════════════════════════════════════════════

def generatePhaseB(lbda: Tensor) -> Dict:
    centers_nm = torch.linspace(385.0, 825.0, 40).tolist()
    sigmas_nm  = [0.5, 1.0, 2.0, 5.0, 10.0]

    spectra, ids, categories = [], [], []
    centers_out, sigmas_out = [], []

    for c in centers_nm:
        for s in sigmas_nm:
            sp = _gauss(lbda, c, s)
            sp = sp / sp.max().clamp(min=1e-12)
            spectra.append(sp)
            ids.append(f"delta_c{c:.1f}_s{s}")
            categories.append("delta")
            centers_out.append(c)
            sigmas_out.append(s)

    device, dtype = lbda.device, lbda.dtype
    return {
        "spectra":    torch.stack(spectra),
        "ids":        ids,
        "categories": categories,
        "phase":      "B",
        "centers":    torch.tensor(centers_out, device=device, dtype=dtype),
        "sigmas":     torch.tensor(sigmas_out,  device=device, dtype=dtype),
    }


# ════════════════════════════════════════════════════════════════════════════
# Phase C — Misc Hard Cases
# ════════════════════════════════════════════════════════════════════════════

def generatePhaseC(lbda: Tensor) -> Dict:
    spectra, ids, categories = [], [], []

    lmin, lmax = lbda[0].item(), lbda[-1].item()
    lspan = lmax - lmin

    def add(s: Tensor, sid: str, cat: str):
        spectra.append(_normalize(s.clamp(min=0.0)))
        ids.append(sid)
        categories.append(cat)

    # Linear ramps (10)
    for i, frac in enumerate([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]):
        t = (lbda - lmin) / lspan
        add(frac + (1.0 - frac) * t, f"ramp_up_{i}", "ramp")
    for i, frac in enumerate([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]):
        t = (lbda - lmin) / lspan
        add(frac + (1.0 - frac) * (1.0 - t), f"ramp_dn_{i}", "ramp")

    # Step functions at varying thresholds (15)
    thresholds = torch.linspace(420.0, 760.0, 15).tolist()
    for i, thr in enumerate(thresholds):
        add(_sigmoid(lbda, thr, 5.0), f"step_{i:02d}", "step")

    # Notch filters — narrow absorption bands (15)
    notch_centers = torch.linspace(420.0, 760.0, 15).tolist()
    for i, nc in enumerate(notch_centers):
        add(1.0 - 0.95 * _gauss(lbda, nc, 8.0), f"notch_{i:02d}", "notch")

    # Broad Gaussians (20)
    broad_centers = torch.linspace(430.0, 760.0, 10).tolist()
    broad_sigmas  = [60.0, 100.0]
    for c in broad_centers:
        for s in broad_sigmas:
            add(_gauss(lbda, c, s), f"broad_c{int(c)}_s{int(s)}", "broad_gauss")

    # Laser lines σ=0.5 nm at 20 specific wavelengths (20)
    laser_wl = [404.7, 435.8, 457.9, 476.5, 488.0, 496.5, 514.5,
                532.0, 543.5, 568.2, 594.1, 611.9, 632.8, 647.1,
                676.4, 694.3, 720.0, 750.0, 785.0, 820.0]
    for wl in laser_wl:
        add(_gauss(lbda, wl, 0.5), f"laser_{wl:.1f}", "laser")

    # Sine-modulated flat (15)
    freqs = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 20, 25, 30]
    for i, freq in enumerate(freqs):
        t = (lbda - lmin) / lspan
        add(0.5 + 0.5 * torch.sin(2.0 * math.pi * freq * t), f"sine_{freq}cyc", "sine_mod")

    # Exponential decay / rise (10)
    for i, rate in enumerate([0.002, 0.004, 0.006, 0.008, 0.010]):
        add(torch.exp(-rate * (lbda - lmin)), f"exp_decay_{i}", "exponential")
        add(torch.exp( rate * (lbda - lmin)), f"exp_rise_{i}",  "exponential")

    # Near-flat with small perturbations (5)
    rng = torch.Generator(device=lbda.device); rng.manual_seed(99)
    for i in range(5):
        perturb = 0.05 * _gauss(lbda, float(450 + i * 60), 30.0)
        add(0.5 + perturb, f"nearflat_{i}", "near_flat")

    # Random smooth — reproducible seeded (30)
    for seed in range(30):
        rng2 = torch.Generator(device=lbda.device); rng2.manual_seed(1000 + seed)
        n_components = 4 + (seed % 4)
        s = torch.zeros_like(lbda)
        for _ in range(n_components):
            c = lmin + torch.rand(1, generator=rng2).item() * lspan
            sg = 20.0 + torch.rand(1, generator=rng2).item() * 60.0
            amp = 0.3 + torch.rand(1, generator=rng2).item() * 0.7
            s = s + _gauss(lbda, c, sg, amp)
        add(s, f"random_smooth_{seed:02d}", "random_smooth")

    # Hyperspectral-like — complex multi-feature (30)
    hyper_params = [
        ([420, 480, 540, 620], [8, 15, 10, 12], [0.8, 0.4, 1.0, 0.6]),
        ([400, 450, 550, 650, 720], [5, 20, 15, 10, 8], [0.5, 0.7, 1.0, 0.6, 0.4]),
        ([430, 510, 580, 670], [10, 12, 18, 15], [0.9, 0.5, 0.8, 0.7]),
        ([415, 465, 520, 590, 680], [8, 10, 15, 12, 10], [0.6, 0.8, 1.0, 0.5, 0.7]),
        ([440, 500, 560, 640, 710], [12, 8, 20, 10, 15], [0.7, 0.9, 0.8, 0.6, 0.5]),
        ([420, 470, 530, 610, 700], [6, 14, 10, 8, 12], [1.0, 0.6, 0.8, 0.9, 0.4]),
        ([400, 460, 540, 600, 680], [10, 8, 15, 20, 10], [0.8, 1.0, 0.6, 0.5, 0.7]),
        ([430, 490, 550, 630], [8, 15, 12, 10], [0.9, 0.7, 1.0, 0.5]),
        ([410, 480, 560, 650, 730], [6, 12, 18, 8, 15], [0.7, 0.8, 1.0, 0.6, 0.4]),
        ([450, 510, 580, 660], [10, 8, 15, 12], [0.6, 1.0, 0.8, 0.7]),
    ]
    # Extend to 30 by shifting centers
    for extra in range(20):
        shift = (extra + 1) * 5.0
        p = hyper_params[extra % 10]
        hyper_params.append(
            ([c + shift for c in p[0]], p[1], p[2])
        )

    for i, (cents, sigs, amps) in enumerate(hyper_params[:30]):
        s = torch.zeros_like(lbda)
        for c, sg, a in zip(cents, sigs, amps):
            s = s + _gauss(lbda, float(c), float(sg), float(a))
        add(s, f"hyper_{i:02d}", "hyperspectral")

    return {"spectra": torch.stack(spectra), "ids": ids, "categories": categories, "phase": "C"}


# ════════════════════════════════════════════════════════════════════════════
# Combined
# ════════════════════════════════════════════════════════════════════════════

def generateAll(lbda: Tensor) -> Dict:
    A = generatePhaseA(lbda)
    B = generatePhaseB(lbda)
    C = generatePhaseC(lbda)

    SA = A["spectra"].shape[0]
    SB = B["spectra"].shape[0]
    SC = C["spectra"].shape[0]

    spectra    = torch.cat([A["spectra"], B["spectra"], C["spectra"]], dim=0)
    ids        = A["ids"]        + B["ids"]        + C["ids"]
    categories = A["categories"] + B["categories"] + C["categories"]
    phases     = (["A"] * SA)    + (["B"] * SB)    + (["C"] * SC)

    # Phase B delta metadata — padded with zeros for A/C rows
    device, dtype = lbda.device, lbda.dtype
    S = SA + SB + SC
    centers = torch.zeros(S, device=device, dtype=dtype)
    sigmas  = torch.zeros(S, device=device, dtype=dtype)
    centers[SA: SA + SB] = B["centers"]
    sigmas [SA: SA + SB] = B["sigmas"]

    # Boolean masks per phase
    phaseA_mask = torch.zeros(S, dtype=torch.bool, device=device)
    phaseB_mask = torch.zeros(S, dtype=torch.bool, device=device)
    phaseC_mask = torch.zeros(S, dtype=torch.bool, device=device)
    phaseA_mask[:SA]            = True
    phaseB_mask[SA: SA + SB]    = True
    phaseC_mask[SA + SB:]       = True

    return {
        "spectra":     spectra,
        "ids":         ids,
        "categories":  categories,
        "phases":      phases,
        "phaseA_mask": phaseA_mask,
        "phaseB_mask": phaseB_mask,
        "phaseC_mask": phaseC_mask,
        "centers":     centers,
        "sigmas":      sigmas,
        "SA": SA, "SB": SB, "SC": SC,
    }
