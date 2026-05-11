import torch
from torch import Tensor

# CIE 1931 2° Standard Observer at 5nm intervals, 380–780 nm (81 values)
_CMF_LAMBDA_NM = [float(x) for x in range(380, 781, 5)]

_X_BAR = [
    0.001368, 0.002236, 0.004243, 0.007650, 0.014310, 0.023190, 0.043510, 0.077630,
    0.134380, 0.214770, 0.283900, 0.328500, 0.348280, 0.348060, 0.336200, 0.318700,
    0.290800, 0.251100, 0.195360, 0.142100, 0.095640, 0.057950, 0.032010, 0.014700,
    0.004900, 0.002400, 0.009300, 0.029100, 0.063270, 0.109600, 0.165500, 0.225750,
    0.290400, 0.359700, 0.433450, 0.512050, 0.594500, 0.678400, 0.762100, 0.842500,
    0.916300, 0.978600, 1.026300, 1.056700, 1.062200, 1.045600, 1.002600, 0.938400,
    0.854450, 0.751400, 0.642400, 0.541900, 0.447900, 0.360800, 0.283500, 0.218700,
    0.164900, 0.121200, 0.087400, 0.063600, 0.046770, 0.032900, 0.022700, 0.015840,
    0.011359, 0.008111, 0.005790, 0.004109, 0.002899, 0.002049, 0.001440, 0.001000,
    0.000690, 0.000476, 0.000332, 0.000235, 0.000166, 0.000117, 0.000083, 0.000059,
    0.000042,
]

_Y_BAR = [
    0.000039, 0.000064, 0.000120, 0.000217, 0.000396, 0.000640, 0.001210, 0.002180,
    0.004000, 0.007300, 0.011600, 0.016840, 0.023000, 0.029800, 0.038000, 0.048000,
    0.060000, 0.073900, 0.090980, 0.112600, 0.139020, 0.169300, 0.208020, 0.258600,
    0.323000, 0.407300, 0.503000, 0.608200, 0.710000, 0.793200, 0.862000, 0.914850,
    0.954000, 0.980300, 0.994950, 1.000000, 0.995000, 0.978600, 0.952000, 0.915400,
    0.870000, 0.816300, 0.757000, 0.694900, 0.631000, 0.566800, 0.503000, 0.441200,
    0.381000, 0.321000, 0.265000, 0.217000, 0.175000, 0.138200, 0.107000, 0.081600,
    0.061000, 0.044580, 0.032000, 0.023200, 0.017000, 0.011920, 0.008210, 0.005723,
    0.004102, 0.002929, 0.002091, 0.001484, 0.001047, 0.000740, 0.000520, 0.000361,
    0.000249, 0.000172, 0.000120, 0.000085, 0.000060, 0.000042, 0.000030, 0.000021,
    0.000015,
]

_Z_BAR = [
    0.006450, 0.010550, 0.020050, 0.036210, 0.067850, 0.110200, 0.207400, 0.371300,
    0.645600, 1.039050, 1.385600, 1.622960, 1.747060, 1.782600, 1.772110, 1.744100,
    1.669200, 1.528100, 1.287640, 1.041900, 0.812950, 0.616200, 0.465200, 0.353300,
    0.272000, 0.212300, 0.158200, 0.111700, 0.078250, 0.057250, 0.042160, 0.029840,
    0.020300, 0.013400, 0.008750, 0.005750, 0.003900, 0.002750, 0.002100, 0.001800,
    0.001650, 0.001400, 0.001100, 0.001000, 0.000800, 0.000600, 0.000340, 0.000240,
    0.000190, 0.000100, 0.000050, 0.000030, 0.000020, 0.000010, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000,
]

# D65 reference white (CIE standard, 2° observer)
_XN = 95.0489
_YN = 100.0000
_ZN = 108.8840


def buildCmfTensor(lbda: Tensor) -> Tensor:
    """
    Interpolates CIE 1931 2° CMF to the working wavelength grid.
    Returns [3, L] tensor (X, Y, Z rows). Values beyond 780nm are zero.
    """
    device, dtype = lbda.device, lbda.dtype

    cmf_lbda = torch.tensor(_CMF_LAMBDA_NM, device=device, dtype=dtype)
    x_tab    = torch.tensor(_X_BAR,         device=device, dtype=dtype)
    y_tab    = torch.tensor(_Y_BAR,         device=device, dtype=dtype)
    z_tab    = torch.tensor(_Z_BAR,         device=device, dtype=dtype)

    def interp(tab: Tensor) -> Tensor:
        # torch.searchsorted needs sorted input (cmf_lbda is already sorted)
        idx = torch.searchsorted(cmf_lbda, lbda).clamp(1, len(cmf_lbda) - 1)
        lo, hi = idx - 1, idx
        lam_lo, lam_hi = cmf_lbda[lo], cmf_lbda[hi]
        t = ((lbda - lam_lo) / (lam_hi - lam_lo)).clamp(0.0, 1.0)
        result = tab[lo] + t * (tab[hi] - tab[lo])
        # Zero out wavelengths outside CMF tabulation
        result = torch.where(lbda > 780.0, torch.zeros_like(result), result)
        result = torch.where(lbda < 380.0, torch.zeros_like(result), result)
        return result

    return torch.stack([interp(x_tab), interp(y_tab), interp(z_tab)], dim=0)  # [3, L]


def computeXYZ(spectra: Tensor, w: Tensor, cmf: Tensor) -> Tensor:
    """
    spectra: [S, L]
    w:       [L]  quadrature weights
    cmf:     [3, L]
    Returns: [S, 3] XYZ tristimulus values
    """
    return (cmf @ (spectra * w).T).T  # [S, 3]


def xyzToLab(xyz: Tensor) -> Tensor:
    """
    xyz: [S, 3] — converts to CIE L*a*b* under D65 reference white.
    Returns [S, 3] (L*, a*, b*).
    """
    device, dtype = xyz.device, xyz.dtype
    ref = torch.tensor([_XN, _YN, _ZN], device=device, dtype=dtype)

    r = xyz / ref.unsqueeze(0)  # [S, 3]

    delta  = 6.0 / 29.0
    delta2 = delta ** 2
    delta3 = delta ** 3

    f = torch.where(
        r > delta3,
        r.pow(1.0 / 3.0),
        r / (3.0 * delta2) + 4.0 / 29.0
    )  # [S, 3]

    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)

    return torch.stack([L, a, b], dim=1)  # [S, 3]


def computeDeltaE2000(lab1: Tensor, lab2: Tensor) -> Tensor:
    """
    Full CIEDE2000 formula.
    lab1, lab2: [S, 3] (L*, a*, b*)
    Returns:    [S]   ΔE₀₀ values
    """
    pi = torch.tensor(torch.pi, device=lab1.device, dtype=lab1.dtype)

    L1, a1, b1 = lab1[:, 0], lab1[:, 1], lab1[:, 2]
    L2, a2, b2 = lab2[:, 0], lab2[:, 1], lab2[:, 2]

    C1ab = torch.sqrt(a1 ** 2 + b1 ** 2)
    C2ab = torch.sqrt(a2 ** 2 + b2 ** 2)
    C_bar_ab = (C1ab + C2ab) * 0.5
    C_bar7   = C_bar_ab ** 7
    G = 0.5 * (1.0 - torch.sqrt(C_bar7 / (C_bar7 + 25.0 ** 7)))

    a1p = a1 * (1.0 + G)
    a2p = a2 * (1.0 + G)
    C1p = torch.sqrt(a1p ** 2 + b1 ** 2)
    C2p = torch.sqrt(a2p ** 2 + b2 ** 2)

    h1p = torch.atan2(b1, a1p) % (2.0 * pi)
    h2p = torch.atan2(b2, a2p) % (2.0 * pi)

    dLp = L2 - L1
    dCp = C2p - C1p

    both_nonzero = (C1p * C2p) > 0.0
    raw_dh = h2p - h1p
    dh = torch.where(both_nonzero,
         torch.where(raw_dh.abs() <= pi, raw_dh,
         torch.where(raw_dh > pi, raw_dh - 2.0 * pi, raw_dh + 2.0 * pi)),
         torch.zeros_like(raw_dh))
    dHp = 2.0 * torch.sqrt(C1p * C2p) * torch.sin(dh * 0.5)

    Lp_bar = (L1 + L2) * 0.5
    Cp_bar = (C1p + C2p) * 0.5

    h_sum = h1p + h2p
    hp_bar = torch.where(both_nonzero,
             torch.where(raw_dh.abs() <= pi, h_sum * 0.5,
             torch.where(h_sum < 2.0 * pi, (h_sum + 2.0 * pi) * 0.5, (h_sum - 2.0 * pi) * 0.5)),
             h_sum)

    T = (1.0
         - 0.17 * torch.cos(hp_bar - pi / 6.0)
         + 0.24 * torch.cos(2.0 * hp_bar)
         + 0.32 * torch.cos(3.0 * hp_bar + pi * 6.0 / 180.0)
         - 0.20 * torch.cos(4.0 * hp_bar - pi * 63.0 / 180.0))

    SL = 1.0 + 0.015 * (Lp_bar - 50.0) ** 2 / torch.sqrt(20.0 + (Lp_bar - 50.0) ** 2)
    SC = 1.0 + 0.045 * Cp_bar
    SH = 1.0 + 0.015 * Cp_bar * T

    Cp_bar7 = Cp_bar ** 7
    RC = 2.0 * torch.sqrt(Cp_bar7 / (Cp_bar7 + 25.0 ** 7))
    d_theta = (pi / 6.0) * torch.exp(-((hp_bar - 275.0 * pi / 180.0) / (25.0 * pi / 180.0)) ** 2)
    RT = -torch.sin(2.0 * d_theta) * RC

    dE = torch.sqrt(
        (dLp / SL) ** 2 +
        (dCp / SC) ** 2 +
        (dHp / SH) ** 2 +
        RT * (dCp / SC) * (dHp / SH)
    )
    return dE
