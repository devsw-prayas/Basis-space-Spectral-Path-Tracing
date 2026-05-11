import torch
from torch import Tensor
from typing import Dict, Optional

from research.phase2a.cmf import computeXYZ, xyzToLab, computeDeltaE2000


def computeBaseMetrics(
    f_true: Tensor,
    f_hat:  Tensor,
    w:      Tensor,
    cmf:    Tensor,
) -> Dict[str, Tensor]:
    """
    f_true, f_hat: [S, L]
    w:             [L]
    cmf:           [3, L]
    Returns dict of [S] tensors: l2, nrmse, maxError, xyzDelta, perceptualDeltaE
    """
    err = f_hat - f_true                                         # [S, L]

    l2      = torch.sqrt((err ** 2 * w).sum(dim=1))             # [S]
    f_norm  = torch.sqrt((f_true ** 2 * w).sum(dim=1)).clamp(min=1e-12)
    nrmse   = l2 / f_norm                                        # [S]
    maxErr  = err.abs().amax(dim=1)                              # [S]

    xyz_true = computeXYZ(f_true, w, cmf)                       # [S, 3]
    xyz_hat  = computeXYZ(f_hat,  w, cmf)                       # [S, 3]
    xyzDelta = (xyz_hat - xyz_true).norm(dim=1)                  # [S]

    lab_true = xyzToLab(xyz_true)                                # [S, 3]
    lab_hat  = xyzToLab(xyz_hat)                                 # [S, 3]
    dE       = computeDeltaE2000(lab_true, lab_hat)              # [S]

    return {
        "l2":               l2,
        "nrmse":            nrmse,
        "maxError":         maxErr,
        "xyzDelta":         xyzDelta,
        "perceptualDeltaE": dE,
    }


def computeDeltaExtras(
    f_true:  Tensor,
    f_hat:   Tensor,
    w:       Tensor,
    lbda:    Tensor,
    centers: Tensor,
    sigmas:  Tensor,
) -> Dict[str, Tensor]:
    """
    f_true, f_hat: [SB, L]  (Phase B spectra only)
    w:             [L]
    lbda:          [L]
    centers:       [SB]  delta center wavelengths in nm
    sigmas:        [SB]  delta sigma values in nm
    Returns dict of [SB] tensors.
    """
    # Energy retention
    energy_true = (f_true ** 2 * w).sum(dim=1).clamp(min=1e-12)  # [SB]
    energy_hat  = (f_hat  ** 2 * w).sum(dim=1)                    # [SB]
    energyRetention = energy_hat / energy_true                     # [SB]

    # Amplitude accuracy
    amp_true = f_true.amax(dim=1).clamp(min=1e-12)                # [SB]
    amp_hat  = f_hat.amax(dim=1)                                   # [SB]
    amplitudeAccuracy = amp_hat / amp_true                         # [SB]

    # Peak position shift in nm
    peak_idx_true = f_true.argmax(dim=1)                           # [SB]
    peak_idx_hat  = f_hat.argmax(dim=1)                            # [SB]
    peakShiftNm   = lbda[peak_idx_hat] - lbda[peak_idx_true]      # [SB]

    # Side-lobe energy: fraction of f_hat energy outside center ± 3σ
    # window: [SB, L] — True inside the central window
    window = (lbda.unsqueeze(0) - centers.unsqueeze(1)).abs() <= 3.0 * sigmas.unsqueeze(1)
    outside_energy = (f_hat ** 2 * w * ~window).sum(dim=1)        # [SB]
    sideLobeEnergy = outside_energy / energy_hat.clamp(min=1e-12)  # [SB]

    return {
        "energyRetention":   energyRetention,
        "amplitudeAccuracy": amplitudeAccuracy,
        "peakShiftNm":       peakShiftNm,
        "sideLobeEnergy":    sideLobeEnergy,
    }


def computeAllMetrics(
    f_true:      Tensor,
    f_hat:       Tensor,
    w:           Tensor,
    cmf:         Tensor,
    lbda:        Tensor,
    phaseB_mask: Tensor,
    centers:     Tensor,
    sigmas:      Tensor,
) -> Dict[str, Tensor]:
    """
    f_true, f_hat: [S, L]
    phaseB_mask:   [S] bool — which rows are Phase B spectra
    centers:       [S] delta centers (nonzero only for Phase B rows)
    sigmas:        [S] delta sigmas  (nonzero only for Phase B rows)

    Returns dict with base metrics [S] and delta extras [S] (zero-padded for non-B rows).
    """
    base = computeBaseMetrics(f_true, f_hat, w, cmf)

    S = f_true.shape[0]
    device, dtype = f_true.device, f_true.dtype

    extras = {
        "energyRetention":   torch.zeros(S, device=device, dtype=dtype),
        "amplitudeAccuracy": torch.zeros(S, device=device, dtype=dtype),
        "peakShiftNm":       torch.zeros(S, device=device, dtype=dtype),
        "sideLobeEnergy":    torch.zeros(S, device=device, dtype=dtype),
    }

    if phaseB_mask.any():
        delta = computeDeltaExtras(
            f_true[phaseB_mask],
            f_hat [phaseB_mask],
            w, lbda,
            centers[phaseB_mask],
            sigmas [phaseB_mask],
        )
        for k, v in delta.items():
            extras[k][phaseB_mask] = v

    return {**base, **extras}
