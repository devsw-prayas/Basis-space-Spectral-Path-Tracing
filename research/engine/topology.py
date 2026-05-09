import torch
from typing import List, Tuple

L_MIN_DEFAULT = 380.0
L_MAX_DEFAULT = 830.0

# ── Raw center generators (public) ────────────────────────────────────────────
# These return List[float] only. Used directly by callers that manage wideIndices
# themselves (e.g. thin-film-terp.py).

def topologyUniform(K: int, lMin: float = L_MIN_DEFAULT, lMax: float = L_MAX_DEFAULT, margin: float = 0.0) -> List[float]:
    return torch.linspace(lMin + margin, lMax - margin, K).tolist()

def topologyBell(K: int, lMin: float = L_MIN_DEFAULT, lMax: float = L_MAX_DEFAULT, margin: float = 0.0) -> List[float]:
    t = torch.linspace(0.0, 1.0, K)
    warped = 0.5 * (1.0 - torch.cos(torch.pi * t))
    centers = (lMin + margin) + warped * ((lMax - margin) - (lMin + margin))
    return centers.tolist()

def topologyTristimulus(K: int, lMin: float = L_MIN_DEFAULT, lMax: float = L_MAX_DEFAULT, margin: float = 0.0) -> List[float]:
    anchors = torch.tensor([450.0, 550.0, 650.0])
    if margin > 0:
        span = lMax - lMin
        newSpan = span - 2 * margin
        scale = newSpan / span
        anchors = (lMin + margin) + (anchors - lMin) * scale

    if K <= 3: return anchors[:K].tolist()
    perBand = K // 3
    remainder = K % 3
    centers = []
    for i in range(3):
        count = perBand + (1 if i < remainder else 0)
        if count == 1:
            centers.append(anchors[i].item())
        else:
            spread = 40.0
            local = torch.linspace(-spread / 2, spread / 2, count)
            centers.extend((anchors[i] + local).tolist())
    return centers[:K]

def topologyValley(K: int, lMin: float = L_MIN_DEFAULT, lMax: float = L_MAX_DEFAULT, margin: float = 0.0) -> List[float]:
    t = torch.linspace(0.0, 1.0, K)
    warped = torch.sin(torch.pi * t / 2.0) ** 2
    centers = (lMin + margin) + warped * ((lMax - margin) - (lMin + margin))
    return centers.tolist()

def topologySawblade(K: int, lMin: float = L_MIN_DEFAULT, lMax: float = L_MAX_DEFAULT, margin: float = 0.0) -> List[float]:
    lMinM = lMin + margin
    lMaxM = lMax - margin
    base = torch.linspace(lMinM, lMaxM, K)
    perturb = (lMaxM - lMinM) / (4.0 * K)
    for i in range(K):
        if i % 2 == 0: base[i] -= perturb
        else: base[i] += perturb
    return torch.clamp(base, lMinM, lMaxM).tolist()


# ── Topology variants ─────────────────────────────────────────────────────────
# Each returns (centers, wideIndices). Topology IDs:
#   0 — Uniform All Wide
#   1 — Uniform All Narrow
#   2 — Uniform First Half Wide  (first K//2 centers wide)
#   3 — Bell  (outermost K//2 centers wide, inner narrow)
#   4 — Tristimulus  (R/G/B band anchor wide, satellites narrow)
#   5 — Valley  (last K//2 centers wide — sparse/red end)
#   6 — Sawblade  (even-indexed centers wide)

def _uniformAllWide(K: int, lMin: float, lMax: float, margin: float) -> Tuple[List[float], List[int]]:
    return topologyUniform(K, lMin, lMax, margin), list(range(K))

def _uniformAllNarrow(K: int, lMin: float, lMax: float, margin: float) -> Tuple[List[float], List[int]]:
    return topologyUniform(K, lMin, lMax, margin), []

def _uniformFirstWide(K: int, lMin: float, lMax: float, margin: float) -> Tuple[List[float], List[int]]:
    return topologyUniform(K, lMin, lMax, margin), list(range(K // 2))

def _bellOuterWide(K: int, lMin: float, lMax: float, margin: float) -> Tuple[List[float], List[int]]:
    centers = topologyBell(K, lMin, lMax, margin)
    numWide = K // 2
    nLeft   = (numWide + 1) // 2
    nRight  = numWide // 2
    wideIndices = list(range(nLeft)) + list(range(K - nRight, K))
    return centers, wideIndices

def _tristimulusAnchorWide(K: int, lMin: float, lMax: float, margin: float) -> Tuple[List[float], List[int]]:
    anchors = torch.tensor([450.0, 550.0, 650.0])
    if margin > 0:
        span = lMax - lMin
        newSpan = span - 2 * margin
        scale = newSpan / span
        anchors = (lMin + margin) + (anchors - lMin) * scale

    if K <= 3:
        return anchors[:K].tolist(), list(range(K))

    perBand = K // 3
    remainder = K % 3
    centers = []
    wideIndices = []
    idx = 0
    for i in range(3):
        count = perBand + (1 if i < remainder else 0)
        if count == 1:
            wideIndices.append(idx)
            centers.append(anchors[i].item())
        else:
            wideIndices.append(idx + count // 2)
            spread = 40.0
            local = torch.linspace(-spread / 2, spread / 2, count)
            centers.extend((anchors[i] + local).tolist())
        idx += count
    return centers[:K], wideIndices

def _valleyLastWide(K: int, lMin: float, lMax: float, margin: float) -> Tuple[List[float], List[int]]:
    centers = topologyValley(K, lMin, lMax, margin)
    numWide = K // 2
    return centers, list(range(K - numWide, K))

def _sawbladeEvenWide(K: int, lMin: float, lMax: float, margin: float) -> Tuple[List[float], List[int]]:
    centers = topologySawblade(K, lMin, lMax, margin)
    return centers, [i for i in range(K) if i % 2 == 0]


_DISPATCH = {
    0: _uniformAllWide,
    1: _uniformAllNarrow,
    2: _uniformFirstWide,
    3: _bellOuterWide,
    4: _tristimulusAnchorWide,
    5: _valleyLastWide,
    6: _sawbladeEvenWide,
}

def generateTopology(
    topologyId: int,
    K: int,
    lMin: float = L_MIN_DEFAULT,
    lMax: float = L_MAX_DEFAULT,
    margin: float = 0.0
) -> Tuple[List[float], List[int]]:
    if topologyId not in _DISPATCH:
        raise ValueError("Invalid topologyId (must be 0–6)")
    return _DISPATCH[topologyId](K, lMin, lMax, margin)
