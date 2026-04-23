import torch
from typing import List

L_MIN_DEFAULT = 380.0
L_MAX_DEFAULT = 830.0

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
            local = torch.linspace(-spread/2, spread/2, count)
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

def generateTopology(topologyId: int, K: int, lMin: float = L_MIN_DEFAULT, lMax: float = L_MAX_DEFAULT, margin: float = 0.0) -> List[float]:
    dispatch = {
        0: topologyUniform,
        1: topologyBell,
        2: topologyTristimulus,
        3: topologyValley,
        4: topologySawblade
    }
    if topologyId not in dispatch:
        raise ValueError("Invalid topologyId (must be 0–4)")
    return dispatch[topologyId](K, lMin, lMax, margin)
