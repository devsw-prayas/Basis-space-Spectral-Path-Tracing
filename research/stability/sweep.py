import os
import torch
import pandas as pd
import traceback
import time
import sys
from typing import List, Tuple

from research.engine.domain import SpectralDomain
from research.engine.basis import GHGSFDualDomainBasis
from research.engine.topology import generateTopology
from research.engine.config import TorchConfig

# ============================================================
# CONFIGURATION & SCHEMA
# ============================================================

CONFIG_COLUMNS = [
    "familyId", "K", "order", "scalingId",
    "wideMin", "wideMax", "narrowMin", "narrowMax", "margin"
]

# Metrics computed for both Raw and Whitened states
CORE_METRICS = [
    "lambdaMin", "lambdaMax", "condition", "logCond",
    "entropy", "eigenGap"
]

METRIC_COLUMNS = []
for m in CORE_METRICS: METRIC_COLUMNS.append(f"raw_{m}")
for m in CORE_METRICS: METRIC_COLUMNS.append(f"wht_{m}")
METRIC_COLUMNS.append("rawSpdFail")
METRIC_COLUMNS.append("whtSpdFail")

SCALING_ID_MAP = {0: "constant", 1: "linear", 2: "sqrt", 3: "power"}

# ============================================================
# CONFIGURATION BUILDER
# ============================================================

def buildSweepConfigs(device=torch.device("cpu")) -> torch.Tensor:
    families = torch.arange(5, device=device, dtype=torch.int64)
    lobes    = torch.arange(4, 13, device=device, dtype=torch.int64)
    orders   = torch.arange(4, 13, device=device, dtype=torch.int64)
    scaling  = torch.arange(4, device=device, dtype=torch.int64)

    base = torch.cartesian_prod(families, lobes, orders, scaling).to(torch.float64)

    sigmaVals = torch.arange(6.0, 12.5, 0.5, device=device, dtype=torch.float64)
    pairs = torch.combinations(sigmaVals, r=2)

    wideExp   = pairs.unsqueeze(1).expand(-1, pairs.shape[0], -1)
    narrowExp = pairs.unsqueeze(0).expand(pairs.shape[0], -1, -1)
    mask = wideExp[:, :, 1] > narrowExp[:, :, 1]

    domains = torch.cat([wideExp[mask], narrowExp[mask]], dim=1)

    B, D = base.shape[0], domains.shape[0]
    baseExp   = base.unsqueeze(1).expand(B, D, -1)
    domainExp = domains.unsqueeze(0).expand(B, D, -1)

    combined = torch.cat([baseExp, domainExp], dim=2).reshape(-1, 10)
    
    # Add margins: 0.0, 10.0, 20.0 nm
    margins = torch.tensor([0.0, 10.0, 20.0], device=device, dtype=torch.float64)
    M = margins.shape[0]
    
    combinedExp = combined.unsqueeze(1).expand(-1, M, -1)
    marginExp   = margins.view(1, M, 1).expand(combined.shape[0], -1, -1)
    
    configs = torch.cat([combinedExp, marginExp], dim=2).reshape(-1, 11)
    return configs[:, [0, 1, 2, 3, 6, 7, 8, 9, 10]] # Correct column mapping

# ============================================================
# METRIC ENGINE
# ============================================================

def computeMetrics(
    basis: GHGSFDualDomainBasis,
    domain: SpectralDomain
) -> torch.Tensor:
    """Computes stability metrics for both Raw and Whitened Gram matrices."""

    # 1. Raw Gram
    G_raw = basis.m_gram
    raw_metrics, raw_fail = calculateMatrixMetrics(G_raw)

    # 2. Whitened Gram (L^-1 G L^-T)
    L = basis.m_chol
    LiG = torch.linalg.solve_triangular(L, G_raw, upper=False)
    G_wht = torch.linalg.solve_triangular(L, LiG.T, upper=False).T
    wht_metrics, wht_fail = calculateMatrixMetrics(G_wht)

    return torch.tensor(raw_metrics + wht_metrics + [raw_fail, wht_fail], dtype=torch.float64)

def calculateMatrixMetrics(G: torch.Tensor) -> Tuple[List[float], float]:
    ev = torch.linalg.eigvalsh(G)
    fail = 1.0 if ev[0] <= 0 else 0.0

    # We use abs() or clamp for log/condition to avoid crashing, 
    # but the fail flag will tell the truth.
    lMin = ev[0].clamp(min=1e-30)
    lMax = ev[-1]
    l2 = ev[1] if ev.shape[0] > 1 else lMin
    cond = lMax / lMin

    prob = ev.clamp(min=1e-30)
    prob = prob / torch.sum(prob)
    entropy = -torch.sum(prob * torch.log(prob + 1e-12))
    gap = l2 / lMin

    metrics = [ev[0].item(), ev[-1].item(), cond.item(), torch.log10(cond).item(), entropy.item(), gap.item()]
    return metrics, fail

# ============================================================
# MAIN SWEEP
# ============================================================

def runStabilitySweep(outputFile: str = "stability_results.parquet"):
    torchInfo = TorchConfig.setMode("reference", verbose=True) # FP64 for stability research
    device, dtype = torchInfo["device"], torchInfo["dtype"]

    # Build Domain ONCE
    print("Initializing Global Spectral Domain (4096 samples)...")
    domain = SpectralDomain(380.0, 830.0, 4096, device=device, dtype=dtype)

    configs = buildSweepConfigs(device)
    total = configs.shape[0]
    print(f"Starting optimized sweep over {total:,} configurations...")

    allResults = []
    t0 = time.time()

    for i in range(total):
        cfg = configs[i].tolist()
        familyId, K, order, scalingId, wMin, wMax, nMin, nMax, margin = cfg

        try:
            centers = generateTopology(int(familyId), int(K), margin=margin)
            basis = GHGSFDualDomainBasis(
                domain=domain,
                centers=centers,
                numWide=int(K)//2,
                wideSigmaMin=wMin, wideSigmaMax=wMax, wideScaleType=SCALING_ID_MAP[int(scalingId)],
                narrowSigmaMin=nMin, narrowSigmaMax=nMax, narrowScaleType=SCALING_ID_MAP[int(scalingId)],
                order=int(order)
            )

            metrics = computeMetrics(basis, domain)
            allResults.append(torch.cat([configs[i], metrics]))

        except Exception:
            # Mark catastrophic failure (e.g. NaN in basis construction)
            failRow = torch.zeros(len(CONFIG_COLUMNS) + len(CORE_METRICS)*2 + 2)
            failRow[:len(CONFIG_COLUMNS)] = configs[i]
            failRow[-2:] = 1.0 # Both failed
            allResults.append(failRow)

        if i == 0 or i % 10 == 0:
            elapsed = time.time() - t0
            eta = (elapsed / (i+1)) * (total - i - 1)
            msg = f" Progress: {i}/{total} ({100*i/total:.2f}%) | ETA: {eta/60:.1f}m"
            sys.stdout.write('\r' + msg)
            sys.stdout.flush()

    print(f"\nSweep complete in {(time.time()-t0)/60:.2f} minutes.")

    # Save results
    os.makedirs("results", exist_ok=True)
    outputFile = os.path.join("results", outputFile)
    
    finalData = torch.stack(allResults).numpy()
    df = pd.DataFrame(finalData, columns=CONFIG_COLUMNS + METRIC_COLUMNS)
    df.to_parquet(outputFile, engine='pyarrow')
    print(f"Results saved to {outputFile}")

if __name__ == "__main__":
    runStabilitySweep()

"""
Torch set to REFERENCE mode (FP64) on cuda.
Initializing Global Spectral Domain (4096 samples)...
Starting optimized sweep over 10,563,696 configurations...
 Progress: 10563690/10563696 (100.00%) | ETA: 0.0m
Sweep complete in 393.17 minutes.
Results saved to results\stability_results.parquet

"""