import os
import glob
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

CORE_METRICS = [
    "lambdaMin", "lambdaMax", "condition", "logCond",
    "entropy", "eigenGap", "traceG", "stdEig"
]

METRIC_COLUMNS = []
for m in CORE_METRICS: METRIC_COLUMNS.append(f"raw_{m}")
for m in CORE_METRICS: METRIC_COLUMNS.append(f"wht_{m}")
METRIC_COLUMNS.append("rawSpdFail")
METRIC_COLUMNS.append("rescued")

SCALING_ID_MAP = {0: "constant", 1: "linear", 2: "sqrt", 3: "power"}

CHECKPOINT_INTERVAL = 50_000   # flush every N configs

# ============================================================
# CONFIGURATION BUILDER
# ============================================================

def buildSweepConfigs(device=torch.device("cpu")) -> torch.Tensor:
    families = torch.arange(7, device=device, dtype=torch.int64)
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

    combined = torch.cat([baseExp, domainExp], dim=2).reshape(-1, 8)

    margins = torch.tensor([0.0, 10.0, 20.0], device=device, dtype=torch.float64)
    M = margins.shape[0]

    combinedExp = combined.unsqueeze(1).expand(-1, M, -1)
    marginExp   = margins.view(1, M, 1).expand(combined.shape[0], -1, -1)

    configs = torch.cat([combinedExp, marginExp], dim=2).reshape(-1, 9)
    return configs

# ============================================================
# METRIC ENGINE
# ============================================================

def computeMetrics(
    basis: GHGSFDualDomainBasis,
    domain: SpectralDomain
) -> torch.Tensor:
    G_raw = basis.m_gram
    raw_metrics, raw_fail = calculateMatrixMetrics(G_raw)

    L = basis.m_chol
    LiG = torch.linalg.solve_triangular(L, G_raw, upper=False)
    G_wht = torch.linalg.solve_triangular(L, LiG.T, upper=False).T
    wht_metrics, _ = calculateMatrixMetrics(G_wht)

    fp32Limit = 1.0 / torch.finfo(torch.float32).eps
    rescued = 1.0 if (raw_fail == 0.0 and raw_metrics[2] > fp32Limit) else 0.0

    return torch.tensor(raw_metrics + wht_metrics + [raw_fail, rescued], dtype=torch.float64)

def calculateMatrixMetrics(G: torch.Tensor) -> Tuple[List[float], float]:
    ev = torch.linalg.eigvalsh(G)
    fail = 1.0 if ev[0] <= 0 else 0.0

    lMin = ev[0].clamp(min=1e-30)
    lMax = ev[-1]
    l2 = ev[1] if ev.shape[0] > 1 else lMin
    cond = lMax / lMin

    prob = ev.clamp(min=1e-30)
    prob = prob / torch.sum(prob)
    entropy = -torch.sum(prob * torch.log(prob + 1e-12))
    gap = l2 / lMin

    traceG = torch.sum(ev).item()
    stdEig = torch.std(ev).item()

    metrics = [ev[0].item(), ev[-1].item(), cond.item(), torch.log10(cond).item(), entropy.item(), gap.item(), traceG, stdEig]
    return metrics, fail

# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def _flushCheckpoint(buffer: list, columns: list, outputFile: str, chunkIdx: int) -> None:
    """Write buffer to a numbered shard and clear it in-place."""
    chunkFile = outputFile.replace(".parquet", f"_chunk{chunkIdx:04d}.parquet")
    data = torch.stack(buffer).numpy()
    pd.DataFrame(data, columns=columns).to_parquet(chunkFile, engine="pyarrow")
    print(f"\n  [checkpoint] Wrote {len(buffer):,} rows -> {os.path.basename(chunkFile)}")
    buffer.clear()


def mergeChunks(outputFile: str, columns: list) -> None:
    """Merge all _chunkNNNN.parquet shards into the final output file and delete shards."""
    pattern = outputFile.replace(".parquet", "_chunk*.parquet")
    chunks  = sorted(glob.glob(pattern))
    if not chunks:
        return
    print(f"Merging {len(chunks)} chunks into {os.path.basename(outputFile)} ...")
    df = pd.concat([pd.read_parquet(c) for c in chunks], ignore_index=True)
    df.to_parquet(outputFile, engine="pyarrow")
    for c in chunks:
        os.remove(c)
    print(f"Merge complete. Final rows: {len(df):,}")

# ============================================================
# MAIN SWEEP
# ============================================================

def runStabilitySweep(outputFile: str = "stability_results.parquet"):
    torchInfo = TorchConfig.setMode("reference", verbose=True)
    device, dtype = torchInfo["device"], torchInfo["dtype"]

    print("Initializing Global Spectral Domain (4096 samples)...")
    domain = SpectralDomain(380.0, 830.0, 4096, device=device, dtype=dtype)

    configs = buildSweepConfigs()  # keep on CPU — metrics tensor is CPU, cat must match
    total   = configs.shape[0]

    _resultsDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(_resultsDir, exist_ok=True)
    outputFile = os.path.join(_resultsDir, outputFile)
    allColumns = CONFIG_COLUMNS + METRIC_COLUMNS

    # Resume: count rows already written across existing chunk files
    existingChunks = sorted(glob.glob(outputFile.replace(".parquet", "_chunk*.parquet")))
    startRow = 0
    chunkIdx = 0
    if existingChunks:
        for c in existingChunks:
            startRow += len(pd.read_parquet(c))
        chunkIdx = len(existingChunks)
        print(f"Resuming from row {startRow:,} (found {len(existingChunks)} existing chunk(s)).")
    else:
        print(f"Starting sweep over {total:,} configurations...")

    buffer = []
    t0 = time.time()

    for i in range(startRow, total):
        cfg = configs[i].tolist()
        familyId, K, order, scalingId, wMin, wMax, nMin, nMax, margin = cfg

        try:
            centers, wideIndices = generateTopology(int(familyId), int(K), margin=margin)
            basis = GHGSFDualDomainBasis(
                domain=domain,
                centers=centers,
                wideIndices=wideIndices,
                wideSigmaMin=wMin,  wideSigmaMax=wMax,
                wideScaleType=SCALING_ID_MAP[int(scalingId)],
                narrowSigmaMin=nMin, narrowSigmaMax=nMax,
                narrowScaleType=SCALING_ID_MAP[int(scalingId)],
                order=int(order)
            )

            ev  = torch.linalg.eigvalsh(basis.m_gram)
            eps = 2.0 * torch.finfo(dtype).eps
            if ev[0].item() < eps:
                failRow = torch.zeros(len(allColumns), dtype=torch.float64)
                failRow[:len(CONFIG_COLUMNS)] = configs[i]
                failRow[-2] = 1.0
                buffer.append(failRow)
            else:
                basis.buildCholesky()
                metrics = computeMetrics(basis, domain)
                buffer.append(torch.cat([configs[i], metrics]))

        except Exception:
            failRow = torch.zeros(len(allColumns), dtype=torch.float64)
            failRow[:len(CONFIG_COLUMNS)] = configs[i]
            failRow[-2] = 1.0
            buffer.append(failRow)

        # Checkpoint flush
        if len(buffer) >= CHECKPOINT_INTERVAL:
            _flushCheckpoint(buffer, allColumns, outputFile, chunkIdx)
            chunkIdx += 1

        if (i - startRow) % 10 == 0:
            elapsed = time.time() - t0
            done    = i - startRow + 1
            eta     = (elapsed / done) * (total - i - 1)
            sys.stdout.write(f"\r Progress: {i:,}/{total:,} ({100*i/total:.2f}%) | ETA: {eta/60:.1f}m")
            sys.stdout.flush()

    # Flush remainder
    if buffer:
        _flushCheckpoint(buffer, allColumns, outputFile, chunkIdx)

    print(f"\nSweep complete in {(time.time() - t0) / 60:.2f} minutes.")
    mergeChunks(outputFile, allColumns)
    print(f"Results saved to {outputFile}")


if __name__ == "__main__":
    runStabilitySweep()