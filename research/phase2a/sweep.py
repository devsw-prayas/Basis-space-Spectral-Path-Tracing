import os
import time
import sys
import glob
import torch
import pandas as pd
import numpy as np
from typing import Dict

from research.engine.domain import SpectralDomain
from research.engine.basis import GHGSFDualDomainBasis
from research.engine.topology import generateTopology
from research.engine.config import TorchConfig
from research.phase2a.metrics import computeAllMetrics
from research.phase2a.schema import PHASE_A_COLUMNS, PHASE_B_COLUMNS, PHASE_C_COLUMNS

CHECKPOINT_INTERVAL = 50_000
SCALING_ID_MAP = {0: "constant", 1: "linear", 2: "sqrt", 3: "power"}

CONFIG_COLS = ["familyId", "K", "order", "scalingId",
               "wideMin", "wideMax", "narrowMin", "narrowMax", "margin"]


def _flushChunk(buf: list, columns: list, path: str, idx: int) -> None:
    chunk_path = path.replace(".parquet", f"_chunk{idx:04d}.parquet")
    df = pd.DataFrame(np.array(buf, dtype=np.float32), columns=columns)
    df.to_parquet(chunk_path, engine="pyarrow", compression="zstd")
    print(f"\n  [ckpt] {len(buf):,} rows → {os.path.basename(chunk_path)}")
    buf.clear()


def _mergeChunks(output_path: str, columns: list) -> None:
    pattern = output_path.replace(".parquet", "_chunk*.parquet")
    chunks  = sorted(glob.glob(pattern))
    if not chunks:
        return
    df = pd.concat([pd.read_parquet(c) for c in chunks], ignore_index=True)
    df.to_parquet(output_path, engine="pyarrow", compression="zstd")
    for c in chunks:
        os.remove(c)
    print(f"  Merged → {os.path.basename(output_path)}  ({len(df):,} rows)")


def _buildBasis(cfg: dict, domain: SpectralDomain) -> GHGSFDualDomainBasis:
    centers, wideIndices = generateTopology(
        int(cfg["familyId"]), int(cfg["K"]), margin=float(cfg["margin"])
    )
    basis = GHGSFDualDomainBasis(
        domain         = domain,
        centers        = centers,
        wideIndices    = wideIndices,
        wideSigmaMin   = float(cfg["wideMin"]),
        wideSigmaMax   = float(cfg["wideMax"]),
        wideScaleType  = SCALING_ID_MAP[int(cfg["scalingId"])],
        narrowSigmaMin = float(cfg["narrowMin"]),
        narrowSigmaMax = float(cfg["narrowMax"]),
        narrowScaleType= SCALING_ID_MAP[int(cfg["scalingId"])],
        order          = int(cfg["order"]),
    )
    basis.buildCholesky()
    return basis


def runSweep(
    inputParquet: str,
    outputDir:    str,
    bundle:       Dict,
    cmf:          "torch.Tensor",
    domain:       SpectralDomain,
) -> None:
    """
    Sweep one Phase 1 filtered parquet → 3 output parquets (A, B, C).

    bundle keys: spectra [S,L], phaseA_mask/B/C [S], centers [S], sigmas [S], SA, SB, SC
    """
    tag = os.path.splitext(os.path.basename(inputParquet))[0]  # e.g. stability_margin_0_stable
    tag = tag.replace("stability_", "phase2a_")

    out_A = os.path.join(outputDir, f"{tag}_A.parquet")
    out_B = os.path.join(outputDir, f"{tag}_B.parquet")
    out_C = os.path.join(outputDir, f"{tag}_C.parquet")

    # Resume: count rows already written in chunks
    def _count_existing(path):
        chunks = sorted(glob.glob(path.replace(".parquet", "_chunk*.parquet")))
        if not chunks:
            return 0, len(chunks)
        n = sum(len(pd.read_parquet(c)) for c in chunks)
        return n, len(chunks)

    SA, SB, SC = bundle["SA"], bundle["SB"], bundle["SC"]
    # Each config produces SA+SB+SC rows total; chunks store all phases interleaved per config
    # We track progress by configs completed = rows_A / SA
    existing_rows_A, n_chunks = _count_existing(out_A)
    start_config = existing_rows_A // SA if SA > 0 else 0
    chunk_idx_A  = n_chunks
    chunk_idx_B  = len(sorted(glob.glob(out_B.replace(".parquet", "_chunk*.parquet"))))
    chunk_idx_C  = len(sorted(glob.glob(out_C.replace(".parquet", "_chunk*.parquet"))))

    df_configs = pd.read_parquet(inputParquet)
    total      = len(df_configs)

    if start_config >= total:
        print(f"  {tag}: already complete ({total:,} configs). Skipping.")
        return

    print(f"\n{'='*60}")
    print(f"  Sweeping: {os.path.basename(inputParquet)}")
    print(f"  Configs: {total:,}  |  Resuming from: {start_config:,}")
    print(f"  Spectra: A={SA}  B={SB}  C={SC}  Total={SA+SB+SC}")
    print(f"{'='*60}")

    device = domain.m_device
    dtype  = domain.m_dtype

    F        = bundle["spectra"].to(device=device, dtype=dtype)    # [S, L]
    maskB    = bundle["phaseB_mask"].to(device=device)
    maskA    = bundle["phaseA_mask"].to(device=device)
    maskC    = bundle["phaseC_mask"].to(device=device)
    centers  = bundle["centers"].to(device=device, dtype=dtype)
    sigmas   = bundle["sigmas"].to(device=device, dtype=dtype)
    lbda     = domain.m_lambda
    w        = domain.m_weights

    spectrum_ids = list(range(SA + SB + SC))
    spec_id_A    = torch.tensor(spectrum_ids[:SA],        dtype=torch.int32, device=device)
    spec_id_B    = torch.tensor(spectrum_ids[SA:SA+SB],   dtype=torch.int32, device=device)
    spec_id_C    = torch.tensor(spectrum_ids[SA+SB:],     dtype=torch.int32, device=device)

    buf_A, buf_B, buf_C = [], [], []
    t0 = time.time()

    for i in range(start_config, total):
        cfg = df_configs.iloc[i]
        config_id = i

        try:
            basis = _buildBasis(cfg, domain)

            B_mat = basis.m_basisRaw       # [M, L]
            L_chol= basis.m_chol           # [M, M]
            B_wht = basis.m_basisWhitened  # [M, L]

            # ── One batched solve for ALL spectra ─────────────────────────
            b_all   = (B_mat * w) @ F.T                                          # [M, S]
            alpha_w = torch.linalg.solve_triangular(L_chol, b_all, upper=False)  # [M, S]
            f_hat   = alpha_w.T @ B_wht                                          # [S, L]

            # ── Metrics (fully vectorized over S) ─────────────────────────
            m = computeAllMetrics(F, f_hat, w, cmf, lbda, maskB, centers, sigmas)

            # ── Pack rows per phase ───────────────────────────────────────
            cid = torch.tensor([config_id], dtype=torch.float32, device=device)

            def _pack(mask, spec_ids, extras=None):
                rows = []
                idxs = mask.nonzero(as_tuple=True)[0]
                for j, si in zip(idxs, spec_ids):
                    row = [float(config_id), float(si.item()),
                           m["l2"][j].item(), m["nrmse"][j].item(),
                           m["maxError"][j].item(), m["xyzDelta"][j].item(),
                           m["perceptualDeltaE"][j].item()]
                    if extras is not None:
                        row += [m[k][j].item() for k in extras]
                    rows.append(row)
                return rows

            buf_A += _pack(maskA, spec_id_A)
            buf_B += _pack(maskB, spec_id_B,
                           ["energyRetention", "amplitudeAccuracy", "peakShiftNm", "sideLobeEnergy"])
            buf_C += _pack(maskC, spec_id_C)

        except Exception:
            # Failed configs: write NaN rows so indices stay aligned
            for si in range(SA):
                buf_A.append([float(i), float(si)] + [float("nan")] * 5)
            for si in range(SB):
                buf_B.append([float(i), float(SA + si)] + [float("nan")] * 9)
            for si in range(SC):
                buf_C.append([float(i), float(SA + SB + si)] + [float("nan")] * 5)

        # ── Checkpoint ────────────────────────────────────────────────────
        completed = i - start_config + 1
        if completed % CHECKPOINT_INTERVAL == 0:
            _flushChunk(buf_A, PHASE_A_COLUMNS, out_A, chunk_idx_A); chunk_idx_A += 1
            _flushChunk(buf_B, PHASE_B_COLUMNS, out_B, chunk_idx_B); chunk_idx_B += 1
            _flushChunk(buf_C, PHASE_C_COLUMNS, out_C, chunk_idx_C); chunk_idx_C += 1

        # ── Progress ──────────────────────────────────────────────────────
        if completed % 10 == 0:
            elapsed = time.time() - t0
            eta     = (elapsed / completed) * (total - i - 1)
            sys.stdout.write(
                f"\r  [{tag}] {i+1:,}/{total:,} ({100*(i+1)/total:.1f}%) | ETA {eta/60:.1f}m"
            )
            sys.stdout.flush()

    # ── Final flush ───────────────────────────────────────────────────────
    if buf_A: _flushChunk(buf_A, PHASE_A_COLUMNS, out_A, chunk_idx_A)
    if buf_B: _flushChunk(buf_B, PHASE_B_COLUMNS, out_B, chunk_idx_B)
    if buf_C: _flushChunk(buf_C, PHASE_C_COLUMNS, out_C, chunk_idx_C)

    print(f"\n  Sweep complete in {(time.time()-t0)/60:.2f} min.")

    _mergeChunks(out_A, PHASE_A_COLUMNS)
    _mergeChunks(out_B, PHASE_B_COLUMNS)
    _mergeChunks(out_C, PHASE_C_COLUMNS)
