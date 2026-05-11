import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from research.engine.domain import SpectralDomain
from research.engine.config import TorchConfig
from research.phase2a.cmf import buildCmfTensor
from research.phase2a.spectrum_gen import generateAll
from research.phase2a.sweep import runSweep

STABILITY_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "stability", "results"
)
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results"
)

INPUT_PARQUETS = [
    "stability_margin_0_stable.parquet",
    "stability_margin_0_rescued.parquet",
    "stability_margin_10_stable.parquet",
    "stability_margin_10_rescued.parquet",
    "stability_margin_20_stable.parquet",
    "stability_margin_20_rescued.parquet",
]


def runPhase2AMaster():
    print("=" * 63)
    print(" SPECTRAL BASIS ENGINE — PHASE 2A MASTER")
    print(" Spectral Reconstruction Strength Sweep")
    print("=" * 63)

    torchInfo = TorchConfig.setMode("reference", verbose=True)
    device, dtype = torchInfo["device"], torchInfo["dtype"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\nInitializing domain (4096 samples, 380–830 nm)...")
    domain = SpectralDomain(380.0, 830.0, 4096, device=device, dtype=dtype)

    print("Building CMF tensor...")
    cmf = buildCmfTensor(domain.m_lambda)

    print("Generating spectrum suite...")
    bundle = generateAll(domain.m_lambda)
    SA, SB, SC = bundle["SA"], bundle["SB"], bundle["SC"]
    print(f"  Phase A: {SA}  |  Phase B: {SB}  |  Phase C: {SC}  |  Total: {SA+SB+SC}")

    t_total = time.time()

    for parquet_name in INPUT_PARQUETS:
        input_path = os.path.join(STABILITY_DIR, parquet_name)
        if not os.path.exists(input_path):
            print(f"\n[SKIP] Not found: {parquet_name}")
            continue
        runSweep(input_path, OUTPUT_DIR, bundle, cmf, domain)

    print(f"\n{'='*63}")
    print(f" PHASE 2A COMPLETE — {(time.time()-t_total)/3600:.2f} hrs total")
    print(f" Results in: {OUTPUT_DIR}")
    print(f"{'='*63}")


if __name__ == "__main__":
    runPhase2AMaster()
