import os
import sys
import time

# Add the current directory to sys.path to ensure local imports work
sys.path.append(os.getcwd())

from research.stability.sweep import runStabilitySweep
from research.stability.split_results import splitStabilityResults
from research.stability.filter_spd import filterSpdSuccess
from research.stability.plot_heatmaps import plotStabilityHeatmaps
from research.stability.analyze_golden_zone import analyzeGoldenZone

def runPhase1Master():
    print("=" * 63)
    print(" SPECTRAL BASIS ENGINE - PHASE 1 MASTER RUNNER")
    print("=" * 63)

    results_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "research", "stability", "results", "stability_results.parquet")

    # Stage 1: Sweep (Data Generation)
    if not os.path.exists(results_file):
        print("\n[STAGE 1] Starting Stability Sweep (Data Generation)...")
        runStabilitySweep()
    else:
        print("\n[STAGE 1] Results already exist. Skipping sweep.")

    # Stage 2: Splitting
    print("\n[STAGE 2] Splitting results by margin...")
    splitStabilityResults()

    # Stage 3: Filtering
    print("\n[STAGE 3] Filtering for Rescued and Naturally Stable sets...")
    for m in [0, 10, 20]:
        filterSpdSuccess(m)

    # Stage 4: Visualization
    print("\n[STAGE 4] Generating stability heatmaps...")
    for m in [0, 10, 20]:
        plotStabilityHeatmaps(m)

    # Stage 5: Analysis
    print("\n[STAGE 5] Identifying the Golden Zone candidates...")
    for m in [0, 10, 20]:
        analyzeGoldenZone(m)

    print("\n" + "=" * 63)
    print(" PHASE 1 COMPLETE: All research artifacts generated.")
    print("=" * 63)

if __name__ == "__main__":
    runPhase1Master()
