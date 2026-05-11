"""
One-time migration: replace whtSpdFail with rescued across all parquets in results/.

rescued = 1  iff  rawSpdFail == 0  AND  raw_condition > fp32_limit
"""

import glob
import os
import numpy as np
import pandas as pd

FP32_LIMIT = 1.0 / np.finfo(np.float32).eps   # ~1.677e7

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

def migrateFile(path: str) -> bool:
    df = pd.read_parquet(path)
    if "whtSpdFail" not in df.columns:
        return False

    df["rescued"] = ((df["rawSpdFail"] == 0.0) & (df["raw_condition"] > FP32_LIMIT)).astype(float)
    df = df.drop(columns=["whtSpdFail"])
    df.to_parquet(path, engine="pyarrow")
    return True


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.parquet")))
    print(f"Found {len(files)} parquet file(s) in {RESULTS_DIR}")

    updated = 0
    for i, f in enumerate(files):
        changed = migrateFile(f)
        if changed:
            updated += 1
        if (i + 1) % 20 == 0 or (i + 1) == len(files):
            print(f"  {i + 1}/{len(files)} processed ({updated} updated)")

    print(f"\nDone. {updated}/{len(files)} files migrated.")
