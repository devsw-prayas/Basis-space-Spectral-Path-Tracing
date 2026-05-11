import glob
import pandas as pd

chunks = sorted(glob.glob("research/stability/results/stability_results_chunk*.parquet"))
print(f"Found {len(chunks)} chunk files")

total = 0
spd_pass = 0
spd_fail = 0
wht_rescued = 0

for c in chunks:
    df = pd.read_parquet(c, columns=["rawSpdFail", "rescued"])
    total += len(df)
    spd_pass += int(((df["rawSpdFail"] == 0) & (df["rescued"] == 0)).sum())
    spd_fail += int((df["rawSpdFail"] == 1).sum())
    wht_rescued += int(((df["rescued"] == 1) & (df["rawSpdFail"] == 0)).sum())

print(f"\nTotal configs processed        : {total:,}")
print(f"Raw SPD pass (naturally stable): {spd_pass:,}  ({100*spd_pass/total:.2f}%)")
print(f"Raw SPD fail                   : {spd_fail:,}  ({100*spd_fail/total:.2f}%)")
if spd_fail > 0:
    print(f"Rescued whitened configurations: {wht_rescued:,}  ({100*wht_rescued/total:.2f}% of beyond fp32 precision)")
