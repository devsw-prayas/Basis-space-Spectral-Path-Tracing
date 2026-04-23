import pandas as pd
import os

def analyzeGoldenZone(margin: int = 20):
    """
    Two-pass analysis to find:
    1. Naturally Stable Resolutions (10^4 to 10^15)
    2. Whitened Extreme Resolutions (10^4 to 10^15)
    """
    results_dir = "results"
    
    stableFile = os.path.join(results_dir, f"stability_margin_{margin}_stable.parquet")
    rescuedFile = os.path.join(results_dir, f"stability_margin_{margin}_rescued.parquet")

    print(f"\n" + "="*80)
    print(f" GOLDEN ZONE ANALYSIS (MARGIN {margin}nm) ")
    print("="*80)

    # Pass 1: Naturally Stable
    if os.path.exists(stableFile):
        df_stable = pd.read_parquet(stableFile)
        # Filter for 10^4 (4.0) up to limit (15.0)
        stable_candidates = df_stable[
            (df_stable["raw_logCond"] >= 4.0) & (df_stable["raw_logCond"] <= 15.0)
        ].sort_values(by="wht_entropy", ascending=False)
        
        print(f"\n[PASS 1] Top Naturally Stable Candidates (Count: {len(stable_candidates):,})")
        print(stable_candidates[["familyId", "K", "order", "raw_logCond", "wht_entropy"]].head(10).to_string(index=False))
    else:
        print(f"\n[PASS 1] Skipping: {stableFile} not found.")

    # Pass 2: Whitened Extreme
    if os.path.exists(rescuedFile):
        df_rescued = pd.read_parquet(rescuedFile)
        # Filter for 10^4 (4.0) up to limit (15.0) for the WHITENED state
        extreme_candidates = df_rescued[
            (df_rescued["wht_logCond"] >= 4.0) & (df_rescued["wht_logCond"] <= 15.0)
        ].sort_values(by="wht_entropy", ascending=False)
        
        print(f"\n[PASS 2] Top Whitened Extreme Candidates (Count: {len(extreme_candidates):,})")
        print(extreme_candidates[["familyId", "K", "order", "wht_logCond", "wht_entropy"]].head(10).to_string(index=False))
    else:
        print(f"\n[PASS 2] Skipping: {rescuedFile} not found.")

if __name__ == "__main__":
    for m in [0, 10, 20]:
        analyzeGoldenZone(m)
