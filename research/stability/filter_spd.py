import pandas as pd
import os

def filterSpdSuccess(margin: int):
    """
    Categorizes successful configurations into two research sets:
    1. Rescued: Configurations that failed Raw SPD but were saved by Whitening.
    2. Naturally Stable: Configurations that were SPD safe in their Raw form.
    """
    inputFile = f"results/stability_margin_{margin}.parquet"
    
    if not os.path.exists(inputFile):
        print(f"Skipping margin {margin}: {inputFile} not found.")
        return

    print(f"\nFiltering Margin {margin} nm...")
    df = pd.read_parquet(inputFile)
    
    # 1. Rescued Set (Raw Fail, Wht Success)
    rescued = df[(df["rawSpdFail"] == 1.0) & (df["whtSpdFail"] == 0.0)].reset_index(drop=True)
    
    # 2. Naturally Stable Set (Raw Success)
    naturally_stable = df[df["rawSpdFail"] == 0.0].reset_index(drop=True)
    
    # Save Rescued
    rescuedFile = f"results/stability_margin_{margin}_rescued.parquet"
    rescued.to_parquet(rescuedFile, engine='pyarrow')
    
    # Save Naturally Stable
    stableFile = f"results/stability_margin_{margin}_stable.parquet"
    naturally_stable.to_parquet(stableFile, engine='pyarrow')
    
    print(f"  Rescued (Wht Hero) : {len(rescued):,}")
    print(f"  Naturally Stable   : {len(naturally_stable):,}")
    print(f"  Saved to '{rescuedFile}' and '{stableFile}'")

if __name__ == "__main__":
    for m in [0, 10, 20]:
        filterSpdSuccess(m)
    print("\nFiltering complete.")
