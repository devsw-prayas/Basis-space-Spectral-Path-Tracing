import pandas as pd
import os

def splitStabilityResults(inputFile: str = "results/stability_results.parquet"):
    """
    Splits the main stability dataset into separate parts based on the margin.
    Creates:
        - results/stability_margin_0.parquet
        - results/stability_margin_10.parquet
        - results/stability_margin_20.parquet
    """
    if not os.path.exists(inputFile):
        print(f"Error: Could not find {inputFile}")
        return

    print(f"Loading dataset: {inputFile}")
    df = pd.read_parquet(inputFile)
    
    total_rows = len(df)
    print(f"Total configurations: {total_rows:,}")

    margins = df['margin'].unique()
    print(f"Detected margins: {margins.tolist()}")

    for m in margins:
        print(f"\nProcessing margin: {m} nm...")
        subset = df[df['margin'] == m].reset_index(drop=True)
        
        outputFile = f"results/stability_margin_{int(m)}.parquet"
        subset.to_parquet(outputFile, engine='pyarrow')
        
        print(f"  Rows      : {len(subset):,}")
        print(f"  Saved to  : {outputFile}")

    print("\nSplitting complete.")

if __name__ == "__main__":
    splitStabilityResults()
