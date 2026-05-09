import pandas as pd
df = pd.read_parquet("research/stability/results/stability_results_chunk0000.parquet")
print(df[["rawSpdFail", "whtSpdFail"]].value_counts(dropna=False))
print(df[["rawSpdFail", "whtSpdFail"]].dtypes)