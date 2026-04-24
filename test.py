import pandas as pd
df = pd.read_parquet("results/stability_margin_0.parquet")
print(df[["rawSpdFail", "whtSpdFail"]].value_counts(dropna=False))
print(df[["rawSpdFail", "whtSpdFail"]].dtypes)