import pandas as pd

# Part B
df1 = pd.read_csv('Lab2 D1A.csv')
df2 = pd.read_csv('Lab2 D1B.csv')

common_cols_AB = list(set(df1.columns) & set(df2.columns))

merged_df = pd.merge(df1, df2, on=common_cols_AB, how='outer')

# Drop duplicate columns
merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

print("(D1A + D1B) Final Columns:", merged_df.columns)
print("(D1A + D1B) Final Shape:", merged_df.shape)

# Part C
df3 = pd.read_csv('Lab2 D1C.csv')

common_cols_AC = list(set(df1.columns) & set(df3.columns))

comboAC = pd.merge(df1, df3, on=common_cols_AC, how='inner')

print("(D1A + D1C) Final Columns:", comboAC.columns)
print("(D1A + D1C) Final Shape:", comboAC.shape)
