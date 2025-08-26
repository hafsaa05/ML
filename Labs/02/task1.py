import pandas as pd

df1 = pd.read_csv('Lab2 D1A.csv')
df2 = pd.read_csv('Lab2 D1B.csv')

common_cols = list(set(df1.columns) & set(df2.columns))

merged_df = pd.merge(df1, df2, on=common_cols, how='outer')

#Drop Duplicate coloumns
merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

print("Final Columns:", merged_df.columns)
print("Final Shape:", merged_df.shape)
