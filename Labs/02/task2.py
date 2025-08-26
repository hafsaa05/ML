import pandas as pd

df1 = pd.read_csv('Lab2 D1A.csv')
df2 = pd.read_csv('Lab2 D1B.csv')
df3 = pd.read_csv('Lab2 D1C.csv')

population_df1 = df1[['population']]
name_df2 = df2[['name']]
city_df3 = df3[['city']]


