import pandas as pd
df_line = pd.read_csv('kepler_XYZ_line.csv')
df1 = df_line[df_line['name'] == 'Mercury']
df2 = df_line[df_line['name'] == 'Venus']
df = pd.read_csv('kepler_XYZ.csv', index_col='time')
planets = df_line.drop_duplicates("name", keep='first')["name"].values
print(planets)