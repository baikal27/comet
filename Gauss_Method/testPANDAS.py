import pandas as pd
import json

df = pd.read_csv('kepler_XYZ.csv', index_col='time')
print(df.head())

planets = ["Mars", "Venus"]
#selected_df = pd.DataFrame(columns=['name', 'x', 'y', 'z'])
selected_li = []

start=0
m = 10000 # data 개수
internum = 100 # 속도와 관련 10~50
for planet in planets:
	new_df = pd.DataFrame({
			'name' : df[df['name']==planet]['name'][start:internum*m:internum].values,
			'x' : df[df['name']==planet]['x'][start:internum*m:internum].values,
			'y' : df[df['name']==planet]['y'][start:internum*m:internum].values,
			'z' : df[df['name']==planet]['z'][start:internum*m:internum].values
		})
	selected_li.append(new_df)

selected_df = pd.concat(selected_li, axis=0)
print(len(selected_df))

'''
cached =  selected_df.to_json()
return_data = pd.read_json(cached)
print(return_data.tail())
'''