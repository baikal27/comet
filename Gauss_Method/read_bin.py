from pyarrow import csv

pf = csv.read_csv('kepler_XYZ.csv').to_pandas()
print(pf)