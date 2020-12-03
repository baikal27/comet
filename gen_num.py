import csv
import time

fieldnames = ['time', 'x', 'y', 'z']
tt, x, y, z = 0, 0, 0, 0

with open('data.csv', 'w') as csv_file:
	csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
	csv_writer.writeheader()

while True:
	with open('data.csv', 'a') as csv_file:
		csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
		info = {
			'time':tt,
			'x': x,
			'y': y,
			'z': z
		}

		csv_writer.writerow(info)
		tt += 1
		x = x + 0.1
		y = y + 0.2
		z = z + 0.3
		print(tt, x, y, z)
	time.sleep(0.5)