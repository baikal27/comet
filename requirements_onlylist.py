with open("f_requirements.txt", 'w') as wfile:
	with open("requirements.txt", 'r') as myFile:
		pkgs = myFile.read()
		pkgs = pkgs.splitlines()

		for pkg in pkgs:
			wfile.write(f'{pkg.split("==")[0]} \n')