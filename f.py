with open("sp_requirements.txt") as wFile:
	with open("requirements.txt") as rFile:
		pkgs = rFile.read()
		pkgs = pkgs.splitlines()
		for pkg in pkgs:
			wFile.write(pkg.split('==')[0]))
