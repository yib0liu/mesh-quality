## .msh Version Convert
Convert to latest msh version (4.1)
```
gmsh -convert output.msh
``` 
## Volume
1. Download https://github.com/yib0liu/legacy-quadpy to local
2. Add library path to your sys path
	```
	import sys
	sys.path.insert(0, "/*your path*/legacy-quadpy/src/")
	```
3. Run
	```
	python volume.py
	```