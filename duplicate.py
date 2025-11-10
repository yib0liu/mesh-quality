import numpy as np
import meshio
from collections import defaultdict
import matplotlib.pyplot as plt
from collections import Counter


for n in [1,2,4]:

	# tets, prisms, pyramids = read_msh_elements(f"data2/output{n}.msh")

	# mesh = meshio.read(f"data2/output{n}.msh")
	# print(f"data2/output{n}.msh")
	mesh = meshio.read(f"data2/output{n}.msh_new",file_format="gmsh")

	# print(mesh)
	# print(mesh.cells_dict.keys())  # np array

	# for elem in mesh.cells_dict["pyramid"]:
	# 	print(elem)

	# print(mesh.cells_dict["tetra"])  # np array

	tetfmap=[
		[0,1,2],
		[0,1,3],
		[0,2,3],
		[3,1,2]
	]
	pyramidfmap=[
		[0,1,2,3],
		[0,1,4],
		[1,2,4],
		[2,3,4],
		[0,3,4]
	]
	wedgefmap=[
		[0,1,4,3],
		[1,2,5,4],
		[2,0,3,5],
		[3,4,5],
		[0,1,2]
	]


	tface_count = defaultdict(int)
	for elem in mesh.cells_dict["tetra"]:
	# for elem in tets:
		for face in tetfmap:
			nodes = tuple(sorted(elem[face]))
			tface_count[nodes] += 1
			# print(elem,face,nodes)

	more_adj1 = [(k,v) for k,v in tface_count.items() if v > 2]
	print(more_adj1)

	pface_count = defaultdict(int)
	for elem in mesh.cells_dict["pyramid"]:
	# for elem in pyramids:
		for face in pyramidfmap:
			nodes = tuple(sorted(elem[face]))
			pface_count[nodes] += 1
			# print(elem,face,nodes)

	more_adj2 = [(k,v) for k,v in pface_count.items() if v > 2]
	print(more_adj2)


	wface_count = defaultdict(int)
	for elem in mesh.cells_dict["wedge"]:
	# for elem in prisms:
		for face in wedgefmap:
			nodes = tuple(sorted(elem[face]))
			wface_count[nodes] += 1
			# print(elem,face,nodes)

	more_adj3 = [(k,v) for k,v in wface_count.items() if v > 2]
	print(more_adj3)


	percent=100*(len(more_adj1)+len(more_adj2)+len(more_adj3)) / (len(mesh.cells_dict["wedge"])+len(mesh.cells_dict["tetra"])+len(mesh.cells_dict["pyramid"]))
	# percent=100*(len(more_adj1)+len(more_adj2)+len(more_adj3)) / (len(tets)+len(pyramids)+len(prisms))
	print(percent)


	total_face_count = Counter(tface_count) + Counter(pface_count) + Counter(wface_count)
	counts = list(total_face_count.values())

	unique, freq = np.unique(counts, return_counts=True)

	plt.figure(figsize=(6,4))
	plt.bar(unique, freq)
	plt.xlabel("Face appearance count")
	plt.ylabel("Number of faces")
	plt.title(f"Distribution of face counts, {percent:.2f}% > 2")
	plt.grid(alpha=0.3)
	plt.savefig(f"data2/percent{n}.png")
	plt.show()
