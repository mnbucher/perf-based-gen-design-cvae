import numpy as np
import json
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

from compas.datastructures import Mesh


fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
#ax2 = fig.add_subplot(122, projection='3d')

#ax1.plot(np.array([1,2]), np.array([1,2]), np.array([1,2]))
#plt.show()
#exit()

#mesh = Mesh.from_json("rhino/export/mesh.json")
mesh = Mesh.from_json("rhino/export/68_mesh.json")

nodes = list(mesh.vertices(data=True))
nodes_arr = torch.zeros((len(nodes),3), dtype=torch.float64)
for idx, node in enumerate(nodes):
	nodes_arr[idx, :] = torch.tensor([node[1].get("x"), node[1].get("y"), node[1].get("z")])


coords = nodes_arr.numpy()
scale_x = np.max(coords[:, 0], axis=0) - np.min(coords[:, 0], axis=0)
scale_y = np.max(coords[:, 1], axis=0) - np.min(coords[:, 1], axis=0)
scale_z = np.max(coords[:, 2], axis=0) - np.min(coords[:, 2], axis=0)
scale_max = np.max([scale_x, scale_y, scale_z])
ax1.set_box_aspect(aspect = (scale_x, scale_y, scale_z))

edges = list(mesh.edges())
edge_arr = torch.zeros((2,len(edges)), dtype=torch.long)
for idx, edge in enumerate(edges):
	edge_arr[0, idx] = edge[0]
	edge_arr[1, idx] = edge[1]



ax1.scatter(nodes_arr[:, 0], nodes_arr[:, 1], nodes_arr[:, 2])

for idx in range(len(edges)):
	indices = edge_arr[:, idx]
	node_1 = nodes_arr[indices[0], :]
	node_2 = nodes_arr[indices[1], :]
	ax1.plot(np.array([node_1[0], node_2[0]]), np.array([node_1[1], node_2[1]]), np.array([node_1[2], node_2[2]]))
	#break

plt.show()

exit()

with open("rhino/export/mesh.json") as file:
	data = json.load(file)

	vertices = data["vertex"]

	coords = np.zeros((len(vertices),3))
	for i in range(len(vertices)):
		x,y,z = vertices[str(i)].get('x'), vertices[str(i)].get('y'), vertices[str(i)].get('z')
		coords[i, :] = np.array([x,y,z])

	scale_x = np.max(coords[:, 0], axis=0) - np.min(coords[:, 0], axis=0)
	scale_y = np.max(coords[:, 1], axis=0) - np.min(coords[:, 1], axis=0)
	scale_z = np.max(coords[:, 2], axis=0) - np.min(coords[:, 2], axis=0)
	scale_max = np.max([scale_x, scale_y, scale_z])

	print(scale_x/scale_max, scale_y/scale_max, scale_z/scale_max)

	#ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2])
	ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2])
	#ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([scale_x, scale_y, scale_z, 1]))
	ax1.set_box_aspect(aspect = (scale_x, scale_y, scale_z))
	#ax.view_init(2,None)

	plt.show()


	# with open("rhino/export/mesh_weld.json") as file:
	# 	data = json.load(file)

	# 	vertices = data["vertex"]

	# 	coords = np.zeros((len(vertices),3))
	# 	for i in range(len(vertices)):
	# 		x,y,z = vertices[str(i)].get('x'), vertices[str(i)].get('y'), vertices[str(i)].get('z')
	# 		coords[i, :] = np.array([x,y,z])

	# 	scale_x = np.max(coords[:, 0], axis=0) - np.min(coords[:, 0], axis=0)
	# 	scale_y = np.max(coords[:, 1], axis=0) - np.min(coords[:, 1], axis=0)
	# 	scale_z = np.max(coords[:, 2], axis=0) - np.min(coords[:, 2], axis=0)
	# 	scale_max = np.max([scale_x, scale_y, scale_z])

	# 	print(scale_x/scale_max, scale_y/scale_max, scale_z/scale_max)

	# 	#ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2])
	# 	ax2.scatter(coords[:, 0], coords[:, 1], coords[:, 2])


		# plt.show()