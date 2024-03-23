import numpy as np
import json
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

from compas.datastructures import Mesh, Network


fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

#network = Network.from_json("rhino/export/22162_graph.json")
#network = Network.from_json("rhino/export/68_graph.json")
network = Network.from_json("rhino/export/68_graph_v2.json")
#print(len(list(network.edges())))

n_nodes = len(list(network.nodes()))
print(n_nodes)
all_nodes = np.zeros((n_nodes, 3))
for node in network.nodes(data=True):
	all_nodes[node[0], :] = np.array([node[1].get("x"), node[1].get("y"), node[1].get("z")])

ax1.scatter(all_nodes[:, 0], all_nodes[:, 1], all_nodes[:, 2])


coords = all_nodes
scale_x = np.max(coords[:, 0], axis=0) - np.min(coords[:, 0], axis=0)
scale_y = np.max(coords[:, 1], axis=0) - np.min(coords[:, 1], axis=0)
scale_z = np.max(coords[:, 2], axis=0) - np.min(coords[:, 2], axis=0)
ax1.set_box_aspect(aspect = (scale_x, scale_y, scale_z))

for edge in network.edges(data=True):
	print(edge)
	edge_coords = edge[0]
	n1 = edge_coords[0]
	n2 = edge_coords[1]
	thickness = edge[1].get("data")[0]
	#print(thickness)
	#print(thickness)
	#plt.plot(np.array([all_nodes[n1, 0], all_nodes[n2, 0]]), np.array([all_nodes[n1, 1], all_nodes[n2, 1]]), np.array([all_nodes[n1, 2], all_nodes[n2, 2]]))
	plt.plot(np.array([all_nodes[n1, 0], all_nodes[n2, 0]]), np.array([all_nodes[n1, 1], all_nodes[n2, 1]]), np.array([all_nodes[n1, 2], all_nodes[n2, 2]]), linewidth=2*thickness/40.790205)
	#print(edge)

plt.show()