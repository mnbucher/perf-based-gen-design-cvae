import compas
from compas.datastructures import Network
import numpy as np
import matplotlib.pyplot as plt

network = Network.from_json("./export/32768_adj.json")

print(network)

#print(network.adjacency_matrix())

#nodes = np.array(network.to_nodes_and_edges()[0])

edges = np.array(network.to_nodes_and_edges()[1])

print(edges.T.shape)

#print(network.network_connectivity_matrix(rtype='coo'))

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(edges[:, 0], edges[:, 1], edges[:, 2])
# plt.show()