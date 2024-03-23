import networkx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

adjmatrix = np.array(pd.read_csv("/Users/mnbucher/git/eth-master-thesis/rhino/export/network_adj.csv", index_col=False, header=None))

#print(adjmatrix.sum(axis=0))
# print(adjmatrix.sum(axis=1))
# print("")

# print(adjmatrix[:, 9])

#exit()


adjmatrix_symm = np.zeros(adjmatrix.shape)
for i in range(41):
	for j in range(0,i):
		if adjmatrix[i,j] == 1 or adjmatrix[j,i] == 1:
			adjmatrix_symm[i,j] = adjmatrix_symm[j,i] = 1
		else:
			adjmatrix_symm[i,j] = adjmatrix_symm[j,i] = 0

print(adjmatrix_symm.sum(axis=0))
print(adjmatrix_symm.sum(axis=1))


#graph = networkx.convert_matrix.from_numpy_array(adjmatrix)
#networkx.draw(graph)




# import compas
# from compas.datastructures import Network
# from compas_plotters import NetworkPlotter

# network = Network.from_json("/Users/mnbucher/git/eth-master-thesis/rhino/export/network.json")

# plotter = NetworkPlotter(network)
# plotter.draw_nodes(
#  text='key',
#  facecolor={key: '#ff0000' for key in network.leaves()},
#  radius=0.15
# )
# plotter.draw_edges()
# plotter.show()