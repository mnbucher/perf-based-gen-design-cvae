import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


adjmatrix = np.array(pd.read_csv("/Users/mnbucher/git/eth-master-thesis/rhino/export/network_adj.csv", index_col=False, header=None))
#print(np.sum(adjmatrix, axis=0))

print(np.allclose(adjmatrix, adjmatrix.T))

fig, axes = plt.subplots(1,2)

bw_map = np.zeros((41, 41, 3))
bw_map[adjmatrix != 0] = [1,1,1]
axes[0].imshow(bw_map)

adjmatrix_symm = np.zeros(adjmatrix.shape)
for i in range(41):
    for j in range(0,i):
        if adjmatrix[i,j] == 1 or adjmatrix[j,i] == 1:
            adjmatrix_symm[i,j] = adjmatrix_symm[j,i] = 1
        else:
            adjmatrix_symm[i,j] = adjmatrix_symm[j,i] = 0

bw_map_symm = np.zeros((41, 41, 3))
bw_map_symm[adjmatrix_symm != 0] = [1,1,1]
axes[1].imshow(bw_map_symm)

plt.show()

exit()



G = nx.convert_matrix.from_numpy_array(adjmatrix)
# G = nx.convert_matrix.from_numpy_array(adjmatrix_symm)

# 3d spring layout
pos = nx.spring_layout(G, dim=3, seed=779)
# Extract node and edge positions from the layout
node_xyz = np.array([pos[v] for v in sorted(G)])
edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])

# Create the 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot the nodes - alpha is scaled by "depth" automatically
ax.scatter(*node_xyz.T, s=100, ec="w")

# Plot the edges
for vizedge in edge_xyz:
    ax.plot(*vizedge.T, color="tab:gray")


def _format_axes(ax):
    """Visualization options for the 3D axes."""
    # Turn gridlines off
    ax.grid(False)
    # Suppress tick labels
    for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
        dim.set_ticks([])
    # Set axes labels
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


_format_axes(ax)
fig.tight_layout()
plt.show()
