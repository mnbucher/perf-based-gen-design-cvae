import numpy as np
import pandas as pd
import torch
import os
import shutil
import logging
import pdb

import smogn
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
import compas
from compas.datastructures import Network, Mesh
from sklearn.preprocessing import OneHotEncoder

from torch_geometric.data import Data as TGData
from torch_geometric.loader import DataLoader as TGDataLoader

import sys
sys.path.append(os.getcwd())

import src.utility as utility
import src.plot as plot


feature_dim_names = [
	"Span [m]", 					#0
	"Rise of Arch [m]", 			#1
	"Width [m]",					#2
	"Hanger Spacing [m]",			#3
	"Alpha0 [°]",					#4
	"Tie and Arch CS-Depth [m]",	#5
	"Hanger CS-Area [m^2]",			#6
	"h/b Box CS [-]",				#7
	"tw/h Box [-]",					#8
	"tf/b Box [-]"					#9
]

label_dim_names = [
	"Material Cost (Thousand CHF)",
    "Maximum Utilisation", # < 1.0
    "Utilisation Loss" # IGNORE atm ... 1*...(1-a)
]


def normalize_data(data, x_min, x_max):
	return (data - x_min) / (x_max - x_min)


def unnormalize_data(data, x_min, x_max, dim_config=None):
	return (data * (x_max - x_min)) + x_min


def compute_label_freqs_and_store_to_file(dataset_target, y_all):
	y_min = np.min(y_all, axis=0)
	y_max = np.max(y_all, axis=0)
	y_normalized = normalize_data(y_all, y_min, y_max)

	for dim in range(2):

		y_dimwise = y_normalized[:, dim]

		hist, bin_edges = np.histogram(y_dimwise, bins='auto')
		print(f"[dim {dim+1}] — # of bins:", len(hist))

		# add some shift to first and last bin
		bin_edges[0] -= 10**-6
		bin_edges[-1] += 10**-6

		torch.save(torch.Tensor(bin_edges), f"data/stats/{dataset_target}/y-bin-edges-dim-{dim+1}.pt",)


def read_output_array_from_files(dataset_root, n_total=32768):

	y_all = np.zeros((n_total, 3), dtype=np.float64)

	for i in tqdm(range(n_total)):
		row = pd.read_csv(f'./data/y/{dataset_root}/{i}_output.csv', header=None, dtype=np.float64).values
		y_all[i, :] = row

	df = pd.DataFrame(y_all, columns=[0, 1, 2])

	np.savetxt(f'./data/y/output-{dataset_root}.csv', np.array(df), delimiter=',')

	return df


def generate_robustness_test_dataset():

	#df_y = read_output_array_from_files("10d-robustness-test", n_total=8192)

	df_x = pd.read_csv(f'./data/x/x-10d-robustness-test-10d-8192.csv', header=None, dtype=np.float64)
	df_y = pd.read_csv(f'./data/y/output-10d-robustness-test.csv', header=None, dtype=np.float64)

	print(df_y.shape)

	df_y = df_y[df_y[0] != 0.0]
	print(df_y.shape)

	df_y = df_y[df_y[1] != 0.0]
	print(df_y.shape)

	df_y = df_y[df_y[1] < 1.0]
	print(df_y.shape)

	print(df_x.shape)
	idxs_y = df_y.index.to_numpy()
	df_x = df_x.iloc[idxs_y]
	print(df_x.shape)

	np.savetxt(f'./data/y/output-10d-robustness-test-filtered.csv', np.array(df_y), delimiter=',')
	np.savetxt(f'./data/x/x-10d-robustness-test-filtered.csv', np.array(df_x), delimiter=',')


def generate_train_test_split_and_stats(dataset_root, dataset_target):

	# initial data
	df_x = pd.read_csv(f'./data/x/sobol-samples-{dataset_root}.csv', header=None, dtype=np.float64)

	df_y = pd.read_csv(f'./data/y/output-{dataset_root}.csv', header=None, dtype=np.float64)
	#df_y = read_output_array_from_files(dataset_root)

	# filter out INVALID samples
	do_prior_filter = True
	if do_prior_filter:
		print(df_y.shape)
		df_y = df_y[df_y[0] != 0.0]
		print(df_y.shape)
		df_y = df_y[df_y[1] != 0.0] # 30276
		print(df_y.shape)

	# before ANY filtering
	# _, axes = plt.subplots(2,2)
	# axes[0, 0].hist(np.array(df_y)[:, 0], 50, color="#26408B", log=True)
	# axes[0, 1].hist(np.array(df_y)[:, 1], 50, color="#26408B", log=True)
	# plt.show()

	# compare 3d space of valid and invalid samples
	# df_y_1 = df_y[df_y[1] >= 2.0]
	# df_y_1_idxs = df_y_1.index.to_numpy()
	# df_x_1 = df_x.iloc[df_y_1_idxs]
	# df_y_2 = df_y[df_y[1] < 2.0]
	# df_y_2_idxs = df_y_2.index.to_numpy()
	# df_x_2 = df_x.iloc[df_y_2_idxs]
	# plot.plot_data_pca_3d(df_x_1, df_x_2)

	if dataset_target == "10d-robust":
		# 10d-robust: filter out extreme outliers (unlikely valid bridges)
		util_threshold = 2.0
		df_y = df_y[df_y[1] < util_threshold] # 30276
		print(df_y.shape)

		# _, axes = plt.subplots(2,2)
		# axes[1, 0].hist(np.array(df_y)[:, 0], 50, log=True, color="#3D60A7")
		# axes[1, 1].hist(np.array(df_y)[:, 1], 50, log=True, color="#3D60A7")
		# plt.show()

	# 10d-simpile:
	elif dataset_target == "10d-simple":
		# filter out with cost > 30000 (total: 280)
		df_y = df_y[df_y[0] < 30000]
		print(df_y.shape)
		# # filter out heavy ouliers with utli == 0.0 or util > 200%
		df_y = df_y[df_y[1] <= 1.0]
		print(df_y.shape)

		# axes[1, 0].hist(np.array(df_y)[:, 0], 50, log=True) #, log=True)
		# axes[1, 1].hist(np.array(df_y)[:, 1], 50, log=True)
		# plt.show()

	# get indexes of y for x and g
	idxs_y = df_y.index.to_numpy()

	# select relevant x
	df_x = df_x.iloc[idxs_y]

	data_x = df_x.to_numpy()
	data_y = df_y.to_numpy()

	x_train, x_test, y_train, y_test, indices_train, indices_test = train_test_split(data_x, data_y, list(idxs_y), test_size=0.1)
	print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

	# save x / y
	np.savetxt(f'./data/train-test-split/train/x/x-{dataset_target}.csv', x_train, delimiter=',')
	np.savetxt(f'./data/train-test-split/train/y/y-{dataset_target}.csv', y_train, delimiter=',')

	np.savetxt(f'./data/train-test-split/test/x/x-{dataset_target}.csv', x_test, delimiter=',')
	np.savetxt(f'./data/train-test-split/test/y/y-{dataset_target}.csv', y_test, delimiter=',')

	os.makedirs(f"./data/stats/{dataset_target}", exist_ok=True)
	torch.save(torch.Tensor(np.min(data_x, axis=0)), f"./data/stats/{dataset_target}/x-stats-min.pt")
	torch.save(torch.Tensor(np.max(data_x, axis=0)), f"./data/stats/{dataset_target}/x-stats-max.pt")
	torch.save(torch.Tensor(np.min(data_y, axis=0)), f"./data/stats/{dataset_target}/y-stats-min.pt")
	torch.save(torch.Tensor(np.max(data_y, axis=0)), f"./data/stats/{dataset_target}/y-stats-max.pt")

	compute_label_freqs_and_store_to_file(dataset_target, data_y)

	# select and save g
	select_compas_data(dataset_root, dataset_target, indices_train, indices_test)

	return x_train, x_test, y_train, y_test


def select_compas_graph(fn_parent, idx, cnt, all_g_graph, all_g_graph_data_stats):
	fn = f"{fn_parent}-graph/{idx}_graph.json"

	if os.path.isfile(fn):
		network = Network.from_json(fn)

		n_nodes = len(list(network.nodes()))
		all_nodes = torch.zeros((n_nodes, 3), dtype=torch.float64)
		for node in network.nodes(data=True):
			all_nodes[node[0], :] = torch.tensor([node[1].get("x"), node[1].get("y"), node[1].get("z")])

		n_edges = len(list(network.edges()))
		all_edges = torch.zeros((2, n_edges), dtype=torch.long)
		all_edge_data = torch.zeros((n_edges, 5), dtype=torch.float64)
		for idx, edge in enumerate(network.edges(data=True)):
			# 1 x 3
			all_edges[:, idx] = torch.tensor([edge[0][0], edge[0][1]])
			# 1 x 5
			# height
			# max-width
			# a
			# iyy
			# izz
			all_edge_data[idx, :] = torch.tensor(edge[1].get("data"))

		data = (all_nodes, all_edges, all_edge_data)
		all_g_graph.append(data)

		all_g_graph_data_stats[0].append(all_nodes)
		all_g_graph_data_stats[1].append(all_edge_data[:, 0])
		all_g_graph_data_stats[2].append(all_edge_data[:, 1])
		all_g_graph_data_stats[3].append(all_edge_data[:, 2])
		all_g_graph_data_stats[4].append(all_edge_data[:, 3])
		all_g_graph_data_stats[5].append(all_edge_data[:, 4])

	else:
		cnt += 1

	return cnt, all_g_graph, all_g_graph_data_stats


def select_compas_mesh(fn_parent, idx, cnt, all_g_mesh):
	fn = f"{fn_parent}-mesh/{idx}_mesh.json"

	if os.path.isfile(fn):
		mesh = Mesh.from_json(fn)

		n_nodes = len(list(mesh.nodes()))
		all_nodes = np.zeros((n_nodes, 3), dtype=torch.float64)
		for node in mesh.nodes(data=True):
			all_nodes[node[0], :] = np.array([node[1].get("x"), node[1].get("y"), node[1].get("z")])

		n_edges = len(list(mesh.edges()))
		all_edges = np.zeros((2, n_edges), dtype=torch.long)
		for idx, edge in enumerate(mesh.edges(data=True)):
			all_edges[:, idx] = np.array([edge[0][0], edge[0][1]])

		data = (torch.tensor(all_nodes), torch.tensor(all_edges))
		all_g_mesh.append(data)
	else:
		cnt += 1

	return cnt, all_g_mesh


def select_compas_data(dataset_root, dataset_target, indices_train, indices_test):

	cnt_invalid_graph = 0

	all_g_graph_train = []
	all_g_graph_test = []

	all_g_graph_data_stats = [ [], [], [], [], [], [] ]

	#cnt_invalid_mesh = 0
	#all_g_mesh = []

	fn_parent = f"./data/g/{dataset_root}"

	for idx in tqdm(indices_train):
		cnt_invalid_graph, all_g_graph_train, all_g_graph_data_stats = select_compas_graph(fn_parent, idx, cnt_invalid_graph, all_g_graph_train, all_g_graph_data_stats)
		#cnt_invalid_mesh, all_g_mesh = select_compas_mesh(fn_parent, idx, cnt_invalid_mesh, all_g_mesh)

	for idx in tqdm(indices_test):
		cnt_invalid_graph, all_g_graph_test, all_g_graph_data_stats = select_compas_graph(fn_parent, idx, cnt_invalid_graph, all_g_graph_test, all_g_graph_data_stats)

	all_g_graph_data_stats = {
		"coords_min_max": (np.min(np.concatenate(all_g_graph_data_stats[0], axis=0)), np.max(np.concatenate(all_g_graph_data_stats[0], axis=0))),
		"height_min_max": (np.min(np.concatenate(all_g_graph_data_stats[1], axis=0)), np.max(np.concatenate(all_g_graph_data_stats[1]), axis=0)),
		"width_min_max": (np.min(np.concatenate(all_g_graph_data_stats[2], axis=0)), np.max(np.concatenate(all_g_graph_data_stats[2]), axis=0)),
		"a_min_max": (np.min(np.concatenate(all_g_graph_data_stats[3], axis=0)), np.max(np.concatenate(all_g_graph_data_stats[3]), axis=0)),
		"iyy_min_max": (np.min(np.concatenate(all_g_graph_data_stats[4], axis=0)), np.max(np.concatenate(all_g_graph_data_stats[4]), axis=0)),
		"izz_min_max": (np.min(np.concatenate(all_g_graph_data_stats[5], axis=0)), np.max(np.concatenate(all_g_graph_data_stats[5]), axis=0))
	}

	print(all_g_graph_data_stats)

	torch.save(all_g_graph_train, f"./data/train-test-split/train/g/g-{dataset_target}-graph.pt")
	torch.save(all_g_graph_test, f"./data/train-test-split/test/g/g-{dataset_target}-graph.pt")
	torch.save(all_g_graph_data_stats, f"./data/stats/{dataset_target}/g-graph-stats.pt")
	#print(torch.load(f"./data/stats/{dataset_target}/g-graph-stats.pt"))

	#torch.save(all_g_graph, f"./data/train-test-split/{mode}/g/g-{dataset_target}-mesh.pt")


def compute_adj_matrix_from_edges(n, edges, add_identity=False):
	A = torch.zeros((n, n))

	A[edges[0, :].T, edges[1, :].T] = 1
	A[edges[1, :].T, edges[0, :].T] = 1

	if add_identity:
		A = A + torch.eye(n)

	return A


def init_bridge_dataset(dataset_root, dataset_target):
	generate_train_test_split_and_stats(dataset_root, dataset_target)


def init_wall_dataset(dataset_target):

	all_x = np.array(pd.read_csv(f'./data/x/x-{dataset_target}.csv', header=None, dtype=np.float64))
	all_y = np.array(pd.read_csv(f'./data/y/y-{dataset_target}.csv', header=None, dtype=np.float64))

	x_train, x_test, y_train, y_test = train_test_split(all_x, all_y, test_size=0.1)
	print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

	np.savetxt(f'./data/train-test-split/train/x/x-{dataset_target}.csv', x_train, delimiter=',')
	np.savetxt(f'./data/train-test-split/train/y/y-{dataset_target}.csv', y_train, delimiter=',')

	np.savetxt(f'./data/train-test-split/test/x/x-{dataset_target}.csv', x_test, delimiter=',')
	np.savetxt(f'./data/train-test-split/test/y/y-{dataset_target}.csv', y_test, delimiter=',')

	os.makedirs(f"./data/stats/{dataset_target}", exist_ok=True)
	torch.save(torch.Tensor(np.min(all_x, axis=0)), f"./data/stats/{dataset_target}/x-stats-min.pt")
	torch.save(torch.Tensor(np.max(all_x, axis=0)), f"./data/stats/{dataset_target}/x-stats-max.pt")
	torch.save(torch.Tensor(np.min(all_y, axis=0)), f"./data/stats/{dataset_target}/y-stats-min.pt")
	torch.save(torch.Tensor(np.max(all_y, axis=0)), f"./data/stats/{dataset_target}/y-stats-max.pt")


class WallDataset(torch.utils.data.Dataset):
	def __init__(self, dataset, mode, va, is_y_augm):
		super().__init__()

		self.dvc = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
		self.dataset = dataset
		self.mode = mode

		self.y_augm = is_y_augm

		self.x = torch.Tensor(np.array(pd.read_csv(f"./data/train-test-split/{mode}/x/x-{dataset}.csv", header=None, dtype=np.float64)))
		self.x_stats_min = torch.load(f"./data/stats/{dataset}/x-stats-min.pt").double().to(self.dvc)
		self.x_stats_max = torch.load(f"./data/stats/{dataset}/x-stats-max.pt").double().to(self.dvc)

		self.y = torch.Tensor(np.array(pd.read_csv(f"./data/train-test-split/{mode}/y/y-{dataset}.csv", header=None, dtype=np.float64)))[:, 0]
		self.y_stats_min = torch.load(f"./data/stats/{dataset}/y-stats-min.pt").double().to(self.dvc)[0]
		self.y_stats_max = torch.load(f"./data/stats/{dataset}/y-stats-max.pt").double().to(self.dvc)[0]

		self.y_x_partial_indices = torch.load(f"data/stats/{dataset}/y-x-partial-indices.pt").to(self.dvc).contiguous()

		if len(self.y.shape) == 1:
			self.y = torch.unsqueeze(self.y, 1)

		txt_dataset_loaded = f"created {mode} WallDataset with {dataset} and len: {self.x.shape[0]}"
		if va is not None:
			utility.logtext(txt_dataset_loaded, va)
		else:
			print(txt_dataset_loaded)

	def __getitem__(self, idx):
		x = self.x[idx, :].clone().to(self.dvc)
		y = self.y[idx, :].clone().to(self.dvc)

		x = self.normalize(x, self.x_stats_min, self.x_stats_max)
		y = self.normalize(y, self.y_stats_min, self.y_stats_max)

		# data augmentation: randomly empty out y condition with 50% probability
		if self.y_augm:
			if torch.randint(0,2,(1,)).bool():
				y = torch.tensor([-1]).to(self.dvc)

		return x, y

	def __len__(self):
		return self.x.shape[0]

	def normalize(self, data, stats_min, stats_max):
		return (data - stats_min) / (stats_max - stats_min)

	def get_dataset_stats(self):
		stats = (self.x_stats_min, self.x_stats_max, self.y_stats_min, self.y_stats_max)
		return stats


class BridgeDataset(torch.utils.data.Dataset):

	def __init__(self, dataset, y_cols, mode, va, do_normalization=True, do_return_g=False, is_y_augm=False):
		super().__init__()

		self.dvc = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
		self.do_normalization = do_normalization
		self.do_return_g = do_return_g
		self.dataset = dataset
		self.mode = mode

		self.y_augm = is_y_augm

		self.x = torch.Tensor(np.array(pd.read_csv(f"./data/train-test-split/{mode}/x/x-{dataset}.csv", header=None, dtype=np.float64)))
		self.x_stats_min = torch.load(f"./data/stats/{dataset}/x-stats-min.pt").double().to(self.dvc)
		self.x_stats_max = torch.load(f"./data/stats/{dataset}/x-stats-max.pt").double().to(self.dvc)

		self.y = torch.Tensor(np.array(pd.read_csv(f"./data/train-test-split/{mode}/y/y-{dataset}.csv", header=None, dtype=np.float64)))
		self.y_cols = y_cols
		self.y = self.y[:, self.y_cols]
		if len(self.y.shape) == 1:
			self.y = torch.unsqueeze(self.y, 1)

		self.y_stats_min = torch.load(f"./data/stats/{dataset}/y-stats-min.pt")[self.y_cols].double().to(self.dvc)
		self.y_stats_max = torch.load(f"./data/stats/{dataset}/y-stats-max.pt")[self.y_cols].double().to(self.dvc)
		self.y_bin_edges = {
			"y_bin_edges_dim_1": torch.load(f"data/stats/{dataset}/y-bin-edges-dim-1.pt").to(self.dvc).contiguous(),
			"y_bin_edges_dim_2": torch.load(f"data/stats/{dataset}/y-bin-edges-dim-2.pt").to(self.dvc).contiguous()
		}

		self.y_x_partial_indices = torch.load(f"data/stats/{dataset}/y-x-partial-indices.pt").to(self.dvc).contiguous()

		if do_return_g:
			self.g = torch.load(f"./data/train-test-split/{mode}/g/g-{dataset}-graph.pt")
			self.g_graph_stats = torch.load(f"./data/stats/{dataset}/g-graph-stats.pt")

		txt_dataset_loaded = f"created {mode} BridgeDataset with {dataset} and len: {self.x.shape[0]}"
		if va is not None:
			utility.logtext(txt_dataset_loaded, va)
		else:
			print(txt_dataset_loaded)

	def __getitem__(self, idx):
		x = self.x[idx, :].clone().to(self.dvc)
		y = self.y[idx, :].clone().to(self.dvc)

		if self.do_normalization:
			x = self.normalize(x, self.x_stats_min, self.x_stats_max)
			y = self.normalize(y, self.y_stats_min, self.y_stats_max)

		# data augmentation: randomly empty out y condition with 50% probability
		if self.y_augm:
			for dim in range(len(self.y_cols)):
				if torch.randint(0,2,(1,)).bool():
					y[dim] = torch.tensor([-1]).to(self.dvc)

		if not self.do_return_g:
			return x, y

		g = self.g[idx]

		return x, y, g

	def __len__(self):
		return self.x.shape[0]

	def normalize(self, data, stats_min, stats_max):
		return (data - stats_min) / (stats_max - stats_min)

	def get_dataset_stats(self):
		stats = (self.x_stats_min, self.x_stats_max, self.y_stats_min, self.y_stats_max)
		return stats

	def get_dataset_stats_g(self):
		stats = (self.g_stats_min, self.g_stats_max)
		return stats


def get_dataset_as_nparray_from_dataset(dataset_train, dataset_test):
	data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=dataset_train.__len__(), shuffle=True)
	data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=dataset_test.__len__(), shuffle=False)

	x_train, y_train = next(iter(data_loader_train))
	x_test, y_test = next(iter(data_loader_test))

	x_train, y_train = x_train.cpu().numpy(), y_train.cpu().numpy()
	x_test, y_test = x_test.cpu().numpy(), y_test.cpu().numpy()

	stats = dataset_train.get_dataset_stats()
	dataset = (x_train, y_train, x_test, y_test)

	return dataset, stats


def get_dataset_as_nparray(va):
	dataset_train, dataset_test, _, _, _, _, _ = get_datasets_and_dataloader(va, va.model_config.get("model_type"), va.model_config.get("model_params").get("batch_size"), va.model_config.get("shift_all_dimensions"), is_y_augm=False)
	return get_dataset_as_nparray_from_dataset(dataset_train, dataset_test)


def get_normalized_edge_data(edges_data, graph_stats):
	edges_data_norm = torch.zeros(edges_data.shape)
	edges_data_norm[:, 0] = normalize_data(edges_data[:, 0], graph_stats.get("height_min_max")[0], graph_stats.get("height_min_max")[1])
	edges_data_norm[:, 1] = normalize_data(edges_data[:, 1], graph_stats.get("width_min_max")[0], graph_stats.get("width_min_max")[1])
	edges_data_norm[:, 2] = normalize_data(edges_data[:, 2], graph_stats.get("a_min_max")[0], graph_stats.get("a_min_max")[1])
	edges_data_norm[:, 3] = normalize_data(edges_data[:, 3], graph_stats.get("iyy_min_max")[0], graph_stats.get("iyy_min_max")[1])
	edges_data_norm[:, 4] = normalize_data(edges_data[:, 4], graph_stats.get("izz_min_max")[0], graph_stats.get("izz_min_max")[1])
	return edges_data_norm


def get_torch_geometric_dataloader(xs, ys, gs, va, do_normalize_coords=False, do_add_self_loops=True, do_node_features_x10d=False, do_edge_features=False):

	all_data = []

	for idx in range(len(gs)):

		(node_coords, edges, edges_data) = gs[idx]

		if do_normalize_coords:
			stats = va.dataset_train.get_dataset_stats_g()
			node_coords = normalize_data(node_coords, stats[0], stats[1])

		if not do_node_features_x10d:
			node_coords_with_data = node_coords[:, [0,2]]
		else:
			node_coords_with_data = torch.zeros(node_coords.shape[0], 2+10)
			node_coords_with_data[:, :2] = node_coords[:, [0,2]]
			node_coords_with_data[:, 2:] = xs[idx, :]


		n_edges = edges.shape[1]
		final_dim = n_edges*2

		# make bidirectional graph
		edges_bidirect = torch.zeros((2, final_dim), dtype=torch.long)
		edges_bidirect[:, :n_edges] = edges
		edges_bidirect[0, n_edges:(n_edges*2)] = edges[1, :]
		edges_bidirect[1, n_edges:(n_edges*2)] = edges[0, :]

		# same weight for both edge directions
		if do_edge_features:
			edge_attr_doubled_norm = torch.zeros((final_dim, 5))
			edges_data_norm = get_normalized_edge_data(edges_data, va.dataset_train.g_graph_stats)
			edge_attr_doubled_norm[:n_edges, :] = edges_data_norm
			edge_attr_doubled_norm[n_edges:, :] = edges_data_norm

			edge_attr = edge_attr_doubled_norm

		else:
			edge_attr = None

		#plot.plot_3d_bridge_from_node_and_edges(node_coords, edges_bidirect)

		data = TGData(x=node_coords_with_data, edge_index=edges_bidirect, edge_attr=edge_attr, y=torch.unsqueeze(ys[idx, :], dim=0))
		all_data.append(data)

	dataloader = TGDataLoader(all_data, batch_size=len(gs)) # no shuffle, no batching

	return dataloader


# def upsample_imbalanced_dataset_naive():

# 	# get valid samples from raw data (without train/test split)
# 	df_x = pd.read_csv(f'./data/train-test-split/x/sobol-samples-10d-v2.csv', header=None, dtype=np.float64)
# 	df_y = pd.read_csv(f'./data/train-test-split/y/output-10d-v2.csv', header=None, dtype=np.float64)
# 	df_y = df_y[df_y[0] != 0.0]
# 	df_y = df_y[df_y[1] != 0.0]
# 	idxs_y = df_y.index.to_numpy()
# 	df_x = df_x.iloc[idxs_y]

# 	x = df_x.to_numpy()
# 	y = df_y.to_numpy()

# 	# ASSUMPTION: make upsampling based on y[0] distribution only
# 	hist, bin_edges = np.histogram(y[:, 0], bins='auto') # 91 bins

# 	n_max_upsampling = np.max(hist)

# 	hist_nonzero = hist[hist != 0]
# 	n_equal_bins = int(df_x.shape[0] / len(hist_nonzero))

# 	print("n_max_upsampling:", n_max_upsampling)
# 	print("n_equal_bins:", n_equal_bins)
# 	print("# of bins:", len(hist))
# 	print("# of nonzero bins:", len(hist_nonzero))

# 	bin_edges[0] -= 10**-6
# 	bin_edges[-1] += 10**-6

# 	bin_idxs = np.digitize(y[:, 0], bin_edges) - 1

# 	all_x_upsampled = None
# 	all_y_upsampled = None

# 	cnt_bins_zero = 0

# 	print("run upsampling...")
# 	for i in range(max(bin_idxs)+1):
# 		bin_mask = bin_idxs == i

# 		bin_x = x[bin_mask, :]
# 		bin_y = y[bin_mask, :]

# 		if bin_y.shape[0] != 0:
# 			upsampled_idxs = np.random.choice(bin_y.shape[0], n_equal_bins)

# 			bin_x_upsampled = bin_x[upsampled_idxs, :]
# 			bin_y_upsampled = bin_y[upsampled_idxs, :]

# 			if all_x_upsampled is None:
# 				all_x_upsampled = bin_x_upsampled
# 				all_y_upsampled = bin_y_upsampled
# 			else:
# 				all_x_upsampled = np.concatenate((all_x_upsampled, bin_x_upsampled))
# 				all_y_upsampled = np.concatenate((all_y_upsampled, bin_y_upsampled))
# 		else:
# 			cnt_bins_zero += 1

# 	print("cnt_bins_zero:", cnt_bins_zero)

# 	print(y.shape)
# 	print(all_y_upsampled.shape)

# 	_, axes = plt.subplots(1, 2)
# 	axes[0].hist(y[:,0], bins=bin_edges)
# 	axes[1].hist(all_y_upsampled[:, 0], bins=bin_edges)
# 	#plt.show()

# 	print("save to files...")
# 	np.savetxt("data/train-test-split/x/sobol-samples-10d-v2-updownsampled.csv", all_x_upsampled, delimiter=",")
# 	np.savetxt("data/train-test-split/y/output-10d-v2-updownsampled.csv", all_y_upsampled, delimiter=",")

# 	return bin_edges


class DynamicDataset(torch.utils.data.Dataset):
	def __init__(self, va, mode, data_x, data_y, stats, dim_config, is_y_augm):
		super().__init__()

		self.dvc = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
		self.mode = mode
		self.y_augm = is_y_augm

		self.dim_config = dim_config
		self.x_dims_conti = self.dim_config.get("x_dims_conti")
		self.x_output_mask_sigmoid = self.dim_config.get("x_output_masks").get("sigmoid")
		self.y_dims_conti = self.dim_config.get("y_dims_conti")
		self.y_output_mask_sigmoid = self.dim_config.get("y_output_masks").get("sigmoid")

		self.x = data_x
		self.x_stats_min = stats.get("x_stats_min")
		self.x_stats_max = stats.get("x_stats_max")

		self.y_dim = len(va.cli_args.y_cols)

		#print(self.x[:50, self.x_output_mask_sigmoid])
		#exit()

		self.y = data_y
		self.y_stats_min = stats.get("y_stats_min")
		self.y_stats_max = stats.get("y_stats_max")

		self.y_x_partial_indices = utility.create_y_x_partial_indices_file(va.dim_config.get("x_dim_orig"), va.dim_config).to(self.dvc).contiguous()

		if len(self.y.shape) == 1:
			self.y = torch.unsqueeze(self.y, 1)

		txt_dataset_loaded = f"created {mode} DynamicDataset with len: {self.x.shape[0]} and {self.y_x_partial_indices.shape[0]} y_x masks"
		if va is not None:
			utility.logtext(txt_dataset_loaded, va)
		else:
			print(txt_dataset_loaded)

	def __getitem__(self, idx):
		x = self.x[idx, :].clone().to(self.dvc)
		y = self.y[idx, :].clone().to(self.dvc)

		#print(self.x_output_mask_sigmoid)
		#print(x[self.x_output_mask_sigmoid], self.x_stats_min[self.x_dims_conti], self.x_stats_max[self.x_dims_conti])

		#print(x[self.x_output_mask_sigmoid])

		x[self.x_output_mask_sigmoid] = self.normalize(x[self.x_output_mask_sigmoid], self.x_stats_min[self.x_dims_conti], self.x_stats_max[self.x_dims_conti])

		#print(x)

		y[self.y_output_mask_sigmoid] = self.normalize(y[self.y_output_mask_sigmoid], self.y_stats_min[self.y_dims_conti], self.y_stats_max[self.y_dims_conti])

		# data augmentation: randomly empty out y condition with 50% probability
		if self.y_augm:
			idx = 0
			for dim in range(self.y_dim):
				if self.dim_config.get(f"y{dim}").get("is_categorical"):
					n = len(self.dim_config.get(f"y{dim}").get("vals"))
				else:
					n = 1
				if torch.randint(0,2,(1,)).bool():
					y[idx:(idx+n)] = torch.full((1,n), -1).to(self.dvc)
				idx += n

		return x, y

	def __len__(self):
		return self.x.shape[0]

	def normalize(self, data, stats_min, stats_max):
		return (data - stats_min) / (stats_max - stats_min)

	def get_dataset_stats(self):
		stats = (self.x_stats_min, self.x_stats_max, self.y_stats_min, self.y_stats_max)
		return stats


def init_dim_config(data_x, data_y, n_x, n_y, col_names):

	# discrete = categorical
	# continuous

	x_dims_categorical = np.array(['_c' in el for el in col_names[:n_x]]).astype(int)
	y_dims_categorical = np.array(['_c' in el for el in col_names[n_x:(n_x+n_y)]]).astype(int)

	dim_config = {
		"x_dims_categorical": x_dims_categorical,
		"y_dims_categorical": y_dims_categorical,
		"x_dims_conti": [],
		"y_dims_conti": [],
		"x_dim_orig": n_x,
		"y_dim_orig": n_y,
	}

	x_dim_enc = 0
	for i, is_col_discrete in enumerate(x_dims_categorical):
		if is_col_discrete:
			unique_vals = np.unique(data_x[:, i])
			n = len(unique_vals)
			dim_config[f"x{i}"] = {}
			dim_config[f"x{i}"]["is_categorical"] = True
			dim_config[f"x{i}"]["vals"] = unique_vals
			x_dim_enc += n
		else:
			dim_config["x_dims_conti"] += [i]
			dim_config[f"x{i}"] = {}
			dim_config[f"x{i}"]["is_categorical"] = False
			x_dim_enc += 1

	dim_config["x_dim_enc"] = x_dim_enc

	y_dim_enc = 0
	for i, is_col_discrete in enumerate(y_dims_categorical):
		if is_col_discrete:
			unique_vals = np.unique(data_y[:, i])
			n = len(unique_vals)
			dim_config[f"y{i}"] = {}
			dim_config[f"y{i}"]["is_categorical"] = True
			dim_config[f"y{i}"]["vals"] = unique_vals
			y_dim_enc += n
		else:
			dim_config["y_dims_conti"] += [i]
			dim_config[f"y{i}"] = {}
			dim_config[f"y{i}"]["is_categorical"] = False
			y_dim_enc += 1

	dim_config["y_dim_enc"] = y_dim_enc

	return dim_config


def encode_dims_for_x(data_x, dim_config):

	x_dim_enc = dim_config.get("x_dim_enc")

	dim_config["x_output_masks"] = {
		"sigmoid": torch.zeros(x_dim_enc).bool(),
		"softmax": []
	}

	data_x_enc = np.zeros((data_x.shape[0], x_dim_enc))

	col_idx = 0
	for i, is_col_discrete in enumerate(dim_config.get("x_dims_categorical")):
		if is_col_discrete:
			n = len(dim_config.get(f"x{i}")["vals"])
			enc = OneHotEncoder(categories='auto', sparse=False)
			data_x_enc[:, col_idx:(col_idx+n)] = enc.fit_transform(np.expand_dims(data_x[:, i], 1))
			dim_config[f"x{i}"]["enc"] = enc
			softmax_masks = torch.zeros(x_dim_enc).bool()
			softmax_masks[col_idx:(col_idx+n)] = True
			dim_config["x_output_masks"]["softmax"].append(softmax_masks)
			col_idx += n
		else:
			dim_config["x_output_masks"]["sigmoid"][col_idx] = 1
			data_x_enc[:, col_idx:(col_idx+1)] = np.expand_dims(data_x[:, i], 1)
			col_idx += 1

	return data_x_enc, dim_config


def encode_dims_for_y(data_y, dim_config):

	y_dim_enc = dim_config.get("y_dim_enc")

	dim_config["y_output_masks"] = {
		"sigmoid": torch.zeros(y_dim_enc).bool(),
	}

	data_y_enc = np.zeros((data_y.shape[0], y_dim_enc))

	col_idx = 0
	for i, is_col_discrete in enumerate(dim_config.get("y_dims_categorical")):
		if is_col_discrete:
			n = len(dim_config.get(f"y{i}")["vals"])
			enc = OneHotEncoder(categories='auto', sparse=False)
			data_y_enc[:, col_idx:(col_idx+n)] = enc.fit_transform(np.expand_dims(data_y[:, i], 1))
			dim_config[f"y{i}"]["enc"] = enc
			col_idx += n
		else:
			dim_config["y_output_masks"]["sigmoid"][col_idx] = 1
			data_y_enc[:, col_idx:(col_idx+1)] = np.expand_dims(data_y[:, i], 1)
			col_idx += 1

	return data_y_enc, dim_config


def encode_dims_for_y_single_sample(data_y, dim_config):

	y_dim_enc = dim_config.get("y_dim_enc")
	data_y_enc = np.zeros((1, y_dim_enc))

	col_idx = 0
	for i, is_col_discrete in enumerate(dim_config.get("y_dims_categorical")):
		if is_col_discrete:
			n = len(dim_config.get(f"y{i}")["vals"])
			enc = dim_config[f"y{i}"]["enc"]

			#print(data_y[0, i])

			if data_y[0, i] != -1.0:
				#print("encode: ", data_y[0, i])
				#print(enc.categories_)
				#print(enc.transform(np.expand_dims(data_y[:, i], 1)))

				data_y_enc[:, col_idx:(col_idx+n)] = enc.transform(np.expand_dims(data_y[:, i], 1))
			else:
				#print("do not encode")
				data_y_enc[:, col_idx:(col_idx+n)] = -1.0

			col_idx += n
		else:
			data_y_enc[:, col_idx:(col_idx+1)] = data_y[:, i]
			col_idx += 1

	return data_y_enc


def encode_dims_for_y_x_single_sample(data_y_x, dim_config):

	x_dim_enc = dim_config.get("x_dim_enc")
	data_y_x_enc = np.zeros((1, x_dim_enc))

	col_idx = 0
	for i, is_col_discrete in enumerate(dim_config.get("x_dims_categorical")):
		if is_col_discrete:
			n = len(dim_config.get(f"x{i}")["vals"])
			enc = dim_config[f"x{i}"]["enc"]

			if data_y_x[0, i] != -1.0:
				#print(f"dim {i} is not -1.0")
				#print("encode: ", data_y_x[0, i])
				#print(enc.categories_)
				data_y_x_enc[:, col_idx:(col_idx+n)] = enc.transform(np.expand_dims(data_y_x[:, i], 1))
			else:
				#print(f"dim {i} is -1.0")
				data_y_x_enc[:, col_idx:(col_idx+n)] = -1.0

			col_idx += n
		else:
			data_y_x_enc[:, col_idx:(col_idx+1)] = data_y_x[0, i]
			col_idx += 1

	data_y_x_enc = torch.tensor(data_y_x_enc).float()

	return data_y_x_enc


def decode_y_x_partial(data_y_x_partial_enc, va, dataset_stats):

	# unnormalize continuous dimensions
	x_dims_conti = va.dim_config.get("x_dims_conti")
	data_y_x_partial_enc_unnorm = data_y_x_partial_enc.clone()
	for i, x_dim in enumerate(np.arange(data_y_x_partial_enc.shape[1])[va.dim_config.get("x_output_masks").get("sigmoid")]):
		x_orig_dim = x_dims_conti[i]
		mask = data_y_x_partial_enc_unnorm[:, x_dim] != -1.0
		data_y_x_partial_enc_unnorm[mask, x_dim] = (data_y_x_partial_enc_unnorm[mask, x_dim] * (dataset_stats[1][x_orig_dim] - dataset_stats[0][x_orig_dim])) + dataset_stats[0][x_orig_dim]
	data_y_x_partial_enc = data_y_x_partial_enc_unnorm

	# decode data
	data_y_x_partial_enc = data_y_x_partial_enc.detach().cpu().numpy()
	x_dim_orig = va.dim_config.get("x_dim_orig")
	data_y_x_partial_dec = np.zeros((data_y_x_partial_enc.shape[0], x_dim_orig))

	col_idx = 0
	for i in range(x_dim_orig):
		xi = va.dim_config.get(f"x{i}")
		if xi.get("is_categorical"):
			n = len(xi.get("vals"))

			row_mask = data_y_x_partial_enc[:, col_idx] != -1.0
			# decode only if at least one dimensions is set
			if sum(row_mask) > 0:
				data_y_x_partial_dec[row_mask, i] = np.squeeze(xi.get("enc").inverse_transform(data_y_x_partial_enc[row_mask, col_idx:(col_idx+n)]))
			#else:
				#utility.logtext("", va)
				#utility.logtext("MASK IS EMPTY!", va)
				#utility.logtext(data_y_x_partial_enc[:, col_idx], va)
				#utility.logtext("", va)

			row_mask = data_y_x_partial_enc[:, col_idx] == -1.0
			data_y_x_partial_dec[row_mask, i] = -1.0

			col_idx += n
		else:
			data_y_x_partial_dec[:, i] = data_y_x_partial_enc[:, col_idx]
			col_idx += 1

	data_x_dec = torch.tensor(data_y_x_partial_dec, device=va.dvc)

	return data_x_dec


def decode_cat_and_unnorm_cont_dims_for_x(data_x_enc, va, dataset_stats):

	# unnormalize continuous dimensions
	x_dims_conti = va.dim_config.get("x_dims_conti")
	data_x_enc_unnorm = data_x_enc.clone()
	data_x_enc_unnorm[:, va.dim_config.get("x_output_masks").get("sigmoid")] = (data_x_enc_unnorm[:, va.dim_config.get("x_output_masks").get("sigmoid")] * (dataset_stats[1][x_dims_conti] - dataset_stats[0][x_dims_conti])) + dataset_stats[0][x_dims_conti]
	data_x_enc = data_x_enc_unnorm

	# decode data
	data_x_enc = data_x_enc.detach().cpu().numpy()
	x_dim_orig = va.dim_config.get("x_dim_orig")
	data_x_dec = np.zeros((data_x_enc.shape[0], x_dim_orig))
	col_idx = 0
	for i in range(x_dim_orig):
		xi = va.dim_config.get(f"x{i}")
		if xi.get("is_categorical"):
			n = len(xi.get("vals"))
			data_x_dec[:, i] = np.squeeze(xi.get("enc").inverse_transform(data_x_enc[:, col_idx:(col_idx+n)]))
			col_idx += n
		else:
			data_x_dec[:, i] = data_x_enc[:, col_idx]
			col_idx += 1

	data_x_dec = torch.tensor(data_x_dec, device=va.dvc)

	return data_x_dec


def decode_cat_and_unnorm_cont_dims_for_y(data_y_enc, va, dataset_stats):

	# unnormalize continuous dimensions
	y_dims_conti = va.dim_config.get("y_dims_conti")
	data_y_enc_unnorm = data_y_enc.clone()
	data_y_enc_unnorm[:, va.dim_config.get("y_output_masks").get("sigmoid")] = (data_y_enc_unnorm[:, va.dim_config.get("y_output_masks").get("sigmoid")] * (dataset_stats[3][y_dims_conti] - dataset_stats[2][y_dims_conti])) + dataset_stats[2][y_dims_conti]
	data_y_enc = data_y_enc_unnorm

	# decode data
	data_y_enc = data_y_enc.detach().cpu().numpy()
	y_dim_orig = va.dim_config.get("y_dim_orig")
	data_y_dec = np.zeros((data_y_enc.shape[0], y_dim_orig))
	col_idx = 0
	for i in range(y_dim_orig):
		yi = va.dim_config.get(f"y{i}")
		if yi.get("is_categorical"):
			n = len(yi.get("vals"))
			data_y_dec[:, i] = np.squeeze(yi.get("enc").inverse_transform(data_y_enc[:, col_idx:(col_idx+n)]))
			col_idx += n
		else:
			data_y_dec[:, i] = data_y_enc[:, col_idx]
			col_idx += 1

	data_y_dec = torch.tensor(data_y_dec, device=va.dvc)

	return data_y_dec


def get_dynamic_dataset(va, ds_path, batch_size, shift_all_dimensions, is_stats_only=False, is_y_augm=False):

	dvc = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	# read raw data
	df = pd.read_csv(ds_path)
	col_names = list(df.columns)

	n_x = sum(['x' in el for el in col_names])
	
	n_y = sum(['y' in el for el in col_names])
	if va.cli_args.y_cols is not None:
		n_y = len(va.cli_args.y_cols)
		col_names = col_names[:n_x] + [ col_names[j] for j in [ n_x + i for i in va.cli_args.y_cols ] ]

	if n_x == 0 and n_y == 0:
		raise Exception('x or y columns empty (not found)! please provide a header row with x_i and y_i for the corresponding columns')

	data = np.array(df)
	
	data_x = data[:, :n_x]

	data_y = data[:, n_x:]
	data_y = data_y[:, va.cli_args.y_cols] # take only provided col indices from cli args

	# shift up +1 if flag is set
	if shift_all_dimensions:
		data_x = data_x + 1.0
		data_y = data_y + 1.0

	# get stats
	stats = {
		"x_stats_min": torch.tensor(np.min(data_x, axis=0)).float().to(dvc),
		"x_stats_max": torch.tensor(np.max(data_x, axis=0)).float().to(dvc),
		"y_stats_min": torch.tensor(np.min(data_y, axis=0)).float().to(dvc),
		"y_stats_max": torch.tensor(np.max(data_y, axis=0)).float().to(dvc),
	}
	utility.logtext(stats, va)

	utility.logtext(ds_path, va)
	utility.logtext(f"before encoding: {data_x.shape}, {data_y.shape}", va)

	# convert discrete cols into one hot encoding
	dim_config = init_dim_config(data_x, data_y, n_x, n_y, col_names)
	data_x, dim_config = encode_dims_for_x(data_x, dim_config)
	data_y, dim_config = encode_dims_for_y(data_y, dim_config)
	va.dim_config = dim_config

	utility.logtext(f"after encoding: {data_x.shape}, {data_y.shape}", va)

	# do train test split
	utility.logtext(f"creating split...", va)
	x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.1)
	x_train = torch.tensor(x_train)
	x_test = torch.tensor(x_test)
	y_train = torch.tensor(y_train)
	y_test = torch.tensor(y_test)
	utility.logtext(f"train: {x_train.shape}, {y_train.shape}", va)
	utility.logtext(f"test: {x_test.shape}, {y_test.shape}", va)

	if is_stats_only:
		return stats, dim_config

	else:
		dataset_train = DynamicDataset(va, "train", x_train, y_train, stats, dim_config, is_y_augm)
		dataset_test = DynamicDataset(va, "test", x_test, y_test, stats, dim_config, is_y_augm)

		data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=False, worker_init_fn=utility.set_seeds(va.cli_args.seed), generator=utility.get_tgseed(va.cli_args.seed))
		data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, worker_init_fn=utility.set_seeds(va.cli_args.seed), generator=utility.get_tgseed(va.cli_args.seed))

		return dataset_train, dataset_test, data_loader_train, data_loader_test, x_train.shape[1], y_train.shape[1], dim_config


def get_datasets_and_dataloader(va, model_type, batch_size, shift_all_dimensions, is_y_augm):

	ds_name = va.cli_args.dataset
	ds_path = va.cli_args.dataset_path_csv

	if ds_name in ["10d-robust", "10d-simple", "10d-valid", "5d-wall", "5d-wall-v2"]:

		if "5d-wall" not in ds_name:
			do_return_g, collate_fn = utility.get_collate_fn(model_type)
			dataset_train = BridgeDataset(ds_name, va.cli_args.y_cols, "train", va, do_normalization=True, do_return_g=do_return_g, is_y_augm=is_y_augm)
			dataset_test = BridgeDataset(ds_name, va.cli_args.y_cols, "test", va, do_normalization=True, do_return_g=do_return_g, is_y_augm=is_y_augm)
			data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=False, worker_init_fn=utility.set_seeds(va.cli_args.seed), generator=utility.get_tgseed(va.cli_args.seed), collate_fn=collate_fn)
			data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, worker_init_fn=utility.set_seeds(va.cli_args.seed), generator=utility.get_tgseed(va.cli_args.seed), collate_fn=collate_fn)

		else:
			collate_fn = None
			dataset_train = WallDataset(va.cli_args.dataset, "train", va, is_y_augm=is_y_augm)
			dataset_test = WallDataset(va.cli_args.dataset, "test", va, is_y_augm=is_y_augm)
			data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=False, worker_init_fn=utility.set_seeds(va.cli_args.seed), generator=utility.get_tgseed(va.cli_args.seed))
			data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, worker_init_fn=utility.set_seeds(va.cli_args.seed), generator=utility.get_tgseed(va.cli_args.seed))

		x, y = dataset_train.__getitem__(0)

		return dataset_train, dataset_test, data_loader_train, data_loader_test, x.shape[0], y.shape[0], None

	return get_dynamic_dataset(va, ds_path, batch_size, shift_all_dimensions, is_stats_only=False, is_y_augm=is_y_augm)


if __name__ == "__main__":

	#init_dataset("10d-orig", "10d-valid")
	#init_dataset("10d-orig", "10d-robust")
	#init_dataset("10d-orig", "10d-simple")

	#init_wall_dataset("5d-wall")
	init_wall_dataset("5d-wall-v2")

	#generate_robustness_test_dataset()
	#read_output_array_from_files("10d-orig")
	#upsample_imbalanced_dataset_naive()
