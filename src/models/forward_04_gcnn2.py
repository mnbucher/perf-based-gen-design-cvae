import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GCN2Conv, global_add_pool, global_max_pool, Linear

import src.plot as plot
import src.utility as utility
import src.dataset as dataset

# gcn layer: use edge weights as sum of A and Iyy (both normalized)

# like gnn4 but with edge weighting
# gnn4: gnn3 + MLP head for improved prediction

class GCNN2(torch.nn.Module):

	def __init__(self, model_config, va):
		super().__init__()

		self.y_dim = model_config.get("y_dim")

		self.gnn_h_dim = model_config.get("model_params").get("gnn_h_dim")
		self.gnn_n_h = model_config.get("model_params").get("gnn_n_h")
		self.gnn_n_h_mlp = model_config.get("model_params").get("gnn_n_h_mlp")
		self.concat_x_with_z_g_for_mlp_head = model_config.get("model_params").get("concat_x_with_z_g_for_mlp_head")

		gnn_res_alpha = model_config.get("model_params").get("gnn_res_alpha")

		self.gnn_edge_weight_n = 6

		self.linear_first = Linear(2, self.gnn_h_dim)

		# GNN
		self.gnn = nn.Sequential()
		for i in range(self.gnn_n_h):
			inp_dim = self.gnn_h_dim

			out_dim = self.gnn_h_dim
			self.gnn.add_module(name=f"gnn5_gnn_gcn_{i}", module=GCNBlock(va, i+1, gnn_res_alpha, inp_dim, out_dim))

		if va.cli_args.is_vanilla_training:
			utility.logtext(f"", va)

		# edge weight MLP
		self.mlp_hidden_sizes = list(np.linspace(5, 1, self.gnn_edge_weight_n, dtype=int))
		self.gnn_edge_weight_mlp = nn.Sequential()
		for k in range(1, self.gnn_edge_weight_n):
			inp_dim = self.mlp_hidden_sizes[k-1]
			out_dim = self.mlp_hidden_sizes[k]
			is_last = False if k < len(self.mlp_hidden_sizes)-1 else True
			self.gnn_edge_weight_mlp.add_module(name=f"gnn5_edge_weight_mlp_{k}", module=MLPBlock(va, k, inp_dim, out_dim, is_last_layer=is_last, do_return_embedding=False))

		if va.cli_args.is_vanilla_training:
			utility.logtext(f"", va)

		# MLP Head
		mlp_first_layer = self.gnn_h_dim+10 if self.concat_x_with_z_g_for_mlp_head else self.gnn_h_dim
		self.mlp_hidden_sizes = list(np.linspace(128, self.y_dim, self.gnn_n_h_mlp, dtype=int))
		self.mlp = nn.Sequential()
		for j in range(len(self.mlp_hidden_sizes)):
			inp_dim = mlp_first_layer if j == 0 else self.mlp_hidden_sizes[j-1]
			out_dim = self.mlp_hidden_sizes[j]
			is_last = False if j < len(self.mlp_hidden_sizes)-1 else True
			self.mlp.add_module(name=f"gnn5_mlp_{j}", module=MLPBlock(va, i, inp_dim, out_dim, is_last_layer=is_last, do_return_embedding=True))



	def aggregate_node_features(self, x, batch):
		aggr = global_add_pool(x, batch)
		aggr = nn.Sigmoid()(aggr)
		return aggr


	def forward(self, data, x_10d):

		edge_weights = data.edge_attr
		edge_weights = self.gnn_edge_weight_mlp(edge_weights)
		edge_weights = nn.Sigmoid()(edge_weights)

		x = data.x
		x = self.linear_first(x)

		x_with_edges = x, x, data.edge_index.long(), edge_weights, None

		# gnn forward pass
		x, _, _, _, _ = self.gnn(x_with_edges)

		# global pooling into single latent embedding
		x_aggr = self.aggregate_node_features(x, data.batch)

		# mlp to perform regression
		if self.concat_x_with_z_g_for_mlp_head:
			x_aggr = torch.cat([x_aggr, x_10d], dim=1)

		x = self.mlp(x_aggr)
		x = nn.Sigmoid()(x)

		return x, x_aggr


class GCNBlock(nn.Module):

	def __init__(self, va, idx, alpha, inp_dim, out_dim):
		super().__init__()

		if va.cli_args.is_vanilla_training:
			utility.logtext(f"GCN Layer {idx} — {inp_dim} > {out_dim}", va)

		self.idx = idx

		self.conv = GCN2Conv(inp_dim, alpha=alpha)
		self.activ = nn.ReLU()
		self.bn = nn.BatchNorm1d(out_dim)

	def forward(self, x_with_edges):

		x, x_orig, edge_indices, edge_weights, _ = x_with_edges
		last_hidd = x

		x1 = self.conv(x, x_orig, edge_indices, edge_weights)

		x1 = self.activ(x1)
		x1 = self.bn(x1)

		x = x1

		x_with_edges = x, x_orig, edge_indices, edge_weights, last_hidd

		return x_with_edges


class MLPBlock(nn.Module):

	def __init__(self, va, idx, inp_dim, out_dim, is_last_layer, do_return_embedding=True):
		super().__init__()

		if va.cli_args.is_vanilla_training:
			utility.logtext(f"MLP Layer {idx} – {inp_dim} > {out_dim}", va)

		self.is_last_layer = is_last_layer
		self.do_return_embedding = do_return_embedding

		self.linear1 = nn.Linear(inp_dim, out_dim)
		self.activ = nn.ReLU()
		self.bn = nn.BatchNorm1d(out_dim)

	def forward(self, x):
		x1 = self.linear1(x)

		x2 = self.activ(x1)
		x2 = self.bn(x2)

		x3 = x2 + x1

		return x3
