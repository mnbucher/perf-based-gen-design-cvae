import torch
from torch import nn
import numpy as np
from tqdm import tqdm

import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool

import src.plot as plot
import src.dataset as dataset

# tries to learn X->Y without any graph information (baseline)

class MLP(torch.nn.Module):

	def __init__(self, model_config):
		super().__init__()

		# gnn_h_dim = 128
		# gnn_n_h = 10

		gnn_h0_dim = model_config.get("model_params").get("gnn_h0_dim")
		gnn_n_h = model_config.get("model_params").get("gnn_n_h")

		self.mlp_hidden_sizes = list(np.linspace(gnn_h0_dim, 2, gnn_n_h, dtype=int))
		#self.mlp_hidden_sizes = list(np.linspace(256, 2, 20, dtype=int))

		#print(self.mlp_hidden_sizes)

		self.mlp = nn.Sequential()

		for i in range(len(self.mlp_hidden_sizes)):
			inp_dim = 10 if i == 0 else self.mlp_hidden_sizes[i-1]
			out_dim = self.mlp_hidden_sizes[i]
			self.mlp.add_module(name=f"gnn0_mlp_{i}", module=MLPBlock(i, inp_dim, out_dim))


	def forward(self, x):
		# mlp to perform regression
		#print(x.shape)
		y = self.mlp(x)
		y = nn.Sigmoid()(y)

		return y


class MLPBlock(nn.Module):

	def __init__(self, idx, inp_dim, out_dim):
		super().__init__()

		#print(f"MLP Layer {idx}", inp_dim, out_dim)

		self.linear1 = nn.Linear(inp_dim, out_dim)
		self.activ = nn.ReLU()
		self.bn = nn.BatchNorm1d(out_dim)

		self.linear2 = nn.Linear(out_dim, out_dim)

	def forward(self, x):
		x1 = self.linear1(x)

		x2 = self.activ(x1)
		x2 = self.bn(x2)

		x2 = self.linear2(x2)

		x = x2 + x1

		return x
