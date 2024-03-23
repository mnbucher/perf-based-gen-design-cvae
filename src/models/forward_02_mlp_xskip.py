import torch
from torch import nn
import numpy as np
from tqdm import tqdm

import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool

import src.plot as plot
import src.dataset as dataset
from src.models.helper import tupleSequential


class MLP2(torch.nn.Module):

	def __init__(self, model_config):
		super().__init__()

		gnn_h0_dim = model_config.get("model_params").get("gnn_h0_dim")
		gnn_n_h = model_config.get("model_params").get("gnn_n_h")

		self.mlp_hidden_sizes = list(np.linspace(gnn_h0_dim, 2, gnn_n_h, dtype=int))
		#print(self.mlp_hidden_sizes)

		self.mlp = tupleSequential()

		for i in range(len(self.mlp_hidden_sizes)):
			inp_dim = 10 if i == 0 else self.mlp_hidden_sizes[i-1]
			out_dim = self.mlp_hidden_sizes[i]
			self.mlp.add_module(name=f"gnn0_mlp_{i}", module=MLPBlock(i, inp_dim, out_dim, 10))


	def forward(self, x):
		y, _ = self.mlp(x, x)
		y = nn.Sigmoid()(y)

		return y


class MLPBlock(nn.Module):

	def __init__(self, idx, inp_dim, out_dim, x_dim):
		super().__init__()

		#print(f"MLP Layer {idx}", inp_dim, out_dim)

		self.linear1 = nn.Linear(inp_dim + x_dim, out_dim)
		self.activ = nn.ReLU()
		self.bn = nn.BatchNorm1d(out_dim)

		#self.linear2 = nn.Linear(out_dim, out_dim)

	def forward(self, x, x_orig):

		x = torch.cat([x, x_orig], 1)

		x1 = self.linear1(x)

		x2 = self.activ(x1)
		x2 = self.bn(x2)

		#x2 = self.linear2(x2)

		x = x2 + x1

		return x, x_orig
