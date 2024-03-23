import torch
from torch import nn
import numpy as np

from src.models.helper import tupleSequential


class CVAE6(nn.Module):

	def __init__(self, model_config):
		super().__init__()

		x_dim = model_config.get("x_dim")
		y_dim = model_config.get("y_dim")

		z_dim = model_config.get("model_params").get("z_dim")
		h_dim = model_config.get("model_params").get("h_dim")
		n_h = model_config.get("model_params").get("n_h")

		self.hidden_sizes = list(np.linspace(h_dim, z_dim, n_h+1, dtype=int))
		#print(self.hidden_sizes)

		self.encoder = Encoder(x_dim, z_dim, y_dim, self.hidden_sizes, n_h)
		self.decoder = Decoder(x_dim, z_dim, y_dim, self.hidden_sizes, n_h)

	def reparametrize(self, mu, logvar):
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		return mu + eps*std

	def forward(self, x, y, y_x_partial):
		z_mu, z_logvar = self.encoder(x, y)
		z = self.reparametrize(z_mu, z_logvar)

		recon_x = self.decoder(z, y, y_x_partial)

		return recon_x, z_mu, z_logvar, z


class Encoder(nn.Module):

	def __init__(self, x_dim, z_dim, y_dim, hidden_sizes, n_h):
		super().__init__()

		self.neuralnet = tupleSequential()

		for i in range(n_h):
			inp_size = x_dim + y_dim if i == 0 else hidden_sizes[i-1] + y_dim
			out_size = hidden_sizes[i]
			self.neuralnet.add_module(name=f"enc_mlp_block_{i}", module=MLPBlock(inp_size, out_size))

		self.z_mean = nn.Linear(hidden_sizes[-2], z_dim)
		self.z_logvar = nn.Linear(hidden_sizes[-2], z_dim)

	def forward(self, x, y):

		x, _ = self.neuralnet(x, y)

		z_mu = self.z_mean(x)
		z_logvar = self.z_logvar(x)

		return z_mu, z_logvar


class Decoder(nn.Module):

	def __init__(self, x_dim, z_dim, y_dim, hidden_sizes, n_h):
		super().__init__()

		self.neuralnet = tupleSequential()

		for i in range(n_h):
			inp_size = (z_dim + y_dim + x_dim) if i == 0 else (hidden_sizes[-(i+1)] + y_dim + x_dim)
			out_size = hidden_sizes[-(i+2)]
			self.neuralnet.add_module(name=f"dec_mlp_block_{i}", module=MLPBlock(inp_size, out_size))

		self.linear_out = nn.Linear(hidden_sizes[0], x_dim)

	def forward(self, z, y, y_x_partial):

		y = torch.cat([y, y_x_partial], 1)
		x, _ = self.neuralnet(z, y)

		x = self.linear_out(x)
		x = nn.Sigmoid()(x)

		return x


class MLPBlock(nn.Module):

	def __init__(self, inp_dim, out_dim):
		super().__init__()

		self.linear1 = nn.Linear(inp_dim, out_dim)
		self.activ = nn.ReLU()
		self.bn = nn.BatchNorm1d(out_dim)


	def forward(self, x, y):
		x1 = torch.cat([x, y], 1)
		x1 = self.linear1(x1)

		x2 = self.activ(x1)
		x2 = self.bn(x2)

		x = x2 + x1

		return x, y
