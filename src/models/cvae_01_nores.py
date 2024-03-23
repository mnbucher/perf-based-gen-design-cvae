import torch
from torch import nn
import numpy as np


class CVAE1(nn.Module):

	def __init__(self, model_config):
		super().__init__()

		x_size = model_config.get("x_dim")
		y_size = model_config.get("y_dim")

		z_size = model_config.get("model_params").get("z_dim")
		h_size = model_config.get("model_params").get("h_dim")
		n_h = model_config.get("model_params").get("n_h")

		self.hidden_sizes = list(np.linspace(h_size, z_size+y_size, n_h+1, dtype=int))
		#print(self.hidden_sizes)

		self.encode = Encoder(x_size, z_size, y_size, self.hidden_sizes, n_h)
		self.decode = Decoder(x_size, z_size, y_size, self.hidden_sizes, n_h)

	def reparametrize(self, mu, logvar):
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		return mu + eps*std

	def forward(self, x, y):
		z_mu, z_logvar = self.encode(x, y)
		z = self.reparametrize(z_mu, z_logvar)
		recon_x = self.decode(z, y)
		return recon_x, z_mu, z_logvar, z


class Encoder(nn.Module):

	def __init__(self, x_size, z_size, y_size, hidden_sizes, n_hidden_layers):
		super().__init__()

		self.neuralnet = nn.Sequential()

		for i in range(n_hidden_layers):

			inp_size = x_size+y_size if i == 0 else hidden_sizes[i-1]
			out_size = hidden_sizes[i]

			self.neuralnet.add_module(name=f"linear_hidd_{i}", module=nn.Linear(inp_size, out_size))
			self.neuralnet.add_module(name=f"activ_hidd_{i}", module=nn.ReLU())
			self.neuralnet.add_module(name=f"bn_hidd_{i}", module=nn.BatchNorm1d(out_size))

		self.out_means = nn.Linear(hidden_sizes[-2], z_size)
		self.out_logvars = nn.Linear(hidden_sizes[-2], z_size)

	def forward(self, x, y):
		x = torch.cat([x, y], 1)
		x = self.neuralnet(x)
		z_mu = self.out_means(x)
		z_logvar = self.out_logvars(x)
		return z_mu, z_logvar


class Decoder(nn.Module):

	def __init__(self, x_size, z_size, y_size,  hidden_sizes, n_hidden_layers):
		super().__init__()

		self.neuralnet = nn.Sequential()

		for i in range(n_hidden_layers):

			inp_size = z_size+y_size if i == 0 else hidden_sizes[-(i+1)]
			out_size = hidden_sizes[-(i+2)]

			self.neuralnet.add_module(name=f"linear_hidd_{i}", module=nn.Linear(inp_size, out_size))
			self.neuralnet.add_module(name=f"activ_hidd_{i}", module=nn.ReLU())
			self.neuralnet.add_module(name=f"bn_hidd_{i}", module=nn.BatchNorm1d(out_size))

		self.neuralnet.add_module(name=f"linear_out", module=nn.Linear(hidden_sizes[0], x_size))
		self.neuralnet.add_module(name=f"sigmoid", module=nn.Sigmoid())

	def forward(self, z, y):
		x = torch.cat([z, y], 1)
		x = self.neuralnet(x)
		return x
