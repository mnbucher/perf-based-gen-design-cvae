import torch
from torch import nn
import numpy as np
import math
import torch.nn.functional as F


class customSequential(nn.Sequential):
	def forward(self, *inputs):
		for module in self._modules.values():
			inputs = module(*inputs)
		return inputs


class CVAE5(nn.Module):

	def __init__(self, model_config, va):
		super().__init__()

		self.dvc = va.dvc

		x_dim = model_config.get("x_dim")
		y_dim = model_config.get("y_dim")

		z_dim = model_config.get("model_params").get("z_dim")
		h_dim = model_config.get("model_params").get("h_dim")
		n_h = model_config.get("model_params").get("n_h")

		h_context_dim = model_config.get("model_params").get("h_context_dim")
		self.k_flows = model_config.get("model_params").get("k_flows")

		self.hidden_sizes = list(np.linspace(h_dim, z_dim, n_h+1, dtype=int))
		#print(self.hidden_sizes)

		self.encoder = Encoder(x_dim, z_dim, y_dim, self.hidden_sizes, n_h, h_context_dim, self.k_flows)
		self.iaf_posterior = IAFPosterior(z_dim, h_context_dim, x_dim, y_dim, self.k_flows)

		self.decoder = Decoder(x_dim, z_dim, y_dim, self.hidden_sizes, n_h)

		self.logpi = torch.log(torch.tensor(2*math.pi, device=self.dvc))

	def reparametrize(self, mu, logvar):
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)

		z0 = mu + eps*std

		log_q_z0 = -(torch.log(std) + 0.5*(eps**2) + 0.5*self.logpi)

		return z0, log_q_z0

	def forward(self, x, y):
		# recognition phase
		z_mu, z_logvar, h_context = self.encoder(x, y)

		z0, log_q_z0 = self.reparametrize(z_mu, z_logvar)

		# IAF
		if self.k_flows > 0:
			zk, logdetjacob = self.iaf_posterior(z0, torch.cat([h_context, x, y], 1), torch.zeros(z0.shape, device=self.dvc))
			kl_posterior = log_q_z0 - logdetjacob
		else:
			kl_posterior = log_q_z0

		# standard normal prior for z_k
		kl_prior = -0.5*(self.logpi + zk**2)

		kl_divergence = kl_posterior - kl_prior

		# generative phase
		recon_x = self.decoder(zk, y)

		return recon_x, kl_divergence, zk


class Encoder(nn.Module):

	def __init__(self, x_dim, z_dim, y_dim, hidden_sizes, n_h, h_context_dim, k_flows):
		super().__init__()

		self.z_dim = z_dim
		self.neuralnet = customSequential()

		for i in range(n_h):
			inp_size = x_dim + y_dim if i == 0 else hidden_sizes[i-1] + y_dim
			out_size = hidden_sizes[i]
			self.neuralnet.add_module(name=f"enc_mlp_block_{i}", module=MLPBlock(inp_size, out_size))

		self.z_mean = nn.Linear(hidden_sizes[-2], z_dim)
		self.z_logvar = nn.Linear(hidden_sizes[-2], z_dim)
		self.h_context = nn.Linear(hidden_sizes[-2], h_context_dim)


	def forward(self, x, y):

		x, _ = self.neuralnet(x, y)

		z_mu = self.z_mean(x)
		z_logvar = self.z_logvar(x)
		h_context = nn.Sigmoid()(self.h_context(x))

		return z_mu, z_logvar, h_context


class Decoder(nn.Module):

	def __init__(self, x_dim, z_dim, y_dim,  hidden_sizes, n_h):
		super().__init__()

		self.neuralnet = customSequential()

		for i in range(n_h):

			inp_size = z_dim + y_dim if i == 0 else hidden_sizes[-(i+1)] + y_dim
			out_size = hidden_sizes[-(i+2)]

			self.neuralnet.add_module(name=f"dec_mlp_block_{i}", module=MLPBlock(inp_size, out_size))

		self.linear_out = nn.Linear(hidden_sizes[0], x_dim)


	def forward(self, z, y):
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


class IAFPosterior(nn.Module):

	def __init__(self, z_dim, h_context_dim, x_dim, y_dim, k_flows):
		super().__init__()

		self.z_dim = z_dim
		self.h_context_dim = h_context_dim + x_dim + y_dim
		self.k_flows = k_flows

		self.neuralnet = customSequential()

		for i in range(self.k_flows):
			self.neuralnet.add_module(name=f"iaf_layer_{i}", module=IAFLayer(self.z_dim, self.h_context_dim))

	def forward(self, z, h_context, logdetjacob):
		z, _, logdetjacob = self.neuralnet(z, h_context, logdetjacob)

		return z, logdetjacob


class IAFLayer(nn.Module):

	def __init__(self, z_dim, h_context_dim):
		super().__init__()

		self.z_dim = z_dim
		self.h_context_dim = h_context_dim

		self.sigmoid_arg_bias = nn.Parameter(torch.ones(z_dim) * 1.0)

		# 2 layer MADE
		self.ar_nn = MADE(self.z_dim, self.z_dim*3, 2, self.h_context_dim)


	def forward(self, z, h_context, logdetjacob):

		m_t, s_t = torch.chunk(self.ar_nn(z, h_context), chunks=2, dim=-1)

		# from IAF paper: shift up to be closer to 1.0 after sigmoid ??
		s_t = s_t + self.sigmoid_arg_bias

		sigma_t = nn.Sigmoid()(s_t)

		z_t = (sigma_t * z) + (1 - sigma_t) * m_t

		logdetjacob += nn.LogSigmoid()(s_t)
		#logdetjacob += torch.sum(nn.LogSigmoid()(s_t), axis=1)

		# TODO: reverse order after each layer ?? (as in IAF paper)

		return z_t, h_context, logdetjacob


class MADE(nn.Module):

	def __init__(self, z_dim, hidd_dim, n_output, hidd_context_dim):
		super().__init__()

		self.m = []
		degrees = create_degrees(input_size=z_dim, hidden_units=[ hidd_dim ] * 2, input_order="left-to-right", hidden_degrees="equal")

		self.masks = create_masks(degrees)
		self.masks[-1] = np.hstack( [ self.masks[-1] for _ in range(n_output) ] )
		self.masks = [torch.from_numpy(m.T) for m in self.masks]

		self.nn_input_context = MaskedLinear(z_dim, hidd_dim, self.masks[0], hidd_context_dim)

		self.nn = nn.Sequential(
			nn.ReLU(),
			MaskedLinear(hidd_dim, hidd_dim, self.masks[1], context_dim=None),
			nn.ReLU(),
			MaskedLinear(hidd_dim, n_output * z_dim, self.masks[2], context_dim=None)
		)

	def forward(self, z, h_context):
		x = self.nn_input_context(z, h_context)
		x = self.nn(x)

		return x


class MaskedLinear(nn.Module):

	def __init__(self, inp_dim, out_dim, mask, context_dim=None):
		super().__init__()

		self.linear = nn.Linear(inp_dim, out_dim)

		self.register_buffer("mask", mask)

		if context_dim is not None:
			self.context_proj = nn.Linear(context_dim, out_dim)

	def forward(self, z, context=None):

		x = F.linear(z, self.mask * self.linear.weight, self.linear.bias)

		if context is not None:
			x = x + self.context_proj(context)

		return x


def create_degrees(input_size, hidden_units, input_order="left-to-right", hidden_degrees="equal"):
	input_order = create_input_order(input_size, input_order)
	degrees = [input_order]
	for units in hidden_units:
		if hidden_degrees == "random":
			# samples from: [low, high)
			degrees.append(
				np.random.randint(
					low=min(np.min(degrees[-1]), input_size - 1),
					high=input_size,
					size=units,
				)
			)
		elif hidden_degrees == "equal":
			min_degree = min(np.min(degrees[-1]), input_size - 1)
			degrees.append(
				np.maximum(
					min_degree,
					# Evenly divide the range `[1, input_size - 1]` in to `units + 1`
					# segments, and pick the boundaries between the segments as degrees.
					np.ceil(
						np.arange(1, units + 1) * (input_size - 1) / float(units + 1)
					).astype(np.int32),
				)
			)
	return degrees


def create_masks(degrees):
	return [ inp[:, np.newaxis] <= out for inp, out in zip(degrees[:-1], degrees[1:]) ] + [ degrees[-1][:, np.newaxis] < degrees[0] ]


def create_input_order(input_size, input_order="left-to-right"):

	if input_order == "left-to-right":
		return np.arange(start=1, stop=input_size + 1)

	elif input_order == "right-to-left":
		return np.arange(start=input_size, stop=0, step=-1)

	elif input_order == "random":
		ret = np.arange(start=1, stop=input_size + 1)
		np.random.shuffle(ret)
		return ret
