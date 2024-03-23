import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt
import torchist
import logging

import src.utility as utility


def gnn_loss(dvc, y, y_hat, va):

	loss_elemwise = torch.sum(torch.nn.BCELoss(reduction='none')(y_hat, y), axis=1)

	# weight loss
	if va.model_config.get("weighted_loss"):
		loss_weights = get_loss_weights(dvc, va, y)
		loss_elemwise_weighted = torch.multiply(loss_elemwise, loss_weights)
		loss = torch.mean(loss_elemwise_weighted)
	else:
		loss = torch.mean(loss_elemwise)

	return loss, loss_elemwise


def get_recon_loss(x_recon, x, va):

	#if is_bce_loss:
		#return torch.mean(torch.nn.BCEWithLogitsLoss(reduction='none')(x_recon, x), axis=1)

	if va.dim_config is not None:

		x_dim_orig = va.dim_config.get("x_dim_orig")

		recon_loss_all = torch.zeros(x_recon.shape[0], x_dim_orig, device=va.dvc, dtype=torch.float64)

		# BCE loss for continuous dimensions
		bce_mask = va.dim_config.get("x_dims_conti")
		sigmoid_mask = va.dim_config.get("x_output_masks").get("sigmoid")
		recon_loss_all[:, bce_mask] = torch.nn.BCELoss(reduction='none')(x_recon[:, sigmoid_mask], x[:, sigmoid_mask])

		# CE loss for one hot encoded categorical dimensions
		softmax_masks = va.dim_config.get("x_output_masks").get("softmax")
		ce_idxs = np.setdiff1d(list(range(x_dim_orig)), bce_mask)
		for ce_idx, softmax_mask in zip(ce_idxs, softmax_masks):
			recon_loss_all[:, ce_idx] = torch.nn.CrossEntropyLoss(reduction='none')(x_recon[:, softmax_mask], x[:, softmax_mask])
	else:
		recon_loss_all = torch.nn.BCELoss(reduction='none')(x_recon, x)

	#print(torch.sum(torch.nn.BCELoss(reduction='none')(x_recon, x), axis=1))
	#print(torch.sum(recon_loss_all, axis=1))
	#exit()

	return torch.sum(recon_loss_all, axis=1)

	#return torch.sum((x_recon - x)**2, axis=1)


def get_loss_kld(dvc, logvar, mu, y, model_type):
	# -KLD = 0.5 * ...
	# +KLD = -0.5 * ...

	kld = 1 + logvar - mu**2 - logvar.exp()

	loss_kld_dimwise = -0.5 * torch.mean(kld, dim = 0)
	loss_kld_elemwise = -0.5 * torch.sum(kld, dim = 1)

	return loss_kld_dimwise, loss_kld_elemwise


def get_loss_x_partially_fixed(dvc, x_recon, y_x_partial):

	if y_x_partial is None:
		return torch.zeros(x_recon.shape[0], device=dvc, dtype=torch.float64)

	mask = y_x_partial != -1.0
	diffs = torch.zeros(y_x_partial.shape, device=dvc, dtype=torch.float64)
	diffs[mask] = torch.nn.BCELoss(reduction='none')(x_recon[mask], y_x_partial[mask])

	loss_x_partially_fixed = torch.sum(diffs, axis=1)

	return loss_x_partially_fixed


def iaf_loss(dvc, x_recon, kl_divergence, x, y, va, epoch, y_x_partial):

	#model_type = va.model_config.get("model_type")
	lambda_y_x_partial = va.model_config.get("lambda_y_x_partial")
	beta = va.model_config.get("model_params").get("beta_kl")
	kld_min_free_bits = va.model_config.get("kld_min_free_bits")

	# ****************************************************************************

	loss_recon_elemwise = get_recon_loss(x_recon, x, va)

	# ****************************************************************************

	loss_yxpart_elemwise = get_loss_x_partially_fixed(dvc, x_recon, y_x_partial)

	# ****************************************************************************

	loss_vae = torch.mean(loss_recon_elemwise) + lambda_y_x_partial*torch.mean(loss_yxpart_elemwise)

	if kl_divergence.shape[1] > 0:
		loss_kld_elemwise = torch.sum(kl_divergence, axis=1)
		loss_kld_dimwise = torch.max(beta*torch.mean(kl_divergence, axis=0), torch.tensor(kld_min_free_bits))
		loss_vae += torch.sum(loss_kld_dimwise)

	else:
		# for cvae7yxpart_fullcov we already have the sum for each sample (logdet...)
		loss_kld_elemwise = kl_divergence
		loss_vae += torch.mean(loss_kld_elemwise)

	# ****************************************************************************

	loss_elemwise = loss_recon_elemwise + loss_kld_elemwise + loss_yxpart_elemwise

	# ****************************************************************************

	return loss_vae, loss_elemwise, loss_recon_elemwise, loss_kld_elemwise, loss_yxpart_elemwise


def vae_loss(dvc, x_recon, x, mu, logvar, y, va, curr_epoch, y_x_partial=None):

	# maximize log likelihood: Reconstr - KLD
	# <=> minimize (- Reconstr) + KLD
	# <=> minimize BCE + KLD

	# set beta for KL divergence
	if va.model_config.get("is_beta_schedule"):
		beta = va.schedule_beta_kld[curr_epoch-1]
	else:
		beta = va.model_config.get("model_params").get("beta_kl")

	kld_min_free_bits = va.model_config.get("kld_min_free_bits")
	lambda_y_x_partial = va.model_config.get("lambda_y_x_partial")
	model_type = va.model_config.get("model_type")

	# ****************************************************************************

	loss_recon_elemwise = get_recon_loss(x_recon, x, va)

	# ****************************************************************************

	loss_kld_dimwise, loss_kld_elemwise = get_loss_kld(dvc, logvar, mu, y, model_type)

	# ****************************************************************************

	loss_yxpart_elemwise = get_loss_x_partially_fixed(dvc, x_recon, y_x_partial)

	# ****************************************************************************

	loss_elemwise = loss_recon_elemwise + loss_kld_elemwise + loss_yxpart_elemwise

	# ****************************************************************************

	#loss_vae = torch.mean(loss_recon) + torch.mean(torch.max(beta*loss_kld_dimwise, torch.tensor(kld_min_free_bits))) + torch.mean(loss_y_x_partially_fixed)

	loss_vae = torch.mean(loss_recon_elemwise) + torch.sum(torch.max(beta*loss_kld_dimwise, torch.tensor(kld_min_free_bits))) + lambda_y_x_partial*torch.mean(loss_yxpart_elemwise)

	return loss_vae, loss_elemwise, loss_recon_elemwise, loss_kld_elemwise, loss_yxpart_elemwise


def get_loss_weights(dvc, va, y):

	lambda_lossweight = va.model_config.get("lambda_lossweight")
	n = y.shape[0]

	# y1d case
	if y.shape[1] == 1:
		label_bin_edges = va.dataset_train.y_bin_edges.get("y_bin_edges_dim_1")

		hist = torchist.histogram(y, edges=label_bin_edges).double().to(dvc)
		hist_nonzeros = hist[hist != 0.0]

		c = len(hist_nonzeros)

		bin_weights = torch.zeros(len(hist), device=dvc)
		bin_weights[hist != 0.0] = torch.pow(n / (c * hist_nonzeros), lambda_lossweight)
		bin_idxs = torch.squeeze(torch.bucketize(y, label_bin_edges) - 1, axis=1) #.to(dvc)

		lw = bin_weights[bin_idxs]

		return lw

	# y2d case
	else:
		label_bin_edges_dim_1 = va.dataset_train.y_bin_edges.get("y_bin_edges_dim_1")
		label_bin_edges_dim_2 = va.dataset_train.y_bin_edges.get("y_bin_edges_dim_2")

		hist = torchist.histogramdd(y[:, :2], edges=[label_bin_edges_dim_1, label_bin_edges_dim_2]).double().to(dvc)
		hist_nonzero_idxs = torch.nonzero(hist, as_tuple=True)

		c = torch.tensor([len(hist_nonzero_idxs[0])], device=dvc)

		bin_weights = torch.zeros(hist.shape, device=dvc, dtype=torch.float64)
		bin_weights[hist_nonzero_idxs] = torch.pow(n / (c * hist[hist_nonzero_idxs]), lambda_lossweight)

		bin_idxs_dim_1 = torch.bucketize(y[:, 0].contiguous(), label_bin_edges_dim_1) - 1
		bin_idxs_dim_2 = torch.bucketize(y[:, 1].contiguous(), label_bin_edges_dim_2) - 1

		return bin_weights[bin_idxs_dim_1, bin_idxs_dim_2]


def weighted_vae_loss(dvc, x_recon, x, mu, logvar, y, va, curr_epoch, y_x_partial=None):

	if va.model_config.get("is_beta_schedule"):
		beta = va.schedule_beta_kld[curr_epoch-1]
	else:
		beta = va.model_config.get("model_params").get("beta_kl")

	lambda_y_x_partial = va.model_config.get("lambda_y_x_partial")
	model_type = va.model_config.get("model_type")

	# ****************************************************************************
	# fix cuda issue

	x_recon = torch.where(torch.isnan(x_recon), torch.zeros_like(x_recon), x_recon)
	x_recon = torch.where(torch.isinf(x_recon), torch.zeros_like(x_recon), x_recon)

	# ****************************************************************************
	# reconstruction loss

	loss_recon = get_recon_loss(x_recon, x, va)

	# ****************************************************************************
	# kl divergence

	loss_kld = get_loss_kld(dvc, logvar, mu, y, model_type)

	# ****************************************************************************

	loss_y_x_partially_fixed = get_loss_x_partially_fixed(dvc, x_recon, y_x_partial)

	# ****************************************************************************
	# weight losses inverse proportional to label frequency

	loss_weights = get_loss_weights(dvc, va, y)

	# ****************************************************************************

	loss_elemwise = loss_recon + (beta*loss_kld) + (lambda_y_x_partial*loss_y_x_partially_fixed)
	loss_elemwise_weighted = torch.multiply(loss_elemwise, loss_weights)

	# ****************************************************************************

	loss_vae = torch.mean(loss_elemwise_weighted)

	# ****************************************************************************

	return loss_vae, loss_elemwise_weighted, loss_recon, loss_kld, loss_y_x_partially_fixed


def mseloss():
	return nn.MSELoss()


def get_loss_function(va):

	model_type = va.model_config.get("model_type")

	if "cvae" in model_type:
		if va.model_config.get("weighted_loss") is True:
			utility.logtext("using weighted loss", va)
			return weighted_vae_loss

		utility.logtext("using NO weighted loss", va)
		return vae_loss

	elif "forward" in model_type:
		utility.logtext("using gnn loss", va)
		return gnn_loss

	return mseloss
