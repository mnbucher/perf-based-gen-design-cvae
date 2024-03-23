import torch
from torch.nn import functional as tfunc
from torchvision.utils import save_image
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from matplotlib import pyplot as plt
from datetime import datetime
from tqdm import tqdm
import numpy as np
import logging

import src.utility as utility
import src.plot as plot
import src.dataset as dataset
from src.loss import iaf_loss


def get_schedule_beta_kld_cyclic_linear(n_cold_start, n_epoch, max_beta, n_cycle=4, ratio=0.5):
	cold_start = np.zeros(n_cold_start)
	n_schedule = n_epoch-n_cold_start
	schedule = np.tile(max_beta, n_schedule)
	period = float(n_schedule) / float(n_cycle)
	step = max_beta / (period*ratio) # linear schedule

	for c in range(n_cycle):
		v = 0.0
		i = 0
		while v <= max_beta and (int(i+c*period) < n_schedule):
			# linear
			schedule[int(i+c*period)] = v
			v += step
			i += 1

	schedule = np.concatenate((cold_start, schedule))
	return schedule


def adapt_learning_rate(optimizer, epoch, va):
	lr = va.lr
	if epoch in va.cli_args.lr_schedule:
		utility.logtext(f"learning rate is changed from {lr} to {lr * va.cli_args.lr_gamma}...", va)
		lr *= va.cli_args.lr_gamma
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr
	return lr


def compute_mape(x, x_recon, va, is_yxpart=False):

	# print("before mape computation")
	# print("")
	# print("x", x)
	# print("x_recon", x_recon)
	# print("")

	if is_yxpart:
		nonzero_idxs = x != -1.0

		x_unnorm_masked = torch.zeros(x.shape, device=va.dvc)
		x_unnorm_masked[nonzero_idxs] = x[nonzero_idxs]

		x_recon_unnorm_masked = torch.zeros(x.shape, device=va.dvc)
		x_recon_unnorm_masked[nonzero_idxs] = x_recon[nonzero_idxs]

		mape = 100 * torch.divide(torch.abs(x_recon_unnorm_masked - x_unnorm_masked), x_unnorm_masked + 1e-10)
		#print(mape)
		return mape

	mape = 100 * torch.divide(torch.abs(x_recon - x), x + 1e-10)
	#print(mape)
	return mape


def compute_mape_yxpart(y_x_partial, x_recon, va):

	mape = compute_mape(y_x_partial, x_recon, va, is_yxpart=True)

	return torch.divide(torch.sum(mape, axis=1), torch.count_nonzero(mape, axis=1) + 1e-10)


def create_y_x_partial(x, va, epoch=None, is_train=False):

	y_x_partial_indices = va.dataset_train.y_x_partial_indices
	y_x_partial = torch.full((x.shape), -1.0, device=va.dvc)

	y_x_indices_sampled = y_x_partial_indices[torch.randint(low=0, high=y_x_partial_indices.shape[0], size=(x.shape[0],)), :]
	y_x_partial_filled = torch.full((x.shape), -1.0, device=va.dvc)
	y_x_partial_filled[y_x_indices_sampled.bool()] = x[y_x_indices_sampled.bool()]

	# 50% probability return some y_x_vector
	if is_train:
		take_y_x_part_mask = torch.randint(0, x.shape[0], (int(x.shape[0]/2.0),)).double().to(va.dvc).long()
	else:
		take_y_x_part_mask = torch.arange(x.shape[0])

	y_x_partial[take_y_x_part_mask, :] = y_x_partial_filled[take_y_x_part_mask, :]

	return y_x_partial


def get_pretrained_gnn_embedding(x, y, g, va):
	graph_dataloader = dataset.get_torch_geometric_dataloader(x, y, g, va, do_normalize_coords=False, do_add_self_loops=True, do_node_features_x10d=False, do_edge_features=True)
	g_batch = next(iter(graph_dataloader))
	_, z_g = va.gnn_model(g_batch, x)
	return z_g


def compute_forward_pass_and_loss(model_type, model, va, epoch, batch):

	if "cvae" in model_type:

		y_x_partial = None

		# (1) extract batch
		if "gcnn" in model_type:
			x, y, g = batch
		else:
			x, y = batch

		# (2) make forward pass
		# NORMALIZING FLOWS MODELS
		if "nf" in model_type:
			if "yxpart" in model_type:
				y_x_partial = create_y_x_partial(x, va, epoch)
				x_recon, kl_divergence, z = model(x, y, y_x_partial)

			elif "gcnn" in model_type:
				z_g = get_pretrained_gnn_embedding(x, y, g, va)
				x_recon, x_g, kl_divergence, z = model(x, y, z_g)
				x = x_g

			else:
				x_recon, kl_divergence, z = model(x, y)

			loss, loss_elemwise, loss_recon, loss_kld, total_loss_yxpart = iaf_loss(va.dvc, x_recon, kl_divergence, x, y, va, epoch, y_x_partial)

		# OTHER MODELS (NO NF)
		else:
			if "yxpart" in model_type:
				y_x_partial = create_y_x_partial(x, va, epoch)
				x_recon, mu, logvar, z = model(x, y, y_x_partial)

			elif "gcnn" in model_type:
				z_g = get_pretrained_gnn_embedding(x, y, g, va)
				x_recon, x_g, mu, logvar, z = model(x, y, z_g)
				x = x_g

			else:
				x_recon, mu, logvar, z = model(x, y)

			loss, loss_elemwise, loss_recon, loss_kld, total_loss_yxpart = va.loss_func(va.dvc, x_recon, x, mu, logvar, y, va, epoch, y_x_partial)

	#print(x[0, :])
	#print(x_recon[0, :])

	# unnormalize continuous dimensions
	# decode one-hot-encoded dimensions
	dataset_stats = va.dataset_train.get_dataset_stats()	
	x = dataset.decode_cat_and_unnorm_cont_dims_for_x(x, va, dataset_stats)
	x_recon = dataset.decode_cat_and_unnorm_cont_dims_for_x(x_recon, va, dataset_stats)

	if y_x_partial is not None:
		y_x_partial = dataset.decode_y_x_partial(y_x_partial, va, dataset_stats)

	# print("")
	# print("after decoding and unnorm")
	# print("")
	# print(x)
	# print(x_recon)
	# print("")

	#exit()

	# 10d-bridge dataset
	# pick only first 10 dims for bridge dataset when using GCNN embeddings
	if va.cli_args.pretrained_gnn:
		x = x[:, :10]
		x_recon = x_recon[:, :10]

	return x, y, x_recon, y_x_partial, loss, loss_elemwise, loss_recon, loss_kld, total_loss_yxpart, z


def train(epoch, model, optimizer, va):
	model.train()

	model_type = va.model_config.get("model_type")
	x_dim = va.model_config.get("x_dim")

	lr = adapt_learning_rate(optimizer, epoch, va)

	len_dataset = va.dataset_train.__len__()
	if model_type == "cvae5":
		n_total = len_dataset*va.model_config.get("y_x_partial_indices_subsample_n")
	elif va.cli_args.is_first_batch_only:
		n_total = va.model_config.get("model_params").get("batch_size")
	else:
		n_total = len_dataset

	total_loss = torch.zeros(n_total, device=va.dvc)
	total_loss_recon = torch.zeros(n_total, device=va.dvc)
	total_loss_kld = torch.zeros(n_total, device=va.dvc)
	total_loss_yxpart = torch.zeros(n_total, device=va.dvc)

	total_mape = torch.zeros((n_total, va.dim_config.get("x_dim_orig")), device=va.dvc)
	total_mape_yxpart_elemwise = torch.zeros((n_total), device=va.dvc)

	idx = 0
	iterator = va.data_loader_train if va.cli_args.is_logging_enabled else tqdm(va.data_loader_train)

	for batch in iterator:

		x, y, x_recon, y_x_partial, loss, loss_elemwise, loss_recon, loss_kld, loss_yxpart, z = compute_forward_pass_and_loss(model_type, model, va, epoch, batch)

		loss.backward()
		optimizer.step()
		optimizer.zero_grad()

		total_loss[idx:(idx+x_recon.shape[0])] = loss_elemwise
		total_loss_recon[idx:(idx+x_recon.shape[0])] = loss_recon
		total_loss_kld[idx:(idx+x_recon.shape[0])] = loss_kld
		total_loss_yxpart[idx:(idx+x_recon.shape[0])] = loss_yxpart
		total_mape[idx:(idx+x_recon.shape[0]), :] = compute_mape(x, x_recon, va)

		if "yxpart" in model_type:
			total_mape_yxpart_elemwise[idx:(idx+x_recon.shape[0])] = compute_mape_yxpart(y_x_partial, x_recon, va)

		idx += x_recon.shape[0]
		if va.cli_args.is_first_batch_only:
			break

	total_loss = torch.mean(total_loss).item()
	total_loss_recon = torch.mean(total_loss_recon).item()
	total_loss_kld = torch.mean(total_loss_kld).item()
	total_loss_yxpart = torch.mean(total_loss_yxpart[torch.nonzero(total_loss_yxpart)]).item()

	total_mape_dimwise = torch.mean(total_mape, axis=0)
	mape_avg = torch.mean(total_mape_dimwise).item()

	total_mape_yxpart_avg = torch.mean(total_mape_yxpart_elemwise[torch.nonzero(total_mape_yxpart_elemwise)]).item()

	if va.cli_args.is_vanilla_training:
		utility.logtext(f"train MAPE (%) per epoch: {total_mape_dimwise}", va)

		if "yxpart" in model_type:
			utility.logtext(f"train MAPE (%) for given y_x_part: {total_mape_yxpart_avg}", va)
			utility.log_metric(va, "train MAPE (%) for given y_x_part:", total_mape_yxpart_avg, epoch)

		utility.logtext(f"train loss per epoch: {total_loss}", va)
		utility.log_metric(va, "train loss per epoch:", total_loss, epoch)

		utility.logtext(f"train recon_loss per epoch: {total_loss_recon}", va)
		utility.log_metric(va, "train recon_loss per epoch", total_loss_recon, epoch)

		utility.logtext(f"train kld_loss per epoch: {total_loss_kld}", va)
		utility.log_metric(va, "train kld_loss per epoch", total_loss_kld, epoch)

		utility.logtext(f"train loss_yxpart per epoch: {total_loss_yxpart}", va)
		utility.log_metric(va, "train loss_yxpart per epoch", total_loss_yxpart, epoch)

		utility.logtext(f"train MAPE (%) per epoch: {mape_avg}", va)
		utility.log_metric(va, "train MAPE (%) per epoch", mape_avg, epoch)

	return lr


def test(epoch, model, va):
	model.eval()

	model_type = va.model_config.get("model_type")
	x_dim = va.model_config.get("x_dim")
	z_dim = va.model_config.get("model_params").get("z_dim")

	len_dataset = va.dataset_test.__len__()
	if model_type == "cvae5":
		n_total = len_dataset*va.model_config.get("y_x_partial_indices_subsample_n")
	elif va.cli_args.is_first_batch_only:
		n_total = va.model_config.get("model_params").get("batch_size")
	else:
		n_total = len_dataset

	total_loss = torch.zeros(n_total, device=va.dvc)
	total_loss_recon = torch.zeros(n_total, device=va.dvc)
	total_loss_kld = torch.zeros(n_total, device=va.dvc)
	total_loss_yxpart = torch.zeros(n_total, device=va.dvc)

	total_mape = torch.zeros((n_total, va.dim_config.get("x_dim_orig")), device=va.dvc)
	total_mape_yxpart_elemwise = torch.zeros((n_total), device=va.dvc)

	all_zs_with_ys = torch.zeros((n_total, z_dim+va.model_config.get("y_dim")), device=va.dvc)
	all_x = torch.zeros((n_total, va.dim_config.get("x_dim_orig")))
	all_x_recon = torch.zeros((n_total, va.dim_config.get("x_dim_orig")))

	idx = 0
	with torch.no_grad():

		for batch in va.data_loader_test:

			x, y, x_recon, y_x_partial, loss, loss_elemwise, loss_recon, loss_kld, loss_yxpart, z = compute_forward_pass_and_loss(model_type, model, va, epoch, batch)

			total_loss[idx:(idx+x_recon.shape[0])] = loss_elemwise
			total_loss_recon[idx:(idx+x_recon.shape[0])] = loss_recon
			total_loss_kld[idx:(idx+x_recon.shape[0])] = loss_kld
			total_loss_yxpart[idx:(idx+x_recon.shape[0])] = loss_yxpart
			total_mape[idx:(idx+x_recon.shape[0]), :] = compute_mape(x, x_recon, va)
			#total_mape_yxpart[] = compute_mape(y_x_partial)

			all_zs_with_ys[idx:(idx+x_recon.shape[0]), :z_dim] = z
			all_zs_with_ys[idx:(idx+x_recon.shape[0]), z_dim:] = y

			all_x[idx:(idx+x_recon.shape[0])] = x
			all_x_recon[idx:(idx+x_recon.shape[0])] = x_recon

			if "yxpart" in model_type:
				total_mape_yxpart_elemwise[idx:(idx+x_recon.shape[0])] = compute_mape_yxpart(y_x_partial, x_recon, va)

			idx += x_recon.shape[0]
			if va.cli_args.is_first_batch_only:
				break


	if not va.cli_args.is_logging_enabled:
		# analyze error
		#plot.plot_error_vs_label_density(all_test_rmse)
		#plot.plot_taylor_diagram(all_x.numpy(), all_x_recon.numpy(), va.run_id)
		#plot.plot_latent_z_binned_x(all_z, all_x, z_dim)
		#plot.plot_latent_z_binned_y(all_z, all_x, z_dim)

		#plot.plot_latent_z_3d_raw(all_zs_with_ys[:, :z_dim])

		#plot.plot_latent_z_3d_colored_by_y(all_zs_with_ys, z_dim)
		#plot.plot_latent_z_3d_colored_by_x(all_zs_with_ys[:, :z_dim], all_x)

		#plot.plot_latent_space_price_ranges(all_zs_with_ys, z_dim, va.dataset_train.get_dataset_stats())

		#plot.plot_mape_hist_dim_wise(all_x_recon, all_x, x_concat_min.numpy(), x_concat_max.numpy())
		pass

	total_loss = torch.mean(total_loss).item()
	total_loss_recon = torch.mean(total_loss_recon).item()
	total_loss_kld = torch.mean(total_loss_kld).item()
	total_loss_yxpart = torch.mean(total_loss_yxpart[torch.nonzero(total_loss_yxpart)]).item()

	total_mape_dimwise = torch.mean(total_mape, axis=0)
	mape_avg = torch.mean(total_mape_dimwise).item()

	total_mape_yxpart_avg = torch.mean(total_mape_yxpart_elemwise[torch.nonzero(total_mape_yxpart_elemwise)]).item()

	if va.cli_args.is_vanilla_training:
		utility.logtext(f"test MAPE (%) per epoch: {total_mape_dimwise}", va)
		for dim in range(total_mape_dimwise.shape[0]):
			utility.log_metric(va, f"test MAPE x_{dim} (%) per epoch", total_mape_dimwise[dim].item(), epoch)

		if "yxpart" in model_type:
			utility.logtext(f"test MAPE (%) for given y_x_part: {total_mape_yxpart_avg}", va)
			utility.log_metric(va, "test MAPE (%) for given y_x_part:", total_mape_yxpart_avg, epoch)

		utility.logtext(f"test loss per epoch: {total_loss}", va)
		utility.log_metric(va, "test loss per epoch", total_loss, epoch)

		utility.logtext(f"test recon_loss per epoch: {total_loss_recon}", va)
		utility.log_metric(va, "test recon_loss per epoch", total_loss_recon, epoch)

		utility.logtext(f"test kld_loss per epoch: {total_loss_kld}", va)
		utility.log_metric(va, "test kld_loss per epoch", total_loss_kld, epoch)

		utility.logtext(f"test loss_yxpart per epoch: {total_loss_yxpart}", va)
		utility.log_metric(va, "test loss_yxpart per epoch", total_loss_yxpart, epoch)

		utility.logtext(f"test MAPE (%) per epoch: {mape_avg}", va)
		utility.log_metric(va, "test MAPE (%) per epoch", mape_avg, epoch)

		utility.logtext("", va)

	return total_loss, total_loss_recon, total_loss_kld
