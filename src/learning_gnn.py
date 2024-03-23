import torch
from torch.nn import functional as tfunc
from torchvision.utils import save_image
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from matplotlib import pyplot as plt
from datetime import datetime
import numpy as np
from tqdm import tqdm
import logging
from sklearn.metrics import mean_squared_error

import src.utility as utility
import src.plot as plot
import src.dataset as dataset
import src.learning as learning
import src.analysis as analysis


def gnn_compute_forward_pass_and_loss(model, va, batch):

	x, y, g = batch

	model_type = va.model_config.get("model_type")

	# (1) forward pass
	if "gcnn" in model_type:
		graph_dataloader = dataset.get_torch_geometric_dataloader(x, y, g, va, do_normalize_coords=False, do_add_self_loops=True, do_node_features_x10d=False, do_edge_features=True)
		batch = next(iter(graph_dataloader))
		y_hat, _ = model(batch, x)

	else:
		y_hat = model(x)

	# if model_type == "gnn0":
	# 	y_hat = model(x)

	# if model_type == "gnn1":
	# 	y_hat = model(g, x)

	# if model_type in ["gnn2", "gnn3"]:
	# 	graph_dataloader = dataset.get_torch_geometric_dataloader(x, y, g, va, do_normalize_coords=False, do_add_self_loops=True, do_node_features_x10d=True)
	# 	batch = next(iter(graph_dataloader))
	# 	y_hat, _ = model(batch)

	# if model_type == "gnn4":
	# 	graph_dataloader = dataset.get_torch_geometric_dataloader(x, y, g, va, do_normalize_coords=False, do_add_self_loops=True, do_node_features_x10d=False)
	# 	batch = next(iter(graph_dataloader))
	# 	y_hat, _, _ = model(batch, x)

	# if model_type == "gnn5":
	# 	graph_dataloader = dataset.get_torch_geometric_dataloader(x, y, g, va, do_normalize_coords=False, do_add_self_loops=True, do_node_features_x10d=False, do_edge_features=True)
	# 	batch = next(iter(graph_dataloader))
	# 	y_hat, _, _ = model(batch, x)

	# if model_type == "gnn6":
	# 	graph_dataloader = dataset.get_torch_geometric_dataloader(x, y, g, va, do_normalize_coords=False, do_add_self_loops=True, do_node_features_x10d=False, do_edge_features=True)
	# 	batch = next(iter(graph_dataloader))
	# 	y_hat, _, _ = model(batch, x)

	loss, loss_elemwise = va.loss_func(va.dvc, y, y_hat, va)

	# since BCEWithLogits is used, perform Sigmoid again here explicitly
	if va.model_config.get("is_bce_loss"):
		y_hat = torch.nn.Sigmoid()(y_hat)

	return loss, loss_elemwise, y, y_hat


def gnn_train(epoch, model, optimizer, va):
	model.train()

	y_dim = va.model_config.get("y_dim")

	lr = learning.adapt_learning_rate(optimizer, epoch, va)

	if va.cli_args.is_first_batch_only:
		n_total = va.model_config.get("model_params").get("batch_size")
	else:
		n_total = va.dataset_train.__len__()

	total_loss = torch.zeros(n_total, device=va.dvc)
	all_y_unnorm = torch.zeros(n_total, y_dim, device=va.dvc)
	all_y_hat_unnorm = torch.zeros(n_total, y_dim, device=va.dvc)
	dataset_stats = va.dataset_train.get_dataset_stats()

	idx = 0

	iterator = va.data_loader_train if va.cli_args.is_logging_enabled else tqdm(va.data_loader_train)
	for batch in iterator:

		loss, loss_elemwise, y, y_hat = gnn_compute_forward_pass_and_loss(model, va, batch)

		loss.backward()
		optimizer.step()
		optimizer.zero_grad()

		total_loss[idx:(idx+y_hat.shape[0])] = loss_elemwise
		all_y_unnorm[idx:(idx+y_hat.shape[0]), :] = dataset.unnormalize_data(y, dataset_stats[2], dataset_stats[3])
		all_y_hat_unnorm[idx:(idx+y_hat.shape[0]), :] = dataset.unnormalize_data(y_hat, dataset_stats[2], dataset_stats[3])

		idx += y_hat.shape[0]

		if va.cli_args.is_first_batch_only:
			break

	mape_dimwise, mape, rmse, pcc = analysis.compute_error_metrics(all_y_hat_unnorm.detach().cpu().numpy(), all_y_unnorm.detach().cpu().numpy(), do_print=False)

	total_loss = torch.mean(total_loss).item()
	mape_avg_total = np.mean(mape_dimwise)

	if va.cli_args.is_vanilla_training:
		utility.logtext(f"[gnn] train mape_dimwise: {mape_dimwise}", va)

		utility.logtext(f"[gnn] train loss per epoch: {total_loss}", va)
		utility.log_metric(va, "[gnn] train loss per epoch", total_loss, epoch)

		utility.logtext(f"[gnn] train MAPE (%) per epoch: {mape_avg_total}", va)
		utility.log_metric(va, "[gnn] train MAPE (%) per epoch", mape_avg_total, epoch)

	return lr


def gnn_test(epoch, model, va):
	model.eval()

	torch.set_printoptions(sci_mode=False)
	np.set_printoptions(suppress=True)

	y_dim = va.model_config.get("y_dim")

	if va.cli_args.is_first_batch_only:
		n_total = va.model_config.get("model_params").get("batch_size")
	else:
		n_total = va.dataset_test.__len__()

	total_loss = torch.zeros(n_total, device=va.dvc)
	all_y_unnorm = torch.zeros(n_total, y_dim, device=va.dvc)
	all_y_hat_unnorm = torch.zeros(n_total, y_dim, device=va.dvc)
	dataset_stats = va.dataset_train.get_dataset_stats()

	idx = 0
	with torch.no_grad():
		iterator = va.data_loader_test if va.cli_args.is_logging_enabled else tqdm(va.data_loader_test)
		for batch in iterator:

			loss, loss_elemwise, y, y_hat = gnn_compute_forward_pass_and_loss(model, va, batch)

			total_loss[idx:(idx+y_hat.shape[0])] = loss_elemwise
			all_y_unnorm[idx:(idx+y_hat.shape[0]), :] = dataset.unnormalize_data(y, dataset_stats[2], dataset_stats[3])
			all_y_hat_unnorm[idx:(idx+y_hat.shape[0]), :] = dataset.unnormalize_data(y_hat, dataset_stats[2], dataset_stats[3])

			idx += y_hat.shape[0]

			if va.cli_args.is_first_batch_only:
				break

	mape_dimwise, mape, rmse, pcc = analysis.compute_error_metrics(all_y_hat_unnorm.detach().cpu().numpy(), all_y_unnorm.detach().cpu().numpy(), do_print=False)

	if not va.cli_args.is_logging_enabled:
		#plot.plot_baseline_y_error_distr_for_y_test(va, mape, va.cli_args.dataset)
		#analysis.check_mape_error_uniformly_sampled(test_mape_y_hat_dim_wise, all_y_unnorm)
		pass

	total_loss = torch.mean(total_loss).item()
	mape_avg_total = np.mean(mape_dimwise)

	if va.cli_args.is_vanilla_training:

		utility.logtext(f"[gnn] test loss per epoch: {total_loss}", va)
		utility.log_metric(va, "[gnn] test loss per epoch", total_loss, epoch)

		utility.logtext(f"[gnn] test mape_dimwise: {mape_dimwise}", va)

		utility.logtext(f"[gnn] test MAPE (%) per epoch: {mape_avg_total}", va)
		utility.log_metric(va, "[[gnn] test MAPE (%) per epoch", mape_avg_total, epoch)

		utility.logtext(f"[gnn] test RMSE per epoch: {rmse}", va)
		utility.logtext(f"[gnn] test PCC per epoch: {pcc} with mean {np.mean(pcc)}", va)

		for dim in range(mape_dimwise.shape[0]):
			utility.log_metric(va, f"[gnn] test MAPE y_{dim} (%) per epoch", mape_dimwise[dim].item(), epoch)

	return total_loss, mape_avg_total, mape_dimwise, rmse, pcc
