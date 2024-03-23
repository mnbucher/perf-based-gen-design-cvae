import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import torch
import base64
import requests
import time
import logging
from requests_toolbelt.adapters import host_header_ssl
from datetime import datetime
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import torchist
from skopt.space import Space
from skopt.sampler import Sobol
from pycel import ExcelCompiler
import pdb
from tqdm import tqdm

import src.dataset as dataset
import src.utility as utility
import src.models as models
import src.plot as plot
import src.analysis as analysis
import src.learning as learning
import src.learning_gnn as learning_gnn


def metrics_compute_mape_error_numpy(y_pred, y_true):

	if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
		mape = np.zeros(y_pred.shape)
		for col in range(y_pred.shape[1]):
			mape[:, col] = 100 * abs(((y_pred[:, col] - y_true[:, col]) / y_true[:, col] + 1e-10))

	else:
		mape = 100 * abs(((y_pred - y_true) / y_true))

	return mape


def metrics_compute_rmse_error_numpy(y_pred, y_true):

	if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
		rmse = np.zeros((1, y_pred.shape[1]))
		for col in range(y_pred.shape[1]):
			mask = y_true[:, col] != 0.0
			rmse[0, col] = mean_squared_error(y_true[mask, col], y_pred[mask, col], multioutput='raw_values', squared=False)
	else:
		mask = y_true != 0.0
		rmse = mean_squared_error(y_true[mask], y_pred[mask], squared=False)

	#if is_yxpart:
		#rmse_dimwise = np.sum(rmse, axis=0) / (np.sum(y_true != 0.0) + 1e-10)

	return rmse


def metrics_compute_pcc_error_numpy(y_pred, y_true):

	if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
		pcc = np.zeros((1, y_pred.shape[1]))

		for col in range(y_pred.shape[1]):
			mask = y_true[:, col] != 0.0
			pcc[0, col], _ = pearsonr(y_pred[mask, col], y_true[mask, col])

		#pcc_y0, _ = pearsonr(y_pred[:, 0], y_true[:, 0])
		#pcc_y1, _ = pearsonr(y_pred[:, 1], y_true[:, 1])
		#pcc = np.array([pcc_y0, pcc_y1])
	else:
		mask = y_true != 0.0
		pcc, _ = pearsonr(np.squeeze(y_pred[mask]), np.squeeze(y_true[mask]))

	return pcc


def compute_error_metrics(y_pred, y_true, is_yxpart=False, va=None, do_print=True, do_plot=False, ds=None):

	mape = metrics_compute_mape_error_numpy(y_pred, y_true)
	rmse = metrics_compute_rmse_error_numpy(y_pred, y_true)
	pcc = metrics_compute_pcc_error_numpy(y_pred, y_true)


	if is_yxpart:
		mape_masked = np.zeros(mape.shape)
		mask = y_true != 0.0
		mape_masked[mask] = mape[mask]
		mape_dimwise = np.sum(mape_masked, axis=0) / (np.sum(mape_masked != 0.0, axis=0) + 1e-10)
	else:
		mape_dimwise = np.mean(mape, axis=0)

	if is_yxpart:
		mape_masked = np.zeros(y_pred.shape)
		mape_masked[y_true != 0.0] = mape[y_true != 0.0]

	if do_print:
		torch.set_printoptions(sci_mode=False)
		np.set_printoptions(suppress=True)

		utility.logtext("MAPE:", va)
		utility.logtext(mape_dimwise, va)
		utility.logtext(np.mean(mape_dimwise), va)
		#utility.logtext("{:.4f}".format(np.mean(np.mean(mape, axis=0))), va)

		utility.logtext("RMSE:", va)
		utility.logtext(rmse, va)
		utility.logtext(np.mean(rmse), va)
		# if type(rmse) == np.ndarray and rmse.shape[0] > 1:
		# 	for dim in rmse:
		# 		utility.logtext("{:.1f}".format(dim), va)
		# else:
		# 	utility.logtext("{:.1f}".format(rmse), va)

		utility.logtext("PCC:", va)
		utility.logtext(pcc, va)
		utility.logtext(np.mean(pcc), va)
		#utility.logtext("{:.4f}".format(np.mean(pcc)), va)

	if do_plot:
		plot.plot_baseline_y_error_distr_for_y_test(va, mape, ds)

	return mape_dimwise, mape, rmse, pcc


def get_x_y_gen_from_runid(run_id, seed, scheme):
	output = None
	newsamples_yzx = None

	output_fn = f"gen/{run_id}/{seed}/{scheme}/output.csv"
	if os.path.isfile(output_fn):
		output = np.array(pd.read_csv(output_fn, header=None))

	newsamples_fn = f"gen/{run_id}/{seed}/{scheme}/newsamples-full.csv"
	if os.path.isfile(newsamples_fn):
		newsamples_yzx = np.array(pd.read_csv(newsamples_fn, header=None))

	return newsamples_yzx, output


def create_masked_nparray_with_zeros(data, ref=None):
	data_masked = np.zeros(data.shape)

	if ref is not None:
		nonzero_idxs = np.nonzero(ref)
	else:
		nonzero_idxs = np.nonzero(data)

	data_masked[nonzero_idxs[0], nonzero_idxs[1]] = data[nonzero_idxs[0], nonzero_idxs[1]]
	return data_masked


def create_masked_tensor(data, dvc, ref=None):

	data_masked = torch.zeros(data.shape).to(dvc)

	if ref is not None:
		mask = ref != -1.0
	else:
		mask = data != -1.0

	if torch.sum(mask.int()) > 0:
		data_masked[mask] = data[mask]

	return data_masked



def evaluate_y_values_from_files(run_id, seed, y_cols, scheme, va=None, do_print=False, do_plot=False, ds=None):

	newsamples_yzx, output = get_x_y_gen_from_runid(run_id, seed, scheme)

	y_dim = len(y_cols)

	if va is not None and ("yxpart" in va.model_config.get("model_type")) and scheme != "ga-comp":
		utility.logtext(scheme, va)
		utility.logtext(scheme != "ga-comp", va)
		utility.logtext("computing metrics for y_x...", va)
		mape_dimwise, mape_all, rmse, pcc = compute_error_metrics(newsamples_yzx[:, (y_dim+va.model_config.get("x_dim")+va.model_config.get("model_params").get("z_dim")):], newsamples_yzx[:, y_dim:(y_dim+va.model_config.get("x_dim"))], True, va, do_print, do_plot, ds)

	if va is None or not va.cli_args.is_only_yx_for_gen:

		y_true = output[:, y_cols]

		utility.logtext(y_true.shape, va)

		y_true_with_idxs = np.zeros((y_true.shape[0], y_true.shape[1] + 1))
		y_true_with_idxs[:, 0] = np.arange(y_true_with_idxs.shape[0])
		y_true_with_idxs[:, 1:] = y_true

		valid_idxs = np.sum(y_true_with_idxs[:, 1:] !=0.0, axis=1) == y_dim

		n_zeros = y_true_with_idxs.shape[0] - valid_idxs.shape[0]
		utility.logtext(f"# of zeros for y_true: {n_zeros}", va)

		y_true = y_true_with_idxs[valid_idxs, 1:]

		y_pred = newsamples_yzx[:, y_cols]
		y_pred = y_pred[valid_idxs, :]

		mape_dimwise, mape_all, rmse, pcc = compute_error_metrics(y_pred, y_true, False, va, do_print, do_plot, ds)

	mape_total = np.mean(mape_dimwise)
	return mape_total, mape_dimwise, mape_all, rmse, pcc


def gen_forward_pass(model, va, dataset_stats, z, y, y_x):

	model_type = va.model_config.get("model_type")

	# @pre: y is already encoded for categorical dimensions
	# forward pass through decoder
	if model_type == "cvae_01_nores":
		batch_data_x_recon = model.decode(z, y)
	elif "yxpart" in model_type:
		batch_data_x_recon = model.decoder(z, y, y_x)
	else:
		batch_data_x_recon = model.decoder(z, y)

	# 10d-bridge dataset
	# pick only first 10 dims for bridge dataset when using GCNN embeddings
	if va.cli_args.pretrained_gnn:
		batch_data_x_recon = batch_data_x_recon[:, :10]

	if va.dim_config is not None:
		
		# decode and unnormalize output
		batch_x_recon_unnorm = dataset.decode_cat_and_unnorm_cont_dims_for_x(batch_data_x_recon, va, dataset_stats).cpu().detach().numpy()
		
		# unnormalize conditioning on y
		batch_y_unnorm = dataset.decode_cat_and_unnorm_cont_dims_for_y(y, va, dataset_stats).cpu().detach().numpy()
		
		# unnormalize conditioning on y_x_partial
		batch_y_x_partial_unnorm_masked = dataset.decode_y_x_partial(y_x, va, dataset_stats)

	else:
		batch_x_recon_unnorm = dataset.unnormalize_data(batch_data_x_recon, dataset_stats[0], dataset_stats[1]).cpu().detach().numpy()
		
		batch_y_unnorm = dataset.unnormalize_data(y, dataset_stats[2], dataset_stats[3]).cpu().detach().numpy()
		
		batch_y_x_partial_unnorm = dataset.unnormalize_data(y_x, dataset_stats[0], dataset_stats[1])
		
		batch_y_x_partial_unnorm_masked = create_masked_tensor(batch_y_x_partial_unnorm, dvc=va.dvc, ref=y_x).cpu().detach().numpy()

	if va.model_config.get("shift_all_dimensions"):
		batch_x_recon_unnorm -= 1.0
		batch_y_unnorm -= 1.0
		batch_y_x_partial_unnorm_masked[batch_y_x_partial_unnorm_masked != -1.0] -= 1.0

	return batch_x_recon_unnorm, batch_y_unnorm, batch_y_x_partial_unnorm_masked


def generate_new_samples(model, va, seed):

	z_dim = va.model_config.get("model_params").get("z_dim")

	model.eval()
	torch.set_printoptions(sci_mode=False)
	np.set_printoptions(suppress=True)

	gen_data = np.zeros((va.gen_n_y*va.gen_n_z, va.dim_config.get("y_dim_orig") + va.dim_config.get("x_dim_orig") + z_dim + va.dim_config.get("x_dim_orig")))

	# fix seed to kill randomness
	utility.set_seeds(va.cli_args.seed)

	ds, ds_stats = dataset.get_dataset_as_nparray(va)
	x_train, y_train = ds[0], ds[1]

	# SAMPLE SCHEMA 1: sample proportionally from y_train
	if va.cli_args.genmode == "proportional":
		utility.logtext("sampling PROPORTIONAL...", va)
		idxs_samples = np.random.choice(y_train.shape[0], va.gen_n_y)
		y_samples = y_train[idxs_samples, :]
		x_samples = x_train[idxs_samples, :]

	# SAMPLE SCHMEA 2: sample uniformly for NON categorical dims
	elif va.cli_args.genmode == "uniform":

		if va.dim_config.get("y_dim_orig") > len(va.dim_config.get("y_dims_conti")):
			raise Exception('Uniform sampling on categorical data not allowed!')

		utility.logtext("sampling UNIFORMLY (Sobol)...", va)
		y_train_dims_min = np.min(y_train, axis=0)
		y_train_dims_max = np.max(y_train, axis=0)
		dims = []
		for dim in range(va.model_config.get("y_dim_orig")):
			dims.append((y_train_dims_min[dim], y_train_dims_max[dim]))
		space = Space(dims)
		sobol = Sobol()
		y_samples = sobol.generate(space.dimensions, va.gen_n_y)
		y_samples = np.array(y_samples)

	if va.model_config.get("y_dim_orig") == 1:
		y_samples = np.expand_dims(y_samples[:, 0], axis=1)

	# sample latent vector z for each y sample
	for idx_y, y in enumerate(y_samples):

		if va.cli_args.is_only_yx_for_gen:
			y = torch.full((y.shape), -1.0, device=va.dvc)

		# set y
		batch_y = np.repeat(np.expand_dims(y, axis=0), va.gen_n_z, axis=0)

		# introduce slight gaussian noise if NOT categorical
		for row_idx in range(va.gen_n_z):
			for dim in range(va.dim_config.get("y_dim_orig")):
				yi = va.dim_config.get(f"y{dim}")
				if not yi.get("is_categorical"):
					batch_y[row_idx, dim] = batch_y[row_idx, dim] + np.random.normal(0, 1) * (batch_y[row_idx, dim] * 0.01)

		#batch_y, _ = dataset.encode_dims_for_y(batch_y, va.dim_config)
		batch_y = torch.Tensor(batch_y).to(va.dvc)

		# set z: draw from standard normal gaussian (assume z_i is i.i.d)
		batch_z = torch.randn(va.gen_n_z, z_dim).to(va.dvc)

		# only for prop scheme ATM
		if va.cli_args.genmode == "proportional":
			# introduce slight gaussian noise:
			batch_x = np.repeat(np.expand_dims(x_samples[idx_y, :], axis=0), va.gen_n_z, axis=0)
			for row_idx in range(va.gen_n_z):
				for dim in range(va.dim_config.get("x_dim_orig")):
					xi = va.dim_config.get(f"x{dim}")
					if not xi.get("is_categorical"):
						batch_x[row_idx, dim] = batch_x[row_idx, dim] + np.random.normal(0, 1) * (batch_x[row_idx, dim] * 0.01)

			batch_yxpartial = learning.create_y_x_partial(torch.Tensor(batch_x).to(va.dvc), va, is_train=False)
		else:
			batch_yxpartial = torch.full((va.gen_n_z, va.model_config.get("x_dim")), -1.0, device=va.dvc).to(va.dvc)

		# forward pass into generative model to get x_hat
		batch_data_x_recon_unnorm, batch_y_unnorm, batch_y_x_partial_unnorm_masked = gen_forward_pass(model, va, ds_stats, batch_z, batch_y, batch_yxpartial)

		y_dim = va.dim_config.get("y_dim_orig")
		x_dim = va.dim_config.get("x_dim_orig")

		gen_data[(idx_y*va.gen_n_z):(idx_y*va.gen_n_z + va.gen_n_z), :y_dim] = batch_y_unnorm
		gen_data[(idx_y*va.gen_n_z):(idx_y*va.gen_n_z + va.gen_n_z), y_dim:(y_dim+x_dim)] = batch_y_x_partial_unnorm_masked
		gen_data[(idx_y*va.gen_n_z):(idx_y*va.gen_n_z + va.gen_n_z), (y_dim+x_dim):(y_dim+x_dim+z_dim)] = batch_z.cpu().numpy()
		gen_data[(idx_y*va.gen_n_z):(idx_y*va.gen_n_z + va.gen_n_z), (y_dim+x_dim+z_dim):] = batch_data_x_recon_unnorm

	utility.logtext(f"generated data has shape: {gen_data.shape}", va)
	gen_path = f"gen/{va.run_id}"
	utility.logtext(f"save files at {gen_path}", va)
	os.makedirs(gen_path, exist_ok=True)
	os.makedirs(f"{gen_path}/{seed}/{va.cli_args.genmode}", exist_ok=True)
	np.savetxt(f"{gen_path}/{seed}/{va.cli_args.genmode}/newsamples-full.csv", gen_data, delimiter=",")
	np.savetxt(f"{gen_path}/{seed}/{va.cli_args.genmode}/newsamples-x-only.csv", gen_data[:, (y_dim+x_dim+z_dim):], delimiter=",")


def make_rhino_request(endpoint, payload_post, use_proxy=True):
	remote_host_rhino = "https://213.167.225.54:443"
	s = requests.Session()
	#s.mount('https://', host_header_ssl.HostHeaderSSLAdapter())
	if use_proxy:
		proxy = {
			"http" : "http://proxy.ethz.ch:3128",
			"https" : "http://proxy.ethz.ch:3128"
		}
		s.proxies.update(proxy)
	try:
		resp = s.post(remote_host_rhino + endpoint, verify=False, json=payload_post, timeout=5.0)
		return resp
	except requests.Timeout:
		return -1


def run_rhino_computation_remote(va, gen_path, xs):

	payload_post = {
		"run_id": va.run_id,
		"xs": base64.b64encode(np.ascontiguousarray(xs)).decode("utf-8"),
		"xs_shape": xs.shape
	}

	try:
		now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
		utility.logtext(f"send POST request to IDL remote host at: {now}", va)
		resp = make_rhino_request("/rhinoceros-start-comp", payload_post, va.cli_args.use_proxy)
		if resp == -1:
			return -1, None, None, None, None
		if resp.status_code != 200:
			utility.logtext("could not start rhino comp via remote", va)
			return -1, None, None, None, None

	except Exception as exc:
		utility.logtext(exc, va)
		utility.logtext(">> [error] could not reach remote host during initial POST", va)
		return -1, None, None, None, None

	try:

		init_secs_to_wait = 0.6 * (va.gen_n_y * va.gen_n_z)

		utility.logtext(f"wait {init_secs_to_wait} seconds for IDL remote host to finish computation...", va)
		time.sleep(init_secs_to_wait)

		payload_get = {
			"run_id": va.run_id,
		}

		while True:
			utility.logtext("fetch response from IDL remote host...", va)
			resp = make_rhino_request("/rhinoceros-get-result", payload_get, va.cli_args.use_proxy)

			if resp == -1:
				utility.logtext(">> [error] could not do initial GET request to remote rhino machine", va)
				return -1, None, None, None, None

			if resp.status_code != 200:
				message = resp.json().get("message")
				utility.logtext(f"[HTTP Error {resp.status_code}] message from server: {message}", va)

				if resp.status_code == 404:
					secs_to_wait = init_secs_to_wait
				else:
					secs_to_wait = 20

				utility.logtext(f"waiting another {secs_to_wait} seconds for IDL remote host...", va)
				time.sleep(secs_to_wait)
			else:
				break

		now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
		utility.logtext(f"got response from IDL remote host at: {now}", va)
		json_data = resp.json()
		buf = base64.b64decode(json_data["ys"].encode("utf-8"))
		ys = np.reshape(np.frombuffer(buf, dtype=np.float64), json_data["ys_shape"])
		np.savetxt(f"{gen_path}/{va.cli_args.seed}/{va.cli_args.genmode}/output.csv", ys, delimiter=",")

		return evaluate_y_values_from_files(va.run_id, va.cli_args.seed, va.cli_args.y_cols, va.cli_args.genmode, va, do_print=True)

	except Exception as exc:
		utility.logtext(exc, va)
		utility.logtext(">> [error] could not reach remote host during GET", va)
		return -1, None, None, None, None


def gen_and_eval_bridge_y_via_rhino_remote(model, va, seed):

	if va.cli_args.is_output_files_only:
		return evaluate_y_values_from_files(va.run_id, va.cli_args.seed, va.cli_args.y_cols, va.cli_args.genmode, va, do_print=True)

	else:
		gen(va, model, seed)

		gen_path = f"gen/{va.run_id}"
		xs_only = np.array(pd.read_csv(f"{gen_path}/{seed}/{va.cli_args.genmode}/newsamples-x-only.csv", header=None))
		utility.logtext(f"{xs_only.shape}", va)

		if va.cli_args.is_only_yx_for_gen:
			return evaluate_y_values_from_files(va.run_id, va.cli_args.seed, va.cli_args.y_cols, va.cli_args.genmode, va, do_print=True)
		else:
			return run_rhino_computation_remote(va, gen_path, xs_only)



def check_performance_of_generative_model(model, optimizer, va, itr, best_loss_test, best_loss_recon, best_loss_kld):

	utility.resume_model(va, va.ckpt_path + f"/ckpt_last_seed_{va.cli_args.seed}.pth.tar", model, optimizer)

	if "5d-wall" not in va.cli_args.dataset and "5d-wall" not in va.cli_args.dataset_path_csv:
		utility.logtext("doing bridge eval", va)
		mape_loss_generated_samples, _, _, _, _ = gen_and_eval_bridge_y_via_rhino_remote(model, va, va.cli_args.seed)
	else:
		utility.logtext("doing wall eval", va)
		mape_loss_generated_samples, _, _, _, _ = gen_and_eval_wall_y_via_excel(model, va, va.cli_args.seed)

	# (optional): save metrics
	if va.cli_args.is_vanilla_training:
		analysis_type = "model-epoch"
		utility.save_metrics_from_itr(analysis_type, itr, va.ckpt_path, mape_loss_generated_samples, best_loss_test, best_loss_recon, best_loss_kld)
	else:
		analysis_type = "bay-opt"
		utility.save_metrics_from_itr(analysis_type, itr, va.ckpt_path, mape_loss_generated_samples, best_loss_test, best_loss_recon, best_loss_kld)

	if mape_loss_generated_samples != -1:
		va.bayopt_running_stats.append(mape_loss_generated_samples)
		utility.logtext(f"bayopt_running_stats: {va.bayopt_running_stats}", va)
		return mape_loss_generated_samples
	else:
		utility.logtext("invalid result for generative phase", va)
		#utility.logtext("taking running median instead of 'np.nan' or 'np.inf' to not disturb the bayesian optimization process too much...", va)
		#running_median = np.median(va.bayopt_running_stats)
		#utility.logtext(f"current running median: {running_median}", va)
		#return running_median

		utility.logtext("taking 1'000'000", va)
		return 1000000


def check_bay_opt_results(run_id):
	filepath = "ckpt/" + run_id + "/bay-opt-results.csv"
	data = np.array(pd.read_csv(filepath, header=None))
	print(data.shape)
	loss_kld = data[:, 3]
	mape = data[:, 4]

	sort_idxs = np.argsort(loss_kld)

	_, axes = plt.subplots(1, 2)

	axes[0].plot(mape[sort_idxs][1:], color='g')
	axes[0].set_title("MAPE for each iteration during BO")
	#axes[0].set_yscale('log')

	print(mape[sort_idxs][1:][0])

	axes[1].plot(loss_kld[sort_idxs][1:], color='b')
	axes[1].set_title("KLD Loss for each iteration during BO")
	#axes[1].set_yscale('log')

	plt.show()


def check_mape_error_uniformly_sampled(test_mape_y_hat_dim_wise, y_unnorm):

	test_mape_y_hat_dim_wise = test_mape_y_hat_dim_wise.cpu().numpy()
	y_unnorm = y_unnorm.cpu().numpy()

	n_bins = 50

	_, axis = plt.subplots(1,2)

	for dim in range(2):
		bin_edges = np.histogram_bin_edges(y_unnorm[:, dim], bins=n_bins)
		bin_edges[0] -= 10**-6
		bin_edges[-1] += 10**-6
		bin_idxs = np.digitize(y_unnorm[:, dim], bin_edges) - 1

		mapes_binwise = np.zeros(n_bins)

		for i in range(n_bins):
			indices = np.where(bin_idxs == i)
			#print(indices[0].shape)
			if indices[0].shape[0] != 0:
				bin_data = test_mape_y_hat_dim_wise[indices[0], dim]

				if bin_data[0] > 20:
					#print(np.mean(np.random.choice(bin_data, 20)))
					mapes_binwise[i] = np.mean(np.random.choice(bin_data, 10))
				else:
					if bin_data.shape[0] != 0 and bin_data[0] != 0:
						#print(np.random.choice(bin_data))
						mapes_binwise[i] = np.random.choice(bin_data)

		x_axis = list(np.linspace(bin_edges[0], bin_edges[-1], n_bins))
		axis[dim].bar(x_axis, mapes_binwise, width=bin_edges[1]-bin_edges[0])
		axis[dim].set_ylim(0, np.max(mapes_binwise))

	plt.show()


def gnn_robustness_test_evaluate_perf(va, model, xs, ys, stats):

	print("forward pass")
	all_y_unnorm = dataset.unnormalize_data(ys, stats[2], stats[3])

	all_y_hat_norm = model(xs)

	if va.model_config.get("is_bce_loss"):
		all_y_hat_norm = torch.nn.Sigmoid()(all_y_hat_norm)

	all_y_hat_unnorm = dataset.unnormalize_data(all_y_hat_norm, stats[2], stats[3])

	return analysis.compute_error_metrics(all_y_hat_unnorm.detach().cpu().numpy(), all_y_unnorm.detach().cpu().numpy())


def gnn_robustness_test(run_id, va, model_config, optimizer):

	_, stats = dataset.get_dataset_as_nparray(va)
	all_x_robustness_test = torch.tensor(np.array(pd.read_csv(f'./data/x/x-10d-robustness-test-filtered.csv', header=None, dtype=np.float64)))
	all_x_robustness_test = dataset.normalize_data(all_x_robustness_test, stats[0], stats[1])
	all_y_robustness_test = torch.tensor(np.array(pd.read_csv(f'./data/y/output-10d-robustness-test-filtered.csv', header=None, dtype=np.float64)))
	all_y_robustness_test = dataset.normalize_data(all_y_robustness_test[:, :2], stats[2], stats[3])

	all_seeds = [ "1234", "5678", "9876"]

	all_stats = {
		"all_pcc": [],
		"all_rmse": [],
		"all_mape": []
	}

	for s in all_seeds:
		utility.set_seeds(int(s))
		va.cli_args.seed = int(s)

		resume_pth = f"ckpt/{run_id}/ckpt_last_seed_{s}.pth.tar"
		model = models.init_model(va, model_config, model_config.get("is_bce_loss"))
		model.eval()
		_, _ = utility.resume_model(va, resume_pth, model, optimizer)

		mape, rmse, pcc = gnn_robustness_test_evaluate_perf(va, model, all_x_robustness_test, all_y_robustness_test, stats)

		all_stats["all_pcc"].append(pcc)
		all_stats["all_rmse"].append(rmse)
		all_stats["all_mape"].append(np.mean(mape, axis=0))

	report_metrics(va, all_stats)


def evaluate_wall_data_via_excel(va, gen_path, xs_only):

	fn = "./misc/dataset-02-walls/mw_ec6_master-tweak.xlsx"
	excel = ExcelCompiler(filename=fn)
	excel.log.setLevel(logging.WARNING)

	try:
		ys = np.zeros((xs_only.shape[0], 4))
		for row_idx, row in enumerate(xs_only):

			excel.evaluate('1!C22')
			excel.evaluate('1!C23')
			excel.evaluate('1!C25')
			excel.evaluate('1!C26')
			excel.evaluate('1!C27')
			excel.evaluate('1!F46')
			excel.evaluate('1!C40')
			excel.evaluate('1!F40')
			excel.evaluate('1!B49')

			excel.set_value('1!C22', row[0])
			excel.set_value('1!C23', row[1])
			excel.set_value('1!C25', row[2])
			excel.set_value('1!C26', row[3])
			excel.set_value('1!C27', row[4])

			y1 = excel.evaluate('1!F46') # eta_w
			y2 = excel.evaluate('1!C40') # phi_1
			y3 = excel.evaluate('1!F40') # phi_2
			y4 = 1 if excel.evaluate('1!B49') == "OK" else 0 # schlankheit

			ys[row_idx, :] = np.array([y1, y2, y3, y4])
			row_idx += 1

		utility.logtext("finished with excel", va)
		utility.logtext(ys.shape, va)

		np.savetxt(f"{gen_path}/{va.cli_args.seed}/{va.cli_args.genmode}/output.csv", ys, delimiter=",")

		return evaluate_y_values_from_files(va.run_id, va.cli_args.seed, va.cli_args.y_cols, va.cli_args.genmode, va, do_print=True)

	except Exception as exc:
		utility.logtext(exc, va)
		utility.logtext("could not perform gen perf via excel file (probably nan values in xs_only)", va)
		return -1, None, None, None, None


def gen(va, model, seed):
	model.eval()
	utility.logtext(f"generate new samples... ({va.gen_n_z * va.gen_n_y})", va)
	generate_new_samples(model, va, seed)


def gen_and_eval_wall_y_via_excel(model, va, seed):

	if va.cli_args.is_output_files_only:
		return evaluate_y_values_from_files(va.run_id, seed, va.cli_args.y_cols, va.cli_args.genmode, va, do_print=True)

	else:
		gen(va, model, seed)

		gen_path = f"gen/{va.run_id}"
		xs_only = np.array(pd.read_csv(f"{gen_path}/{seed}/{va.cli_args.genmode}/newsamples-x-only.csv", header=None))
		utility.logtext(f"{xs_only.shape}", va)

		if va.cli_args.is_only_yx_for_gen:
			return evaluate_y_values_from_files(va.run_id, va.cli_args.seed, va.cli_args.y_cols, va.cli_args.genmode, va, do_print=True)
		else:
			return evaluate_wall_data_via_excel(va, gen_path, xs_only)



def report_metrics(va, all_stats):

	torch.set_printoptions(sci_mode=False)
	np.set_printoptions(suppress=True)

	utility.logtext("", va)
	utility.logtext(all_stats, va)

	all_stats["all_pcc"] = np.array(all_stats["all_pcc"])
	all_stats["all_rmse"] = np.array(all_stats["all_rmse"])
	all_stats["all_mape"] = np.array(all_stats["all_mape"])

	utility.logtext("", va)
	utility.logtext("dimwise:", va)
	utility.logtext("pcc: " + str(np.mean(all_stats["all_pcc"], axis=0)) + " +/- " + str(np.std(all_stats["all_pcc"], axis=0)), va)
	utility.logtext("rmse: " + str(np.mean(all_stats["all_rmse"], axis=0)) + " +/- " + str(np.std(all_stats["all_rmse"], axis=0)), va)
	utility.logtext("mape: " + str(np.mean(all_stats["all_mape"], axis=0)) + " +/- " + str(np.std(all_stats["all_mape"], axis=0)), va)

	utility.logtext("", va)
	utility.logtext("total:", va)
	utility.logtext("pcc: " + str(np.mean(np.mean(all_stats["all_pcc"], axis=0))) + " +/- " + str(np.std(np.mean(all_stats["all_pcc"], axis=0))), va)
	utility.logtext("rmse: " + str(np.mean(np.mean(all_stats["all_rmse"], axis=0))) + " +/- " + str(np.std(np.mean(all_stats["all_rmse"], axis=0))), va)
	utility.logtext("mape: " + str(np.mean(np.mean(all_stats["all_mape"], axis=0))) + " +/- " + str(np.std(np.mean(all_stats["all_mape"], axis=0))), va)
	utility.logtext("", va)


def gen_eval_repro_thesis(run_id, va, model_config, optimizer):

	#all_seeds = [ "1234", "5678", "9876"]
	all_seeds = [ "1234" ]

	all_stats = {
		"all_pcc": [],
		"all_rmse": [],
		"all_mape": []
	}

	for s in all_seeds:
		utility.set_seeds(int(s))
		va.cli_args.seed = int(s)

		resume_pth = f"ckpt/{run_id}/ckpt_last_seed_{s}.pth.tar"
		model = models.init_model(va, model_config, model_config.get("is_bce_loss"))
		_, _ = utility.resume_model(va, resume_pth, model, optimizer)

		if "5d-wall" not in va.cli_args.dataset and "5d-wall" not in va.cli_args.dataset_path_csv:
			utility.logtext("doing bridge eval", va)
			_, mape_dimwise, _, rmse, pcc = analysis.gen_and_eval_bridge_y_via_rhino_remote(model, va, s)
		else:
			utility.logtext("doing wall eval", va)
			_, mape_dimwise, _, rmse, pcc = analysis.gen_and_eval_wall_y_via_excel(model, va, s)

		all_stats["all_pcc"].append(pcc)
		all_stats["all_rmse"].append(rmse)
		all_stats["all_mape"].append(mape_dimwise)

	report_metrics(va, all_stats)


def gen_eval_walls_comparison_case_study(case_study_no, va, model, n_total):

	z_dim = va.model_config.get("model_params").get("z_dim")
	_, ds_stats = dataset.get_dataset_as_nparray(va)

	gen_data = np.zeros((n_total, 1+5+5))

	y_samples = np.random.uniform(low=0.98, high=1.02, size=n_total)
	gen_data[:, 0] = y_samples

	x_samples = np.zeros((n_total, 5))
	x_samples.fill(-1.0)

	if case_study_no == 2:
		x_samples[:, 0] = 100.0
		x_samples[:, 3] = 2.65

	elif case_study_no == 3:
		x_samples[:, 0] = 100.0
		x_samples[:, 2] = 150.0
		x_samples[:, 3] = 2.65

	# utility.logtext(x_samples, va)
	gen_data[:, 1:(1+5)] = x_samples

	for idx in tqdm(range(n_total)):

		y_sample = y_samples[idx]
		x_sample = x_samples[idx, :]
		
		batch_y = torch.unsqueeze(torch.unsqueeze(torch.Tensor(np.array(y_sample)).to(va.dvc), dim=0), dim=0)
		batch_y = dataset.normalize_data(batch_y, ds_stats[2], ds_stats[3])
		batch_z = torch.randn(1, z_dim).to(va.dvc) # i.i.d

		x_mask = x_sample != -1.0
		batch_yxpartial = torch.Tensor(x_sample).to(va.dvc)
		batch_yxpartial[x_mask] = dataset.normalize_data(torch.Tensor(x_sample[x_mask]).to(va.dvc), ds_stats[0][x_mask], ds_stats[1][x_mask])
		batch_yxpartial = torch.unsqueeze(batch_yxpartial, dim=0)

		# forward pass into generative model to get x_hat
		batch_data_x_recon_unnorm, batch_y_unnorm, batch_y_x_partial_unnorm_masked = gen_forward_pass(model, va, ds_stats, batch_z, batch_y, batch_yxpartial)

		y_dim = va.dim_config.get("y_dim_orig")
		x_dim = va.dim_config.get("x_dim_orig")

		gen_data[idx, (1+5):(1+5+5)] = batch_data_x_recon_unnorm

	gen_path = f"gen/{va.run_id}"
	utility.logtext(f"save files at {gen_path}", va)
	os.makedirs(gen_path, exist_ok=True)
	os.makedirs(f"{gen_path}/{va.cli_args.seed}/{va.cli_args.genmode}", exist_ok=True)
	np.savetxt(f"{gen_path}/{va.cli_args.seed}/{va.cli_args.genmode}/newsamples-full.csv", gen_data, delimiter=",")
	np.savetxt(f"{gen_path}/{va.cli_args.seed}/{va.cli_args.genmode}/newsamples-x-only.csv", gen_data[:, (1+5):(1+5+5)], delimiter=",")
	xs_only = np.array(pd.read_csv(f"{gen_path}/{va.cli_args.seed}/{va.cli_args.genmode}/newsamples-x-only.csv", header=None))

	evaluate_wall_data_via_excel(va, gen_path, xs_only)

	output_fn = f"gen/{va.run_id}/{va.cli_args.seed}/{va.cli_args.genmode}/output.csv"
	if os.path.isfile(output_fn):
		output = np.array(pd.read_csv(output_fn, header=None))

	y_true = output[:, 0]

	print(((y_true >= 0.95) & (y_true <= 1.05)).sum())
	print(y_true)
	# print(gen_data[:, 0])

	np.savetxt(f"{gen_path}/{va.cli_args.seed}/{va.cli_args.genmode}/case-study-{case_study_no}.csv", np.concatenate([gen_data, np.expand_dims(y_true, axis=1)], axis=1), delimiter=",")


def gen_eval_walls_comparison(run_id, va, model_config, optimizer):
	
	seed = 1234
	utility.set_seeds(seed)
	va.cli_args.seed = seed

	resume_pth = f"ckpt/{run_id}/ckpt_last_seed_{seed}.pth.tar"
	model = models.init_model(va, model_config, model_config.get("is_bce_loss"))
	_, _ = utility.resume_model(va, resume_pth, model, optimizer)
	model.eval()

	torch.set_printoptions(sci_mode=False)
	np.set_printoptions(suppress=True)

	#n_total = 1000
	n_total = 1000 * 20

	utility.logtext("CASE STUDY 1...", va)
	utility.set_seeds(seed)
	gen_eval_walls_comparison_case_study(1, va, model, n_total)

	utility.logtext("CASE STUDY 2...", va)
	utility.set_seeds(seed)
	gen_eval_walls_comparison_case_study(2, va, model, n_total)

	utility.logtext("CASE STUDY 3...", va)
	utility.set_seeds(seed)
	gen_eval_walls_comparison_case_study(3, va, model, n_total)



def eval_y_repro_thesis(run_id, va, model_config, optimizer):

	all_seeds = [ "1234", "5678", "9876"]

	all_stats = {
		"all_pcc": [],
		"all_rmse": [],
		"all_mape": []
	}

	for s in all_seeds:
		utility.set_seeds(int(s))
		va.cli_args.seed = int(s)

		#resume_pth = f"ckpt/{run_id}/ckpt_last_seed_{s}.pth.tar"
		resume_pth = f"ckpt/{run_id}/ckpt_best_seed_{s}.pth.tar"

		model = models.init_model(va, model_config, model_config.get("is_bce_loss"))
		_, _ = utility.resume_model(va, resume_pth, model, optimizer)

		_, _, mape, rmse, pcc = learning_gnn.gnn_test(0, model, va)

		all_stats["all_pcc"].append(pcc)
		all_stats["all_rmse"].append(rmse)
		all_stats["all_mape"].append(mape)

	report_metrics(va, all_stats)


if __name__ == "__main__":

	#run_id = "report-22-04-15-5dwall-cvae5nf-v2"
	#run_id = "report-22-04-16-10dy1d-cvae5nf"
	#run_id = "report-22-04-16-10dy2d-cvae5nf"

	#evaluate_y_values_from_files("10d-simple", run_id, y_cols=[0], scheme="uniform", do_print=True, plot=True)
	#evaluate_y_values_from_files("10d-simple", run_id, y_cols=[0,1], scheme="proportional", va=None, do_print=True, plot=True)

	#run_id = "report-22-04-15-5dwall-cvae5nf-v2"
	#evaluate_y_values_from_files(run_id, 1234, y_cols=[0], scheme="proportional", va=None, do_print=True, do_plot=True, ds="5d-wall-v2")

	#run_id = "22-04-19-y1d-cvae5nf-k2"
	#run_id = "report-22-04-18-10dy1d-cvae5nf"

	run_id = "report-22-04-24-y2d-cvae8nfyxpart"

	evaluate_y_values_from_files(run_id, 1234, y_cols=[0], scheme="proportional", va=None, do_print=True, do_plot=True, ds="10d-simple")

	#run_id = "report-22-04-18-10dy2d-cvae5nf"
	#evaluate_y_values_from_files(run_id, 1234, y_cols=[0,1], scheme="proportional", va=None, do_print=True, do_plot=True, ds="10d-simple")

	#evaluate_y_values_from_files(run_id, 1234, y_cols=[0,1], scheme="proportional", va=None, do_print=True)

	#run_id = "euler-22-02-17-y2d-cvae45-pretr-gnn-r02-gnnmodelineval"
	#evaluate_y_values_from_files("10d-simple", run_id, y_cols=[0,1], scheme="uniform", plot=True)

	#check_bay_opt_results(run_id)

	#evaluate_gen_x_samples_y_values("10d-v2", run_id, n_y=100, n_z=5, is_only_y1d=True)
	#evaluate_gen_x_distr_for_given_y(gen_path)cu
