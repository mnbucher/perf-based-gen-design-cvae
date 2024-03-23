import os
import numpy as np
import pandas as pd
import torch

import matplotlib
from matplotlib import pyplot as plt
import matplotlib.font_manager

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import collections
from skopt.space import Space
from skopt.sampler import Sobol
from skopt.sampler import Grid
import logging
from tqdm import tqdm

import src.models as models
import src.utility as utility
import src.dataset as dataset
import src.learning as learning
import src.analysis as analysis


def plot_error_vs_label_density(all_test_rmse, n_bins=20):

	all_test_rmse = all_test_rmse.detach().numpy()

	label_min = np.min(all_test_rmse[:, 0])
	label_max = np.max(all_test_rmse[:, 0])

	bin_edges = np.linspace(label_min, label_max, n_bins)
	bin_edges[0] -= 10**-6
	bin_edges[-1] += 10**-6

	bin_idxs = np.digitize(all_test_rmse[:, 0], bin_edges) - 1

	bins_rmse = [ [] for i in range(n_bins) ]

	for i, row in enumerate(all_test_rmse):
		bin_idx = bin_idxs[i]
		bins_rmse[bin_idx].append(row[1])

	#for elem in bins_rmse:
		#print(len(elem))
		#print(elem)
	bins_rmse = [ np.mean(elem) for elem in bins_rmse ]

	#print(bins_rmse)

	fig, ax1 = plt.subplots()

	ax1.hist(all_test_rmse[:, 0], n_bins)
	ax1.set_title("Histogram for y[0]")

	ax2 = ax1.twinx()
	x_ticks = np.linspace(label_min, label_max, n_bins)
	bar_width = (x_ticks[1]-x_ticks[0])
	x_ticks += bar_width/2.0
	ax2.bar(x_ticks, bins_rmse, width=bar_width, color='r', fill=False)
	#ax2.set_title("RMSE")

	plt.show()


def do_pca_3d(data):
	print(f"doing PCA with shape:{data.shape}")
	pca = PCA(n_components=3)
	pca.fit(data)
	data_proj = pca.transform(data)
	return data_proj


def plot_data_pca_3d(data, data2=None):

	data_proj = do_pca_3d(data)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	ax.scatter(data_proj[:, 0], data_proj[:, 1], data_proj[:, 2], c='b')
	ax.set_title("PCA down to d=3")

	if data2 is not None:
		data2_proj = do_pca_3d(data2)
		ax.scatter(data_proj[:, 0], data_proj[:, 1], data_proj[:, 2], c='g')

	plt.show()


def plot_taylor_diagram(values_pred, values_true, run_id):

	#for dim in range(all_x.shape[1]):
		#taylor_plot([f"x-dim-{dim}"], [np.expand_dims(all_x_recon[:, dim], 1)], np.expand_dims(all_x[:, dim], 1), ['x0'])

	from rafael.plot_functions import taylor_plot

	#print(values_pred.shape)
	#print(values_true.shape)

	values_pred = np.expand_dims(values_pred, axis=1)
	values_true	 = np.expand_dims(values_true, axis=1)

	#print(values_pred.shape)
	#print(values_true.shape)

	img_path = f'img/{run_id}'
	os.makedirs(img_path, exist_ok=True)

	taylor_plot([f"CVAE"], values_true, [values_pred], [f'dim_{i}' for i in range(values_true.shape[1])], path=img_path)

	#taylor_plot(["CVAE-x-dim-0"], [np.expand_dims(all_x_recon[:, 0], 1)], np.expand_dims(all_x[:, 0], 1), ['x0'])
	#taylor_plot(["CVAE-x-dim-1"], [np.expand_dims(all_x_recon[:, 1], 1)], np.expand_dims(all_x[:, 1], 1), ['x1'])


def plot_latent_z_plain(all_z):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(all_z[:, 0], all_z[:, 1], all_z[:, 2], c='b')
	ax.set_title("Latent vec z for CVAE2 x2d")
	plt.show()


def plot_latent_z_binned_y(all_z, all_x, z_dim):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.set_title("Latent vector (z) for test set")

	sc = ax.scatter(all_z[:, 0], all_z[:, 1], all_z[:, 2], c=all_x[:, 2], cmap="Spectral")
	plt.colorbar(sc)

	plt.show()


def plot_latent_z_3d_raw(all_z):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.set_title("Latent vector (z) for test set")

	print("doing PCA...")
	pca = PCA(n_components=3)
	pca.fit(all_z)
	all_z_proj = pca.transform(all_z)
	ax.scatter(all_z_proj[:, 0], all_z_proj[:, 1], all_z_proj[:, 2], c='#237D74')

	plt.show()


def plot_latent_z_3d_colored_by_y(all_z_y, z_dim):

	z = all_z_y[:, :z_dim]
	y = all_z_y[:, z_dim:]

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	#ax.set_title("Latent vector (z) for test set")

	print("doing PCA...")
	pca = PCA(n_components=3)
	pca.fit(z)
	all_z_proj = pca.transform(z)
	sc = ax.scatter(all_z_proj[:, 0], all_z_proj[:, 1], all_z_proj[:, 2], c=y[:, 0], cmap="Spectral")

	plt.colorbar(sc)

	#fig.savefig("/Users/mnbucher/Downloads/latent-space-3d-y1d.svg", bbox_inches='tight', pad_inches=0)
	#fig.savefig("/Users/mnbucher/Downloads/latent-space-3d-y2d.svg", bbox_inches='tight', pad_inches=0)
	fig.savefig("/Users/mnbucher/Downloads/latent-space-3d-5dwall.svg", bbox_inches='tight', pad_inches=0)

	plt.show()


def plot_latent_z_3d_colored_by_x(all_z, all_x):
	fig = plt.figure()

	print("doing PCA...")
	pca = PCA(n_components=3)
	pca.fit(all_z)
	all_z_proj = pca.transform(all_z)

	for dim in range(3):
		ax = fig.add_subplot(111, projection='3d')
		sc = ax.scatter(all_z_proj[:, 0], all_z_proj[:, 1], all_z_proj[:, 2], c=all_x[:, dim], cmap="Spectral")
		plt.colorbar(sc)

		#fig.savefig(f"/Users/mnbucher/Downloads/latent-space-3d-y1d-dim-{dim}.svg", bbox_inches='tight', pad_inches=0)

		#fig.savefig(f"/Users/mnbucher/Downloads/latent-space-3d-y2d-dim-{dim}.svg", bbox_inches='tight', pad_inches=0)
		fig.savefig(f"/Users/mnbucher/Downloads/latent-space-3d-y2d-dim-{dim}.pdf", bbox_inches='tight', pad_inches=0)

		#fig.savefig(f"/Users/mnbucher/Downloads/latent-space-3d-5dwall-dim-{dim}.svg", bbox_inches='tight', pad_inches=0)


def plot_latent_z_binned_x(all_z, all_x, z_dim):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	bins_dim0 = np.digitize(all_x[:, 0], np.linspace(np.min(all_x[:, 0]), np.max(all_x[:, 0]), 2, endpoint=False))
	bins_dim1 = np.digitize(all_x[:, 1], np.linspace(np.min(all_x[:, 1]), np.max(all_x[:, 1]), 2, endpoint=False))

	#print(bins_dim0)
	#print(bins_dim1)
	# print(np.unique(bins_dim0))
	# print(np.unique(bins_dim1))
	# print(collections.Counter(bins_dim0))
	# print(collections.Counter(bins_dim1))
	# plt.hist(all_x[:, 0], 10)
	# plt.show()

	bins_rmse = { (0,0): [], (0,1): [], (1,0): [], (1,1): []}

	for idx, row in enumerate(all_z):
		key = (bins_dim0[idx]-1, bins_dim1[idx]-1)
		#print(key)
		bins_rmse[key].append(row)

	#for key, value in bins_rmse.items():
		#print(len(value))

	#for pair in [(0,0), (0,1), (1,0), (1,1)]:
		#pair[0]
	colors = ['r', 'g', 'b', 'c']

	for idx, pair in enumerate([(0,0),(0,1),(1,0),(1,1)]):

		third_dim = np.array(bins_rmse.get(pair))[:, 2]
		#third_dim = np.ones((np.array(bins_rmse.get(pair))[:, 0].shape[0], 1))

		ax.scatter(np.array(bins_rmse.get(pair))[:, 0], np.array(bins_rmse.get(pair))[:, 1], third_dim, c=colors[idx])

	#ax.scatter(np.array(bins_rmse.get((0,1)))[:, 0], np.array(bins_rmse.get((0,1)))[:, 1], all_z[:, 2], c='g')
	#ax.scatter(np.array(bins_rmse.get((1,0)))[:, 0], all_z[:, 1], all_z[:, 2], c='b')
	#ax.scatter(np.array(bins_rmse.get((1,1)))[:, 0], all_z[:, 1], all_z[:, 2], c='c')

	ax.set_title("Latent vector (z) for test set")
	plt.show()


def plot_x10d_distr_by_histograms():

	# original
	x_10d_orig = np.array(pd.read_csv('/Users/mnbucher/git/eth-master-thesis/data/x/sobol-samples-10d-orig.csv', header=None, dtype=np.float64))
	y_10d_orig = np.array(pd.read_csv('/Users/mnbucher/git/eth-master-thesis/data/y/output-10d-orig.csv', header=None, dtype=np.float64))
	#print(y_10d_orig.shape)
	#y_10d_orig = np.array(dataset.read_output_array_from_files("10d-orig"))
	print("x_10d_orig:", x_10d_orig.shape)

	# 10d-valid
	x_train = np.array(pd.read_csv('/Users/mnbucher/git/eth-master-thesis/data/train-test-split/train/x/x-10d-valid.csv', header=None, dtype=np.float64))
	x_test = np.array(pd.read_csv('/Users/mnbucher/git/eth-master-thesis/data/train-test-split/test/x/x-10d-valid.csv', header=None, dtype=np.float64))
	y_train = np.array(pd.read_csv('/Users/mnbucher/git/eth-master-thesis/data/train-test-split/train/y/y-10d-valid.csv', header=None, dtype=np.float64))
	y_test = np.array(pd.read_csv('/Users/mnbucher/git/eth-master-thesis/data/train-test-split/test/y/y-10d-valid.csv', header=None, dtype=np.float64))
	x_10d_valid = np.concatenate((x_train, x_test))
	y_10d_valid = np.concatenate((y_train, y_test))
	print("x_10d_valid:", x_10d_valid.shape)

	# 10d-robust
	x_train = np.array(pd.read_csv('/Users/mnbucher/git/eth-master-thesis/data/train-test-split/train/x/x-10d-robust.csv', header=None, dtype=np.float64))
	x_test = np.array(pd.read_csv('/Users/mnbucher/git/eth-master-thesis/data/train-test-split/test/x/x-10d-robust.csv', header=None, dtype=np.float64))
	y_train = np.array(pd.read_csv('/Users/mnbucher/git/eth-master-thesis/data/train-test-split/train/y/y-10d-robust.csv', header=None, dtype=np.float64))
	y_test = np.array(pd.read_csv('/Users/mnbucher/git/eth-master-thesis/data/train-test-split/test/y/y-10d-robust.csv', header=None, dtype=np.float64))
	x_10d_robust = np.concatenate((x_train, x_test))
	y_10d_robust = np.concatenate((y_train, y_test))
	print("x_10d_robust:", x_10d_robust.shape)

	# 10d-simple
	x_train = np.array(pd.read_csv('/Users/mnbucher/git/eth-master-thesis/data/train-test-split/train/x/x-10d-simple.csv', header=None, dtype=np.float64))
	x_test = np.array(pd.read_csv('/Users/mnbucher/git/eth-master-thesis/data/train-test-split/test/x/x-10d-simple.csv', header=None, dtype=np.float64))
	y_train = np.array(pd.read_csv('/Users/mnbucher/git/eth-master-thesis/data/train-test-split/train/y/y-10d-simple.csv', header=None, dtype=np.float64))
	y_test = np.array(pd.read_csv('/Users/mnbucher/git/eth-master-thesis/data/train-test-split/test/y/y-10d-simple.csv', header=None, dtype=np.float64))
	x_10d_simple = np.concatenate((x_train, x_test))
	y_10d_simple = np.concatenate((y_train, y_test))
	print("x_10d_simple:", x_10d_simple.shape)

	fig = plt.figure()
	axes = fig.subplots(2, 5)

	font = {'size' : 18}
	matplotlib.rc('font', **font)
	matplotlib.rcParams['xtick.major.pad']='10'
	matplotlib.rcParams['ytick.major.pad']='10'
	plt.subplots_adjust(top=0.8)

	for dim in range(10):
		idx_row = 0 if dim < 5 else 1
		idx_col = dim % 5
		#axes[idx_row, idx_col].set_title(f"$x_{dim}$: " + dataset.feature_dim_names[dim])
		axes[idx_row, idx_col].set_title(f"dim $x_{dim}$", fontsize=20)

		x_range = (np.min([np.min(x_10d_orig[:, dim]), np.min(x_10d_robust[:, dim]), np.min(x_10d_simple[:, dim])]), np.max([np.max(x_10d_orig[:, dim]), np.max(x_10d_robust[:, dim]), np.max(x_10d_simple[:, dim])]))

		axes[idx_row, idx_col].set_xticks([min(x_10d_orig[:, dim]), max(x_10d_orig[:, dim])])

		axes[idx_row, idx_col].hist(x_10d_orig[:, dim], bins=30, range=x_range, color="#0F084B", label='"10d-orig": Original dataset from Sobol Sampling')
		axes[idx_row, idx_col].hist(x_10d_valid[:, dim], bins=30, range=x_range, color="#26408B", label='"10d-valid": Valid dataset')
		axes[idx_row, idx_col].hist(x_10d_robust[:, dim], bins=30, range=x_range, color="#3D60A7", label='"10d-robust": All samples with $y_1$ <= 2.0')
		axes[idx_row, idx_col].hist(x_10d_simple[:, dim], bins=30, range=x_range, color="#81B1D5", label='"10d-simple": All samples with $y_1$ <= 1.0')

		axes[idx_row, idx_col].tick_params(axis='both', which='major', labelsize=20)
		axes[idx_row, idx_col].tick_params(axis='both', which='minor', labelsize=20)

	#handles, labels = ax.get_lines()
	#handles = axes[0,1].get_lines()
	#fig.legend(handles, labels, loc='upper center')
	#lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
	#lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
	#fig.legend(lines, labels)
	#axes[0,4].legend(bbox_to_anchor=(1.05, 0), loc='lower left', borderaxespad=0.)
	lines, labels = fig.axes[-1].get_legend_handles_labels()
	fig.legend(lines, labels, 'upper center')

	#fig.tight_layout()
	#fig.tight_layout(pad=0.01)

	fig.set_size_inches(19.5, 10.5)
	plt.subplots_adjust(top=0.75)
	fig.savefig("/Users/mnbucher/Downloads/dataset-histogram-x.svg", bbox_inches='tight', pad_inches=0)
	#plt.show()

	fig2, axes2 = plt.subplots(1, 2)
	for dim in range(2):
		axes2[dim].set_title(f"dim $y_{dim}$", fontsize=20)

		y_range = (np.min([np.min(y_10d_orig[:, dim]), np.min(y_10d_robust[:, dim])]), np.max([np.max(y_10d_orig[:, dim]), np.max(y_10d_robust[:, dim])]))

		is_log = True
		#if dim == 1:
			#is_log=True

		axes2[dim].set_xticks([min(y_10d_orig[:, dim]), max(y_10d_orig[:, dim])])

		axes2[dim].hist(y_10d_orig[:, dim], bins=30, log=is_log, range=y_range, color="#0F084B", label='"10d-orig": Original dataset from Sobol Sampling')
		axes2[dim].hist(x_10d_valid[:, dim], bins=30, log=is_log, range=y_range, color="#26408B", label='"10d-valid": Valid dataset')
		axes2[dim].hist(y_10d_robust[:, dim], bins=30, log=is_log, range=y_range, color="#3D60A7", label='"10d-robust": All samples with $y_1$ <= 2.0')
		axes2[dim].hist(y_10d_simple[:, dim], bins=30, log=is_log, range=y_range, color="#81B1D5", label='"10d-simple": All samples with $y_1$ <= 1.0')

		axes2[dim].tick_params(axis='both', which='major', labelsize=20)
		axes2[dim].tick_params(axis='both', which='minor', labelsize=20)

	lines, labels = fig2.axes[-1].get_legend_handles_labels()
	fig2.legend(lines, labels, 'upper center') #, prop={'size': 20})
	fig2.set_size_inches(19.5, 10.5)
	plt.subplots_adjust(top=0.75)
	#plt.subplots_adjust(top=0.8)

	#fig2.tight_layout()
	#fig2.tight_layout(pad=0.2)
	#fig2.set_size_inches(18.5, 10.5)

	fig2.savefig("/Users/mnbucher/Downloads/dataset-histogram-y.svg", bbox_inches='tight', pad_inches=0)

	#plt.show()



def plot_mape_hist_dim_wise(all_x_recon, all_x, x_concat_min, x_concat_max):

	all_x = dataset.unnormalize_data(all_x, x_concat_min, x_concat_max)
	all_x_recon = dataset.unnormalize_data(all_x_recon, x_concat_min, x_concat_max)
	all_mape = np.divide(np.abs(all_x - all_x_recon), all_x) * 100

	fig, axes = plt.subplots(2, 5)

	for dim in range(all_x.shape[1]):
		idx_row = 0 if dim < 5 else 1
		idx_col = dim % 5
		axes[idx_row, idx_col].set_title(dataset.feature_dim_names[dim])
		axes[idx_row, idx_col].hist(all_mape[:, dim], 20, color='b')

	plt.show()


def plot_latent_space_price_ranges(all_zs_with_ys, z_dim, dataset_stats):

	z = all_zs_with_ys[:, :z_dim].detach().cpu().numpy()
	y = all_zs_with_ys[:, z_dim:].detach().cpu().numpy()

	y = dataset.unnormalize_data(y, dataset_stats[2].numpy(), dataset_stats[3].numpy())

	print(y)

	print("doing PCA on latent space (z)")
	pca = PCA(n_components=3)
	pca.fit(z)
	#print(pca.explained_variance_ratio_)
	#print(pca.components_)
	#n_pcs= pca.components_.shape[0]
	#most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
	#print(most_important)
	z_proj_3d = pca.transform(z)

	#indices_filtered = np.where((y_all[:, 0] < 2745) & (y_all[:, 0] > 2743) & (y_all[:, 1] > 0.4) & (y_all[:, 1] < 0.5))

	idxs_cheap = np.where(y[:, 0] < 2700.0)
	idxs_med = np.where((y[:, 0] >= 2700.0) & (y[:, 0] < 6700.0))
	idxs_exp = np.where(y[:, 0] >= 6700.0)

	# idxs_cheap = np.where(y[:, 0] < 0.2)
	# idxs_med = np.where((y[:, 0] >= 0.2) & (y[:, 0] < 1.0))
	# idxs_exp = np.where(y[:, 0] >= 1.0)

	bins = [idxs_cheap[0], idxs_med[0], idxs_exp[0]]
	colors = ['b', 'g', 'r']
	labels = ["y0 < 2700k CHF", "2700k CHF <= y0 < 6700k CHF", "y0 >= 6700k CHF"]

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	for i in range(3):
		idxs = bins[i]
		#print(idxs)
		ax.scatter(z_proj_3d[idxs, 0], z_proj_3d[idxs, 1], z_proj_3d[idxs, 2], c=colors[i], label=labels[i])

	ax.set_title("Latent space of z down to 3D with PCA")
	ax.legend()

	plt.show()


def mape_vs_kldivergence():
	pass
	#mape = [65.01552808093416, 43.56814696122703, 41.77817832389367, 41.51306901385489, 39.717186803089696, 116.5533332491141, 131.56732059042525, 112.48716928724735]
	#kldiv = [0.13841555905763922, 0.015054032755645989, 0.005359445453240289, 0.003605279462238144, 0.0022057874491841047, 0.0012373808399887202, 0.0023402837004646293, 0.0006617709636472761]


def plot_beta_schedule():
	#schedule = learning.adapt_beta_schedule_cyclic_linear(0.0, 1.0, 500, n_cycle=1, ratio=0.25)
	#schedule = learning.get_schedule_beta_kld_cyclic_linear(0.0, 5.0, 600, n_cycle=1, ratio=0.50)

	#schedule = learning.get_schedule_beta_kld_cyclic_linear(200, 2000, n_cycle=1, ratio=0.25)
	schedule = learning.get_schedule_beta_kld_cyclic_linear(100, 600, max_beta=1.0, n_cycle=1, ratio=0.3)

	plt.scatter(np.arange(0, schedule.shape[0], 1), schedule)
	plt.show()


def plot_sobol_vs_random():

	#n_samples = 128
	n_samples = 2**5

	#space = Space([(-5., 10.), (0., 15.)])
	space = Space([(20., 200.),(5., 100.),(10., 25.), (1., 40.),
		(10., 90.),(0.2, 3.5), (0.001, 0.01),(0.5, 2.),
		(0.005, 0.04),(0.01, 0.07)])

	#x = space.rvs(n_samples)

	#sobol = Sobol()
	#x = sobol.generate(space.dimensions, n_samples)

	print("sample")
	grid = Grid(border="include", use_full_layout=False)
	x = grid.generate(space.dimensions, n_samples)

	x = np.array(x)

	fig, axes = plt.subplots(1,1)

	print("pca")
	pca = PCA(n_components=3)
	pca.fit(x)
	x_proj_3d = pca.transform(x)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(x_proj_3d[:, 0], x_proj_3d[:, 1], x_proj_3d[:, 2])

	#axes.plot(x[:, 0], x[:, 1], 'bo')
	#axes.margins(0,0)

	fig.tight_layout()
	fig.savefig("/Users/mnbucher/Downloads/method_sampling_2d_grid.svg", bbox_inches='tight', pad_inches=0)


def check_g_dimension_distr():
	all_dims = []
	for i in range(26089):
		if i % 1000 == 0:
			print(i)
		filepath = f"./data/g/10d-orig/{i}_adj.csv"
		if os.path.isfile(filepath):
			adj = np.array(pd.read_csv(filepath))
			all_dims.append(adj.shape[0])

	print("# of samples:", len(all_dims))

	plt.hist(all_dims, bins=40)
	plt.show()


def plot_baseline_y_error_distr_for_y_test(va, y_mape, ds):

	y_test = np.array(pd.read_csv(f'/Users/mnbucher/git/eth-master-thesis/data/train-test-split/test/y/y-{ds}.csv', header=None, dtype=np.float64))

	#fig, axes = plt.subplots(2,1)

	font = {'size' : 18}
	matplotlib.rc('font', **font)
	matplotlib.rcParams['xtick.major.pad']='10'
	matplotlib.rcParams['ytick.major.pad']='10'

	for dim in range(y_mape.shape[1]):

		fig, axes = plt.subplots(1,1)

		if va is not None:
			bin_edges = va.dataset_train.y_bin_edges.get(f"y_bin_edges_dim_{dim+1}")
			bin_edges = dataset.unnormalize_data(bin_edges, va.dataset_train.get_dataset_stats()[2][dim], va.dataset_train.get_dataset_stats()[3][dim])
		else:
			_, bin_edges = np.histogram(y_test[:, dim], bins=100)

			# if "5d-wall" not in ds:
			# 	dataset_train = dataset.BridgeDataset(ds, 0 if y_mape.shape[1] == 0 else 'y2d', "train", None, do_normalization=True, do_return_g=False)
			# 	dataset_test = dataset.BridgeDataset(ds, 0 if y_mape.shape[1] == 0 else 'y2d', "test", None, do_normalization=True, do_return_g=False)
			# else:
			# 	dataset_train = dataset.WallDataset(ds, "train", None)
			# 	dataset_test = dataset.WallDataset(ds, "test", None)

			# bin_edges = dataset_train.y_bin_edges.get(f"y_bin_edges_dim_{dim+1}")
			# bin_edges = dataset.unnormalize_data(bin_edges, dataset_train.get_dataset_stats()[2][dim], dataset_train.get_dataset_stats()[3][dim])

		n_bins = bin_edges.shape[0]
		bin_idxs = np.digitize(y_test[:, dim], bin_edges) - 1

		bins_mape = [ [] for i in range(n_bins) ]
		for i in range(y_mape[:, dim].shape[0]):
			bin_idx = bin_idxs[i]
			bins_mape[bin_idx].append(y_mape[i, dim])

		bins_mape_avg = [ np.mean(elem) for elem in bins_mape ]
		bins_mape_stdv = [ np.std(elem) for elem in bins_mape ]
		bins_cnt = [ len(elem) for elem in bins_mape ]

		x_axis = list(np.linspace(bin_edges[0], bin_edges[-1], n_bins))

		ax2 = axes.twinx()
		ax2.bar(x_axis, bins_cnt, width=(bin_edges[1]-bin_edges[0]), edgecolor='lightgray', color='lightgray', zorder=10)

		#axes[dim].hist(y_test[:, dim], bins=bin_edges, color="#26408B")
		#ax2.set_title(f"Distr. $y_{dim}$", fontsize=20)

		axes.errorbar(x_axis, bins_mape_avg, yerr=bins_mape_stdv, fmt='o', color="#26408B")
		axes.set_title(f"Error comparison (MAPE) for $y_{dim}$", fontsize=20)

		axes.set_ylabel('MAPE (%)', fontsize=20)
		ax2.set_ylabel('Bin Count', fontsize=20)
		axes.set_xlabel(f'$y_{dim}$', fontsize=20)

		axes.set_zorder(ax2.get_zorder()+1)
		axes.set_frame_on(False)

		axes.tick_params(axis='both', which='major', labelsize=20)
		axes.tick_params(axis='both', which='minor', labelsize=20)

		#fig.set_size_inches(10.5, 14.5)
		fig.set_size_inches(12.5, 8.5)

		fig.tight_layout()
		fig.savefig(f"/Users/mnbucher/Downloads/error-distr-{ds}-dim-{dim}.svg", bbox_inches='tight', pad_inches=0)
		plt.show()


def plot_3d_bridge_from_node_and_edges(node_coords, edges):

	node_coords = node_coords.numpy()
	edges = edges.numpy()

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	#ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2])
	ax.scatter(node_coords[:, 0], 0.0, node_coords[:, 2])
	ax.set_box_aspect(aspect = (np.max(node_coords[:, 0], axis=0) - np.min(node_coords[:, 0], axis=0), np.max(node_coords[:, 1], axis=0) - np.min(node_coords[:, 1], axis=0), np.max(node_coords[:, 2], axis=0) - np.min(node_coords[:, 2], axis=0)))

	#print(edges.shape)

	for col_idx in range(edges.shape[1]):
		edge_coords = edges[:, col_idx]
		n1 = edge_coords[0]
		n2 = edge_coords[1]
		#thickness = edge[1].get("data")[0]
		#print(thickness)
		plt.plot(np.array([node_coords[n1, 0], node_coords[n2, 0]]), np.array([node_coords[n1, 1], node_coords[n2, 1]]), np.array([node_coords[n1, 2], node_coords[n2, 2]]))
		#plt.plot(np.array([all_nodes[n1, 0], all_nodes[n2, 0]]), np.array([all_nodes[n1, 1], all_nodes[n2, 1]]), np.array([all_nodes[n1, 2], all_nodes[n2, 2]]), linewidth=thickness/474.1693967491105)

	plt.show()


def plot_error_distr_for_model():

	run_id = "euler-22-03-16-cvae71-5dwall"
	#sampling_scheme = "proportional"
	sampling_scheme = "uniform"
	y_dims = 1

	fig, axes = plt.subplots(1, y_dims)
	font = {'size' : 18}
	matplotlib.rc('font', **font)
	matplotlib.rcParams['xtick.major.pad']='10'
	matplotlib.rcParams['ytick.major.pad']='10'

	for dim in range(y_dims):

		curr_ax = axes[dim] if y_dims > 1 else axes

		newsamples_yzx, output = analysis.get_x_y_gen_from_runid(run_id, sampling_scheme)

		y = newsamples_yzx[:, 0]
		_, bin_edges = np.histogram(y, bins='auto')
		bin_idxs = np.digitize(y, bin_edges) - 1
		n_bins = len(bin_edges)

		mape, _, _ = analysis.evaluate_y_values_from_files(run_id, 0, sampling_scheme)

		bins_mape = [ [] for i in range(n_bins) ]

		for i in range(newsamples_yzx.shape[0]):
			bin_idx = bin_idxs[i]
			bins_mape[bin_idx].append(mape[i, 0])

		bins_mape_avg = [ np.mean(elem) if elem != [] else np.nan for elem in bins_mape ]
		bins_mape_stdv = [ np.std(elem) if elem != [] else np.nan for elem in bins_mape ]
		bins_cnt = [ len(elem) for elem in bins_mape ]

		x_axis = list(np.linspace(bin_edges[0], bin_edges[-1], n_bins))

		ax2 = curr_ax.twinx()
		ax2.bar(x_axis, bins_cnt, width=(bin_edges[1]-bin_edges[0]), edgecolor='lightgray', color='lightgray', zorder=10)

		curr_ax.errorbar(x_axis, bins_mape_avg, yerr=bins_mape_stdv, fmt='o', color="#3D60A7")
		curr_ax.set_title(f"Error comparison (MAPE) for $y_{dim}$", fontsize=20)

		curr_ax.set_ylabel('MAPE (%)', fontsize=20)
		ax2.set_ylabel('Bin Count', fontsize=20)
		curr_ax.set_xlabel(f'$y_{dim}$', fontsize=20)

		curr_ax.set_zorder(ax2.get_zorder()+1)
		curr_ax.set_frame_on(False)

		curr_ax.tick_params(axis='both', which='major', labelsize=20)
		curr_ax.tick_params(axis='both', which='minor', labelsize=20)

	# fig.set_size_inches(10.5, 14.5)
	# fig.tight_layout()
	# fig.savefig(f"/Users/mnbucher/Downloads/error-distr-gnn0-{ds}.svg", bbox_inches='tight', pad_inches=0)

	plt.show()


def plot_error_distr_y_gen_for_models_y1d():

	#run_ids = ["report-22-02-22-y1d-r01-cvae3", "report-22-02-19-y1d-r02-cvae4", "report-22-02-21-y1d-r01-cvae4-weightedloss", "report-22-02-23-y1d-cvae4-lw-lambda-0.1"]
	#model_names = [ "CVAE-1d-1-NoR", "CVAE-1d-2-Res", "CVAE-1d-3-LW-1.0", "CVAE-1d-4-LW-0.1"]

	#run_ids = [ "report-22-02-19-y1d-r02-cvae4", "report-22-02-23-y1d-cvae4-lw-lambda-0.1", "report-22-02-23-y1d-cvae4-lw-lambda-1.0"]
	#model_names = [ "CVAE-1d-2-Res", "CVAE-1d-4-LW-0.1", "CVAE-1d-3-LW-1.0"]

	#run_ids = ["report-22-02-19-y1d-r02-cvae4", "report-22-02-21-y1d-r01-cvae4-weightedloss"]
	#model_names = ["CVAE-1d-2-Res", "CVAE-1d-3-LW-1.0"]

	#sampling_scheme = "uniform"
	#sampling_scheme = "proportional"

	#run_ids = ["rep-check-22-02-25-y1d-r01-cvae3", "rep-check-22-02-25-y1d-r02-cvae4", "rep-check-22-02-25-y1d-r03-cvae4-lw1.0", "rep-check-22-02-25-y1d-r04-cvae4-lw0.1", "rep-check-22-02-26-y1d-r01-cvae4-no2ndlinear"]

	#run_ids = [ "rep-check-22-02-25-y1d-r01-cvae3", "rep-check-22-02-26-y1d-r01-cvae4-no2ndlinear", "rep-check-22-02-26-y1d-r03-cvae4-no2nd-lw1.0"]
	#model_names = [ "CVAE-1d-1-NoR", "CVAE-1d-2-Res", "CVAE-1d-3-LW-1.0" ]

	run_ids = ["euler-22-03-16-cvae71-5dwall"]
	model_names = run_ids

	#run_ids = ["report-22-02-22-y1d-r01-cvae3", "report-22-02-23-y1d-cvae4-lw-lambda-1.0", "report-22-02-19-y1d-r02-cvae4"]
	#model_names = run_ids

	#colors = [ "#D34747", "#D38647", "#2B7E7E", "#39A939" ]
	#colors = ['b', 'g']
	colors = [ (155, 180, 96), (25, 129, 154), (238, 125, 55), (217, 4, 43), (159, 25, 108)]

	# prep bin edges
	dataset_train = dataset.BridgeDataset("10d-simple", 0, "train", None, do_normalization=True, do_return_g=False)
	bin_edges = dataset_train.y_bin_edges.get(f"y_bin_edges_dim_1")
	bin_edges = dataset.unnormalize_data(bin_edges, dataset_train.get_dataset_stats()[2], dataset_train.get_dataset_stats()[3])
	n_bins = bin_edges.shape[0]

	#sfig, ax = plt.subplots()
	#fig, ax = plt.subplots(1, 2)
	fig, ax = plt.subplots(2,3)

	font = {'size' : 18}
	matplotlib.rc('font', **font)

	for idx, run_id in enumerate(run_ids):

		for idx2, sampling_scheme in enumerate(["proportional", "uniform"]):

			print(run_id, sampling_scheme)

			newsamples_yzx, output = analysis.get_x_y_gen_from_runid(run_id, sampling_scheme)
			if newsamples_yzx is not None:
				mape, rmse, pcc = analysis.evaluate_y_values_from_files(run_id, 0, sampling_scheme)
				bin_idxs = np.digitize(newsamples_yzx[:, 0], bin_edges) - 1
				bins_mape = [ [] for i in range(n_bins) ]
				for i in range(newsamples_yzx.shape[0]):
					bin_idx = bin_idxs[i]
					bins_mape[bin_idx].append(mape[i, 0])
				bins_mape_avg = [ np.mean(elem) if elem != [] else np.nan for elem in bins_mape ]
				bins_mape_stdv = [ np.std(elem) if elem != [] else np.nan for elem in bins_mape ]
				bins_cnt = [ len(elem) for elem in bins_mape ]

				#ax1.hist(y_test[:, dim], bins=bin_edges, color="#26408B")

				x_axis = list(np.linspace(bin_edges[0], bin_edges[-1], n_bins))
				#ax.bar(x_axis, bins_rmse_avg, width=(bin_edges[1]-bin_edges[0]), color=(None if idx < 3 else "#CE8025"), edgecolor=colors[idx], fill=(False if idx < 3 else True), label=model_names[idx])
				#ax.bar(x_axis, bins_cnt, width=(bin_edges[1]-bin_edges[0]), color=(None if idx < 3 else "#CE8025"), edgecolor=colors[idx], fill=(False if idx < 3 else True), label=model_names[idx])

				#ax.bar(x_axis, bins_rmse_avg, width=(bin_edges[1]-bin_edges[0]), edgecolor=colors[idx], fill=False, label=model_names[idx])

				ax2 = ax[idx2, idx].twinx()
				ax2.bar(x_axis, bins_cnt, width=(bin_edges[1]-bin_edges[0]), edgecolor='lightgray', color='lightgray', zorder=10)
				ax[idx2, idx].errorbar(x_axis, bins_mape_avg, yerr=bins_mape_stdv, fmt='o', color=tuple(te/255. for te in colors[idx]), ecolor=tuple(te/255. for te in colors[idx]) + (0.5,), elinewidth=3, capsize=0, zorder=20)

				ax[idx2, idx].set_zorder(ax2.get_zorder()+1)
				ax[idx2, idx].set_frame_on(False)

				#axes[dim, 1].set_title(f"Distr. MAPE $y_{dim}$", fontsize=16)

				#axes[dim, 0].tick_params(axis='both', which='major', labelsize=14)
				#axes[dim, 0].tick_params(axis='both', which='minor', labelsize=14)
				#axes[dim, 1].tick_params(axis='both', which='major', labelsize=14)
				#axes[dim, 1].tick_params(axis='both', which='minor', labelsize=14)
				#plt.show()

				ax[idx2, idx].set_title(model_names[idx])
				ax[idx2, idx].set_ylabel('MAPE (%)', fontsize=20)
				ax2.set_ylabel('Bin Count', fontsize=20)
				ax[idx2, idx].set_xlabel('$y_0$', fontsize=20)
				ax[idx2, idx].set_xticks([0, bin_edges[-1]])
				ax[idx2, idx].tick_params(axis='both', which='major', labelsize=20)
				ax[idx2, idx].tick_params(axis='both', which='minor', labelsize=20)

		print("")


	#lines, labels = fig.axes[-1].get_legend_handles_labels()
	#fig.legend(lines, labels, 'upper center')
	#fig.legend(lines, model_names, 'upper center')

	fig.set_size_inches(14.5, 10.5)
	plt.subplots_adjust(top=0.75)

	fig.tight_layout()
	fig.savefig(f"/Users/mnbucher/Downloads/error-distr-cs1-uniform.svg", bbox_inches='tight', pad_inches=0)
	plt.show()


def plot_x_distr_for_fixed_y():

	#run_id = "report-22-02-25-y1d-r01-cvae3"
	run_id = "report-22-02-26-y1d-r02-cvae4"
	sampling_scheme = "proportional"

	np.set_printoptions(suppress=True)

	newsamples_yzx, output = analysis.get_x_y_gen_from_runid(run_id, sampling_scheme)
	mape, rmse, pcc = analysis.evaluate_y_values_from_files("10d-simple", run_id, 0, sampling_scheme, do_print=False, plot=False)

	#print(np.min(mape), np.max(mape))
	#plt.hist(mape)
	#plt.show()

	mape_with_idxs = np.zeros((640, 2))
	mape_with_idxs[:, 0] = np.arange(640)
	mape_with_idxs[:, 1] = np.squeeze(mape)

	#mape_filtered = mape_with_idxs[mape_with_idxs[:, 1] < 5.0]
	#mape_filtered = mape_with_idxs[mape_with_idxs[:, 1] > 20.0]
	mape_filtered = mape_with_idxs

	ys = newsamples_yzx[mape_filtered[:, 0].astype(int), 0]
	xs = newsamples_yzx[mape_filtered[:, 0].astype(int), 7:]

	ys = pd.read_csv(f'./data/train-test-split/train/y/y-10d-simple.csv', header=None, dtype=np.float64).to_numpy()[:, 0]
	xs = pd.read_csv(f'./data/train-test-split/train/x/x-10d-simple.csv', header=None, dtype=np.float64).to_numpy()

	#print(ys)

	# low price range
	# 5820 - 5870
	#print(ys)
	#print(np.min(ys), np.max(ys))

	mask_01 = ys < 7000
	mask_02 = (ys > 10000)

	ys_01 = ys[mask_01]
	xs_01 = xs[mask_01, :]
	print(xs_01.shape)

	ys_02 = ys[mask_02]
	xs_02 = xs[mask_02, :]
	print(xs_02.shape)

	x_min = torch.load("./data/stats/10d-simple/x-stats-min.pt").numpy()
	x_max = torch.load("./data/stats/10d-simple/x-stats-max.pt").numpy()
	x_train = np.array(pd.read_csv('/Users/mnbucher/git/eth-master-thesis/data/train-test-split/train/x/x-10d-simple.csv', header=None, dtype=np.float64))
	x_test = np.array(pd.read_csv('/Users/mnbucher/git/eth-master-thesis/data/train-test-split/test/x/x-10d-simple.csv', header=None, dtype=np.float64))
	y_train = np.array(pd.read_csv('/Users/mnbucher/git/eth-master-thesis/data/train-test-split/train/y/y-10d-simple.csv', header=None, dtype=np.float64))
	y_test = np.array(pd.read_csv('/Users/mnbucher/git/eth-master-thesis/data/train-test-split/test/y/y-10d-simple.csv', header=None, dtype=np.float64))
	x_10d_simple = np.concatenate((x_train, x_test))
	#y_10d_simple = np.concatenate((y_train, y_test))

	subset_idxs = np.random.choice(np.arange(x_train.shape[0]), 1000)
	subset_idxs = np.random.choice(np.arange(x_train.shape[0]), x_train.shape[0])

	fig, axes = plt.subplots(2,5)
	for dim in range(10):

		if dim > 4:
			dim_1 = 1
		else:
			dim_1 = 0

		print(dim_1, dim % 5)

		axes[dim_1, dim % 5].hist(x_10d_simple[subset_idxs, dim], color=(0, 0, 1, 0.3), log=True)
		axes[dim_1, dim % 5].hist(xs_01[:, dim], color=(0, 0, 1, 1.0), log=True, label="y0 < 7000k CHF")
		axes[dim_1, dim % 5].hist(xs_02[:, dim], color=(1, 0, 0, 1.0), log=True, label="y0 > 10000k CHF")
		axes[dim_1, dim % 5].set_title(dataset.feature_dim_names[dim])
		axes[dim_1, dim % 5].set_xlim([x_min[dim], x_max[dim]])

	lines, labels = fig.axes[-1].get_legend_handles_labels()
	fig.legend(lines, labels, 'upper center')
	plt.show()


def plot_graph_norms():

	g_train = torch.load("./data/train-test-split/train/g/g-10d-simple-graph.pt")
	g_test = torch.load("./data/train-test-split/test/g/g-10d-simple-graph.pt")
	g_graph_stats = torch.load(f"./data/stats/10d-simple/g-graph-stats.pt")

	all_frob_norms = []
	all_frob_norms_weighted = []

	data = [ g_train, g_test ]

	for gs in data:
		for idx in tqdm(range(len(gs))):
			nodes, edges, edges_data = gs[idx]
			nodes, edges, edges_data = nodes.numpy(), edges.numpy(), edges_data.numpy()

			adj = np.zeros((nodes.shape[0], nodes.shape[0]))
			adj[edges[0, :].astype(int), edges[1, :].astype(int)] = 1
			adj[edges[1, :].astype(int), edges[0, :].astype(int)] = 1
			frob_norm = np.linalg.norm(adj, ord='fro')
			all_frob_norms.append(frob_norm)

			edges_dat_norm = dataset.get_normalized_edge_data(torch.tensor(edges_data), g_graph_stats).numpy()
			adj = np.zeros((nodes.shape[0], nodes.shape[0]))
			adj[edges[0, :].astype(int), edges[1, :].astype(int)] = edges_dat_norm.sum(axis=1)
			adj[edges[1, :].astype(int), edges[0, :].astype(int)] = edges_dat_norm.sum(axis=1)
			frob_norm = np.linalg.norm(adj, ord='fro')
			all_frob_norms_weighted.append(frob_norm)

	fig, axis = plt.subplots(1,2)
	axis[0].hist(all_frob_norms, bins=100)
	axis[0].set_title("Frobenius norm of Graph Adj (no weight)")

	axis[1].hist(all_frob_norms_weighted, bins=100)
	axis[1].set_title("Frobenius norm of Graph Adj (weighted)")

	plt.show()


def plot_y_distr_heatmap():

	#run_id = "report-22-02-26-y1d-r02-cvae4"

	run_id = "report-22-03-03-y2d-cvae4-relu"

	scheme = "uniform"

	newsamples_yzx, _ = analysis.get_x_y_gen_from_runid(run_id, scheme)

	print(newsamples_yzx.shape)

	#ys = newsamples_yzx[:, 0]
	ys = newsamples_yzx[:, :2]

	#plt.hist(ys)
	h, _, _, _ = plt.hist2d(ys[:, 0], ys[:, 1])

	print(h.shape)
	print(h)

	print(np.sum(h, axis=0))
	print(np.sum(h, axis=1))

	plt.show()


def plot_x_distr_wall_data(show_plot=False, colors_x_y=None, is_log=False):

	dataset_name = "5d-wall-v2"

	if colors_x_y is None:
		colors_x_y = ["#26408B", "#81B1D5"]

	# 10d-simple
	x_all = np.array(pd.read_csv(f'./data/x/x-{dataset_name}.csv', header=None, dtype=np.float64))
	y_all = np.array(pd.read_csv(f'./data/y/y-{dataset_name}.csv', header=None, dtype=np.float64))[:, 0]
	print(x_all.shape, y_all.shape)

	fig = plt.figure()
	axes = fig.subplots(2, 3)

	font = {'size' : 18}
	matplotlib.rc('font', **font)
	matplotlib.rcParams['xtick.major.pad']='10'
	matplotlib.rcParams['ytick.major.pad']='10'
	#plt.subplots_adjust(top=0.8)

	for dim in range(5):
		idx_row = 0 if dim < 3 else 1
		idx_col = dim % 3
		axes[idx_row, idx_col].set_title(f"dim $x_{dim}$", fontsize=20)

		x_range = (np.min(x_all[:, dim]), np.max(x_all[:, dim]))
		axes[idx_row, idx_col].set_xticks([np.min(x_all[:, dim]), np.max(x_all[:, dim])])
		axes[idx_row, idx_col].tick_params(axis='both', which='major', labelsize=20)
		axes[idx_row, idx_col].tick_params(axis='both', which='minor', labelsize=20)

		axes[idx_row, idx_col].hist(x_all[:, dim], bins=30, range=x_range, log=is_log, color=colors_x_y[0])

	axes[1, 2].set_title(f"dim $y$", fontsize=20)
	y_range = (np.min(y_all), np.max(y_all))
	axes[1, 2].set_xticks([np.min(y_all), np.max(y_all)])
	axes[1, 2].tick_params(axis='both', which='major', labelsize=20)
	axes[1, 2].tick_params(axis='both', which='minor', labelsize=20)

	axes[1, 2].hist(y_all, bins=30, range=y_range, log=is_log, color=colors_x_y[1])

	fig.suptitle("Histogram for dimensions of 5d-wall")

	fig.set_size_inches(19.5, 10.5)
	fig.savefig("/Users/mnbucher/Downloads/dataset-wall-histogram-x-y.svg", bbox_inches='tight', pad_inches=0)

	if show_plot:
		plt.show()

	return fig, axes


def plot_data_distr_for_gen_samples_wall_data():

	colors_x_y = ["#26408B", "#81B1D5"]

	fig, axes = plot_x_distr_wall_data(show_plot=False, colors_x_y=["#c2c2c2", "#c2c2c2"], is_log=True)

	newsamples_yzx, output = analysis.get_x_y_gen_from_runid("euler-22-03-23-cvae71yxpart-yaugm-5dwall-z207", "proportional")

	gen_x_all = newsamples_yzx[:, 213:]

	for dim in range(5):
		idx_row = 0 if dim < 3 else 1
		idx_col = dim % 3
		axes[idx_row, idx_col].hist(gen_x_all[:, dim], bins=30, log=True, color=colors_x_y[0])

	axes[1, 2].hist(output[:, 0], bins=30, log=True, color=colors_x_y[1])

	plt.show()


def plot_histogram_yxpartial_bins(va):

	y_x_partial_indices = va.dataset_train.y_x_partial_indices

	#tot_n_elems = []

	indices = y_x_partial_indices[torch.randint(low=0, high=y_x_partial_indices.shape[0], size=(640,)), :]
	n_elems = torch.count_nonzero(indices, dim=1).double().numpy()

	#bins = np.arange(0, n_elems.max() + 1.5) - 0.5

	fig, ax = plt.subplots()
	ax.bar(*np.unique(n_elems, return_counts=True), color="#26408B")

	ax.tick_params(axis='both', which='major', labelsize=20)
	ax.tick_params(axis='both', which='minor', labelsize=20)

	ax.set_xticks(list(range(int(n_elems.max()))))

	fig.set_size_inches(12.5, 8.5)
	fig.savefig(f"/Users/mnbucher/Downloads/results-histogram-yxpart-{va.cli_args.dataset}.svg", bbox_inches='tight', pad_inches=0)

	plt.show()

	#print(np.mean(n_elems))


if __name__ == '__main__' :

	#plot_x1_x2_y0()

	#plot_latent_spaces_comparison()

	#plot_y_2d()

	#plot_beta_schedule()

	#plot_x0_y0_density()

	#plot_sobol_vs_random()

	#plot_x10d_pca()
	#plot_x10d_distr_by_histograms()
	#check_g_dimension_distr()

	#plot_lambda_lw_vs_mape()
	#plot_error_distr_y_gen_for_models_y1d()

	#plot_error_distr_for_model()

	#plot_data_distr_for_gen_samples_wall_data()

	#plot_y_distr_heatmap()

	#plot_x_distr_for_fixed_y()
	#plot_graph_norms()

	plot_x_distr_wall_data(show_plot=True)

	#check_invalid_x_y_pairs()

	#plot_x_y_distr()
	#plot_y_distr()

	pass
