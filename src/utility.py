from datetime import datetime
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
from torchvision.utils import save_image
import torch
import random
import yaml
import requests
import scipy.special
import shutil
import logging

import sys
sys.path.append(os.getcwd())

import src.dataset as dataset


class VanillaArgs():
	dataset_train = None
	dataset_test = None
	data_loader_train = None
	data_loader_test = None
	loss_func = None
	lr = None
	cli_args = None
	dvc = None
	ckpt_path = None
	model_config = None
	run_id = None
	gen_n_y = None
	gen_n_z = None
	logger = None


def arg_int_or_str(value):
	try:
		return int(value)
	except:
		return value


def set_seeds(seed):
	torch.manual_seed(seed)
	random.seed(seed)
	np.random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	torch.cuda.manual_seed(seed)
	torch.use_deterministic_algorithms(True)
	torch.backends.cudnn.deterministic = True


def get_tgseed(seed):
	g = torch.Generator()
	g.manual_seed(seed)
	return g


def collate_xyg(batch):
	x = torch.stack([item[0] for item in batch], dim=0)
	y = torch.stack([item[1] for item in batch], dim=0)
	g = [item[2] for item in batch]
	return x, y, g


def get_collate_fn(model_type):
	if "gcnn" in model_type or "forward" in model_type:
		do_return_g = True
		collate_fn = collate_xyg
	else:
		do_return_g = False
		collate_fn = None

	return do_return_g, collate_fn


def log_metric(va, name, value, epoch=None):
	if va.wand_exp:
		if epoch:
			va.wand_exp.log({ name: value, "epoch": epoch })
		else:
			va.wand_exp.log({ name: value })


def save_ckpt(state, ckpt_path, seed, is_best=True):
	if is_best:
		#torch.save(state, os.path.join(ckpt_path, f'ckpt_best_seed_{seed}.pth.tar'))
		torch.save(state, os.path.join(ckpt_path, f'ckpt_last_seed_{seed}.pth.tar'))
	else:
		torch.save(state, os.path.join(ckpt_path, f'ckpt_last_seed_{seed}.pth.tar'))


def sample(latent_size, model, run_id, epoch, n):
	with torch.no_grad():

		all_samples = torch.zeros((n, 10, 1, 28, 28))

		for i in range(n):
			c = torch.eye(10, 10)
			sample = torch.randn(10, latent_size)
			sample = model.decode(sample, c)

			all_samples[i] = sample.view(10, 1, 28, 28)

			save_image(sample.view(10, 1, 28, 28), f'img/{run_id}/newsamples/sample_{i}_' + str(epoch) + '.png')

		print(all_samples.shape)

		all_samples = torch.mean(all_samples, axis=0)

		print(all_samples.shape)

		save_image(all_samples.view(10, 1, 28, 28), f'img/{run_id}/newsamples/mean_sample_' + str(epoch) + '.png')
		save_image(sample.view(10, 1, 28, 28), f'img/{run_id}/newsamples/sample_' + str(epoch) + '.png')


def do_initial_checks(args):

	if args.run_id and not args.resume:

		if not args.overwrite and os.path.exists(f'ckpt/{args.run_id}'):
			raise Exception(f"Attention! 'ckpt/{args.run_id}' already exists! Choose a unique run_id")

		if args.overwrite and os.path.exists(f'ckpt/{args.run_id}'):
			best_model_name = f'ckpt/{args.run_id}/ckpt_best_seed_{args.seed}.pth.tar'
			last_model_name = f'ckpt/{args.run_id}/ckpt_last_seed_{args.seed}.pth.tar'
			if os.path.isfile(best_model_name):
				os.remove(best_model_name)
			if os.path.isfile(last_model_name):
				os.remove(last_model_name)

		run_id = args.run_id

	elif args.resume:
		run_id = args.resume.split("/")[-2]
	else:
		now = datetime.now()
		run_id = now.strftime("%Y-%m-%d-%H-%M-%S")

	ckpt_path = f'ckpt/{run_id}'
	os.makedirs(ckpt_path, exist_ok=True)

	if not args.is_vanilla_training and os.path.isfile(ckpt_path + '/bay-opt-results.csv'):
		os.remove(ckpt_path + '/bay-opt-results.csv')

	if args.is_vanilla_training and os.path.isfile(ckpt_path + '/model-epoch-results.csv'):
		os.remove(ckpt_path + '/model-epoch-results.csv')

	return run_id, ckpt_path


def check_proxy():
	remote_host_rhino = "http://brokki.ch:7001"
	#proxies = {"http": "http://proxy.ethz.ch:3128"}
	#r = requests.get(remote_host_rhino, json=None, proxies=proxies)
	r = requests.get(remote_host_rhino, json=None)
	print(r.text)


def save_metrics_from_itr(analysis_type, itr, ckpt_path, mape, loss_test, loss_recon, loss_kld):
	with open(ckpt_path + "/" + analysis_type + "-results.csv", 'a') as file:
		file.write(f"{itr},{loss_test},{loss_recon},{loss_kld},{mape}\n")


def get_relevant_model_configs(model_config):
	return model_config

def print_params(model):
	for name, param in model.named_parameters():
		print(name, param)


def resume_model(va, resume_path, model, optimizer):
	logtext(f"resume ckpt from {resume_path}...", va)
	logtext(f"make backup at {resume_path}.backup", va)
	shutil.copyfile(resume_path, resume_path + ".backup")
	ckpt = torch.load(resume_path, map_location=va.dvc)
	lr = ckpt['lr']
	model.load_state_dict(ckpt['state_dict'])

	if optimizer is not None:
		optimizer.load_state_dict(ckpt['optimizer'])

	logtext(f"Ckpt loaded (epoch: {ckpt['epoch']} | best_err: {ckpt['err']})", va)
	return lr, ckpt['epoch']


def read_model_config_from_file(ckpt_path, va):
	logtext("read config file...", va)
	with open(ckpt_path + "/config.yaml", "r") as file:
		config = yaml.safe_load(file)
		logtext(f"config: {config}", va)
		return config


def write_model_config_to_file(model_config, ckpt_path, va):
	logtext("write model configs to file...", va)
	with open(ckpt_path + "/config.yaml", "w+") as file:
		yaml.dump(model_config, file)


def get_logger(run_id):
	curr_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
	logger_fn = f"./log/train/train-{run_id}-{curr_time}.log"
	lh = logging.FileHandler(logger_fn, encoding="utf-8")
	lh.setFormatter(logging.Formatter("%(message)s"))
	logger = logging.getLogger()
	logger.setLevel(logging.INFO)
	logger.addHandler(lh)
	return logger, logger_fn


def logtext(text, va=None):
	if va is not None and va.cli_args is not None and va.cli_args.is_logging_enabled:
		va.logger.info(text)
	else:
		print(text)


def get_total_n(max_k, x_dim):
	total_n = 0
	total_n_elems = 0
	for k in range(1, max_k+1):
		sub_n = int(scipy.special.binom(x_dim, k))
		total_n += sub_n
		total_n_elems += sub_n*k
	#print(f"total_n: {total_n}, total_n_elems: {total_n_elems}")
	return total_n, total_n_elems


def get_indices(n, x_shape, k=1):
	x = torch.zeros(n, x_shape)
	if k == 0:
		return x
	idx = 0
	for i in range(x_shape):
		sub_n = int(scipy.special.binom(x_shape-1-i, k-1))
		x[idx:(idx+sub_n), i] = 1
		if i < x_shape-1:
			new = get_indices(sub_n, x_shape-1-i, k-1)
			x[idx:(idx+sub_n), (i+1):] = new
		idx += sub_n
	return x


def create_y_x_partial_indices_file(x_dim, dim_config=None, max_n=None, store_file=False, ds_name=None):

	if max_n is None:
		max_n = x_dim

	total_n, total_n_elems = get_total_n(max_n, x_dim)
	y_x_partial_mask = torch.zeros(total_n, x_dim)
	x_shape = y_x_partial_mask.shape[1]
	idx = 0

	for k in range(1, max_n+1):
		n = int(scipy.special.binom(x_dim, k))
		#print(f"k = {k}, n = {n}")
		y_x_partial_mask[idx:(idx+n)] = get_indices(n, x_shape, k)
		idx += n

	assert total_n_elems == torch.count_nonzero(y_x_partial_mask)
	assert y_x_partial_mask.shape == torch.unique(y_x_partial_mask, dim=0).shape, "masks are not unique!"

	if dim_config:
		x_dim_enc = dim_config.get("x_dim_enc")
		x_dim_orig = dim_config.get("x_dim_orig")

		y_x_partial_mask_enc = torch.zeros(total_n, x_dim_enc)

		col_idx = 0
		for i in range(x_dim_orig):
			xi = dim_config.get(f"x{i}")
			if xi.get("is_categorical"):
				n = len(xi.get("vals"))
				y_x_partial_mask_enc[:, col_idx:(col_idx+n)] = torch.repeat_interleave(torch.unsqueeze(y_x_partial_mask[:, i], 1), repeats=n, dim=1)
				col_idx += n
			else:
				y_x_partial_mask_enc[:, col_idx] = y_x_partial_mask[:, i]
				col_idx += 1

		y_x_partial_mask = y_x_partial_mask_enc

	if store_file:
		torch.save(y_x_partial_mask, f"./data/stats/{ds_name}/y-x-partial-indices.pt")

	return y_x_partial_mask


def get_dvc():
	return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


if __name__ == "__main__":

	create_y_x_partial_indices_file(x_dim=5, max_n=None, store_file=True, ds_name="5d-wall-v2")
	#create_y_x_partial_indices_file(x_dim=10, max_n=None, store_file=True, ds_name="10d-simple")
