import os
import numpy as np
import pandas as pd
from datetime import datetime
import yaml
import torch
import numpy as np
import logging
from flask import Flask, render_template
from flask import json, jsonify
from flask import request
from pycel import ExcelCompiler

import src.utility as utility
import src.models as models
import src.analysis as analysis
import src.dataset as dataset


# ****************************************************************************************************** #
# consts and global vars

CONFIGS = {
	"wall5d": "web/configs/wall-5d.yaml",
	#"bridge10d": "web/configs/bridge-10d.yaml",
	"hall10d": "web/configs/hall-10d.yaml",
	"room5d": "web/configs/room-5d.yaml"
}

va = utility.VanillaArgs()

# ****************************************************************************************************** #

def prep_logger():
	logger_fn = f"./web/log/log_gen_webserver.log"
	lh = logging.FileHandler(logger_fn, encoding="utf-8")
	lh.setFormatter(logging.Formatter("%(message)s"))
	logger = logging.getLogger()
	logger.setLevel(logging.INFO)
	logger.addHandler(lh)
	va.logger = logger


def load_all_configs():
	va.all_configs = {}
	for k, v in CONFIGS.items():
		va.logger.info(f"load_gen_config for {k}")
		with open(v, "r") as file:
			va.all_configs[k] = lambda: None
			va.all_configs[k].gen_config = yaml.safe_load(file)


def load_models():
	va.dvc = utility.get_dvc()

	va.model_configs = {}
	for config in CONFIGS:

		va.all_configs[config].model_va = lambda: None
		va.all_configs[config].model_va.cli_args = lambda: None
		va.all_configs[config].model_va.logger = va.logger
		va.all_configs[config].model_va.cli_args.is_logging_enabled = True

		gen_config = va.all_configs[config].gen_config
		model_pth = gen_config.get("model_dir")

		model_config = utility.read_model_config_from_file("ckpt/" + model_pth.split("/")[1], va=va.all_configs[config].model_va)

		va.all_configs[config].model_va.model_config = model_config
		va.all_configs[config].model_va.dvc = utility.get_dvc()
		va.all_configs[config].model_va.cli_args.y_augm = False
		va.all_configs[config].model_va.cli_args.y_cols = gen_config.get("y_cols")
		va.all_configs[config].model_va.cli_args.seed = 1234
		va.all_configs[config].model_va.cli_args.pretrained_gnn = gen_config.get("pretrained_gnn")

		if gen_config.get("dataset") == "dynamic":
			stats, dim_config = dataset.get_dynamic_dataset(va.all_configs[config].model_va, gen_config.get("dataset_path_csv"), va.all_configs[config].model_va.model_config.get("batch_size"), va.all_configs[config].model_va.model_config.get("shift_all_dimensions"), is_stats_only=True)
			va.all_configs[config].dataset_stats = (stats.get("x_stats_min"), stats.get("x_stats_max"), stats.get("y_stats_min"), stats.get("y_stats_max"))
			va.all_configs[config].model_va.dim_config = dim_config

			for j in range(len(va.all_configs[config].gen_config.get("dimensions").get("x"))):
				if dim_config.get(f"x{j}").get("vals") is not None:
					xi_vals = dim_config.get(f"x{j}").get("vals")
					if va.all_configs[config].model_va.model_config.get("shift_all_dimensions"):
						xi_vals -= 1.0
					values_to_choose = "[" + ', '.join([str(elem) for elem in xi_vals]) + "]"
				else:
					values_to_choose = ""
				va.all_configs[config].gen_config["dimensions"]["x"][j]["values_to_choose"] = values_to_choose

			for i in range(len(va.all_configs[config].gen_config.get("dimensions").get("y"))):
				if dim_config.get(f"y{i}").get("vals") is not None:
					yi_vals = dim_config.get(f"y{i}").get("vals")
					if va.all_configs[config].model_va.model_config.get("shift_all_dimensions"):
						yi_vals -= 1.0
					values_to_choose = "[" + ', '.join([str(elem) for elem in yi_vals]) + "]"
				else:
					values_to_choose = ""
				va.all_configs[config].gen_config["dimensions"]["y"][i]["values_to_choose"] = values_to_choose

		else:
			ds_name = gen_config.get("dataset")
			va.all_configs[config].dataset_stats = (torch.load(f"data/stats/{ds_name}/x-stats-min.pt"), torch.load(f"data/stats/{ds_name}/x-stats-max.pt"), torch.load(f"data/stats/{ds_name}/y-stats-min.pt")[gen_config.get("y_cols")], torch.load(f"data/stats/{ds_name}/y-stats-max.pt")[gen_config.get("y_cols")])

			# TODO: implement dim_config for "fixed" datasets (bridge/wall) ?
			va.all_configs[config].model_va.dim_config = None

			for j in range(len(va.all_configs[config].gen_config.get("dimensions").get("x"))):
				va.all_configs[config].gen_config["dimensions"]["x"][j]["values_to_choose"] = ""

			for i in range(len(va.all_configs[config].gen_config.get("dimensions").get("y"))):
				va.all_configs[config].gen_config["dimensions"]["y"][i]["values_to_choose"] = ""

		# load model
		va.all_configs[config].model = models.init_model(va.all_configs[config].model_va, model_config)
		utility.resume_model(va.all_configs[config].model_va, model_pth, va.all_configs[config].model, None)
		va.all_configs[config].model.eval()


def before_startup():
	prep_logger()
	load_all_configs()
	load_models()


def compute_relative_error(eval_fn, y, y_x, x_hat):

	# relative error for x
	y_x_mask = y_x[0, :] != -1.0
	x_mape = torch.full(x_hat.shape, -1.0, dtype=torch.float32)

	if sum(y_x_mask) > 0:
		x_mape[0, y_x_mask] = 100 * torch.divide(torch.abs(x_hat[0, y_x_mask] - y_x[0, y_x_mask]), y_x[0, y_x_mask] + 1e-10)

	total_err_x = np.round(torch.mean(x_mape[0, y_x_mask]).cpu().detach().numpy(), decimals=2)

	all_errors = {
		"relative_errors_x": [np.round(elem.cpu().detach().numpy(), decimals=2) for elem in x_mape[0, :]],
		"total_err_x": total_err_x if not np.isnan(total_err_x) else None
	}

	print("y_x")
	print(y_x)
	print(x_hat)
	print("")

	# relative error for y
	if eval_fn is not None:
		y_true = globals()[eval_fn](x_hat)
		y_mask = y[0, :] != -1.0
		y_mape = torch.full(y.shape, -1.0, dtype=torch.float32)
		if sum(y_mask) > 0:
			y_mape[0, y_mask] = 100 * torch.divide(torch.abs(y[0, y_mask] - y_true[0, y_mask]), y_true[0, y_mask] + 1e-10)

		total_err_y = np.round(torch.mean(y_mape[0, y_mask]).cpu().detach().numpy(), decimals=2)

		all_errors["relative_errors_y"] = [np.round(elem.cpu().detach().numpy(), decimals=2) for elem in y_mape[0, :]]
		all_errors["total_err_y"] = total_err_y if not np.isnan(total_err_y) else None
	else:
		y_true = None

	return all_errors, y_true


def evaluate_wall5d_via_excel(x_hat):
	fn = "./misc/dataset-02-walls/mw_ec6_master-tweak.xlsx"
	excel = ExcelCompiler(filename=fn)

	x_hat = torch.squeeze(x_hat)

	try:
		excel.evaluate('1!C22')
		excel.evaluate('1!C23')
		excel.evaluate('1!C25')
		excel.evaluate('1!C26')
		excel.evaluate('1!C27')
		excel.evaluate('1!F46')
		excel.evaluate('1!C40')
		excel.evaluate('1!F40')
		excel.evaluate('1!B49')

		excel.set_value('1!C22', x_hat[0].item())
		excel.set_value('1!C23', x_hat[1].item())
		excel.set_value('1!C25', x_hat[2].item())
		excel.set_value('1!C26', x_hat[3].item())
		excel.set_value('1!C27', x_hat[4].item())

		y1 = excel.evaluate('1!F46') # eta_w
		# y2 = excel.evaluate('1!C40') # phi_1
		# y3 = excel.evaluate('1!F40') # phi_2
		# y4 = 1 if excel.evaluate('1!B49') == "OK" else 0 # schlankheit

		#y_true = torch.Tensor([y1, y2, y3, y4])
		y_true = torch.Tensor([[ y1 ]])

		return y_true

	except Exception as exc:
		va.logger.info(exc)
		va.logger.info("could not perform gen eval via excel file (probably nan values in x_hat)")
		return -1


def generate_single_sample(config, y, y_x):

	z = torch.randn(1, config.model_va.model_config.get("model_params").get("z_dim"))

	dim_config = config.model_va.dim_config

	# shift data for forward pass if done during training
	if config.model_va.model_config.get("shift_all_dimensions"):
		y[y != -1.0] += 1.0
		y_x[y_x != -1.0] += 1.0

	# encode y vector if dim_config provided
	if dim_config is not None:
		#print(y)
		y_norm = dataset.encode_dims_for_y_single_sample(y, dim_config)
		#print(y_norm)
		y_output_mask_sigmoid = dim_config.get("y_output_masks").get("sigmoid")
		y_dims_conti = dim_config.get("y_dims_conti")
		y_norm = torch.tensor(y_norm).float()
		if len(y_dims_conti) > 0:
			y_norm[:, y_output_mask_sigmoid] = dataset.normalize_data(y_norm[:, y_output_mask_sigmoid], config.dataset_stats[2][y_dims_conti], config.dataset_stats[3][y_dims_conti]).float()
		#print(y_norm)
		#print("")

		#exit()

		#print(y_x)
		y_x_norm = dataset.encode_dims_for_y_x_single_sample(y_x, dim_config)
		#print(y_x_norm)
		x_output_mask_sigmoid = dim_config.get("x_output_masks").get("sigmoid")
		x_dims_conti = dim_config.get("x_dims_conti")
		if len(x_dims_conti) > 0:
			y_x_norm[:, x_output_mask_sigmoid] = dataset.normalize_data(y_x_norm[:, x_output_mask_sigmoid], config.dataset_stats[0][x_dims_conti], config.dataset_stats[1][x_dims_conti]).float()
		#print(y_x_norm)
		#print("")
		#print("")
	else:
		y_norm = dataset.normalize_data(y, config.dataset_stats[2], config.dataset_stats[3]).float()
		y_norm_masked = torch.full(y_norm.shape, -1.0)
		y_norm_masked[y != -1.0] = y_norm[y != -1.0]
		y_norm = y_norm_masked

		y_x_norm = dataset.normalize_data(y_x, config.dataset_stats[0], config.dataset_stats[1])
		y_x_norm_masked = torch.full(y_x.shape, -1.0)
		y_x_norm_masked[y_x != -1.0] = y_x_norm[y_x != -1.0]
		y_x_norm = y_x_norm_masked

	# get output of model (already decoded)
	x_hat_dec, _, _ = analysis.gen_forward_pass(config.model, config.model_va, config.dataset_stats, z, y_norm, y_x_norm)

	return torch.tensor(x_hat_dec)


def gen_sample_and_evaluate_error(config, y, y_x):

	# generate sample
	x_hat = generate_single_sample(config, y.float(), y_x.float())

	if config.model_va.model_config.get("shift_all_dimensions"):
		y_x[0, y_x[0, :] != -1.0] -= 1.0

	# evaluate relative error if provided
	all_errors, y_true = compute_relative_error(config.gen_config.get("eval_fn"), y.float(), y_x.float(), x_hat.float())

	return x_hat, y_true, all_errors


def get_input_from_form(request_form, n_xs, n_ys):

	y = torch.full((1, n_ys), -1.0)
	req_y_strs = []
	err_y = np.zeros(n_ys)

	for i in range(n_ys):
		yi = request_form.get(f"y{i}")
		req_y_strs.append(yi)
		if yi is not None and yi != "":
			try:
				y[0, i] = float(yi)
			except ValueError:
				err_y[i] = 1

	y_x = torch.full((1, n_xs), -1.0)
	req_y_x_strs = []
	err_y_x = np.zeros(n_xs)

	for j in range(n_xs):
		xi = request_form.get(f"x{j}")
		req_y_x_strs.append(xi)
		if xi is not None and xi != "":
			try:
				y_x[0, j] = float(xi)
			except ValueError:
				err_y_x[i] = 1

	return y, y_x, req_y_strs, req_y_x_strs, err_y, err_y_x


def get_config_state(ds):
	config = va.all_configs[ds]
	n_xs = len(config.gen_config.get("dimensions").get("x"))
	n_ys = len(config.gen_config.get("dimensions").get("y"))

	return config, n_xs, n_ys


def get_template_pre(ds):
	config, n_xs, n_ys = get_config_state(ds)
	return render_template('./index.html', title=ds, gen_config=config.gen_config, request_form=None, n_xs=n_xs, n_ys=n_ys)


def get_template_post(ds):
	config, n_xs, n_ys = get_config_state(ds)
	timestamp_now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

	y, y_x, req_y_strs, req_y_x_strs, err_y, err_y_x = get_input_from_form(request.form, n_xs, n_ys)

	x_hat, y_true, all_errors = gen_sample_and_evaluate_error(config, y, y_x)

	if y_true is not None:
		y_true = torch.squeeze(y_true, dim=0)
		y_true = [ str(elem) for elem in list(y_true.cpu().detach().numpy()) ]

	x_hat = torch.squeeze(x_hat, dim=0)
	x_hat = [ str(elem) for elem in list(x_hat.cpu().detach().numpy()) ]

	return render_template('./index.html',
							title=ds,
							gen_config=config.gen_config,
							request_form=request.form,
							req_y_strs=req_y_strs,
							req_y_x_strs=req_y_x_strs,
							n_xs=n_xs,
							n_ys=n_ys,
							now=timestamp_now,
							y_true=y_true,
							x_hat=x_hat,
							err_y=err_y,
							err_y_x=err_y_x,
							all_errors=all_errors)

# ****************************************************************************************************** #

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True


@app.route("/walls")
def pre_walls():
	return get_template_pre("wall5d")

@app.route("/walls", methods=['POST'])
def post_walls():
	return get_template_post("wall5d")


# @app.route("/halls")
# def pre_halls():
# 	return get_template_pre("hall10d")

# @app.route("/halls", methods=['POST'])
# def post_halls():
# 	return get_template_post("hall10d")


@app.route("/rooms")
def pre_rooms():
	return get_template_pre("room5d")

@app.route("/rooms", methods=['POST'])
def post_rooms():
	return get_template_post("room5d")

# ****************************************************************************************************** #

# @app.before_first_request
# def initialize():
# 	load_gen_config()
# 	load_model()
