import numpy as np
from xgboost import XGBRegressor
import torch
import argparse
from datetime import datetime
import requests
import logging
import os
from matplotlib import pyplot as plt

import sys
sys.path.append(os.getcwd())

import src.dataset as dataset
import src.models as models
import src.learning as learning
import src.learning_gnn as learning_gnn
import src.plot as plot
import src.optimization as optimization
import src.utility as utility
import src.loss as loss
import src.analysis as analysis
import src.baselines_xy as baselines_xy


def main(args):

	# *************************************************************************************************
	# prep

	torch.set_default_dtype(torch.float64)
	torch.autograd.set_detect_anomaly(True)

	run_id, ckpt_path = utility.do_initial_checks(args)
	dvc = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	logger, logger_fn = utility.get_logger(run_id)

	va = utility.VanillaArgs()
	va.cli_args = args
	va.dvc = dvc
	va.logger = logger

	if args.is_logging_enabled:
		os.makedirs("log", exist_ok=True)
		print(f"logging at: {logger_fn}")

	utility.logtext(f"device (cpu/gpu): {dvc}", va)
	utility.logtext(f"dataset: {args.dataset}", va)

	# reproducibility
	utility.set_seeds(args.seed)

	# *************************************************************************************************

	if args.case_study == "bridge-y1d-no-nf":
		model_params = {
			"batch_size": 972,
			"h_dim": 209,
			"n_h": 38,
			"z_dim": 33,
			"beta_kl": 1.0,
		}

	elif args.case_study == "bridge-y1d-nf":
		model_params = {
			"batch_size": 2602,
			"h_dim": 151,
			"n_h": 80,
			"z_dim": 206,
			"beta_kl": 1.0,
			"h_context_dim": 16,
			"k_flows": args.k_flows,
		}

	elif args.case_study == "bridge-y2d-no-nf":
		model_params = {
			"batch_size": 1446,
			"h_dim": 60,
			"n_h": 65,
			"z_dim": 107,
			"beta_kl": 1.0,
		}

	elif args.case_study == "bridge-y2d-nf":
		model_params = {
			"batch_size": 3079,
			"h_dim": 223,
			"n_h": 31,
			"z_dim": 132,
			"beta_kl": 1.0,
			"h_context_dim": 16,
			"k_flows": args.k_flows,
		}

	elif args.case_study == "wall-y1d-no-nf":
		model_params = {
			"batch_size": 2288,
			"h_dim": 187,
			"n_h": 32,
			"z_dim": 147,
			"beta_kl": 1.0,
		}

	elif args.case_study == "wall-y1d-nf":
		model_params = {
			"batch_size": 1223,
			"h_dim": 168,
			"n_h": 77,
			"z_dim": 132,
			"beta_kl": 1.0,
			"h_context_dim": 16,
			"k_flows": args.k_flows,
		}

	elif args.case_study == "bridge-gnn":
		model_params = {
			"batch_size": 64,
			"gnn_h_dim": 96,
			"gnn_n_h": 3,
			"gnn_n_h_mlp": 16,
			"gnn_res_alpha": args.gnnres_alpha,
			"concat_x_with_z_g_for_mlp_head": False
		}

	else:
		model_params = {
			"batch_size": 64,
			"gnn_h0_dim": 128,
			"gnn_n_h": 10,
		}

	# *************************************************************************************************

	dataset_train, dataset_test, data_loader_train, data_loader_test, x_dim, y_dim, _ = dataset.get_datasets_and_dataloader(va, args.model, model_params.get("batch_size"), args.shift_all_dimensions, va.cli_args.y_augm)

	# *************************************************************************************************

	if args.datamode == 'train' and (not args.resume or args.datamode == "gnn-baselines"):
		model_config = {
			"dataset_trained_on": args.dataset,
			"seed_trained_on": args.seed,
			"shift_all_dimensions": args.shift_all_dimensions,
			"model_type": args.model,
			"case_study": args.case_study,
			"model_params": model_params,

			"x_dim": x_dim,
			"y_dim": y_dim,

			"weighted_loss" : args.is_weighted_loss,
			"lambda_lossweight": args.lw_lambda,

			"z_g_dim": 96,
			"z_g_lowdim": 8,

			# "beta_schedule": {
			# 	"n_cold_start": 0,
			# 	"n_cycles": 1,
			# 	"ratio": 0.15
			# },
			"is_beta_schedule": args.is_beta_schedule,

			"kld_min_free_bits": 0.05,
			"y_augm": args.y_augm,
			"lambda_y_x_partial": 1.0
		}
		utility.write_model_config_to_file(model_config, ckpt_path, va)
	else:
		model_config = utility.read_model_config_from_file(ckpt_path, va)

	va.model_config = model_config

	# *************************************************************************************************

	va.wand_exp = None
	if args.is_logging_enabled:
		import wandb
		wand_exp = wandb.init(project="master-thesis", entity="mnbucher", config=model_config, mode=("disabled" if not args.is_logging_enabled else "online"))
		wand_exp.name = run_id
		va.wand_exp = wand_exp

	# *************************************************************************************************

	model = models.init_model(va, model_config)
	lr = args.lr
	loss_func = loss.get_loss_function(va)
	optimizer = optimization.get_optimizer(model, lr)

	# *************************************************************************************************

	start_epoch = 1
	if args.resume:
		lr, start_epoch = utility.resume_model(va, args.resume, model, optimizer)

	# *************************************************************************************************

	va.dataset_train = dataset_train
	va.dataset_test = dataset_test
	va.data_loader_train = data_loader_train
	va.data_loader_test = data_loader_test
	va.loss_func = loss_func
	va.lr = lr
	va.ckpt_path = ckpt_path
	va.run_id = run_id
	va.start_epoch = start_epoch

	# prior mean for bay opt to circumvent nan issue for first model during BO...
	va.bayopt_running_stats = [ 50.0 ]

	if args.is_beta_schedule:
		va.schedule_beta_kld = learning.get_schedule_beta_kld_cyclic_linear(n_cold_start=model_config.get("beta_schedule").get("n_cold_start"), n_epoch=args.epochs, max_beta=model_config.get("model_params").get("beta_kl"), n_cycle=model_config.get("beta_schedule").get("n_cycles"), ratio=model_config.get("beta_schedule").get("ratio"))

	va.gen_n_y = 160
	va.gen_n_z = 4

	# va.gen_n_y = 32
	# va.gen_n_z = 1

	# *************************************************************************************************

	if "gcnn" in model_config.get("model_type") and args.pretrained_gnn:
		gnn_config = utility.read_model_config_from_file(f"ckpt/{args.pretrained_gnn.split('/')[1]}", va)
		gnn_config['model_params']['gnn_res_alpha'] = 0.1
		gnn_model = models.init_model(va, gnn_config, "forward_04_gcnn2")
		gnn_model.eval()
		utility.resume_model(va, args.pretrained_gnn, gnn_model, None)
		va.gnn_model = gnn_model

	# *************************************************************************************************

	if args.datamode == 'train':

		if args.is_vanilla_training:

			if "forward" in args.model:
				optimization.full_training_vanilla_gnn(model, optimizer, va)
			else:
				gen_loss, _, _ = optimization.full_training_vanilla(model, optimizer, va, itr=None)

				utility.logtext(f"test loss for model after training: {gen_loss}", va)
				if args.is_logging_enabled:
					va.wand_exp.log({"test loss for model after training": gen_loss})

		else:
			utility.logtext("start bayesian optimization for hyperparameter tuning...", va)

			if args.bayopt == "cvae":
				hyperparams_bounds = {
					"z_dim" : (8, 256),
					"h_dim" : (30, 250),
					"n_h" : (4, 80),
					"batch_size": (320, 3200)
				}
			elif args.bayopt == "gnn":
				hyperparams_bounds = {
					"gnn_h_dim": (3,256),
					"gnn_n_h": (1,8),
					"gnn_n_h_mlp": (2, 30),
					"batch_size" : (16, 3200),
				}
			else:
				hyperparams_bounds = {
					"gnn_h0_dim_x32": (3, 16),
					"gnn_n_h": (5, 50),
					"batch_size": (16, 1800)
				}

			optimization.full_training_with_bay_opt_wrapper(va, hyperparams_bounds, args.bayopt)

	elif args.datamode == "gnn-baselines":

		baselines_xy.run_gnn_baselines_01_rfr(va)
		baselines_xy.run_gnn_baselines_02_gbt(va)

		# xgb_hyperparams_bounds = {
		# 	"n_estimators": (50, 500),
		# 	"max_depth": (1, 50),
		# 	"eta": (0.0, 1.0),
		# 	"subsample": (0.3, 1.0)
		# }
		# baselines_xy.run_gnn_baselines_bayopt(va, xgb_hyperparams_bounds, "xgboost")

		# rfr_hyperparams_bounds = {
		# 	"n_estimators": (10, 1000),
		# 	"max_depth": (20, 100),
		# 	"min_samples_leaf": (1, 5),
		# }
		# baselines_xy.run_gnn_baselines_bayopt(va, rfr_hyperparams_bounds, "rfr")

	elif args.datamode == 'test':
		learning.test(0, model, va)

	elif args.datamode == "eval-y":
		learning_gnn.gnn_test(0, model, va)

	elif args.datamode == "eval-y-robust":
		analysis.gnn_robustness_test(run_id, va, model_config, optimizer)

	elif args.datamode == "eval-y-repro-thesis":
		analysis.eval_y_repro_thesis(run_id, va, model_config, optimizer)

	elif args.datamode == "gen":
		analysis.gen(va, model, va.cli_args.seed)

	elif args.datamode == "gen-eval":

		if "10d-bridge" in args.dataset:
			analysis.gen_and_eval_bridge_y_via_rhino_remote(model, va, va.cli_args.seed)

		elif "5d-wall" in args.dataset:
			analysis.gen_and_eval_wall_y_via_excel(model, va, va.cli_args.seed)

		else:
			analysis.gen(va, model, va.cli_args.seed)

	elif args.datamode == "gen-eval-repro-thesis":
		analysis.gen_eval_repro_thesis(run_id, va, model_config, optimizer)

	elif args.datamode == "gen-eval-walls-comparison":
		analysis.gen_eval_walls_comparison(run_id, va, model_config, optimizer)

	else:
		pass

	if args.is_logging_enabled:
		wandb.finish()

	end_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
	utility.logtext(f"end running code at: {end_time}", va)


if __name__ == '__main__' :

	parser = argparse.ArgumentParser(description='Author: Martin Juan José Bucher. Master Thesis in Computer Science @ ETHZ')

	parser.add_argument('--is-logging-enabled', dest='is_logging_enabled', action='store_true', default=False, help='Enable logging via cloud provider')

	parser.add_argument('--is-first-batch-only', dest='is_first_batch_only', action='store_true', default=False, help='Run first batch only')

	parser.add_argument('--is-vanilla-training', dest='is_vanilla_training', action='store_true', default=False, help='Choose between vanilla training or bayesian optimization')

	parser.add_argument('--is-gen-eval-enabled', dest='is_gen_eval_enabled', action='store_true', default=False, help="Evaluate generative performance of model")

	parser.add_argument('--seed', dest='seed', type=int, default=1234, help="Seed for reproducibility of trained model")

	parser.add_argument('--datamode', dest='datamode', type=str, choices=['train', 'test', 'eval-y', 'eval-y-repro-thesis', 'eval-y-robust', 'gnn-baselines', 'gen', 'gen-eval', 'gen-eval-repro-thesis', 'gen-eval-walls-comparison', 'gen-single'], default='train', help='Choose datamode for main.py')

	parser.add_argument('--case-study', dest='case_study', type=str, choices=["bridge-mlp", "bridge-gnn", "bridge-y1d-nf", "bridge-y1d-no-nf", "bridge-y2d-nf", "bridge-y2d-no-nf", "wall-y1d-nf", "wall-y1d-no-nf"] )

	parser.add_argument('--genmode', dest='genmode', type=str, choices=['proportional', 'uniform', 'ga-comp'], default='proportional', help='Choose genmode for the generation of new samples')

	parser.add_argument('--is-output-files-only', dest='is_output_files_only', action='store_true', default=False, help='read output.csv directly instead of generating new samples (read cached output)')

	parser.add_argument('--model', dest='model', type=str, default='cvae4')

	parser.add_argument('--pretrained-gnn', dest='pretrained_gnn', type=str, help='Ckpt path for pretrained GNN model')

	parser.add_argument('--resume', type=str, help='Resume from pretrained model. Specify path to model relative to /ckpt directory.')

	parser.add_argument('--batch-size', dest='batch_size', type=int, default=32, help='Batch Size for NN model')

	parser.add_argument('--epochs', dest='epochs', type=int, default=700, help='# of Epochs')

	parser.add_argument('--lr', dest='lr', type=float, default=1e-3, help='Learning Rate')

	parser.add_argument('--lr-schedule', dest='lr_schedule', type=int, nargs='+', default=[], help='Decrease learning rate at these epochs.')

	parser.add_argument('--lr-gamma', dest='lr_gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')

	parser.add_argument('--is-weighted-loss', dest='is_weighted_loss', action='store_true', default=False, help='Use weighted loss')

	parser.add_argument('--is-beta-schedule', dest='is_beta_schedule', action='store_true', default=False, help="Use beta-schedule for regularization of KL divergence")

	parser.add_argument('--bayopt', dest='bayopt', choices=['gnn', 'gnn0-baseline', 'cvae'])

	parser.add_argument('--is-epochwise-check', dest='is_epochwise_check', action='store_true', default=False, help="Check gen eval each 100 epochs")

	parser.add_argument('--lw-lambda', dest='lw_lambda', type=float, default=1.0, help='Hyperparameter for loss weight importance in the range [0,1]')

	parser.add_argument('--y-augm', dest='y_augm', action="store_true", default=False, help="Return '-1' for y data with 50% probability for data augmentation")

	parser.add_argument('--shift-all-dimensions', dest='shift_all_dimensions', action="store_true", default=False, help="Shift all dimensions in datatest (both x and y) by +1 to circumvent zeros for MAPE (%) computation")

	parser.add_argument('--gnnres-alpha', dest='gnnres_alpha', type=float, default=0.1, help="hyperparam for GNN-RES model with GCN2 layer from PyG")

	parser.add_argument('--is-only-yx-for-gen', dest='is_only_yx_for_gen', action="store_true", default=False, help="Set -1 on entire y vector (only test for y_x_partial reconstruction")

	parser.add_argument('--y-cols', dest='y_cols', type=int, nargs='+', default=[], help='Restrict output dim (y)')

	parser.add_argument('--k-flows', dest='k_flows', type=int, default=6, help="# of flows for IAF posterior")

	parser.add_argument('--dataset', dest='dataset', type=str, choices=["10d-robust", "10d-simple", "10d-valid", "5d-wall", "5d-wall-v2", "dynamic"], default="10d", help='Choose dataset')

	parser.add_argument('--dataset-path-csv', dest='dataset_path_csv', type=str, help="Dataset path for CSV for dynamic dataloader. Assumed to have a x_i and y_i header column indicating the shape for x and y")

	parser.add_argument('--run-id', dest='run_id', type=str,
						help='Unique identifier for /ckpt, /log, and other important stuff related to current model. Should be unique! Otherwise previous entries will be overwritten')

	parser.add_argument('--overwrite', dest='overwrite', action='store_true', default=False, help='Overwrite existing model with same run_id')

	parser.add_argument('--use-proxy', dest='use_proxy', action='store_true', default=False, help='Use internal Proxy on Euler (ETHZ only)')

	main(parser.parse_args())
