import numpy as np
from torch import optim
from bayes_opt import BayesianOptimization
import sys
import torch
import logging
import time

import src.models as models
import src.utility as utility
import src.learning as learning
import src.learning_gnn as learning_gnn
import src.analysis as analysis


def full_training_vanilla_gnn(model, optimizer, va, itr=None):
	utility.logtext("", va)
	utility.logtext("[gnn] start vanilla training..." if itr is None else f"start vanilla training (param iter {itr})...", va)
	utility.logtext(f"[gnn] model_config: {utility.get_relevant_model_configs(va.model_config)}", va)

	best_loss_test = np.inf
	best_loss_mape = np.inf

	utility.set_seeds(va.cli_args.seed)

	for epoch in range(va.start_epoch, va.cli_args.epochs + 1):

		if va.cli_args.is_vanilla_training:
			utility.logtext(f"[gnn] epoch {epoch}", va)
		else:
			if epoch % 100 == 0:
				utility.logtext(f"[gnn] epoch {epoch}", va)

		va.lr = learning_gnn.gnn_train(epoch, model, optimizer, va)
		curr_loss_test, mape_avg, _, _, _ = learning_gnn.gnn_test(epoch, model, va)

		is_best = curr_loss_test < best_loss_test

		if is_best:
			best_loss_test = curr_loss_test
			best_loss_mape = mape_avg
			if va.cli_args.is_vanilla_training:
				utility.logtext(">> NEW BEST MODEL FOUND", va)

		utility.save_ckpt({'epoch': epoch,
						   'lr': va.lr,
						   'err': best_loss_test,
						   'state_dict': model.state_dict(),
						   'optimizer': optimizer.state_dict()
						   }, ckpt_path=va.ckpt_path, seed=va.cli_args.seed, is_best=is_best)

	return best_loss_test, best_loss_mape


def full_training_vanilla(model, optimizer, va, itr=None):
	utility.logtext("", va)
	utility.logtext("start vanilla training..." if itr is None else f"start vanilla training (param iter {itr})...", va)
	utility.logtext(f"model_config: {utility.get_relevant_model_configs(va.model_config)}", va)

	best_loss_test = np.inf
	best_loss_recon = np.inf
	best_loss_kld = np.inf

	utility.set_seeds(va.cli_args.seed)

	for epoch in range(va.start_epoch, va.cli_args.epochs + 1):

		if va.cli_args.is_vanilla_training:
			utility.logtext(f"epoch {epoch}", va)

		va.lr = learning.train(epoch, model, optimizer, va)

		curr_loss_test, curr_loss_recon, curr_loss_kld = learning.test(epoch, model, va)

		is_best = curr_loss_test < best_loss_test
		if is_best:
			best_loss_test = curr_loss_test
			best_loss_recon = curr_loss_recon
			best_loss_kld = curr_loss_kld

			if va.cli_args.is_vanilla_training:
				utility.logtext(">> NEW BEST MODEL FOUND", va)

		utility.save_ckpt({'epoch': epoch,
						   'lr': va.lr,
						   'err': best_loss_test,
						   'state_dict': model.state_dict(),
						   'optimizer': optimizer.state_dict()
						   }, ckpt_path=va.ckpt_path, seed=va.cli_args.seed, is_best=is_best)

		THRESHOLD_EPOCHWISE_CHECK = 400
		#THRESHOLD_EPOCHWISE_CHECK = 0

		#if epoch > va.start_epoch and epoch >= THRESHOLD_EPOCHWISE_CHECK and epoch % 100 == 0:
		#if epoch > va.start_epoch and epoch >= THRESHOLD_EPOCHWISE_CHECK and epoch % 50 == 0:
		if epoch > va.start_epoch and epoch >= THRESHOLD_EPOCHWISE_CHECK and epoch % 500 == 0:
			utility.logtext(f"epoch {epoch} done", va)
			if va.cli_args.is_epochwise_check:
				analysis.check_performance_of_generative_model(model, optimizer, va, epoch, curr_loss_test, curr_loss_recon, curr_loss_kld)

	if va.cli_args.is_gen_eval_enabled:
		gen_loss = analysis.check_performance_of_generative_model(model, optimizer, va, itr, curr_loss_test, curr_loss_recon, curr_loss_kld)
		return gen_loss, best_loss_recon, best_loss_kld
	else:
		utility.logtext("", va)
		utility.logtext(f"not checking generative phase. taking best_loss_test: {best_loss_test}", va)
		utility.logtext("", va)
		return best_loss_test, best_loss_recon, best_loss_kld


def full_training_with_bay_opt_wrapper(va, hyperparams_bounds, bayopt_mode):
	itr = 1
	default_lr = va.lr

	def prep_bayopt_training_iter(new_model_config):
		nonlocal va
		nonlocal default_lr

		_, collate_fn = utility.get_collate_fn(va.model_config.get("model_type"))

		va.model_config = new_model_config
		va.lr = default_lr

		utility.set_seeds(va.cli_args.seed)

		model = models.init_model(va, new_model_config, va.cli_args.is_bce_loss)
		optimizer = get_optimizer(model, lr=va.lr)

		va.data_loader_train = torch.utils.data.DataLoader(va.dataset_train, batch_size=va.model_config.get("model_params").get("batch_size"), shuffle=False, worker_init_fn=utility.set_seeds(va.cli_args.seed), generator=utility.get_tgseed(va.cli_args.seed), collate_fn=collate_fn)
		va.data_loader_test = torch.utils.data.DataLoader(va.dataset_test, batch_size=va.model_config.get("model_params").get("batch_size"), shuffle=False, worker_init_fn=utility.set_seeds(va.cli_args.seed), generator=utility.get_tgseed(va.cli_args.seed), collate_fn=collate_fn)

		return model, optimizer


	def full_training_with_bay_opt_cvae(z_dim, h_dim, n_h, batch_size):
		nonlocal va
		nonlocal itr

		new_model_config = va.model_config.copy()
		model_params = new_model_config["model_params"]

		model_params["z_dim"] = round(z_dim)
		model_params["h_dim"] = round(h_dim)
		model_params["n_h"] = round(n_h)
		model_params["batch_size"] = round(batch_size)

		new_model_config["model_params"] = model_params
		#new_model_config["beta_kl"] = round(beta_kl)

		model, optimizer = prep_bayopt_training_iter(new_model_config)
		loss, loss_recon, loss_kld = full_training_vanilla(model, optimizer, va, itr)

		utility.logtext(f"test loss for model during bo: {loss}", va)
		va.wand_exp.log({"test loss for model during bo": loss, "epoch": itr})

		utility.logtext(f"test_recon loss for model during bo: {loss_recon}", va)
		va.wand_exp.log({"test_recon loss for model during bo": loss_recon, "epoch": itr})

		utility.logtext(f"test_kld loss for model during bo: {loss_kld}", va)
		va.wand_exp.log({"test_kld loss for model during bo": loss_kld, "epoch": itr})

		itr += 1
		return -loss


	def full_training_with_bay_opt_gnn(gnn_h_dim, gnn_n_h, gnn_n_h_mlp, batch_size):
		nonlocal va
		nonlocal itr

		new_model_config = va.model_config.copy()
		model_params = new_model_config["model_params"]

		model_params["gnn_h_dim"] = round(gnn_h_dim)
		model_params["gnn_n_h"] = round(gnn_n_h)
		model_params["gnn_n_h_mlp"] = round(gnn_n_h_mlp)
		model_params["batch_size"] = round(batch_size)

		new_model_config["model_params"] = model_params

		model, optimizer = prep_bayopt_training_iter(new_model_config)
		loss, best_loss_mape = full_training_vanilla_gnn(model, optimizer, va, itr)

		utility.logtext(f"[gnn] bayopt test loss for model: {loss}", va)
		va.wand_exp.log({"[gnn] bayopt test loss for model": loss, "epoch": itr})

		utility.logtext(f"[gnn] bayopt MAPE test loss for model: {best_loss_mape}", va)
		va.wand_exp.log({"[gnn] bayopt MAPE test loss for model": best_loss_mape, "epoch": itr})

		itr += 1
		return -loss

	def full_training_with_bay_opt_gnn0_baseline(gnn_h0_dim_x32, gnn_n_h, batch_size):
		nonlocal va
		nonlocal itr

		new_model_config = va.model_config.copy()

		model_params_subset = new_model_config["model_params_gnn0_baseline"]
		model_params_subset["gnn_h0_dim"] = round(gnn_h0_dim_x32) * 32
		model_params_subset["gnn_n_h"] = round(gnn_n_h)
		model_params_subset["batch_size"] = batch_size

		new_model_config["model_params_gnn0_baseline"] = model_params_subset

		model, optimizer = prep_bayopt_training_iter(new_model_config)
		loss, best_loss_mape = full_training_vanilla_gnn(model, optimizer, va, itr)

		utility.logtext(f"[gnn] bayopt test loss for model: {loss}", va)
		va.wand_exp.log({"[gnn] bayopt test loss for model": loss, "epoch": itr})

		utility.logtext(f"[gnn] bayopt MAPE test loss for model: {best_loss_mape}", va)
		va.wand_exp.log({"[gnn] bayopt MAPE test loss for model": best_loss_mape, "epoch": itr})

		itr += 1
		return -loss


	seed = np.random.RandomState(va.cli_args.seed)

	if bayopt_mode == "gnn":
		optimizer = BayesianOptimization(f=full_training_with_bay_opt_gnn, pbounds=hyperparams_bounds, random_state=seed, verbose=2)
		optimizer.maximize(n_iter=30, init_points=10, acq='ei')

	elif bayopt_mode == "gnn0-baseline":
		optimizer = BayesianOptimization(f=full_training_with_bay_opt_gnn0_baseline, pbounds=hyperparams_bounds, random_state=seed, verbose=2)
		optimizer.maximize(n_iter=200, init_points=40, acq='ei')

	else:
		optimizer = BayesianOptimization(f=full_training_with_bay_opt_cvae, pbounds=hyperparams_bounds, random_state=seed, verbose=2)

		#optimizer.maximize(n_iter=120, init_points=40, acq='ei')
		#optimizer.maximize(n_iter=80, init_points=20, acq='ei')
		#optimizer.maximize(n_iter=60, init_points=15, acq='ei')

		optimizer.maximize(n_iter=40, init_points=10, acq='ei')
		#optimizer.maximize(n_iter=8, init_points=8, acq='ei')

	utility.logtext("best performance during bayesian optimization:", va)
	utility.logtext(optimizer.max, va)


def get_optimizer(model, lr):

	#return optim.AdamW(model.parameters(), lr=lr)

	#return optim.SGD(model.parameters(), lr=lr, momentum=0.7)

	#return optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
	#return optim.RMSprop(model.parameters(), lr=lr)

	return optim.Adam(model.parameters(), lr=lr)
