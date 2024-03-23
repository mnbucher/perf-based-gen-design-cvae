import numpy as np
import torch
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from tqdm import tqdm
from bayes_opt import BayesianOptimization
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

import src.learning_gnn as learning_gnn
import src.utility as utility
import src.dataset as dataset
import src.analysis as analysis


class BaselineArgs():
	dataset = None
	stats = None
	dim = None
	model_type = None


def baselines_compute_metrics(y_pred, y_true, stats, dim):
	# assumes that model is trained on single y dimension

	y_pred = dataset.unnormalize_data(y_pred, stats[2].numpy()[dim], stats[3].numpy()[dim])
	y_true = dataset.unnormalize_data(y_true, stats[2].numpy()[dim], stats[3].numpy()[dim])

	mape, rmse, pcc = analysis.compute_error_metrics_y(y_pred, y_true)

	print("mape", np.mean(mape, axis=0))
	print("rmse", rmse)
	print("pcc", pcc, np.mean(pcc))


def run_baseline_fit_model_and_predict(model, ds, stats, dim):

	print("fit model...")
	model = model.fit(ds[0], ds[1][:, dim])

	print("predict...")
	y_pred = model.predict(ds[2])

	baselines_compute_metrics(y_pred, ds[3][:, dim], stats, dim)


def run_gnn_baselines_01_rfr(va):
	print("run baseline_01 with RFR...")

	ds, stats = dataset.get_dataset_as_nparray(va)

	# dim y0
	if va.cli_args.dataset == "10d-robust":
		model = RandomForestRegressor(n_estimators=969, max_depth=73, min_samples_leaf=2)
	elif va.cli_args.dataset == "10d-simple":
		model = RandomForestRegressor(n_estimators=848, max_depth=55, min_samples_leaf=2)
	run_baseline_fit_model_and_predict(model, ds, stats, dim=0)

	# dim y1
	if va.cli_args.dataset == "10d-robust":
		model = RandomForestRegressor(n_estimators=892, max_depth=45, min_samples_leaf=2)
	elif va.cli_args.dataset == "10d-simple":
		model = RandomForestRegressor(n_estimators=998 , max_depth=73, min_samples_leaf=2)
	run_baseline_fit_model_and_predict(model, ds, stats, dim=1)


def run_gnn_baselines_02_gbt(va):
	print("run baseline_02 with GBT...")

	ds, stats = dataset.get_dataset_as_nparray(va)

	# dim y0
	if va.cli_args.dataset == "10d-robust":
		model = XGBRegressor(n_estimators=357, max_depth=9, eta=0.0590899223529231, subsample=0.5877054391972742)
	elif va.cli_args.dataset == "10d-simple":
		model = XGBRegressor(n_estimators=488, max_depth=5, eta=0.07244489131931174, subsample=0.5099878263864388)
	run_baseline_fit_model_and_predict(model, ds, stats, dim=0)

	# dim y1
	if va.cli_args.dataset == "10d-robust":
		model = XGBRegressor(n_estimators=446, max_depth=9, eta=0.03818825108970769, subsample=0.682664699254909)
	elif va.cli_args.dataset == "10d-simple":
		model = XGBRegressor(n_estimators=397, max_depth=8, eta=0.07390670398201771, subsample=0.5454734952145629)
	run_baseline_fit_model_and_predict(model, ds, stats, dim=1)


def run_gnn_baselines_bayopt(va, hyperparams_bounds, model_type):

	def prep_data():
		nonlocal va
		nonlocal ba
		ba.dataset, ba.stats = dataset.get_dataset_as_nparray(va)

	def prep_model(curr_hyperparams):
		nonlocal va
		nonlocal ba
		if ba.model_type == "xgboost":
			#model = XGBRegressor()
			model = XGBRegressor(n_estimators=round(curr_hyperparams.get("n_estimators")), max_depth=round(curr_hyperparams.get("max_depth")), eta=curr_hyperparams.get("eta"), subsample=curr_hyperparams.get("subsample"))
		if ba.model_type == "rfr":
			#model = RandomForestRegressor()
			model = RandomForestRegressor(n_estimators=round(curr_hyperparams.get("n_estimators")), max_depth=round(curr_hyperparams.get("max_depth")), min_samples_leaf=round(curr_hyperparams.get("min_samples_leaf")))
		return model

	def run_baseline(curr_hyperparams):
		nonlocal va
		nonlocal ba
		nonlocal itr

		utility.logtext(f"[baseline {ba.model_type}] [y_{ba.dim}] start training for iter {itr}", va)
		utility.logtext(f"[baseline {ba.model_type}] [y_{ba.dim}] params: {curr_hyperparams}", va)

		x_train = ba.dataset[0]
		y_train = ba.dataset[1][:, ba.dim]

		kf = KFold(n_splits=5, random_state=None, shuffle=False)

		rmses = []
		mapes = []

		iterator = kf.split(x_train) if va.cli_args.is_logging_enabled else tqdm(kf.split(x_train))
		for train_idx, test_idx in iterator:
			kfold_x_train, kfold_y_train, kfold_x_test, kfold_y_test = x_train[train_idx, :], y_train[train_idx], x_train[test_idx, :], y_train[test_idx]

			model = prep_model(curr_hyperparams)
			model = model.fit(kfold_x_train, kfold_y_train)

			y_pred = baseline_predict_from_model(model, kfold_x_test)
			rmse, mape = baselines_compute_metrics(y_pred, kfold_y_test, ba.stats, ba.dim)

			rmses.append(rmse)
			mapes.append(mape)

		total_rmse = np.mean(rmses)
		total_mape = np.mean(mapes)

		utility.logtext(f"[baseline {ba.model_type}] [y_{ba.dim}] bayopt test loss for model: {total_rmse}", va)
		utility.logtext(f"[baseline {ba.model_type}] [y_{ba.dim}] bayopt test MAPE for model: {total_mape}", va)

		va.wand_exp.log({f"[baseline {ba.model_type}] [y_{ba.dim}] bayopt test loss for model": total_rmse, "epoch": itr})

		itr += 1
		return -total_rmse

	def run_baseline_1_bayopt_xgb(n_estimators, max_depth, eta, subsample):
		curr_hyperparams = {
			"n_estimators": n_estimators,
			"max_depth": max_depth,
			"eta": eta,
			"subsample": subsample
		}
		return run_baseline(curr_hyperparams)

	def run_baseline_2_bayopt_rfr(n_estimators, max_depth, min_samples_leaf):
		curr_hyperparams = {
			"n_estimators": n_estimators,
			"max_depth": max_depth,
			"min_samples_leaf": min_samples_leaf
		}
		return run_baseline(curr_hyperparams)

	def run_best_model(best_params):
		utility.logtext("", va)
		utility.logtext(f"[baseline {ba.model_type}] [y_{ba.dim}] training model on best hyperparams on full test set...", va)
		utility.logtext("", va)

		x_train = ba.dataset[0]
		y_train = ba.dataset[1][:, ba.dim]
		x_test = ba.dataset[2]
		y_test = ba.dataset[3][:, ba.dim]

		model = prep_model(best_params)
		model = model.fit(x_train, y_train)

		y_pred = baseline_predict_from_model(model, x_test)
		rmse, mape = baselines_compute_metrics(y_pred, y_test, ba.stats, ba.dim)

		utility.logtext(f"[baseline {ba.model_type}] [y_{ba.dim}] best model RMSE: {rmse}", va)
		utility.logtext(f"[baseline {ba.model_type}] [y_{ba.dim}] best model MAPE: {mape}", va)

	def start_bay_opt(hyperparams_bounds):
		nonlocal va
		nonlocal ba

		utility.logtext(f"starting bay opt for dim {ba.dim}", va)

		optimizer = BayesianOptimization(f=run_baseline_1_bayopt_xgb if ba.model_type == "xgboost" else run_baseline_2_bayopt_rfr, pbounds=hyperparams_bounds, verbose=2)

		optimizer.maximize(n_iter=120, init_points=30, acq='ei')
		#optimizer.maximize(n_iter=1, init_points=0, acq='ei')

		utility.logtext(f"[baseline {ba.model_type}] [y_{ba.dim}] best parameters during bay opt:", va)
		utility.logtext(optimizer.max, va)

		run_best_model(optimizer.max.get("params"))

	# prep
	ba = BaselineArgs()
	ba.model_type = model_type
	prep_data()

	# do bay opt for each dimension independently
	itr = 1
	ba.dim = 0
	start_bay_opt(hyperparams_bounds)

	itr = 1
	ba.dim = 1
	start_bay_opt(hyperparams_bounds)
