import src.models.cvae_01_nores as cvae_01_nores
import src.models.cvae_02_res as cvae_02_res
import src.models.cvae_03_yskip as cvae_03_yskip
import src.models.cvae_04_tril as cvae_04_tril
import src.models.cvae_05_nf as cvae_05_nf
import src.models.cvae_06_yskip_yxpart as cvae_06_yskip_yxpart
import src.models.cvae_07_tril_yxpart as cvae_07_tril_yxpart
import src.models.cvae_08_nf_yxpart as cvae_08_nf_yxpart
import src.models.cvae_09_yskip_gcnn as cvae_09_yskip_gcnn
import src.models.cvae_10_tril_gcnn as cvae_10_tril_gcnn
import src.models.cvae_11_nf_gcnn as cvae_11_nf_gcnn

import src.models.forward_01_mlp_noskip as forward_01_mlp_noskip
import src.models.forward_02_mlp_xskip as forward_02_mlp_xskip
import src.models.forward_03_gcnn as forward_03_gcnn
import src.models.forward_04_gcnn2 as forward_04_gcnn2


def init_model(va, model_config, model_type=None):

	if model_type is None:
		model_type = model_config.get("model_type")

	if model_type == "cvae_01_nores":
		return cvae_01_nores.CVAE1(model_config).to(va.dvc)

	if model_type == "cvae_02_res":
		return cvae_02_res.CVAE2(model_config).to(va.dvc)

	if model_type == "cvae_03_yskip":
		return cvae_03_yskip.CVAE3(model_config).to(va.dvc)

	if model_type == "cvae_04_tril":
		return cvae_04_tril.CVAE4(model_config).to(va.dvc)

	if model_type == "cvae_05_nf":
		return cvae_05_nf.CVAE5(model_config, va).to(va.dvc)


	# yxpart

	if model_type == "cvae_06_yskip_yxpart":
		return cvae_06_yskip_yxpart.CVAE6(model_config).to(va.dvc)

	if model_type == "cvae_07_tril_yxpart":
		return cvae_07_tril_yxpart.CVAE7(model_config).to(va.dvc)

	if model_type == "cvae_08_nf_yxpart":
		return cvae_08_nf_yxpart.CVAE8(model_config, va).to(va.dvc)


	# gcnn

	if model_type == "cvae_09_yskip_gcnn":
		return cvae_09_yskip_gcnn.CVAE9(model_config).to(va.dvc)

	if model_type == "cvae_10_tril_gcnn":
		return cvae_10_tril_gcnn.CVAE10(model_config).to(va.dvc)

	if model_type == "cvae_11_nf_gcnn":
		return cvae_11_nf_gcnn.CVAE11(model_config, va).to(va.dvc)


	# forward / mlp / gcnn

	if model_type == "forward_01_mlp_noskip":
		return forward_01_mlp_noskip.MLP(model_config).to(va.dvc)

	if model_type == "forward_02_mlp_xskip":
		return forward_02_mlp_xskip.MLP2(model_config).to(va.dvc)

	if model_type == "forward_03_gcnn":
		return forward_03_gcnn.GCNN(model_config, va).to(va.dvc)

	if model_type == "forward_04_gcnn2":
		return forward_04_gcnn2.GCNN2(model_config, va).to(va.dvc)


	# default
	return cvae2.CVAE2(x_size=10, z_size=4, y_size=3, h_size=50, n_hidden_layers=4).to(va.dvc)
