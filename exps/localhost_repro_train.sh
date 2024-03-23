cd ..

python -m venv .venv
source .venv/bin/activate

export PYTHONPATH="./:$PYTHONPATH"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_LAUNCH_BLOCKING=1

# step 1
pip install wheel
pip install -r requirements.txt

# step 2
# with GPU
#pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu102.html
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
# without GPU
# pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cpu.html

# step 3
pip install --upgrade git+https://github.com/VincentStimper/normalizing-flows.git

mkdir -p log
mkdir -p log/train


# ************************************************************************************************************************************************************
# FORWARD MAPPING

# MLP-NOSKIP
# python3 src/main.py --run-id='22-04-18-mlp-10dsimple' --is-vanilla-training --y-cols 0 1 --dataset='10d-simple' --overwrite --epochs=1000 --model="forward_01_mlp_noskip" --case-study="bridge-mlp" --is-logging-enabled --seed=1234
# python3 src/main.py --run-id='22-04-18-mlp-10dsimple' --is-vanilla-training --y-cols 0 1 --dataset='10d-simple' --overwrite --epochs=1000 --model="forward_01_mlp_noskip" --case-study="bridge-mlp" --is-logging-enabled --seed=5678
# python3 src/main.py --run-id='22-04-18-mlp-10dsimple' --is-vanilla-training --y-cols 0 1 --dataset='10d-simple' --overwrite --epochs=1000 --model="forward_01_mlp_noskip" --case-study="bridge-mlp" --is-logging-enabled --seed=9876
# python3 src/main.py --run-id='22-04-18-mlp-10drobust' --is-vanilla-training --y-cols 0 1 --dataset='10d-robust' --overwrite --epochs=1000 --model="forward_01_mlp_noskip" --case-study="bridge-mlp" --is-logging-enabled --seed=1234
# python3 src/main.py --run-id='22-04-18-mlp-10drobust' --is-vanilla-training --y-cols 0 1 --dataset='10d-robust' --overwrite --epochs=1000 --model="forward_01_mlp_noskip" --case-study="bridge-mlp" --is-logging-enabled --seed=5678
# python3 src/main.py --run-id='22-04-18-mlp-10drobust' --is-vanilla-training --y-cols 0 1 --dataset='10d-robust' --overwrite --epochs=1000 --model="forward_01_mlp_noskip" --case-study="bridge-mlp" --is-logging-enabled --seed=9876

# MLP-XSKIP
# python3 src/main.py --run-id='22-04-18-mlpxskip-10drobust' --is-vanilla-training --y-cols 0 1 --dataset='10d-robust' --overwrite --epochs=1000 --model="forward_02_mlp_xskip" --case-study="bridge-mlp" --is-logging-enabled --seed=1234
# python3 src/main.py --run-id='22-04-18-mlpxskip-10drobust' --is-vanilla-training --y-cols 0 1 --dataset='10d-robust' --overwrite --epochs=1000 --model="forward_02_mlp_xskip" --case-study="bridge-mlp" --is-logging-enabled --seed=5678
# python3 src/main.py --run-id='22-04-18-mlpxskip-10drobust' --is-vanilla-training --y-cols 0 1 --dataset='10d-robust' --overwrite --epochs=1000 --model="forward_02_mlp_xskip" --case-study="bridge-mlp" --is-logging-enabled --seed=9876
# python3 src/main.py --run-id='22-04-18-mlpxskip-10dsimple' --is-vanilla-training --y-cols 0 1 --dataset='10d-simple' --overwrite --epochs=1000 --model="forward_02_mlp_xskip" --case-study="bridge-mlp" --is-logging-enabled --seed=1234
# python3 src/main.py --run-id='22-04-18-mlpxskip-10dsimple' --is-vanilla-training --y-cols 0 1 --dataset='10d-simple' --overwrite --epochs=1000 --model="forward_02_mlp_xskip" --case-study="bridge-mlp" --is-logging-enabled --seed=5678
# python3 src/main.py --run-id='22-04-18-mlpxskip-10dsimple' --is-vanilla-training --y-cols 0 1 --dataset='10d-simple' --overwrite --epochs=1000 --model="forward_02_mlp_xskip" --case-study="bridge-mlp" --is-logging-enabled --seed=9876

# GCNN
#python3 src/main.py --run-id='22-04-18-gcnn-10dsimple' --is-vanilla-training --y-cols 0 1 --dataset='10d-simple' --overwrite --epochs=700 --model="forward_03_gcnn" --case-study="bridge-gnn" --is-logging-enabled --seed=1234
#python3 src/main.py --run-id='22-04-18-gcnn-10dsimple' --is-vanilla-training --y-cols 0 1 --dataset='10d-simple' --overwrite --epochs=700 --model="forward_03_gcnn" --case-study="bridge-gnn" --is-logging-enabled --seed=5678
#python3 src/main.py --run-id='22-04-18-gcnn-10dsimple' --is-vanilla-training --y-cols 0 1 --dataset='10d-simple' --overwrite --epochs=700 --model="forward_03_gcnn" --case-study="bridge-gnn" --is-logging-enabled --seed=9876
#python3 src/main.py --run-id='22-04-18-gcnn-10drobust' --is-vanilla-training --y-cols 0 1 --dataset='10d-robust' --overwrite --epochs=700 --model="forward_03_gcnn" --case-study="bridge-gnn" --is-logging-enabled --seed=1234
#python3 src/main.py --run-id='22-04-18-gcnn-10drobust' --is-vanilla-training --y-cols 0 1 --dataset='10d-robust' --overwrite --epochs=700 --model="forward_03_gcnn" --case-study="bridge-gnn" --is-logging-enabled --seed=5678
#python3 src/main.py --run-id='22-04-18-gcnn-10drobust' --is-vanilla-training --y-cols 0 1 --dataset='10d-robust' --overwrite --epochs=700 --model="forward_03_gcnn" --case-study="bridge-gnn" --is-logging-enabled --seed=9876

# GCNN-RES-a0.1
# python3 src/main.py --run-id='22-04-18-gcnnresa0.1-10dsimple' --is-vanilla-training --y-cols 0 1 --dataset='10d-simple' --overwrite --epochs=700 --model="forward_04_gcnn2" --case-study="bridge-gnn" --is-logging-enabled --seed=1234
# python3 src/main.py --run-id='22-04-18-gcnnresa0.1-10dsimple' --is-vanilla-training --y-cols 0 1 --dataset='10d-simple' --overwrite --epochs=700 --model="forward_04_gcnn2" --case-study="bridge-gnn" --is-logging-enabled --seed=5678
# python3 src/main.py --run-id='22-04-18-gcnnresa0.1-10dsimple' --is-vanilla-training --y-cols 0 1 --dataset='10d-simple' --overwrite --epochs=700 --model="forward_04_gcnn2" --case-study="bridge-gnn" --is-logging-enabled --seed=9876
# python3 src/main.py --run-id='22-04-18-gcnnresa0.1-10drobust' --is-vanilla-training --y-cols 0 1 --dataset='10d-robust' --overwrite --epochs=700 --model="forward_04_gcnn2" --case-study="bridge-gnn" --is-logging-enabled --seed=1234
# python3 src/main.py --run-id='22-04-18-gcnnresa0.1-10drobust' --is-vanilla-training --y-cols 0 1 --dataset='10d-robust' --overwrite --epochs=700 --model="forward_04_gcnn2" --case-study="bridge-gnn" --is-logging-enabled --seed=5678
# python3 src/main.py --run-id='22-04-18-gcnnresa0.1-10drobust' --is-vanilla-training --y-cols 0 1 --dataset='10d-robust' --overwrite --epochs=700 --model="forward_04_gcnn2" --case-study="bridge-gnn" --is-logging-enabled --seed=9876

# GCNN-RES-a0.4
# python3 src/main.py --run-id='22-04-18-gcnnresa0.4-10dsimple' --is-vanilla-training --y-cols 0 1 --dataset='10d-simple' --overwrite --epochs=700 --model="forward_04_gcnn2" --case-study="bridge-gnn" --is-logging-enabled --seed=1234 --gnnres-alpha=0.4
# python3 src/main.py --run-id='22-04-18-gcnnresa0.4-10dsimple' --is-vanilla-training --y-cols 0 1 --dataset='10d-simple' --overwrite --epochs=700 --model="forward_04_gcnn2" --case-study="bridge-gnn" --is-logging-enabled --seed=5678 --gnnres-alpha=0.4
# python3 src/main.py --run-id='22-04-18-gcnnresa0.4-10dsimple' --is-vanilla-training --y-cols 0 1 --dataset='10d-simple' --overwrite --epochs=700 --model="forward_04_gcnn2" --case-study="bridge-gnn" --is-logging-enabled --seed=9876 --gnnres-alpha=0.4
# python3 src/main.py --run-id='22-04-18-gcnnresa0.4-10drobust' --is-vanilla-training --y-cols 0 1 --dataset='10d-robust' --overwrite --epochs=700 --model="forward_04_gcnn2" --case-study="bridge-gnn" --is-logging-enabled --seed=1234 --gnnres-alpha=0.4
# python3 src/main.py --run-id='22-04-18-gcnnresa0.4-10drobust' --is-vanilla-training --y-cols 0 1 --dataset='10d-robust' --overwrite --epochs=700 --model="forward_04_gcnn2" --case-study="bridge-gnn" --is-logging-enabled --seed=5678 --gnnres-alpha=0.4
# python3 src/main.py --run-id='22-04-18-gcnnresa0.4-10drobust' --is-vanilla-training --y-cols 0 1 --dataset='10d-robust' --overwrite --epochs=700 --model="forward_04_gcnn2" --case-study="bridge-gnn" --is-logging-enabled --seed=9876 --gnnres-alpha=0.4

# GCNN-RES-a0.7
# python3 src/main.py --run-id='22-04-18-gcnnresa0.7-10dsimple' --is-vanilla-training --y-cols 0 1 --dataset='10d-simple' --overwrite --epochs=700 --model="forward_04_gcnn2" --case-study="bridge-gnn" --is-logging-enabled --seed=1234 --gnnres-alpha=0.7
# python3 src/main.py --run-id='22-04-18-gcnnresa0.7-10dsimple' --is-vanilla-training --y-cols 0 1 --dataset='10d-simple' --overwrite --epochs=700 --model="forward_04_gcnn2" --case-study="bridge-gnn" --is-logging-enabled --seed=5678 --gnnres-alpha=0.7
# python3 src/main.py --run-id='22-04-18-gcnnresa0.7-10dsimple' --is-vanilla-training --y-cols 0 1 --dataset='10d-simple' --overwrite --epochs=700 --model="forward_04_gcnn2" --case-study="bridge-gnn" --is-logging-enabled --seed=9876 --gnnres-alpha=0.7
# python3 src/main.py --run-id='22-04-18-gcnnresa0.7-10drobust' --is-vanilla-training --y-cols 0 1 --dataset='10d-robust' --overwrite --epochs=700 --model="forward_04_gcnn2" --case-study="bridge-gnn" --is-logging-enabled --seed=1234 --gnnres-alpha=0.7
# python3 src/main.py --run-id='22-04-18-gcnnresa0.7-10drobust' --is-vanilla-training --y-cols 0 1 --dataset='10d-robust' --overwrite --epochs=700 --model="forward_04_gcnn2" --case-study="bridge-gnn" --is-logging-enabled --seed=5678 --gnnres-alpha=0.7
# python3 src/main.py --run-id='22-04-18-gcnnresa0.7-10drobust' --is-vanilla-training --y-cols 0 1 --dataset='10d-robust' --overwrite --epochs=700 --model="forward_04_gcnn2" --case-study="bridge-gnn" --is-logging-enabled --seed=9876 --gnnres-alpha=0.7


# ************************************************************************************************************************************************************
# CASE STUDY 1: BRIDGE DATASET — COSTS ONLY (y1d)

# round 1 with seed: 1234

# python3 src/main.py --run-id='22-04-18-y1d-cvae1nores' --is-vanilla-training --y-cols 0 --y-augm --case-study='bridge-y1d-no-nf' --dataset='10d-simple' --overwrite --epochs=500 --model="cvae_01_nores" --seed=1234 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-y1d-cvae2res' --is-vanilla-training --y-cols 0 --y-augm --case-study='bridge-y1d-no-nf' --dataset='10d-simple' --overwrite --epochs=500 --model="cvae_02_res" --seed=1234 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-y1d-cvae3yskip' --is-vanilla-training --y-cols 0 --y-augm --case-study='bridge-y1d-no-nf' --dataset='10d-simple' --overwrite --epochs=500 --model="cvae_03_yskip" --seed=1234 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-y1d-cvae4tril' --is-vanilla-training --y-cols 0 --y-augm --case-study='bridge-y1d-no-nf' --dataset='10d-simple' --overwrite --epochs=500 --model="cvae_04_tril" --seed=1234 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-y1d-cvae5nf' --is-vanilla-training --y-cols 0 --y-augm --case-study='bridge-y1d-nf' --dataset='10d-simple' --overwrite --epochs=1500 --model="cvae_05_nf" --seed=1234 --is-logging-enabled  --lr-schedule 10 300 800
# python3 src/main.py --run-id='22-04-18-y1d-cvae9yskipgcnn' --pretrained-gnn='ckpt/22-04-18-gcnnresa0.4-10dsimple/ckpt_best_seed_1234.pth.tar' --is-vanilla-training --y-cols 0 --y-augm --case-study='bridge-y1d-no-nf' --dataset='10d-simple' --overwrite --epochs=500 --model="cvae_09_yskip_gcnn" --seed=1234 --is-logging-enabled  --lr-schedule 15
# python3 src/main.py --run-id='22-04-18-y1d-cvae10trilgcnn' --pretrained-gnn='ckpt/22-04-18-gcnnresa0.4-10dsimple/ckpt_best_seed_1234.pth.tar' --is-vanilla-training --y-cols 0 --y-augm --case-study='bridge-y1d-no-nf' --dataset='10d-simple' --overwrite --epochs=500 --model="cvae_10_tril_gcnn" --seed=1234 --is-logging-enabled  --lr-schedule 15
# python3 src/main.py --run-id='22-04-18-y1d-cvae11nfgcnn' --pretrained-gnn='ckpt/22-04-18-gcnnresa0.4-10dsimple/ckpt_best_seed_1234.pth.tar' --is-vanilla-training --y-cols 0 --y-augm --case-study='bridge-y1d-nf' --dataset='10d-simple' --overwrite --epochs=1500 --model="cvae_11_nf_gcnn" --seed=1234 --is-logging-enabled  --lr-schedule 10 300 800

# round 2 with seed: 5678

# python3 src/main.py --run-id='22-04-18-y1d-cvae1nores' --is-vanilla-training --y-cols 0 --y-augm --case-study='bridge-y1d-no-nf' --dataset='10d-simple' --overwrite --epochs=500 --model="cvae_01_nores" --seed=5678 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-y1d-cvae2res' --is-vanilla-training --y-cols 0 --y-augm --case-study='bridge-y1d-no-nf' --dataset='10d-simple' --overwrite --epochs=500 --model="cvae_02_res" --seed=5678 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-y1d-cvae3yskip' --is-vanilla-training --y-cols 0 --y-augm --case-study='bridge-y1d-no-nf' --dataset='10d-simple' --overwrite --epochs=500 --model="cvae_03_yskip" --seed=5678 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-y1d-cvae4tril' --is-vanilla-training --y-cols 0 --y-augm --case-study='bridge-y1d-no-nf' --dataset='10d-simple' --overwrite --epochs=1500 --model="cvae_04_tril" --seed=5678 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-y1d-cvae5nf' --is-vanilla-training --y-cols 0 --y-augm --case-study='bridge-y1d-nf' --dataset='10d-simple' --overwrite --epochs=500 --model="cvae_05_nf" --seed=5678 --is-logging-enabled  --lr-schedule 10 300 800
# python3 src/main.py --run-id='22-04-18-y1d-cvae9yskipgcnn' --pretrained-gnn='ckpt/22-04-18-gcnnresa0.4-10dsimple/ckpt_best_seed_1234.pth.tar' --is-vanilla-training --y-cols 0 --y-augm --case-study='bridge-y1d-no-nf' --dataset='10d-simple' --overwrite --epochs=500 --model="cvae_09_yskip_gcnn" --seed=5678 --is-logging-enabled  --lr-schedule 15
# python3 src/main.py --run-id='22-04-18-y1d-cvae10trilgcnn' --pretrained-gnn='ckpt/22-04-18-gcnnresa0.4-10dsimple/ckpt_best_seed_1234.pth.tar' --is-vanilla-training --y-cols 0 --y-augm --case-study='bridge-y1d-no-nf' --dataset='10d-simple' --overwrite --epochs=500 --model="cvae_10_tril_gcnn" --seed=5678 --is-logging-enabled  --lr-schedule 15
# python3 src/main.py --run-id='22-04-18-y1d-cvae11nfgcnn' --pretrained-gnn='ckpt/22-04-18-gcnnresa0.4-10dsimple/ckpt_best_seed_1234.pth.tar' --is-vanilla-training --y-cols 0 --y-augm --case-study='bridge-y1d-nf' --dataset='10d-simple' --overwrite --epochs=1500 --model="cvae_11_nf_gcnn" --seed=5678 --is-logging-enabled  --lr-schedule 10 300 800

# round 3 with seed: 9876
# python3 src/main.py --run-id='22-04-18-y1d-cvae1nores' --is-vanilla-training --y-cols 0 --y-augm --case-study='bridge-y1d-no-nf' --dataset='10d-simple' --overwrite --epochs=500 --model="cvae_01_nores" --seed=9876 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-y1d-cvae2res' --is-vanilla-training --y-cols 0 --y-augm --case-study='bridge-y1d-no-nf' --dataset='10d-simple' --overwrite --epochs=500 --model="cvae_02_res" --seed=9876 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-y1d-cvae3yskip' --is-vanilla-training --y-cols 0 --y-augm --case-study='bridge-y1d-no-nf' --dataset='10d-simple' --overwrite --epochs=500 --model="cvae_03_yskip" --seed=9876 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-y1d-cvae4tril' --is-vanilla-training --y-cols 0 --y-augm --case-study='bridge-y1d-no-nf' --dataset='10d-simple' --overwrite --epochs=1500 --model="cvae_04_tril" --seed=9876 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-y1d-cvae5nf' --is-vanilla-training --y-cols 0 --y-augm --case-study='bridge-y1d-nnf' --dataset='10d-simple' --overwrite --epochs=500 --model="cvae_05_nf" --seed=9876 --is-logging-enabled  --lr-schedule 10 300 800
# python3 src/main.py --run-id='22-04-18-y1d-cvae9yskipgcnn' --pretrained-gnn='ckpt/22-04-18-gcnnresa0.4-10dsimple/ckpt_best_seed_1234.pth.tar' --is-vanilla-training --y-cols 0 --y-augm --case-study='bridge-y1d-no-nf' --dataset='10d-simple' --overwrite --epochs=500 --model="cvae_09_yskip_gcnn" --seed=9876 --is-logging-enabled  --lr-schedule 15
# python3 src/main.py --run-id='22-04-18-y1d-cvae10trilgcnn' --pretrained-gnn='ckpt/22-04-18-gcnnresa0.4-10dsimple/ckpt_best_seed_1234.pth.tar' --is-vanilla-training --y-cols 0 --y-augm --case-study='bridge-y1d-no-nf' --dataset='10d-simple' --overwrite --epochs=500 --model="cvae_10_tril_gcnn" --seed=9876 --is-logging-enabled  --lr-schedule 15
# python3 src/main.py --run-id='22-04-18-y1d-cvae11nfgcnn' --pretrained-gnn='ckpt/22-04-18-gcnnresa0.4-10dsimple/ckpt_best_seed_1234.pth.tar' --is-vanilla-training --y-cols 0 --y-augm --case-study='bridge-y1d-nf' --dataset='10d-simple' --overwrite --epochs=1500 --model="cvae_11_nf_gcnn" --seed=9876 --is-logging-enabled  --lr-schedule 10 300 800


# ************************************************************************************************************************************************************
# CASE STUDY 2: BRIDGE DATASET — COSTS AND UTILIZATION (y2d)

# round 1 with seed: 1234

# python3 src/main.py --run-id='22-04-18-y2d-cvae1nores' --is-vanilla-training --y-cols 0 1 --y-augm --case-study='bridge-y2d-no-nf' --dataset='10d-simple' --overwrite --epochs=500 --model="cvae_01_nores" --seed=1234 --lr-schedule 300 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-y2d-cvae2res' --is-vanilla-training --y-cols 0 1 --y-augm --case-study='bridge-y2d-no-nf' --dataset='10d-simple' --overwrite --epochs=500 --model="cvae_02_res" --seed=1234 --lr-schedule 300 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-y2d-cvae3yskip' --is-vanilla-training --y-cols 0 1 --y-augm --case-study='bridge-y2d-no-nf' --dataset='10d-simple' --overwrite --epochs=500 --model="cvae_03_yskip" --seed=1234 --lr-schedule 300 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-y2d-cvae4tril' --is-vanilla-training --y-cols 0 1 --y-augm --case-study='bridge-y2d-no-nf' --dataset='10d-simple' --overwrite --epochs=500 --model="cvae_04_tril" --seed=1234 --lr-schedule 300 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-y2d-cvae5nf' --is-vanilla-training --y-cols 0 1 --y-augm --case-study='bridge-y2d-nf' --dataset='10d-simple' --overwrite --epochs=1500 --model="cvae_05_nf" --seed=1234 --lr-schedule 10 300 800 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-y2d-cvae9yskipgcnn' --pretrained-gnn='ckpt/22-04-18-gcnnresa0.4-10dsimple/ckpt_best_seed_1234.pth.tar' --is-vanilla-training --y-cols 0 1 --y-augm --case-study='bridge-y2d-no-nf' --dataset='10d-simple' --overwrite --epochs=500 --model="cvae_09_yskip_gcnn" --seed=1234 --is-logging-enabled  --lr-schedule 15
# python3 src/main.py --run-id='22-04-18-y2d-cvae10trilgcnn' --pretrained-gnn='ckpt/22-04-18-gcnnresa0.4-10dsimple/ckpt_best_seed_1234.pth.tar' --is-vanilla-training --y-cols 0 1 --y-augm --case-study='bridge-y2d-no-nf' --dataset='10d-simple' --overwrite --epochs=500 --model="cvae_10_tril_gcnn" --seed=1234 --is-logging-enabled  --lr-schedule 15
# python3 src/main.py --run-id='22-04-18-y2d-cvae11nfgcnn' --pretrained-gnn='ckpt/22-04-18-gcnnresa0.4-10dsimple/ckpt_best_seed_1234.pth.tar' --is-vanilla-training --y-cols 0 1 --y-augm --case-study='bridge-y2d-nf' --dataset='10d-simple' --overwrite --epochs=1500 --model="cvae_11_nf_gcnn" --seed=1234 --is-logging-enabled  --lr-schedule 10 300 800

# round 2 with seed: 5678

# python3 src/main.py --run-id='22-04-18-y2d-cvae1nores' --is-vanilla-training --y-cols 0 1 --y-augm --case-study='bridge-y2d-no-nf' --dataset='10d-simple' --overwrite --epochs=500 --model="cvae_01_nores" --seed=5678 --lr-schedule 300 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-y2d-cvae2res' --is-vanilla-training --y-cols 0 1 --y-augm --case-study='bridge-y2d-no-nf' --dataset='10d-simple' --overwrite --epochs=500 --model="cvae_02_res" --seed=5678 --lr-schedule 300 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-y2d-cvae3yskip' --is-vanilla-training --y-cols 0 1 --y-augm --case-study='bridge-y2d-no-nf' --dataset='10d-simple' --overwrite --epochs=500 --model="cvae_03_yskip" --seed=5678 --lr-schedule 300 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-y2d-cvae4tril' --is-vanilla-training --y-cols 0 1 --y-augm --case-study='bridge-y2d-no-nf' --dataset='10d-simple' --overwrite --epochs=500 --model="cvae_04_tril" --seed=1234 --lr-schedule 300 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-y2d-cvae5nf' --is-vanilla-training --y-cols 0 1 --y-augm --case-study='bridge-y2d-nf' --dataset='10d-simple' --overwrite --epochs=1500 --model="cvae_05_nf" --seed=5678 --lr-schedule 10 300 800 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-y2d-cvae9yskipgcnn' --pretrained-gnn='ckpt/22-04-18-gcnnresa0.4-10dsimple/ckpt_best_seed_1234.pth.tar' --is-vanilla-training --y-cols 0 1 --y-augm --case-study='bridge-y2d-no-nf' --dataset='10d-simple' --overwrite --epochs=500 --model="cvae_09_yskip_gcnn" --seed=5678 --is-logging-enabled  --lr-schedule 15
# python3 src/main.py --run-id='22-04-18-y2d-cvae10trilgcnn' --pretrained-gnn='ckpt/22-04-18-gcnnresa0.4-10dsimple/ckpt_best_seed_1234.pth.tar' --is-vanilla-training --y-cols 0 1 --y-augm --case-study='bridge-y2d-no-nf' --dataset='10d-simple' --overwrite --epochs=500 --model="cvae_10_tril_gcnn" --seed=5678 --is-logging-enabled  --lr-schedule 15
# python3 src/main.py --run-id='22-04-18-y2d-cvae11nfgcnn' --pretrained-gnn='ckpt/22-04-18-gcnnresa0.4-10dsimple/ckpt_best_seed_1234.pth.tar' --is-vanilla-training --y-cols 0 1 --y-augm --case-study='bridge-y2d-nf' --dataset='10d-simple' --overwrite --epochs=1500 --model="cvae_11_nf_gcnn" --seed=5678 --is-logging-enabled  --lr-schedule 10 300 800

# round 3 with seed: 9876

# python3 src/main.py --run-id='22-04-18-y2d-cvae1nores' --is-vanilla-training --y-cols 0 1 --y-augm --case-study='bridge-y2d-no-nf' --dataset='10d-simple' --overwrite --epochs=500 --model="cvae_01_nores" --seed=9876 --lr-schedule 300 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-y2d-cvae2res' --is-vanilla-training --y-cols 0 1 --y-augm --case-study='bridge-y2d-no-nf' --dataset='10d-simple' --overwrite --epochs=500 --model="cvae_02_res" --seed=9876 --lr-schedule 300 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-y2d-cvae3yskip' --is-vanilla-training --y-cols 0 1 --y-augm --case-study='bridge-y2d-no-nf' --dataset='10d-simple' --overwrite --epochs=500 --model="cvae_03_yskip" --seed=9876 --lr-schedule 300 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-y2d-cvae4tril' --is-vanilla-training --y-cols 0 1 --y-augm --case-study='bridge-y2d-no-nf' --dataset='10d-simple' --overwrite --epochs=500 --model="cvae_04_tril" --seed=1234 --lr-schedule 300 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-y2d-cvae5nf' --is-vanilla-training --y-cols 0 1 --y-augm --case-study='bridge-y2d-nf' --dataset='10d-simple' --overwrite --epochs=1500 --model="cvae_05_nf" --seed=9876 --lr-schedule 10 300 800 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-y2d-cvae9yskipgcnn' --pretrained-gnn='ckpt/22-04-18-gcnnresa0.4-10dsimple/ckpt_best_seed_1234.pth.tar' --is-vanilla-training --y-cols 0 1 --y-augm --case-study='bridge-y2d-no-nf' --dataset='10d-simple' --overwrite --epochs=500 --model="cvae_09_yskip_gcnn" --seed=9876 --is-logging-enabled  --lr-schedule 15
# python3 src/main.py --run-id='22-04-18-y2d-cvae10trilgcnn' --pretrained-gnn='ckpt/22-04-18-gcnnresa0.4-10dsimple/ckpt_best_seed_1234.pth.tar' --is-vanilla-training --y-cols 0 1 --y-augm --case-study='bridge-y2d-no-nf' --dataset='10d-simple' --overwrite --epochs=500 --model="cvae_10_tril_gcnn" --seed=9876 --is-logging-enabled  --lr-schedule 15
# python3 src/main.py --run-id='22-04-18-y2d-cvae11nfgcnn' --pretrained-gnn='ckpt/22-04-18-gcnnresa0.4-10dsimple/ckpt_best_seed_1234.pth.tar' --is-vanilla-training --y-cols 0 1 --y-augm --case-study='bridge-y2d-nf' --dataset='10d-simple' --overwrite --epochs=1500 --model="cvae_11_nf_gcnn" --seed=9876 --is-logging-enabled  --lr-schedule 10 300 800


# ************************************************************************************************************************************************************
# CASE STUDY 3: WALL DATASET - UTILIZATION

# round 1 with seed: 1234
# python3 src/main.py --run-id='22-04-18-w5d-cvae1nores' --is-vanilla-training --y-cols 0 --y-augm --case-study='wall-y1d-no-nf' --dataset='5d-wall-v2' --overwrite --epochs=300 --model="cvae_01_nores" --seed=1234 --lr-schedule 70 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-w5d-cvae2res' --is-vanilla-training --y-cols 0 --y-augm --case-study='wall-y1d-no-nf' --dataset='5d-wall-v2' --overwrite --epochs=300 --model="cvae_02_res" --seed=1234 --lr-schedule 70 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-w5d-cvae3yskip' --is-vanilla-training --y-cols 0 --y-augm --case-study='wall-y1d-no-nf' --dataset='5d-wall-v2' --overwrite --epochs=300 --model="cvae_03_yskip" --seed=1234 --lr-schedule 70 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-w5d-cvae4tril' --is-vanilla-training --y-cols 0 --y-augm --case-study='wall-y1d-no-nf' --dataset='5d-wall-v2' --overwrite --epochs=300 --model="cvae_04_tril" --seed=1234 --lr-schedule 70 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-w5d-cvae5nf' --is-vanilla-training --y-cols 0 --y-augm --case-study='wall-y1d-nf' --dataset='5d-wall-v2' --overwrite --epochs=500 --model="cvae_05_nf" --seed=1234 --lr-schedule 10 300 --is-logging-enabled

# round 2 with seed: 5678
# python3 src/main.py --run-id='22-04-18-w5d-cvae1nores' --is-vanilla-training --y-cols 0 --y-augm --case-study='wall-y1d-no-nf' --dataset='5d-wall-v2' --overwrite --epochs=300 --model="cvae_01_nores" --seed=5678 --lr-schedule 70 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-w5d-cvae2res' --is-vanilla-training --y-cols 0 --y-augm --case-study='wall-y1d-no-nf' --dataset='5d-wall-v2' --overwrite --epochs=300 --model="cvae_02_res" --seed=5678 --lr-schedule 70 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-w5d-cvae3yskip' --is-vanilla-training --y-cols 0 --y-augm --case-study='wall-y1d-no-nf' --dataset='5d-wall-v2' --overwrite --epochs=300 --model="cvae_03_yskip" --seed=5678 --lr-schedule 70 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-w5d-cvae4tril' --is-vanilla-training --y-cols 0 --y-augm --case-study='wall-y1d-no-nf' --dataset='5d-wall-v2' --overwrite --epochs=300 --model="cvae_04_tril" --seed=5678 --lr-schedule 70 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-w5d-cvae5nf' --is-vanilla-training --y-cols 0 --y-augm --case-study='wall-y1d-nf' --dataset='5d-wall-v2' --overwrite --epochs=500 --model="cvae_05_nf" --seed=5678 --lr-schedule 10 300 --is-logging-enabled

# round 3 with seed: 9876
# python3 src/main.py --run-id='22-04-18-w5d-cvae1nores' --is-vanilla-training --y-cols 0 --y-augm --case-study='wall-y1d-no-nf' --dataset='5d-wall-v2' --overwrite --epochs=300 --model="cvae_01_nores" --seed=9876 --lr-schedule 70 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-w5d-cvae2res' --is-vanilla-training --y-cols 0 --y-augm --case-study='wall-y1d-no-nf' --dataset='5d-wall-v2' --overwrite --epochs=300 --model="cvae_02_res" --seed=9876 --lr-schedule 70 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-w5d-cvae3yskip' --is-vanilla-training --y-cols 0 --y-augm --case-study='wall-y1d-no-nf' --dataset='5d-wall-v2' --overwrite --epochs=300 --model="cvae_03_yskip" --seed=9876 --lr-schedule 70 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-w5d-cvae4tril' --is-vanilla-training --y-cols 0 --y-augm --case-study='wall-y1d-no-nf' --dataset='5d-wall-v2' --overwrite --epochs=300 --model="cvae_04_tril" --seed=9876 --lr-schedule 70 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-w5d-cvae5nf' --is-vanilla-training --y-cols 0 --y-augm --case-study='wall-y1d-nf' --dataset='5d-wall-v2' --overwrite --epochs=500 --model="cvae_05_nf" --seed=9876 --lr-schedule 10 300 --is-logging-enabled



#python3 src/main.py --run-id='23-05-19-w5d-cvae5nf' --k-flows=6 --is-vanilla-training --y-cols 0 --y-augm --case-study='wall-y1d-nf' --dataset='5d-wall-v2' --overwrite --epochs=500 --model="cvae_05_nf" --seed=1234 --lr-schedule 10 300 --is-logging-enabled

python3 src/main.py --run-id='23-05-19-w5d-cvae5nf' --k-flows=6 --is-vanilla-training --y-cols 0 --y-augm --case-study='wall-y1d-nf' --dataset='dynamic' --dataset-path-csv='misc/dataset-02-walls/5d-wall-v2.csv' --overwrite --epochs=500 --model="cvae_05_nf" --seed=1234 --lr-schedule 10 300 --is-logging-enabled

#python3 src/main.py --run-id='23-05-19-w5d-cvae8nfyxpart' --k-flows=6 --is-vanilla-training --y-cols 0 --y-augm --case-study='wall-y1d-nf' --dataset='5d-wall-v2' --overwrite --epochs=2500 --model="cvae_08_nf_yxpart" --seed=1234 --lr-schedule 10 300 800 --is-logging-enabled

python3 src/main.py --run-id='23-05-19-w5d-cvae8nfyxpart' --k-flows=6 --is-vanilla-training --y-cols 0 --y-augm --case-study='wall-y1d-nf' --dataset='dynamic' --dataset-path-csv='misc/dataset-02-walls/5d-wall-v2.csv' --overwrite --epochs=2500 --model="cvae_08_nf_yxpart" --seed=1234 --lr-schedule 10 300 800 --is-logging-enabled


python3 src/main.py --resume='ckpt/23-05-19-w5d-cvae8nfyxpart/ckpt_last_seed_1234.pth.tar' --y-cols 0 --y-augm --case-study='wall-y1d-nf' --dataset='dynamic' --dataset-path-csv='misc/dataset-02-walls/5d-wall-v2.csv' --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="proportional" --is-only-yx-for-gen


python3 src/main.py --resume='ckpt/23-05-19-w5d-cvae8nfyxpart/ckpt_last_seed_1234.pth.tar' --y-cols 0 --y-augm --case-study='wall-y1d-nf' --dataset='dynamic' --dataset-path-csv='misc/dataset-02-walls/5d-wall-v2.csv' --is-vanilla-training --datamode="gen-eval-walls-comparison" --genmode="ga-comp"

# ************************************************************************************************************************************************************
# CASE STUDY 4: PARTIAL OUTPUT FIXING ON ALL 3 DATASETS

# 10d-bridge-y1d (3 models with 3 seeds)

# python3 src/main.py --run-id='22-04-18-y1d-cvae6yskipyxpart' --is-vanilla-training --y-cols 0 --y-augm --case-study='bridge-y1d-no-nf' --dataset='10d-simple' --overwrite --epochs=15000 --model="cvae_06_yskip_yxpart" --seed=1234 --lr-schedule 20 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-y1d-cvae6yskipyxpart' --is-vanilla-training --y-cols 0 --y-augm --case-study='bridge-y1d-no-nf' --dataset='10d-simple' --overwrite --epochs=15000 --model="cvae_06_yskip_yxpart" --seed=5678 --lr-schedule 20 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-y1d-cvae6yskipyxpart' --is-vanilla-training --y-cols 0 --y-augm --case-study='bridge-y1d-no-nf' --dataset='10d-simple' --overwrite --epochs=15000 --model="cvae_06_yskip_yxpart" --seed=9876 --lr-schedule 20 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-y1d-cvae7trilyxpart' --is-vanilla-training --y-cols 0 --y-augm --case-study='bridge-y1d-no-nf' --dataset='10d-simple' --overwrite --epochs=15000 --model="cvae_07_tril_yxpart" --seed=1234 --lr-schedule 20 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-y1d-cvae7trilyxpart' --is-vanilla-training --y-cols 0 --y-augm --case-study='bridge-y1d-no-nf' --dataset='10d-simple' --overwrite --epochs=15000 --model="cvae_07_tril_yxpart" --seed=5678 --lr-schedule 20 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-y1d-cvae7trilyxpart' --is-vanilla-training --y-cols 0 --y-augm --case-study='bridge-y1d-no-nf' --dataset='10d-simple' --overwrite --epochs=15000 --model="cvae_07_tril_yxpart" --seed=9876 --lr-schedule 20 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-y1d-cvae8nfyxpart' --is-vanilla-training --y-cols 0 --y-augm --case-study='bridge-y1d-nf' --dataset='10d-simple' --overwrite --epochs=15000 --model="cvae_08_nf_yxpart" --seed=1234 --lr-schedule 5 2500 8000 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-y1d-cvae8nfyxpart' --is-vanilla-training --y-cols 0 --y-augm --case-study='bridge-y1d-nf' --dataset='10d-simple' --overwrite --epochs=15000 --model="cvae_08_nf_yxpart" --seed=5678 --lr-schedule 5 2500 8000 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-y1d-cvae8nfyxpart' --is-vanilla-training --y-cols 0 --y-augm --case-study='bridge-y1d-nf' --dataset='10d-simple' --overwrite --epochs=15000 --model="cvae_08_nf_yxpart" --seed=9876 --lr-schedule 5 2500 8000 --is-logging-enabled

# 10d-bridge-y2d (3 models with 3 seeds)
# python3 src/main.py --run-id='22-04-18-y2d-cvae6yskipyxpart' --is-vanilla-training --y-cols 0 1 --y-augm --case-study='bridge-y2d-no-nf' --dataset='10d-simple' --overwrite --epochs=15000 --model="cvae_06_yskip_yxpart" --seed=1234 --lr-schedule 20 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-y2d-cvae6yskipyxpart' --is-vanilla-training --y-cols 0 1 --y-augm --case-study='bridge-y2d-no-nf' --dataset='10d-simple' --overwrite --epochs=15000 --model="cvae_06_yskip_yxpart" --seed=5678 --lr-schedule 20 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-y2d-cvae6yskipyxpart' --is-vanilla-training --y-cols 0 1 --y-augm --case-study='bridge-y2d-no-nf' --dataset='10d-simple' --overwrite --epochs=15000 --model="cvae_06_yskip_yxpart" --seed=9876 --lr-schedule 20 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-y2d-cvae7trilyxpart' --is-vanilla-training --y-cols 0 1 --y-augm --case-study='bridge-y2d-no-nf' --dataset='10d-simple' --overwrite --epochs=15000 --model="cvae_07_tril_yxpart" --seed=1234 --lr-schedule 20 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-y2d-cvae7trilyxpart' --is-vanilla-training --y-cols 0 1 --y-augm --case-study='bridge-y2d-no-nf' --dataset='10d-simple' --overwrite --epochs=15000 --model="cvae_07_tril_yxpart" --seed=5678 --lr-schedule 20 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-y2d-cvae7trilyxpart' --is-vanilla-training --y-cols 0 1 --y-augm --case-study='bridge-y2d-no-nf' --dataset='10d-simple' --overwrite --epochs=15000 --model="cvae_07_tril_yxpart" --seed=9876 --lr-schedule 20 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-y2d-cvae8nfyxpart' --is-vanilla-training --y-cols 0 1 --y-augm --case-study='bridge-y2d-nf' --dataset='10d-simple' --overwrite --epochs=15000 --model="cvae_08_nf_yxpart" --seed=1234 --lr-schedule 5 2500 8000 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-y2d-cvae8nfyxpart' --is-vanilla-training --y-cols 0 1 --y-augm --case-study='bridge-y2d-nf' --dataset='10d-simple' --overwrite --epochs=15000 --model="cvae_08_nf_yxpart" --seed=5678 --lr-schedule 5 2500 8000 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-y2d-cvae8nfyxpart' --is-vanilla-training --y-cols 0 1 --y-augm --case-study='bridge-y2d-nf' --dataset='10d-simple' --overwrite --epochs=15000 --model="cvae_08_nf_yxpart" --seed=9876 --lr-schedule 5 2500 8000 --is-logging-enabled

# 5d-wall (3 models with 3 seeds)
# python3 src/main.py --run-id='22-04-18-w5d-cvae6yskipyxpart' --is-vanilla-training --y-cols 0 --y-augm --case-study='wall-y1d-no-nf' --dataset='5d-wall-v2' --overwrite --epochs=2500 --model="cvae_06_yskip_yxpart" --seed=1234 --lr-schedule 70 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-w5d-cvae6yskipyxpart' --is-vanilla-training --y-cols 0 --y-augm --case-study='wall-y1d-no-nf' --dataset='5d-wall-v2' --overwrite --epochs=2500 --model="cvae_06_yskip_yxpart" --seed=5678 --lr-schedule 70 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-w5d-cvae6yskipyxpart' --is-vanilla-training --y-cols 0 --y-augm --case-study='wall-y1d-no-nf' --dataset='5d-wall-v2' --overwrite --epochs=2500 --model="cvae_06_yskip_yxpart" --seed=9876 --lr-schedule 70 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-w5d-cvae7trilyxpart' --is-vanilla-training --y-cols 0 --y-augm --case-study='wall-y1d-no-nf' --dataset='5d-wall-v2' --overwrite --epochs=2500 --model="cvae_07_tril_yxpart" --seed=1234 --lr-schedule 70 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-w5d-cvae7trilyxpart' --is-vanilla-training --y-cols 0 --y-augm --case-study='wall-y1d-no-nf' --dataset='5d-wall-v2' --overwrite --epochs=2500 --model="cvae_07_tril_yxpart" --seed=5678 --lr-schedule 70 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-w5d-cvae7trilyxpart' --is-vanilla-training --y-cols 0 --y-augm --case-study='wall-y1d-no-nf' --dataset='5d-wall-v2' --overwrite --epochs=2500 --model="cvae_07_tril_yxpart" --seed=9876 --lr-schedule 70 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-w5d-cvae8nfyxpart' --is-vanilla-training --y-cols 0 --y-augm --case-study='wall-y1d-nf' --dataset='5d-wall-v2' --overwrite --epochs=2500 --model="cvae_08_nf_yxpart" --seed=1234 --lr-schedule 10 300 800 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-w5d-cvae8nfyxpart' --is-vanilla-training --y-cols 0 --y-augm --case-study='wall-y1d-nf' --dataset='5d-wall-v2' --overwrite --epochs=2500 --model="cvae_08_nf_yxpart" --seed=5678 --lr-schedule 10 300 800 --is-logging-enabled
# python3 src/main.py --run-id='22-04-18-w5d-cvae8nfyxpart' --is-vanilla-training --y-cols 0 --y-augm --case-study='wall-y1d-nf' --dataset='5d-wall-v2' --overwrite --epochs=2500 --model="cvae_08_nf_yxpart" --seed=9876 --lr-schedule 10 300 800 --is-logging-enabled


# ************************************************************************************************************************************************************
# MISC

# python3 src/main.py --run-id='22-04-19-hallen-cvae5nf-s1234' --k-flows=6 --is-vanilla-training --y-col=0 --y-augm --case-study='wall-y1d-nf' --dataset='dynamic' --dataset-path-csv='misc/dataset-03-halls/hallen_xy.csv' --overwrite --epochs=1500 --model="cvae_05_nf" --seed=9876 --is-gen-eval-enabled --lr-schedule 10 300 800 --is-logging-enabled
# python3 src/main.py --run-id='22-04-19-hallen-cvae4tril-s1234' --is-vanilla-training --y-col=0 --y-augm --case-study='wall-y1d-nf' --dataset='dynamic' --dataset-path-csv='misc/dataset-03-halls/hallen_xy.csv' --overwrite --epochs=1500 --model="cvae_04_tril" --seed=9876 --is-gen-eval-enabled --lr-schedule 10 300 800 --is-logging-enabled
# python3 src/main.py --run-id='22-04-20-hallen-cvae5nfyxpartial-s1234' --k-flows=6 --is-vanilla-training --y-cols 0 1 --y-augm --case-study='wall-y1d-nf' --dataset='dynamic' --dataset-path-csv='misc/dataset-03-halls/hallen_xy.csv' --overwrite --epochs=1000 --model="cvae_08_nf_yxpart" --seed=1234 --is-gen-eval-enabled --lr-schedule 10 300 800 --is-logging-enabled

# python3 src/main.py --run-id='22-05-10-rooms-cvae5nfyxpartial' --k-flows=6 --is-vanilla-training --y-cols 0 1 2 3 4 --y-augm --case-study='wall-y1d-nf' --dataset='dynamic' --dataset-path-csv='misc/dataset-04-rooms/rooms_xy.csv' --overwrite --epochs=15000 --model="cvae_08_nf_yxpart" --seed=1234 --lr-schedule 5 2500 8000

# python3 src/main.py --run-id='22-05-10-rooms-cvae5nfyxpartial' --k-flows=6 --is-vanilla-training --y-cols 0 1 2 3 4 --y-augm --shift-all-dimensions --case-study='wall-y1d-nf' --dataset='dynamic' --dataset-path-csv='misc/dataset-04-rooms/rooms_xy_v2.csv' --overwrite --epochs=15000 --model="cvae_08_nf_yxpart" --seed=1234 --lr-schedule 5 2500 8000
# python3 src/main.py --run-id='22-05-10-rooms-cvae5nfyxpartial' --k-flows=6 --is-vanilla-training --y-cols 0 1 2 3 4 --y-augm --shift-all-dimensions --case-study='wall-y1d-nf' --dataset='dynamic' --dataset-path-csv='misc/dataset-04-rooms/rooms_xy_fake.csv' --overwrite --epochs=15000 --model="cvae_08_nf_yxpart" --seed=1234 --lr-schedule 5 2500 8000
