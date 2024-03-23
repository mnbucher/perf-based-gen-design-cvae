# localhost_repro_test.sh
# author: Martin Juan José Bucher

cd ..

# ************************************************************************************************************************************************************
# FORWARD MAPPING APPROXIMATION

# ************************************************************************************************************************************************************

# RFR and GBT
# python3 src/main.py --run-id='baselines-test' --overwrite --is-nn-model --y-col="y2d" --dataset="10d-robust" --is-vanilla-training --datamode="gnn-baselines"

# MLP-NOSKIP
# python3 src/main.py --resume='ckpt/report-22-04-18-mlp-10dsimple/ckpt_best_seed_1234.pth.tar' --is-nn-model --y-col="y2d" --dataset="10d-simple" --is-vanilla-training --datamode="eval-y-repro-thesis"
# python3 src/main.py --resume='ckpt/report-22-04-18-mlp-10drobust/ckpt_best_seed_1234.pth.tar' --is-nn-model --y-col="y2d" --dataset="10d-robust" --is-vanilla-training --datamode="eval-y-repro-thesis"

# MLP-XSKIP
# python3 src/main.py --resume='ckpt/report-22-04-18-mlpxskip-10dsimple/ckpt_best_seed_1234.pth.tar' --is-nn-model --y-col="y2d" --dataset="10d-simple" --is-vanilla-training --datamode="eval-y-repro-thesis"
# python3 src/main.py --resume='ckpt/report-22-04-18-mlpxskip-10drobust/ckpt_best_seed_1234.pth.tar' --is-nn-model --y-col="y2d" --dataset="10d-robust" --is-vanilla-training --datamode="eval-y-repro-thesis"

# GCNN
# python3 src/main.py --resume='ckpt/report-22-04-18-gnn5-10dsimple/ckpt_best_seed_1234.pth.tar' --is-nn-model --y-col="y2d" --dataset="10d-simple" --is-vanilla-training --datamode="eval-y-repro-thesis"
# python3 src/main.py --resume='ckpt/report-22-04-18-gnn5-10drobust/ckpt_best_seed_1234.pth.tar' --is-nn-model --y-col="y2d" --dataset="10d-robust" --is-vanilla-training --datamode="eval-y-repro-thesis"

# GCNN-RES-a0.1
# python3 src/main.py --resume='ckpt/22-04-18-gcnnresa0.1-10dsimple/ckpt_best_seed_1234.pth.tar' --is-nn-model --y-col="y2d" --dataset="10d-simple" --is-vanilla-training --datamode="eval-y-repro-thesis"
# python3 src/main.py --resume='ckpt/22-04-18-gcnnresa0.1-10drobust/ckpt_best_seed_1234.pth.tar' --is-nn-model --y-col="y2d" --dataset="10d-robust" --is-vanilla-training --datamode="eval-y-repro-thesis"

# GCNN-RES-a0.4
# python3 src/main.py --resume='ckpt/22-04-18-gcnnresa0.4-10dsimple/ckpt_best_seed_1234.pth.tar' --is-nn-model --y-col="y2d" --dataset="10d-simple" --is-vanilla-training --datamode="eval-y-repro-thesis"
# python3 src/main.py --resume='ckpt/22-04-18-gcnnresa0.4-10drobust/ckpt_best_seed_1234.pth.tar' --is-nn-model --y-col="y2d" --dataset="10d-robust" --is-vanilla-training --datamode="eval-y-repro-thesis"

# GCNN-RES-a0.7
# python3 src/main.py --resume='ckpt/22-04-18-gcnnresa0.7-10dsimple/ckpt_best_seed_1234.pth.tar' --is-nn-model --y-col="y2d" --dataset="10d-simple" --is-vanilla-training --datamode="eval-y-repro-thesis"
# python3 src/main.py --resume='ckpt/22-04-18-gcnnresa0.7-10drobust/ckpt_best_seed_1234.pth.tar' --is-nn-model --y-col="y2d" --dataset="10d-robust" --is-vanilla-training --datamode="eval-y-repro-thesis"

# MLP: Robustness analysis
# python3 src/main.py --resume='ckpt/report-22-04-18-mlpxskip-10dsimple/ckpt_best_seed_1234.pth.tar' --is-nn-model --y-col="y2d" --dataset="10d-simple" --is-vanilla-training --datamode="eval-y-robust"
# python3 src/main.py --resume='ckpt/report-22-04-18-mlpxskip-10drobust/ckpt_best_seed_1234.pth.tar' --is-nn-model --y-col="y2d" --dataset="10d-robust" --is-vanilla-training --datamode="eval-y-robust"


# ************************************************************************************************************************************************************
# GENERATIVE MODELS

# ************************************************************************************************************************************************************
# CASE STUDY 1: BRIDGE DATASET — COSTS ONLY (y2d)

# proportional sampling
# python3 src/main.py --resume='ckpt/22-04-18-y1d-cvae1nores/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 --dataset="10d-simple" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="proportional"
# python3 src/main.py --resume='ckpt/22-04-18-y1d-cvae2res/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 --dataset="10d-simple" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="proportional"
# python3 src/main.py --resume='ckpt/22-04-18-y1d-cvae3yskip/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 --dataset="10d-simple" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="proportional"
# python3 src/main.py --resume='ckpt/22-04-18-y1d-cvae4tril/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 --dataset="10d-simple" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="proportional"
# python3 src/main.py --resume='ckpt/22-04-18-y1d-cvae5nf-k2/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 --dataset="10d-simple" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="proportional"
# python3 src/main.py --resume='ckpt/22-04-18-y1d-cvae5nf-k6/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 --dataset="10d-simple" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="proportional"
# python3 src/main.py --resume='ckpt/22-04-18-y1d-cvae5nf-k10/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 --dataset="10d-simple" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="proportional"
# python3 src/main.py --resume='ckpt/22-04-18-y1d-cvae9yskipgcnn/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 --dataset="10d-simple" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="proportional"
# python3 src/main.py --resume='ckpt/22-04-18-y1d-cvae10trilgcnn/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 --dataset="10d-simple" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="proportional"
# python3 src/main.py --resume='ckpt/22-04-18-y1d-cvae11nfgcnn/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 --dataset="10d-simple" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="proportional"

# uniform sampling
# python3 src/main.py --resume='ckpt/22-04-18-y1d-cvae1nores/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 --dataset="10d-simple" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="uniform"
# python3 src/main.py --resume='ckpt/22-04-18-y1d-cvae2res/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 --dataset="10d-simple" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="uniform"
# python3 src/main.py --resume='ckpt/22-04-18-y1d-cvae3yskip/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 --dataset="10d-simple" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="uniform"
# python3 src/main.py --resume='ckpt/22-04-18-y1d-cvae4tril/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 --dataset="10d-simple" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="uniform"
# python3 src/main.py --resume='ckpt/22-04-18-y1d-cvae5nf-k2/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 --dataset="10d-simple" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="uniform"
# python3 src/main.py --resume='ckpt/22-04-18-y1d-cvae5nf-k6/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 --dataset="10d-simple" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="uniform"
# python3 src/main.py --resume='ckpt/22-04-18-y1d-cvae5nf-k10/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 --dataset="10d-simple" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="uniform"
# python3 src/main.py --resume='ckpt/22-04-18-y1d-cvae9yskipgcnn/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 --dataset="10d-simple" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="uniform"
# python3 src/main.py --resume='ckpt/22-04-18-y1d-cvae10trilgcnn/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 --dataset="10d-simple" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="uniform"
# python3 src/main.py --resume='ckpt/22-04-18-y1d-cvae11nfgcnn/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 --dataset="10d-simple" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="uniform"


# ************************************************************************************************************************************************************
# CASE STUDY 2: BRIDGE DATASET — COSTS AND UTILIZATION (y2d)

# proportional sampling
# python3 src/main.py --resume='ckpt/22-04-18-y2d-cvae1nores/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 1 --dataset="10d-simple" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="proportional"
# python3 src/main.py --resume='ckpt/22-04-18-y2d-cvae2res/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 1 --dataset="10d-simple" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="proportional"
# python3 src/main.py --resume='ckpt/22-04-18-y2d-cvae3yskip/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 1 --dataset="10d-simple" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="proportional"
# python3 src/main.py --resume='ckpt/22-04-18-y2d-cvae4tril/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 1 --dataset="10d-simple" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="proportional"
# python3 src/main.py --resume='ckpt/22-04-18-y2d-cvae5nf-k2/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 1 --dataset="10d-simple" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="proportional"
# python3 src/main.py --resume='ckpt/22-04-18-y2d-cvae5nf-k6/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 1 --dataset="10d-simple" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="proportional"
# python3 src/main.py --resume='ckpt/22-04-18-y2d-cvae5nf-k10/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 1 --dataset="10d-simple" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="proportional"
# python3 src/main.py --resume='ckpt/22-04-18-y2d-cvae9yskipgcnn/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 1 --dataset="10d-simple" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="proportional"
# python3 src/main.py --resume='ckpt/22-04-18-y2d-cvae10trilgcnn/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 1 --dataset="10d-simple" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="proportional"
# python3 src/main.py --resume='ckpt/22-04-18-y2d-cvae11nfgcnn/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 1 --dataset="10d-simple" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="proportional"

# uniform sampling
# python3 src/main.py --resume='ckpt/22-04-18-y2d-cvae1nores/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 1 --dataset="10d-simple" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="uniform"
# python3 src/main.py --resume='ckpt/22-04-18-y2d-cvae2res/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 1 --dataset="10d-simple" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="uniform"
# python3 src/main.py --resume='ckpt/22-04-18-y2d-cvae3yskip/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 1 --dataset="10d-simple" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="uniform"
# python3 src/main.py --resume='ckpt/22-04-18-y2d-cvae4tril/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 1 --dataset="10d-simple" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="uniform"
# python3 src/main.py --resume='ckpt/22-04-18-y2d-cvae5nf-k2/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 1 --dataset="10d-simple" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="uniform"
# python3 src/main.py --resume='ckpt/22-04-18-y2d-cvae5nf-k6/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 1 --dataset="10d-simple" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="uniform"
# python3 src/main.py --resume='ckpt/22-04-18-y2d-cvae5nf-k10/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 1 --dataset="10d-simple" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="uniform"
# python3 src/main.py --resume='ckpt/22-04-18-y2d-cvae9yskipgcnn/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 1 --dataset="10d-simple" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="uniform"
# python3 src/main.py --resume='ckpt/22-04-18-y2d-cvae10trilgcnn/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 1 --dataset="10d-simple" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="uniform"
# python3 src/main.py --resume='ckpt/22-04-18-y2d-cvae11nfgcnn/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 1 --dataset="10d-simple" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="uniform"


# ************************************************************************************************************************************************************
# CASE STUDY 3: WALL DATASET - UTILIZATION

# proportional sampling
# python3 src/main.py --resume='ckpt/22-04-18-w5d-cvae1nores/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 --dataset="5d-wall-v2" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="proportional"
# python3 src/main.py --resume='ckpt/22-04-18-w5d-cvae2res/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 --dataset="5d-wall-v2" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="proportional"
# python3 src/main.py --resume='ckpt/22-04-18-w5d-cvae3yskip/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 --dataset="5d-wall-v2" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="proportional"
# python3 src/main.py --resume='ckpt/22-04-18-w5d-cvae4tril/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 --dataset="5d-wall-v2" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="proportional"
# python3 src/main.py --resume='ckpt/22-04-20-w5d-cvae5nf-k2/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 --dataset="5d-wall-v2" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="proportional"
# python3 src/main.py --resume='ckpt/22-04-18-w5d-cvae5nf-k6/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 --dataset="5d-wall-v2" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="proportional"
# python3 src/main.py --resume='ckpt/22-04-20-w5d-cvae5nf-k10/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 --dataset="5d-wall-v2" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="proportional"

# uniform sampling
# python3 src/main.py --resume='ckpt/22-04-18-w5d-cvae1nores/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 --dataset="5d-wall-v2" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="uniform"
# python3 src/main.py --resume='ckpt/22-04-18-w5d-cvae2res/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 --dataset="5d-wall-v2" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="uniform"
# python3 src/main.py --resume='ckpt/22-04-18-w5d-cvae3yskip/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 --dataset="5d-wall-v2" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="uniform"
# python3 src/main.py --resume='ckpt/22-04-18-w5d-cvae4tril/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 --dataset="5d-wall-v2" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="uniform"
# python3 src/main.py --resume='ckpt/22-04-20-w5d-cvae5nf-k2/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 --dataset="5d-wall-v2" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="uniform"
# python3 src/main.py --resume='ckpt/22-04-18-w5d-cvae5nf-k6/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 --dataset="5d-wall-v2" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="uniform"
# python3 src/main.py --resume='ckpt/22-04-20-w5d-cvae5nf-k10/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 --dataset="5d-wall-v2" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="uniform"

# ************************************************************************************************************************************************************
# CASE STUDY 4: PARTIAL OUTPUT FIXING ON ALL 3 DATASETS

# y1d
# python3 src/main.py --resume='ckpt/22-04-18-y1d-cvae6yskipyxpart/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 --dataset="10d-simple" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="proportional" --is-only-yx-for-gen
# python3 src/main.py --resume='ckpt/22-04-18-y1d-cvae7trilyxpart/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 --dataset="10d-simple" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="proportional" --is-only-yx-for-gen
# python3 src/main.py --resume='ckpt/22-04-18-y1d-cvae8nfyxpart/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 --dataset="10d-simple" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="proportional" --is-only-yx-for-gen

# y2d
# python3 src/main.py --resume='ckpt/22-04-18-y2d-cvae6yskipyxpart/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 1 --dataset="10d-simple" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="proportional" --is-only-yx-for-gen
# python3 src/main.py --resume='ckpt/22-04-18-y2d-cvae7trilyxpart/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 1 --dataset="10d-simple" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="proportional" --is-only-yx-for-gen
# python3 src/main.py --resume='ckpt/22-04-18-y2d-cvae8nfyxpart/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 1 --dataset="10d-simple" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="proportional" --is-only-yx-for-gen

# w5d
# python3 src/main.py --resume='ckpt/22-04-18-w5d-cvae6yskipyxpart/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 --dataset="5d-wall-v2" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="proportional" --is-only-yx-for-gen
# python3 src/main.py --resume='ckpt/22-04-18-w5d-cvae7trilyxpart/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 --dataset="5d-wall-v2" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="proportional" --is-only-yx-for-gen
# python3 src/main.py --resume='ckpt/22-04-18-w5d-cvae8nfyxpart/ckpt_last_seed_1234.pth.tar' --is-nn-model --y-cols 0 --dataset="5d-wall-v2" --is-vanilla-training --datamode="gen-eval-repro-thesis" --genmode="proportional" --is-only-yx-for-gen

#python3 src/main.py --resume='ckpt/22-05-04-rooms-cvae5nfyxpartial/ckpt_last_seed_1234.pth.tar' --is-vanilla-training --y-cols 0 1 2 3 4 --dataset='dynamic' --dataset-path-csv='misc/dataset-04-rooms/rooms_xy.csv' --datamode="test"
#python3 src/main.py --resume='ckpt/22-05-10-rooms-cvae5nfyxpartial/ckpt_last_seed_1234.pth.tar' --is-vanilla-training --y-cols 0 1 2 3 4 --dataset='dynamic' --dataset-path-csv='misc/dataset-04-rooms/rooms_xy_v2.csv' --datamode="test"

python3 src/main.py --resume='ckpt/22-05-12-rooms-cvae8nf-onehot-full/ckpt_last_seed_1234.pth.tar' --is-vanilla-training --y-cols 0 1 2 3 4 --dataset='dynamic' --dataset-path-csv='misc/dataset-04-rooms/rooms_xy_v2.csv' --datamode="test"
python3 src/main.py --resume='ckpt/22-05-12-rooms-cvae8nf-onehot-full/ckpt_last_seed_1234.pth.tar' --is-vanilla-training --y-cols 0 1 2 3 4 --dataset='dynamic' --dataset-path-csv='misc/dataset-04-rooms/rooms_xy_v2.csv' --datamode="gen-eval"

# python3 src/main.py --resume='ckpt/22-05-10-rooms-cvae5nfyxpartial/ckpt_last_seed_1234.pth.tar' --is-vanilla-training --y-cols 0 1 2 3 4 --dataset='dynamic' --dataset-path-csv='misc/dataset-04-rooms/rooms_xy_fake.csv' --datamode="test"

# python3 src/main.py --resume='ckpt/22-05-10-rooms-cvae5nfyxpartial/ckpt_last_seed_1234.pth.tar' --is-vanilla-training --y-cols 0 1 2 3 4 --dataset='dynamic' --dataset-path-csv='misc/dataset-04-rooms/rooms_xy_fake.csv' --datamode="gen-eval"

# python3 src/main.py --resume='ckpt/22-05-12-rooms-cvae8nf-onehot-full/ckpt_last_seed_1234.pth.tar' --is-vanilla-training --y-cols 0 1 2 3 4 --dataset='dynamic' --dataset-path-csv='misc/dataset-04-rooms/rooms_xy_fake.csv' --datamode="gen-eval"
