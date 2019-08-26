#!/bin/sh

#SBATCH --job-name="train_DCRSR"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=argon-tesla1
#SBATCH --gres=gpu:tesla:2
srun python3 main.py --model DCRSR --scale 2 --save dcrsr_x2_3_3_64  --n_dresblocks 3 --patch_size 64  --n_resblocks 3 --n_feats 64 --res_scale 0.1 --reset --n_GPUs 2 --batch_size 8

#srun python3 main.py --scale 2  --cpu  --model XXEDSR --pre_train ../experiment/ddxxedsr_x2_4_3_512_half_NTIRE/model/model_latest.pt  --n_dresblocks 4 --test_only --save_results  --data_test Demo --n_resblocks 3 --n_feats 512 --res_scale 0.1 --self_ensemble
