#! /bin/bash
#SBATCH -J georcg_fm
#SBATCH -p IAI_SLURM_HGX
#SBATCH -N 1
#SBATCH --nodelist=hgx004
#SBATCH --qos=16gpu-hgx
#SBATCH --gres=gpu:2
#SBATCH --time=72:00:00
#SBATCH -o logs/%j.out.txt
#SBATCH -e logs/%j.err.txt
#SBATCH -c 1
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29002 \
    src/self_condition_train_drug_efm.py