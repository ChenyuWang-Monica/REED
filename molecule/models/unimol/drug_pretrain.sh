data_path=/home/lizian/Self-Conditioned-3D-Diffusion/data/geom # replace to your data path
save_dir=/home/lizian/Self-Conditioned-3D-Diffusion/unimol_pretrain_save # replace to your save path
n_gpu=2
MASTER_PORT=10086
lr=1e-4
wd=1e-4
batch_size=64
update_freq=1
masked_token_loss=1
masked_coord_loss=5
masked_dist_loss=10
x_norm_loss=0.01
delta_pair_repr_norm_loss=0.01
mask_prob=0.15
only_polar=-1 # We use all H here.
noise_type="uniform"
noise=1.0
seed=1
warmup_steps=10000
max_steps=1000000
device_idx=0,1

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=$device_idx
python -m torch.distributed.run --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path  --user-dir /home/lizian/Self-Conditioned-3D-Diffusion/models/unimol/unimol --train-subset train --valid-subset valid \
       --num-workers 8 --ddp-backend=c10d \
       --task unimol_drug --loss unimol --arch unimol_base  \
       --optimizer adam --adam-betas "(0.9, 0.99)" --adam-eps 1e-6 --clip-norm 1.0 --weight-decay $wd \
       --lr-scheduler polynomial_decay --lr $lr --warmup-updates $warmup_steps --total-num-update $max_steps \
       --update-freq $update_freq --seed $seed \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --tensorboard-logdir $save_dir/tsb \
       --max-update $max_steps --log-interval 10 --log-format simple \
       --save-interval-updates 10000 --validate-interval-updates 10000 --keep-interval-updates 10 --no-epoch-checkpoints  \
       --masked-token-loss $masked_token_loss --masked-coord-loss $masked_coord_loss --masked-dist-loss $masked_dist_loss \
       --x-norm-loss $x_norm_loss --delta-pair-repr-norm-loss $delta_pair_repr_norm_loss \
       --mask-prob $mask_prob --noise-type $noise_type --noise $noise --batch-size $batch_size \
       --save-dir $save_dir  --only-polar $only_polar