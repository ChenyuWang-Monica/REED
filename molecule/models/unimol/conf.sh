data_path="./conformation_generation"  # replace to your data path
save_dir="./save_confgen"  # replace to your save path
n_gpu=1
MASTER_PORT=10086
dict_name="dict.txt"
weight_path="./ckpts/drugs_220908.pt"  # replace to your ckpt path
task_name="qm9"  # or "drugs", conformation generation task name, as a part of complete data path
recycles=4
coord_loss=1
distance_loss=1
beta=4.0
smooth=0.1
topN=20
lr=2e-5
batch_size=128
epoch=50
warmup=0.06
update_freq=1

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path --task-name $task_name --user-dir ./unimol --train-subset train --valid-subset valid \
       --num-workers 8 --ddp-backend=c10d \
       --task mol_confG --loss mol_confG --arch mol_confG  \
       --optimizer adam --adam-betas "(0.9, 0.99)" --adam-eps 1e-6 --clip-norm 1.0 \
       --lr-scheduler polynomial_decay --lr $lr --warmup-ratio $warmup --max-epoch $epoch --batch-size $batch_size \
       --update-freq $update_freq --seed 1 \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
       --log-interval 100 --log-format simple --tensorboard-logdir $save_dir/tsb \
       --validate-interval 1 --keep-last-epochs 10 \
       --keep-interval-updates 10 --best-checkpoint-metric loss  --patience 50 --all-gather-list-size 102400 \
       --finetune-mol-model $weight_path --save-dir $save_dir \
       --coord-loss $coord_loss --distance-loss $distance_loss \
       --num-recycles $recycles --beta $beta --smooth $smooth --topN $topN \
       --find-unused-parameters