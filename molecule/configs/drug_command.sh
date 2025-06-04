# Unimol

# PCDM

python scripts/monitor.py monitor_interval=120 subprocess_args="pcdm_args.exp_name\=unimol_noise0.3_attn_block3 pcdm_args.attn_block_num\=3 pcdm_args.noise_sigma\=0.3 pcdm_args.encoder_type\=unimol pcdm_args.encoder_path\=./checkpoints/encoder_ckpts/unimol_drug.pt pcdm_args.rep_nf\=512" IAI=true IAI_job_name=drug nodelist=hgx001 time=72:00:00  gres="gpu:2" DDP=true DDP_port=29002 DDP_nproc_per_node=2 version_arg_name="pcdm_args.exp_name" resume_arg_name="pcdm_args.resume" last_ckpt_name=null script_name=self_condition_train_drug output_dir="outputs"


python scripts/monitor.py monitor_interval=120 subprocess_args="pcdm_args.exp_name\=unimol_noise0_attn_block3_attn_dropout0.5 pcdm_args.attn_block_num\=3 pcdm_args.noise_sigma\=0.0 pcdm_args.attn_dropout\=0.5 pcdm_args.encoder_type\=unimol pcdm_args.encoder_path\=./checkpoints/encoder_ckpts/unimol_drug.pt pcdm_args.rep_nf\=512 pcdm_args.resume\=outputs/unimol_noise0.3_attn_block3_middle" IAI=true IAI_job_name=drug nodelist=hgx001 time=72:00:00  gres="gpu:2" DDP=true DDP_port=29002 DDP_nproc_per_node=2 version_arg_name="pcdm_args.exp_name" resume_arg_name="pcdm_args.resume" last_ckpt_name=null script_name=self_condition_train_drug output_dir="outputs"

#RDM s

python scripts/monitor.py monitor_interval=120 subprocess_args="rdm_args.exp_name\=unimol_huge rdm_args.encoder_type\=unimol rdm_args.encoder_path\=./checkpoints/encoder_ckpts/unimol_drug.pt model_args.params.channels\=512 model_args.params.unet_config.params.in_channels\=512 model_args.params.unet_config.params.out_channels\=512 model_args.params.unet_config.params.time_embed_dim\=512 model_args.params.unet_config.params.num_res_blocks\=24 model_args.params.unet_config.params.context_channels\=512 model_args.params.cond_stage_config.params.embed_dim\=512" IAI=true IAI_job_name=drug_rdm nodelist=hgx006 time=72:00:00 gres="gpu:1" DDP=false version_arg_name="rdm_args.exp_name" resume_arg_name="rdm_args.rdm_ckpt" last_ckpt_name="model/checkpoint-last.pth" script_name=self_condition_train_drug_RDM output_dir="outputs/rdm"



#Frad

#PCDM

python scripts/monitor.py monitor_interval=120 subprocess_args="pcdm_args.exp_name\=unimol_noise0.3_attn_block3 pcdm_args.attn_block_num\=3 pcdm_args.noise_sigma\=0.3 pcdm_args.encoder_type\=frad pcdm_args.encoder_path\=./checkpoints/encoder_ckpts/Drug_epoch7.ckpt pcdm_args.rep_nf\=256" gres="gpu:2" DDP=true DDP_port=29001 DDP_nproc_per_node=2 version_arg_name="pcdm_args.exp_name" resume_arg_name="pcdm_args.resume" last_ckpt_name=null script_name=self_condition_train_drug time=72:00:00


#RDM 

python scripts/monitor.py monitor_interval=120 subprocess_args="rdm_args.exp_name\=frad_large model_args.params.channels\=256 model_args.params.unet_config.params.in_channels\=256 model_args.params.unet_config.params.time_embed_dim\=512 model_args.params.unet_config.params.out_channels\=256 model_args.params.unet_config.params.num_res_blocks\=28 model_args.params.unet_config.params.context_channels\=512 model_args.params.unet_config.params.model_channels\=2048 model_args.params.unet_config.params.bottleneck_channels\=2048 model_args.params.cond_stage_config.params.embed_dim\=512 rdm_args.encoder_type\=frad rdm_args.encoder_path\=./checkpoints/encoder_ckpts/Drug_epoch7.ckpt" gres="gpu:1" DDP=false version_arg_name="rdm_args.exp_name" resume_arg_name="rdm_args.rdm_ckpt" last_ckpt_name="model/checkpoint-last.pth" script_name=self_condition_train_drug_RDM output_dir="outputs/rdm" time=72:00:00 


