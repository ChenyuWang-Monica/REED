import sys
sys.path.append(".")
import copy
import utils
import argparse
import wandb
from configs.datasets_config import get_dataset_info
from os.path import join
from qm9 import dataset
from qm9.models import get_optim, get_model
from equivariant_diffusion import en_diffusion
from equivariant_diffusion.utils import assert_correctly_masked
from equivariant_diffusion import utils as flow_utils
import torch
import time
import pickle
from train_test import train_epoch, test, analyze_and_save
import os
import torch
from omegaconf import OmegaConf
import torch.distributed as dist
import models.util.misc as misc
from models.rdm.models.diffusion.ddim import DDIMSampler
from utils import  reduced_mean
from qm9.utils import prepare_context, compute_mean_mad
import copy
import datetime
from omegaconf.listconfig import ListConfig
from models.rep_samplers import *
from models.encoders import initialize_encoder


def dist_setup():
    assert torch.cuda.device_count() > 1, "Only one cuda but using distributed training."
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=50))
    assert dist.is_initialized() and dist.is_available()
    rank, world_size = dist.get_rank(), dist.get_world_size()
    return rank, world_size

def check_mask_correct(variables, node_mask):
    for variable in variables:
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)
import hydra           
@hydra.main(config_path="../configs", config_name="qm9_pcdm_config.yaml", version_base="1.3")
def main(args):
    OmegaConf.set_struct(args, False)
    pcdm_args = args.pcdm_args

    
    if pcdm_args.debug:
        print("Warning: You are using the debug mode!!!")
        pcdm_args.dp = False
        pcdm_args.exp_name = "debug"
        pcdm_args.no_wandb = True

    
    # Set up for DP
    if pcdm_args.dp:
        rank, world_size = dist_setup()
        pcdm_args.rank = rank
        pcdm_args.world_size = world_size
        print("World_size", pcdm_args.world_size)
        print("Rank", pcdm_args.rank)
    else:
        rank = 0
        world_size = 1
        pcdm_args.rank = rank
        pcdm_args.world_size = world_size
        print("World_size", 1)
        print("Rank", 0)
        
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda")
    assert torch.cuda.is_available() # only support cuda training 
    
    
    # Set up for resume (NOTE: Careful with this!)
    if pcdm_args.resume is not None:
        with open(join(pcdm_args.resume, 'args.pickle'), 'rb') as f:
            resumed_args = pickle.load(f)
        new_args = copy.deepcopy(args)

        current_epoch = resumed_args.pcdm_args.current_epoch
        resumed_exp_name = resumed_args.pcdm_args.exp_name
        
        
        # Now, compare them. You should yourself ensure the same batch size
        if new_args.pcdm_args != resumed_args.pcdm_args:
            print(
                f"WARNING: detected difference in pcdm args between current args and the resumed args:\n"
            )
            union_keys = list(set(new_args.pcdm_args.keys()).union(set(resumed_args.pcdm_args.keys()))) 
            for key in union_keys:
                if new_args.pcdm_args.get(key, "UNKNOWN") != resumed_args.pcdm_args.get(key, "UNKNOWN"):
                    print(f"     Different in {key}: new_args {new_args.pcdm_args.get(key, 'UNKNOWN')}, resumed_args {resumed_args.pcdm_args.get(key, 'UNKNOWN')}")

            
        # pcdm_args.exp_name = resumed_exp_name + "_resume"
        pcdm_args.start_epoch = current_epoch
        

    if rank == 0:
        if pcdm_args.no_wandb:
            mode = 'disabled'
        else:
            mode = 'online' if pcdm_args.online else 'offline'
        kwargs = {'entity': pcdm_args.wandb_usr, 'name': pcdm_args.exp_name, 'project': 'e3_diffusion', 'config': {k: v for k, v in args.items()},
                'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
        wandb.init(**kwargs)
        wandb.save('*.txt')
    import setproctitle
    setproctitle.setproctitle(f"{pcdm_args.exp_name}")
    
    # Set up for encoder
    encoder = initialize_encoder(encoder_type=pcdm_args.encoder_type, device=device, encoder_ckpt_path=pcdm_args.encoder_path)
    
    if pcdm_args.finetune_encoder:
        print("\n\nIMPORTANT: You are finetuning the encoder in diffusion tasks!!!\n\n")
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
        encoder_optimizer = AdamW(
            encoder.parameters(),
            lr=pcdm_args.encoder_lr,
            weight_decay=pcdm_args.encoder_weight_decay,
        )
        encoder_scheduler = ReduceLROnPlateau(
            encoder_optimizer,
            "min",
            factor=pcdm_args.encoder_factor,
            patience=pcdm_args.encoder_patience,
            min_lr=pcdm_args.encoder_min_lr,
        )
        
        
        # We change the EGNN for denoising to a light output head, so that the fine-tuning is concentrated on the encoder.
        pcdm_args.n_layers = pcdm_args.light_n_layers
        pcdm_args.nf = pcdm_args.light_nf
        # We use no regularization during encoder finetuning.
        if pcdm_args.attn_dropout > 0. or pcdm_args.noise_sigma > 0.:
            print(f"Warning: You are finetuning the encoder with attn_dropout {pcdm_args.attn_dropout} and noise_sigma {pcdm_args.noise_sigma}. We by default set them to 0.")
            pcdm_args.attn_dropout = 0.
            pcdm_args.noise_sigma = 0.
        
        
    else:
        for param in encoder.parameters():
            param.requires_grad = False
        encoder.eval()
        encoder_optimizer = None
        encoder_scheduler = None
        
    if pcdm_args.finetune_encoder and pcdm_args.dp:
        encoder_dp = torch.nn.parallel.DistributedDataParallel(encoder, find_unused_parameters=True)
        assert encoder_dp.module is encoder
    else:
        encoder_dp = encoder
    
    
    

    # Set up for the datasets
    dataset_info = get_dataset_info(pcdm_args.dataset, pcdm_args.remove_h)
    dataloaders, charge_scale = dataset.retrieve_dataloaders(pcdm_args, which_split=pcdm_args.get("which_split", "edm"))
    data_dummy = next(iter(dataloaders['train']))
    
    
    # Set up for conditioning
    if len(pcdm_args.conditioning) > 0:
        if rank == 0:
            print(f'Conditioning on {pcdm_args.conditioning}')
        property_norms = compute_mean_mad(dataloaders, pcdm_args.conditioning, pcdm_args.dataset)
        context_dummy = prepare_context(pcdm_args.conditioning, data_dummy, property_norms)
        context_node_nf = context_dummy.size(2)
        
    else:
        property_norms = None
        context_node_nf = 0
    pcdm_args.context_node_nf = context_node_nf
        
    # Set up for point cloud diffusion model
    model, nodes_dist, prop_dist = get_model(pcdm_args, device, dataset_info, dataloaders['train'])
        
    if prop_dist is not None:
        prop_dist.set_normalizer(property_norms)
    model = model.to(device)
    optim = get_optim(pcdm_args, model)
    
    
    if pcdm_args.dp:
        model_dp = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        model_dp = model_dp.cuda()
    else:
        model_dp = model

    if pcdm_args.ema_decay > 0:
        model_ema = copy.deepcopy(model)
        ema = flow_utils.EMA(pcdm_args.ema_decay)

        if pcdm_args.dp and torch.cuda.device_count() > 1:
            model_ema_dp = torch.nn.parallel.DistributedDataParallel(model_ema, find_unused_parameters=True)
        else:
            model_ema_dp = model_ema
    else:
        ema = None
        model_ema = model
        model_ema_dp = model_dp
    
    model_ema.eval()
    model_ema_dp.eval()

    if pcdm_args.resume is not None:
        flow_state_dict = torch.load(join(pcdm_args.resume, 'generative_model.npy'))
        ema_flow_state_dict = torch.load(join(pcdm_args.resume, 'generative_model_ema.npy'))
        optim_state_dict = torch.load(join(pcdm_args.resume, 'optim.npy'))
        model.load_state_dict(flow_state_dict)
        optim.load_state_dict(optim_state_dict)
        model_ema.load_state_dict(ema_flow_state_dict)


    # Set up the sampler
    rep_sampler = initilize_rep_sampler(pcdm_args, device, pcdm_args)
    from models.wrapper import SelfConditionWrappedSampler
    sampler = SelfConditionWrappedSampler(pcdm_sampler=model_ema, rdm_sampler=rep_sampler)


    # Other preparations
    utils.create_folders(pcdm_args)
    dtype = torch.float32
    
    best_nll_val = 1e8
    best_nll_test = 1e8
    
    gradnorm_queue = utils.Queue()
    gradnorm_queue.add(3000)  # Add large value that will be flushed.
    
    if pcdm_args.finetune_encoder:
        encoder_gradnorm_queue = utils.Queue()
        encoder_gradnorm_queue.add(3000)  # Add large value that will be flushed.
    else:
        encoder_gradnorm_queue=None
    # Print meta informations
    if rank == 0:
        print(f"Args: {args}")
        print(f"Training Using {world_size} GPUs")
        print(f"Point Cloud Diffusion Model: {model}")
        print(f"Encoder Model: {encoder}")
    santi_check_about_sampling = False
    if pcdm_args.resume is not None:
        # When resuming, we reset the initial global step of wandb. But notice that the global step calculated in the following way is not very precise.
        global_step = (pcdm_args.start_epoch) * len(dataloaders["train"]) 
        if rank == 0: wandb.log({}, step=global_step) 
        santi_check_about_sampling = True
    else: global_step = -1
    
    for epoch in range(pcdm_args.start_epoch, pcdm_args.n_epochs):
        start_epoch = time.time()
        train_epoch(args=pcdm_args, loader=dataloaders['train'], epoch=epoch, model=model, model_dp=model_dp,
                    model_ema=model_ema, ema=ema, device=device, dtype=dtype, property_norms=property_norms,
                    nodes_dist=nodes_dist, dataset_info=dataset_info,
                    gradnorm_queue=gradnorm_queue, optim=optim, prop_dist=prop_dist,
                    encoder=encoder_dp, rank=rank, encoder_optimizer=encoder_optimizer, encoder_dp=encoder_dp, encoder_gradnorm_queue=encoder_gradnorm_queue, sampler=sampler
                    )
        print(f"Epoch took {time.time() - start_epoch:.1f} seconds.")
        if epoch % pcdm_args.test_epochs == 0 or santi_check_about_sampling:
            santi_check_about_sampling = False
            if isinstance(model, en_diffusion.EnVariationalDiffusion) and rank == 0:
                wandb.log(model.log_info(), commit=True)

            if not pcdm_args.break_train_epoch and rank == 0  and not pcdm_args.finetune_encoder and len(pcdm_args.conditioning) == 0:
                analyze_and_save(args=pcdm_args, epoch=epoch, model_sample=sampler, nodes_dist=nodes_dist,
                                 dataset_info=dataset_info, device=device,
                                 prop_dist=prop_dist, n_samples=pcdm_args.n_stability_samples)
            if pcdm_args.dp: dist.barrier()
            nll_epoch_val, n_samples_val = test(args=pcdm_args, loader=dataloaders['valid'], epoch=epoch, eval_model=model_ema_dp,
                           partition='Val', device=device, dtype=dtype, nodes_dist=nodes_dist,
                           property_norms=property_norms, encoder=encoder, rank=rank)
            nll_epoch_test, n_samples_test = test(args=pcdm_args, loader=dataloaders['test'], epoch=epoch, eval_model=model_ema_dp,
                            partition='Test', device=device, dtype=dtype,
                            nodes_dist=nodes_dist, property_norms=property_norms, encoder=encoder, rank=rank)
            
            if pcdm_args.dp:
                nll_val = reduced_mean(nll_epoch_val, n_samples_val)
                nll_test = reduced_mean(nll_epoch_test, n_samples_test)
            else:
                nll_val = nll_epoch_val / n_samples_val
                nll_test = nll_epoch_test / n_samples_test
            
            if pcdm_args.finetune_encoder:
                encoder_scheduler.step(nll_val)
            
            if rank == 0:
                if nll_val < best_nll_val:
                    best_nll_val = nll_val
                    best_nll_test = nll_test
                    if pcdm_args.save_model and rank == 0:
                        pcdm_args.current_epoch = epoch + 1
                        utils.save_model(optim, 'outputs/%s/optim.npy' % pcdm_args.exp_name)
                        utils.save_model(model, 'outputs/%s/generative_model.npy' % pcdm_args.exp_name)
                        if pcdm_args.ema_decay > 0:
                            utils.save_model(model_ema, 'outputs/%s/generative_model_ema.npy' % pcdm_args.exp_name)
                        with open('outputs/%s/args.pickle' % pcdm_args.exp_name, 'wb') as f:
                            pickle.dump(args, f)
                        
                        if pcdm_args.finetune_encoder:
                            encoder_checkpoint = {
                                'state_dict': encoder.state_dict(),
                            }
                            torch.save(encoder_checkpoint, pcdm_args.finetune_save_path)

                    if pcdm_args.save_model and rank == 0:
                        utils.save_model(optim, 'outputs/%s/optim_%d.npy' % (pcdm_args.exp_name, epoch))
                        utils.save_model(model, 'outputs/%s/generative_model_%d.npy' % (pcdm_args.exp_name, epoch))
                        if pcdm_args.ema_decay > 0:
                            utils.save_model(model_ema, 'outputs/%s/generative_model_ema_%d.npy' % (pcdm_args.exp_name, epoch))
                        with open('outputs/%s/args_%d.pickle' % (pcdm_args.exp_name, epoch), 'wb') as f:
                            pickle.dump(args, f)
                print('Reduced mean Val loss: %.4f \t Reduced mean Test loss:  %.4f' % (nll_val, nll_test))
                print('Reduced mean Best val loss: %.4f \t Reduced mean Best test loss:  %.4f' % (best_nll_val, best_nll_test))
                
                wandb.log({"Val loss ": nll_val})
                wandb.log({"Test loss ": nll_test})
                wandb.log({"Best cross-validated test loss ": best_nll_test})
            if pcdm_args.dp: dist.barrier()


if __name__ == "__main__":
    main()
