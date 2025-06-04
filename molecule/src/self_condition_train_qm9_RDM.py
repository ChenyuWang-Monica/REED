import sys, os
sys.path.append(".")
import argparse
import numpy as np
import torch
import models.util.misc as misc
from models.engine_rdm import train_one_epoch
from omegaconf import OmegaConf
import torch.distributed as dist
import wandb
from initialize_models import initialize_RDM
from qm9 import dataset
import datetime
import time
from qm9.models import DistributionNodes, DistributionProperty
from configs.datasets_config import get_dataset_info
from models.rep_samplers import initilize_rep_sampler
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from qm9.utils import prepare_context, compute_mean_mad
from omegaconf.listconfig import ListConfig
from sklearn.metrics import silhouette_score


def vis_tsne(running_rdm_args, save_dir, epoch, n_datapoints=10000, device="cuda", inv_temp=None, running_batch_size=1000):

    dataloaders, charge_scale = dataset.retrieve_dataloaders(running_rdm_args)
    dataloader_train = dataloaders["train"]

    dataset_info = get_dataset_info(running_rdm_args.dataset, running_rdm_args.remove_h)

    histogram = dataset_info['n_nodes']
    in_node_nf = len(dataset_info['atom_decoder']) + int(running_rdm_args.include_charges)
    nodes_dist = DistributionNodes(histogram)

    prop_dist = None
    if len(running_rdm_args.conditioning) > 0:
        dataloaders, charge_scale = dataset.retrieve_dataloaders(running_rdm_args)
        prop_dist = DistributionProperty(dataloader_train, running_rdm_args.conditioning)
        property_norms = compute_mean_mad(dataloaders, running_rdm_args.conditioning, running_rdm_args.dataset)
        prop_dist.set_normalizer(property_norms)
    
    # GtSampler
    sampler = "GtSampler"
    Gt_dataset = "train"
    encoder_path = running_rdm_args.encoder_path


    rep_sampler_args = {
        "sampler": sampler,
        "Gt_dataset": Gt_dataset,
        "encoder_path": encoder_path,
        "encoder_type": running_rdm_args.encoder_type
    }
    rep_sampler_args = OmegaConf.create(rep_sampler_args)

    gtsampler = initilize_rep_sampler(rep_sampler_args, device, dataset_args=running_rdm_args)
    
    
    
    # PCSampler
    sampler = "PCSampler"
    rdm_ckpt = running_rdm_args.output_dir + "/checkpoint-last.pth"
    inv_temp = inv_temp
    n_steps = 5

    rep_sampler_args = {
        "sampler": sampler,
        "rdm_ckpt": rdm_ckpt,
        "inv_temp": inv_temp,
        "n_steps": n_steps,
        "snr": 0.01
    }
    rep_sampler_args = OmegaConf.create(rep_sampler_args)

    pcsampler = initilize_rep_sampler(rep_sampler_args, device, dataset_args=running_rdm_args)
    
    # Sampling
    print("Sampling GT Reps...")
    gt_nodesxsample = nodes_dist.sample(n_datapoints)
    gt_addtional_cond = None
    if prop_dist is not None:
        gt_addtional_cond = prop_dist.sample_batch(gt_nodesxsample)
    gt_reps = gtsampler.sample(
        device=device,
        nodesxsample=gt_nodesxsample,
        additional_cond=gt_addtional_cond,
        running_batch_size=running_batch_size,
    )
    gt_y = torch.zeros((gt_reps.shape[0]), device=device)
    print("Finished Sampling GT Reps.")
    
    print("Sampling PC Reps...")
    pc_nodesxsample = nodes_dist.sample(n_datapoints)
    pc_addtional_cond = None
    if prop_dist is not None:
        pc_addtional_cond = prop_dist.sample_batch(pc_nodesxsample)
    pc_reps = pcsampler.sample(
        device=device,
        nodesxsample=pc_nodesxsample,
        additional_cond=pc_addtional_cond,
        running_batch_size=running_batch_size,
    )
    pc_y = torch.ones((pc_reps.shape[0]), device=device)
    print("Finished Sampling PC Reps.")
    
    # Step 1: Combine representations and labels
    combined_reps = torch.cat((gt_reps, pc_reps), dim=0).cpu().numpy()
    combined_y = torch.cat((gt_y, pc_y), dim=0).cpu().numpy()

    # Step 2: Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(combined_reps)
    
    
    # now, calculate the SS score of the clustering result.
    silhouette_avg = silhouette_score(tsne_results, combined_y)
    silhouette_avg_4_abs = abs(silhouette_avg * 1e4)
    wandb.log({"SS_4_abs": silhouette_avg_4_abs}, commit=False)
    

    # Step 3: Visualize the results
    plt.figure(figsize=(20, 16))
    plt.scatter(tsne_results[combined_y == 0, 0], tsne_results[combined_y == 0, 1], label='gt_reps', alpha=0.6)
    plt.scatter(tsne_results[combined_y == 1, 0], tsne_results[combined_y == 1, 1], label='pc_reps', alpha=0.6)
    plt.legend()
    plt.title('t-SNE Visualization of gt_reps and pc_reps')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')


    save_path = f"{save_dir}/epoch{epoch}_inv_temp{inv_temp}.pdf"
    
    plt.savefig(save_path)
    plt.close()
        
def dist_setup():
    assert torch.cuda.device_count() > 1, "Only one cuda but using distributed training."
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=100))
    assert dist.is_initialized() and dist.is_available()
    rank, world_size = dist.get_rank(), dist.get_world_size()
    return rank, world_size
    
    
import hydra           
@hydra.main(config_path="../configs", config_name="qm9_rdm_config.yaml", version_base="1.3")
def main(args):
    OmegaConf.set_struct(args, False)
    args = args.qm9_rdm
    
    rdm_args = args.rdm_args
    model_args = args.model_args
    
    if type(model_args.params.cond_stage_key) is ListConfig:
        assert len(model_args.params.cond_stage_key) == 2, "cond_stage_key should only contain [node_num, property]"
        rdm_args.conditioning = [model_args.params.cond_stage_key[-1]]
        assert "node_num" not in rdm_args.conditioning, "Please place the property behind the node num in cond_stage_key."
    else:
        rdm_args.conditioning = []
    
    
    if rdm_args.debug:
        print("Warning: You are using the debug mode!!!")
        rdm_args.dp = False
        rdm_args.exp_name = "debug"
        rdm_args.no_wandb = True
    
    
    # Set up for DP
    if rdm_args.dp:
        rank, world_size = dist_setup()
        rdm_args.rank = rank
        rdm_args.world_size = world_size
    else:
        rank = 0
        world_size = 1
        rdm_args.rank = rank
        rdm_args.world_size = world_size
        
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda")
    dtype = torch.float32
    assert torch.cuda.is_available(), "Only support cuda training!"

    # Fix the seed for reproducibility
    seed = rdm_args.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)

    
    # Set up for the datasets, dataloaders, dataset_info
    data_loaders, charge_scale = dataset.retrieve_dataloaders(rdm_args)
    data_loader_train = data_loaders['train']
    data_dummy = next(iter(data_loader_train))
    
    
    # Set up for class_cond and lr and dirs
    rdm_args.class_cond = model_args.params.get("class_cond", False)
    assert rdm_args.class_cond == True, "At least, we need to condition on node number"
    
    
    eff_batch_size = rdm_args.batch_size * rdm_args.accum_iter * world_size
    assert rdm_args.lr is None, "We calculate the learning rate by blr."
    rdm_args.lr = rdm_args.blr * eff_batch_size
    rdm_args.output_dir = f'./outputs/rdm/{rdm_args.exp_name}/model'
    rdm_args.vis_output_dir = f'./outputs/rdm/{rdm_args.exp_name}/vis'
    rdm_args.log_dir = f'./outputs/rdm/{rdm_args.exp_name}/log'
    exp_dir = f'./outputs/rdm/{rdm_args.exp_name}'
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    if not os.path.exists(rdm_args.output_dir):
        os.mkdir(rdm_args.output_dir)
    if not os.path.exists(rdm_args.log_dir):
        os.mkdir(rdm_args.log_dir)
    if not os.path.exists(rdm_args.vis_output_dir):
        os.mkdir(rdm_args.vis_output_dir)
    
    # Set up for basic flowmodels
    model, model_without_ddp, loss_scaler, optimizer = initialize_RDM(rdm_args, model_args, device)

    
    # Set up for wandb logging
    if rank == 0:
        if rdm_args.no_wandb:
            mode = 'disabled'
        else:
            mode = 'online' if rdm_args.online else 'offline'
        kwargs = {'entity': rdm_args.wandb_usr, 'name': rdm_args.exp_name, 'project': 'e3_diffusion', 'config': {k: v for k, v in rdm_args.items()},
                'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
        wandb.init(**kwargs)
        wandb.save('*.txt')
        
        
    if rdm_args.rdm_ckpt is not None and rdm_args.rdm_ckpt != "":
        # When resuming, we reset the initial global step of wandb.
        global_step = rdm_args.start_epoch
        if rank == 0: wandb.log({}, step=global_step) 
    else: global_step = -1
    
    # Prepare conditioning variables
    if len(rdm_args.conditioning) > 0:
        property_norms = compute_mean_mad(data_loaders, rdm_args.conditioning, rdm_args.dataset)
    else:
        property_norms = None
    
    for epoch in range(rdm_args.start_epoch, rdm_args.epochs):
        data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args=rdm_args, property_norms=property_norms, dtype=dtype
        )
        
        if rank == 0:
            misc.save_model_last(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
            if rdm_args.output_dir and (epoch % 200 == 0 or epoch + 1 == rdm_args.epochs):
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)
                s = time.time()
                vis_tsne(running_rdm_args=rdm_args, save_dir=rdm_args.vis_output_dir, epoch=epoch, inv_temp=1.0, device=device)
                vis_tsne(running_rdm_args=rdm_args, save_dir=rdm_args.vis_output_dir, epoch=epoch, inv_temp=2.0, device=device)
                vis_tsne(running_rdm_args=rdm_args, save_dir=rdm_args.vis_output_dir, epoch=epoch, inv_temp=3.0, device=device)
                vis_tsne(running_rdm_args=rdm_args, save_dir=rdm_args.vis_output_dir, epoch=epoch, inv_temp=4.0, device=device)
                print(f"Visualization took {time.time() - s}s.")
            
            wandb.log(train_stats, commit=True)
        if rdm_args.dp:
            dist.barrier()




if __name__ == "__main__":
    main()
