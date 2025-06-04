import wandb
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked, sample_center_gravity_zero_gaussian_with_mask
import numpy as np
import qm9.visualizer as vis
from qm9.analyze import analyze_stability_for_molecules
from qm9.sampling import sample_chain, sample, sample_sweep_conditional
import utils
import qm9.utils as qm9utils
from qm9 import losses
import time
import torch
from utils import dist_wandb_log, dist_print, reduced_mean
from models.encoders import get_global_representation
import torch.distributed as dist
from torch.nn import functional as F



def train_epoch(args, loader, epoch, model, model_dp, model_ema, ema, device, dtype, property_norms, optim,
                nodes_dist, gradnorm_queue, dataset_info, prop_dist, encoder, encoder_dp, sampler, rank, encoder_optimizer=None, encoder_gradnorm_queue=None):
    model_dp.train()
    model.train()
    if args.finetune_encoder:
        encoder.train()
        encoder_dp.train()
    else:
        encoder.eval()
        encoder_dp.eval()
    nll_epoch = []
    n_iterations = len(loader)
    loader.sampler.set_epoch(epoch)
    for i, data in enumerate(loader):
        
        x = data['positions'].to(device, dtype)
        node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
        edge_mask = data['edge_mask'].to(device, dtype)
        one_hot = data['one_hot'].to(device, dtype)
        charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)
        
            

        x = remove_mean_with_mask(x, node_mask)

        if args.augment_noise > 0:
            # Add noise eps ~ N(0, augment_noise) around points.
            eps = sample_center_gravity_zero_gaussian_with_mask(x.size(), x.device, node_mask)
            x = x + eps * args.augment_noise

        x = remove_mean_with_mask(x, node_mask)
        if args.data_augmentation:
            x = utils.random_rotation(x).detach()

        check_mask_correct([x, one_hot, charges], node_mask)
        assert_mean_zero_with_mask(x, node_mask)
        if not args.include_charges:
            assert len(data['charges']) > 0 and len(data['charges'].unique().flatten().tolist()) > 0, "When not including charges, please still ensure we can access atom types from data, so that representations can be calculated."
            h = {'categorical': one_hot, 'integer': charges, 'rep_integer': data['charges'].to(device, dtype)} # This is only for rep calculation.
        else:
            h = {'categorical': one_hot, 'integer': charges}

        if len(args.conditioning) > 0:
            context = qm9utils.prepare_context(args.conditioning, data, property_norms).to(device, dtype)
        else:
            context = None
            
        rep = get_global_representation(node_mask, encoder_dp, x, h, training_encoder=args.finetune_encoder, noise_sigma=args.noise_sigma)
        assert (rep.requires_grad and args.finetune_encoder) or (not rep.requires_grad and not args.finetune_encoder)
        
        
        
        optim.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()

        # transform batch through flow
        nll, reg_term, mean_abs_z, denoised_xh = losses.compute_loss_and_nll(args, model_dp, nodes_dist,
                                                                x, h, node_mask, edge_mask, context, rep=rep, no_logpn=args.finetune_encoder)
        
        # Compute auxiliary loss: denoised rep alignment loss
        if args.rep_align_loss > 0.:
            assert 0, "Not used."
            assert args.noise_sigma == 0. # Not carefully designed
            assert not args.finetune_encoder # Since if encoder is not fixed, a large rep aligment loss could urge the encoder to produce uniform representations for all structures, which is not what we want
            
            
            denoised_x = denoised_xh[:, :, :3]
            denoised_h_int = denoised_xh[:, :, -1:] if args.include_charges else torch.zeros(0).to(denoised_xh.device)
            
            unnorm_func = model_dp.unnormalize if not args.dp else model_dp.module.unnormalize
            
            denoised_x, _, _ = unnorm_func(denoised_x, denoised_xh[:, :, 3:-1], denoised_h_int, node_mask)
            
            # NOTE: We use original h for denoised rep calculation, since argmax stops gradient. (But actually we can adopt Straight-through estimator to estimate the gradient. Implement later if necessary.)
            denoised_rep = get_global_representation(node_mask, encoder_dp, denoised_x, h, training_encoder=args.finetune_encoder)
        
            rep_l2_loss = F.mse_loss(denoised_rep, rep)
        else:
            rep_l2_loss = torch.tensor(0.)
        
        # standard nll from forward KL
        loss = nll + args.ode_regularization * reg_term
        
        loss = loss + rep_l2_loss * args.rep_align_loss
        
        loss.backward()

        if args.clip_grad:
            grad_norm = utils.gradient_clipping(model, gradnorm_queue)
            if args.finetune_encoder:
                encoder_gradnorm = utils.gradient_clipping(encoder, encoder_gradnorm_queue) 
            else:
                encoder_gradnorm = 0.
        else:
            grad_norm = 0.
            encoder_gradnorm = 0.
        optim.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()
        
        # Update EMA if enabled.
        if args.ema_decay > 0:
            ema.update_model_average(model_ema, model)

        nll_epoch.append(nll.item())
        
        if i % args.n_report_steps == 0:
            print(f"\rEpoch: {epoch}, rank: {rank}, iter: {i}/{n_iterations}, "
                f"Loss {loss.item():.2f}, NLL: {nll.item():.2f}, "
                f"RegTerm: {reg_term.item():.1f}, "
                f"GradNorm: {grad_norm:.1f}, "
                f"Encoder GradNorm: {encoder_gradnorm:.3f}",
                f"Rep Align Loss: {rep_l2_loss:.3f}")
        
        if rank == 0:
            wandb.log({"Batch NLL": nll.item()}, commit=False)
            wandb.log({"Rep Align Loss": rep_l2_loss.item()}, commit=True)
            if (epoch % args.test_epochs == 0) and (i % 1e8 == 0) and not (epoch == 0 and i == 0) and not args.finetune_encoder and len(args.conditioning) == 0: # Only perform in time evaluation when unconditionally training
                # If sample from train dataset, then visulization is not supported.
                start = time.time()
                save_and_sample_chain(sampler, args, device, dataset_info, prop_dist, epoch=epoch)
                sample_different_sizes_and_save(sampler, nodes_dist, args, device, dataset_info,
                                                prop_dist, epoch=epoch)
                
                vis.visualize(f"outputs/{args.exp_name}/epoch_{epoch}", dataset_info=dataset_info, wandb=wandb)
                vis.visualize_chain(f"outputs/{args.exp_name}/epoch_{epoch}/chain/", dataset_info, wandb=wandb)
                print(f'Sampling took {time.time() - start:.2f} seconds')

                if len(args.conditioning) > 0:
                    vis.visualize_chain("outputs/%s/epoch_%d/conditional/" % (args.exp_name, epoch), dataset_info,
                                        wandb=wandb, mode='conditional')
        if args.dp: dist.barrier()
        
        if args.break_train_epoch:
            break
        
    
    
    # NOTE: This logging is not carefully treated. We are only monitoring rank 0's metrics without reduction. 
    #       Nevertheless, this is OK since we only need a coarse track for training metrics.
    if rank == 0:
        wandb.log({"Train Epoch NLL": np.mean(nll_epoch)}, commit=False)

def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def test(args, loader, epoch, eval_model, device, dtype, property_norms, nodes_dist, encoder, partition='Test',  rank=None):
    assert encoder is not None
    assert rank is not None
    eval_model.eval()
    eval_model.cal_nll = True
    with torch.no_grad():
        nll_epoch = 0
        n_samples = 0
        n_iterations = len(loader)

        for i, data in enumerate(loader):
            x = data['positions'].to(device, dtype)
            batch_size = x.size(0)
            node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
            edge_mask = data['edge_mask'].to(device, dtype)
            one_hot = data['one_hot'].to(device, dtype)
            charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

            if args.augment_noise > 0:
                # Add noise eps ~ N(0, augment_noise) around points.
                eps = sample_center_gravity_zero_gaussian_with_mask(x.size(),
                                                                    x.device,
                                                                    node_mask)
                x = x + eps * args.augment_noise

            x = remove_mean_with_mask(x, node_mask)
            check_mask_correct([x, one_hot, charges], node_mask)
            assert_mean_zero_with_mask(x, node_mask)

            if not args.include_charges:
                assert len(data['charges']) > 0 and len(data['charges'].unique().flatten().tolist()) > 0, "When not including charges, please still ensure we can access the real charges from data, so that representations can be calculated."
                h = {'categorical': one_hot, 'integer': charges, 'rep_integer': data['charges'].to(device, dtype)} # This is only for rep calculation.
            else:
                h = {'categorical': one_hot, 'integer': charges}

            if len(args.conditioning) > 0:
                context = qm9utils.prepare_context(args.conditioning, data, property_norms).to(device, dtype)
            else:
                context = None
            rep = get_global_representation(node_mask, encoder, x, h)
                
            # transform batch through flow
            nll, _, _ = losses.compute_loss_and_nll(args, eval_model, nodes_dist, x, h,
                                                    node_mask, edge_mask, context, rep=rep)
            # standard nll from forward KL

            nll_epoch += nll.item() * batch_size
            n_samples += batch_size
            
            if args.dp:
                reduced_mean_ = reduced_mean(nll_epoch, n_samples)
            else:
                reduced_mean_ = nll_epoch / n_samples
            if i % args.n_report_steps == 0 and rank == 0:
                print(f"\r {partition} NLL \t epoch: {epoch}, iter: {i}/{n_iterations}, "
                      f"Reduced mean NLL: {reduced_mean_:.2f}")
    eval_model.cal_nll = False
    

    
    return nll_epoch, n_samples


def save_and_sample_chain(model, args, device, dataset_info, prop_dist,
                          epoch=0, id_from=0):
    one_hot, charges, x = sample_chain(args=args, device=device, flow=model,
                                       n_tries=1, dataset_info=dataset_info, prop_dist=prop_dist)

    vis.save_xyz_file(f'outputs/{args.exp_name}/epoch_{epoch}/chain/',
                      one_hot, charges, x, dataset_info, id_from, name='chain')

    return one_hot, charges, x


def sample_different_sizes_and_save(model, nodes_dist, args, device, dataset_info, prop_dist,
                                    n_samples=5, epoch=0, batch_size=10):
    batch_size = min(batch_size, n_samples)
    for counter in range(int(n_samples/batch_size)):
        nodesxsample = nodes_dist.sample(batch_size)
        one_hot, charges, x, node_mask = sample(args, device, model, prop_dist=prop_dist,
                                                nodesxsample=nodesxsample,
                                                dataset_info=dataset_info)
        print(f"Generated molecule: Positions {x[:-1, :, :]}")
        vis.save_xyz_file(f'outputs/{args.exp_name}/epoch_{epoch}/', one_hot, charges, x, dataset_info,
                          batch_size * counter, name='molecule')


def analyze_and_save(epoch, model_sample, nodes_dist, args, device, dataset_info, prop_dist,
                     n_samples=1000, batch_size=100, tolog=True):
    
    s = time.time()
    print(f'Analyzing molecule stability at epoch {epoch}...')
    batch_size = min(batch_size, n_samples)
    assert n_samples % batch_size == 0
    molecules = {'one_hot': [], 'x': [], 'node_mask': []}
    for i in range(int(n_samples/batch_size)):
        print(f"Analyze and save: Sampling process {i}/{n_samples/batch_size}")
        
        nodesxsample = nodes_dist.sample(batch_size)
        
        one_hot, charges, x, node_mask = sample(args, device, model_sample, dataset_info, prop_dist,
                                                nodesxsample=nodesxsample)

        molecules['one_hot'].append(one_hot.detach().cpu())
        molecules['x'].append(x.detach().cpu())
        molecules['node_mask'].append(node_mask.detach().cpu())
    print(f"Sample End, took time {time.time() - s}")
    molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}
    print(f"Analyzing...")
    validity_dict, rdkit_tuple = analyze_stability_for_molecules(molecules, dataset_info)
    print(f"Analyze End.")

    assert rdkit_tuple is not None, "Please install rdkit for more eval information"
    if tolog:
        wandb.log(validity_dict)
        if rdkit_tuple is not None:
            wandb.log({'Validity': rdkit_tuple[0][0], 'Uniqueness': rdkit_tuple[0][1], 'Novelty': rdkit_tuple[0][2]})
    return validity_dict

