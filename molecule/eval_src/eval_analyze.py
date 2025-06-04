import sys
sys.path.append(".")
import argparse
import os
import torch
import time
from qm9.sampling import sample
from models.rep_samplers import *
from eval_src.eval_utils import prepare_model_and_dataset_info, analyze_all_metrics
import logging
import torch.distributed as dist
import datetime
from pathlib import Path

def sample_loop(
    pcdm_args, device, generative_model,
    nodes_dist, prop_dist, dataset_info, n_samples=10,
    batch_size=10, save_molecules=False, pcdm_model_path=None, return_prop=False
    ):
    
    batch_size = min(batch_size, n_samples)
    assert n_samples % batch_size == 0
    molecules = {'one_hot': [], 'x': [], 'node_mask': []}
    start_time = time.time()
    if return_prop:
        prop = []
    
    for i in range(int(n_samples/batch_size)):
        nodesxsample = nodes_dist.sample(batch_size)
        if i == 0:
            logging.info(nodesxsample[:50])
        if prop_dist is not None:
            rep_context = prop_dist.sample_batch(nodesxsample).to(device)
            assert len([key for key in prop_dist.distributions.keys()]) == 1
            property_name = [key for key in prop_dist.distributions.keys()][0]
            if return_prop:
                prop.extend((rep_context * prop_dist.normalizer[property_name]["mad"] + prop_dist.normalizer[property_name]["mean"]).flatten().cpu().tolist())
        else:
            rep_context = None
        
        
        one_hot, charges, x, node_mask = sample(
            pcdm_args, device, generative_model, dataset_info, prop_dist=None, nodesxsample=nodesxsample,
            rep_context=rep_context, context=None
            )

        molecules['one_hot'].append(one_hot.detach().cpu())
        molecules['x'].append(x.detach().cpu())
        molecules['node_mask'].append(node_mask.detach().cpu())


    molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}

    if save_molecules: # NOTE: This saves molecules in .pth form (dict), not xyz. The dict is like {'one_hot': tensor of shape (sample_num, max_node_num, one_hot_dim), 'x': ..., 'node_mask': ...}. Please manually convert it to xyz or other files you like, with the specific dataset_info.
        assert pcdm_model_path is not None, "Please specify the pcdm_model_path for molecule saving!"
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        save_molecules_file_name = f"molecules_{n_samples}_{timestamp}.pt"
        assert Path(pcdm_model_path).exists(), f"Please ensure {Path(pcdm_model_path)}'s existance!"
        if not Path(pcdm_model_path).is_dir():
            save_molecules_dir = Path(pcdm_model_path).parent / "eval"/ "molecules"
        else:
            save_molecules_dir = Path(pcdm_model_path) / "eval"/  "molecules"
        save_molecules_dir.mkdir(parents=True, exist_ok=True)
        save_molecules_path = save_molecules_dir / save_molecules_file_name
            
        torch.save(molecules, save_molecules_path)
        print(f"Molecules have been saved to {save_molecules_path}"); logging.info(f"Molecules have been saved to {save_molecules_path}")
    
    if return_prop:
        return molecules, prop
    
    return molecules





def molecule_sampling(eval_args, device, self_conditioned_sampler, pcdm_args, nodes_dist, prop_dist, dataset_info):
    molecules = sample_loop(
        pcdm_args, 
        device, 
        self_conditioned_sampler,
        nodes_dist, 
        prop_dist, 
        dataset_info, 
        n_samples=eval_args.n_samples,
        batch_size=eval_args.batch_size_gen, 
        save_molecules=eval_args.save_molecules,
        pcdm_model_path=eval_args.pcdm_model_path
        )
    return molecules




def molecule_sampling_ddp(eval_args, device, self_conditioned_sampler, pcdm_args, nodes_dist, prop_dist, dataset_info):
    def dist_setup():
        assert torch.cuda.device_count() > 1, "Only one cuda but using distributed training."
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=4320))
        assert dist.is_initialized() and dist.is_available()
        rank, world_size = dist.get_rank(), dist.get_world_size()
        return rank, world_size
    
    def get_common_timestamp(rank, device):
        if rank == 0:
            # Only rank 0 gets the current timestamp
            timestamp = datetime.datetime.now().timestamp()
            timestamp_tensor = torch.tensor([timestamp], dtype=torch.float64).to(device)
        else:
            timestamp_tensor = torch.tensor([0.0], dtype=torch.float64).to(device)

        # Broadcast the timestamp from rank 0 to all other ranks
        dist.broadcast(timestamp_tensor, src=0)

        # Convert the tensor back to a datetime object
        common_timestamp = datetime.datetime.fromtimestamp(timestamp_tensor.item())
        # Format the common timestamp as a string (e.g., "YYYY-MM-DD HH:MM:SS")
        formatted_timestamp = common_timestamp.strftime("%Y%m%d%H%M%S")
    
        return formatted_timestamp
    
    
    def main_worker(rank, world_size, eval_args, device):        


        # Prepare the model and dataset information
        self_conditioned_sampler, pcdm_args, nodes_dist, prop_dist, dataset_info = prepare_model_and_dataset_info(eval_args, device)
        
        # Wrap the model with DistributedDataParallel
        generative_model = self_conditioned_sampler.to(device)

        # Calculate the number of samples each process will handle
        samples_per_rank = eval_args.n_samples // world_size

        # Perform the sampling
        molecules = sample_loop(
            pcdm_args, device, generative_model,
            nodes_dist, prop_dist, dataset_info,
            n_samples=samples_per_rank, batch_size=eval_args.batch_size_gen
        )

        # Gather results from all processes
        time_str = eval_args.time_str
        if not Path("./temp").exists():
            Path("./temp").mkdir(parents=True, exist_ok=True)
        temp_file = f'./temp/{time_str}_molecules_{samples_per_rank}_rank_{rank}.pkl'
        with open(temp_file, 'wb') as f:
            torch.save(molecules, f)

        dist.barrier()  # Synchronize all processes
        
        # To gather and combine results
        if rank == 0:
            logging.info(f"All processes have finished sampling. Now rank 0 will analyse the sampled molecules.")
            all_molecules = {'one_hot': [], 'x': [], 'node_mask': []}
            for i in range(world_size):
                try:
                    with open(f'./temp/{time_str}_molecules_{samples_per_rank}_rank_{i}.pkl', 'rb') as f:
                        rank_molecules = torch.load(f)
                        for key in all_molecules:
                            all_molecules[key].append(rank_molecules[key])
                    logging.info(f"Successfully loaded ./temp/{time_str}_molecules_{samples_per_rank}_rank_{i}.pkl")
                except Exception as e:
                    logging.info(f"Meet exception {e} when loading ./temp/{time_str}_molecules_{samples_per_rank}_rank_{i}.pkl. Dropping it.")
                    
            all_molecules = {key: torch.cat(all_molecules[key], dim=0) for key in all_molecules}
            logging.info(f"Gathered {all_molecules['x'].shape[0]} molecules.")
            if all_molecules['x'].shape[0] != eval_args.n_samples:
                logging.info(f"Warning: only gathered {all_molecules['x'].shape[0]}/{eval_args.n_samples} molecules.")
            if eval_args.save_molecules:
                pcdm_model_path = eval_args.pcdm_model_path
                assert pcdm_model_path is not None, "Please specify the pcdm_model_path for molecule saving!"
                timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
                if not Path(pcdm_model_path).is_dir():
                    save_molecules_path = Path(pcdm_model_path).parent / f"molecules_{eval_args.n_samples}_{timestamp}.pt"
                else:
                    save_molecules_path = Path(pcdm_model_path) / f"molecules_{eval_args.n_samples}_{timestamp}.pt"
                torch.save(all_molecules, save_molecules_path)
                logging.info(f"Molecules have been saved to {save_molecules_path}")
            
            # Cleanup
            for i in range(world_size):
                try:
                    os.remove(f'./temp/{time_str}_molecules_{samples_per_rank}_rank_{i}.pkl')
                except Exception as e:
                    print(f"Meet exception {e} when removing ./temp/{time_str}_molecules_{samples_per_rank}_rank_{i}.pkl.")
        else:
            all_molecules = None
        dist.barrier()
        return all_molecules
            
    rank, world_size = dist_setup()
    assert eval_args.n_samples % world_size == 0, "Better to ensure that n_samples % world_size == 0, to avoid potential inconsistency."
    # Set device
    torch.cuda.set_device(rank)
    device = "cuda"
    # Gather a common time_str as evaluation identifier
    time_str = get_common_timestamp(rank, device)
    logging.info(f"The file identifier of rank {rank} is {time_str}")
    eval_args.time_str = time_str
    logging.info(f"{rank + 1}/{world_size} initizalized.")
    
    all_molecules = main_worker(rank, world_size, eval_args, device)
    return all_molecules, rank

import hydra





@hydra.main(config_path="../configs", config_name="eval_unconditional.yaml", version_base='1.3')
def eval(args):
    OmegaConf.set_struct(args, False)
    # Initialize Sampler and dataset info. 
    device = "cuda"
    self_conditioned_sampler, pcdm_args, nodes_dist, prop_dist, dataset_info = prepare_model_and_dataset_info(args, device, only_dataset_info=(args.saved_molecules_path is not None)) # If one load saved molecules, then only dataset info is needed.
    
    logging.info(torch.seed())
    # Load molecules
    if args.saved_molecules_path is not None:
        # For saved molecules
        assert not args.use_dist, "Distributed mode is only used for molecule sampling. When you use saved molecules, there is no need to launch distributed mode!"
        saved_molecules_path = Path(args.saved_molecules_path)
        assert saved_molecules_path.is_file() and str(saved_molecules_path).endswith(".pt") and saved_molecules_path.exists(), "Please ensure the existence of the molecule"
        with open(saved_molecules_path, "rb") as f:
            molecules = torch.load(f, map_location=device)
        rank = 0
    else:
        # For molecule sampling
        if args.use_dist:
            logging.info("You are using DDP sampling!!")
            molecules, rank = molecule_sampling_ddp(args, device, self_conditioned_sampler, pcdm_args, nodes_dist, prop_dist, dataset_info)
        else:
            molecules = molecule_sampling(args, device, self_conditioned_sampler, pcdm_args, nodes_dist, prop_dist, dataset_info)
            rank = 0


    # For evaluation. Only evaluate on the main process.
    if rank == 0:
        if args.eval_midi:
            # eval with midi metrics
            assert dataset_info["name"] == "qm9"
            from eval_src.eval_midi_utils import main_midi
            metrics = main_midi(cfg=args.midi_args, edm_molecules=molecules)
        else:
            # eval with edm metrics
            metrics = analyze_all_metrics(molecules, dataset_info)
        for k, val in metrics.items():
            logging.info(f"{k}: {val}")
    if args.use_dist:
        dist.barrier()
    
    
if __name__ == "__main__":
    eval()