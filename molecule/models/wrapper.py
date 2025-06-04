import torch
import logging
from time import time


class SelfConditionWrappedSampler(torch.nn.Module):
    def __init__(
            self,
            pcdm_sampler,
            rdm_sampler
    ):
        super().__init__()
        self.rdm_sampler = rdm_sampler
        self.pcdm_sampler = pcdm_sampler
        
        
    @torch.no_grad()
    def sample(
        self, 
        n_samples, 
        n_nodes, 
        node_mask, 
        edge_mask, 
        context, 
        fix_noise=False,
        fixed_rep=None,
        rep_context=None
        ):
        assert len(node_mask.shape) == 2 or node_mask.shape == (node_mask.shape[0], node_mask.shape[1], 1), f"node mask should have shape (batch size, max n nodes) or (batch size, max n nodes, 1), while it now has shape {node_mask.shape}"
        assert context == None, "We always use unconditional sampling in PCDM."
        
        
        device = node_mask.device
        batch_size, max_n_nodes = node_mask.shape[0], node_mask.shape[1]
        nodesxsample = node_mask.sum(-1) if len(node_mask.shape) == 2 else node_mask.sum(-1).sum(-1)
        nodesxsample = nodesxsample.to(torch.int64)
        
        
        # To sample rep first. We only use context in rep samplers, and pcdm sampler is always unconidtional.
        self.rdm_sampler.eval()
        if fixed_rep is None:
            print("Sampling Reps ...")
            
            s = time()
            sampled_rep = self.rdm_sampler.sample(
                nodesxsample=nodesxsample,
                device=device,
                additional_cond=rep_context,
                running_batch_size=batch_size
            )
            logging.info(f"RDM sampling of {batch_size} samples took {time() - s}s. {(time() - s)/batch_size}s for each sample in average.")
            
            print("Reps Sampling Done!")
        else:
            print("Using provided fixed rep.")
            sampled_rep = fixed_rep
            
        # Now sample from PCDM conditioning on the rep
        print("Sampling Molecules conditioning on Reps ...")
        
        self.pcdm_sampler.eval()
        s = time()
        x, h = self.pcdm_sampler.sample(
            n_samples=batch_size, 
            n_nodes=max_n_nodes,
            node_mask=node_mask, 
            edge_mask=edge_mask, 
            context=None, # NOTE: Always unconditional!
            fix_noise=fix_noise, 
            rep=sampled_rep
            )
        logging.info(f"PCDM sampling of {batch_size} samples took {time() - s}s. {(time() - s)/batch_size}s for each sample in average.")
        
        print("Molecules Sampling Done!")

        return x, h
    
    @torch.no_grad()
    def sample_chain(
        self, 
        n_samples, 
        n_nodes, 
        node_mask, 
        edge_mask, 
        context, 
        keep_frames=None, 
        rep=None,
        fixed_rep=None,
        rep_context=None
        ):
        assert n_samples == 1, f"For chain sampling, we should ensure that n_samples equals 1, but it now equals {n_samples}"
        assert len(node_mask.shape) == 2 or node_mask.shape == (node_mask.shape[0], node_mask.shape[1], 1), f"node mask should have shape (batch size, max n nodes) or (batch size, max n nodes, 1), while it now has shape {node_mask.shape}"
        assert context == None, "We always use unconditional sampling in PCDM."
        
        device = node_mask.device
        batch_size, max_n_nodes = node_mask.shape[0], node_mask.shape[1]
        nodesxsample = node_mask.sum(-1) if len(node_mask.shape) == 2 else node_mask.sum(-1).sum(-1)
        nodesxsample = nodesxsample.to(torch.int64)
        
        
        # To sample rep first. We only use context in rep samplers, and pcdm sampler is always unconidtional.
        self.rdm_sampler.eval()
        if fixed_rep is None:
            print("Sampling Reps ...")
            sampled_rep = self.rdm_sampler.sample(
                nodesxsample=nodesxsample,
                device=device,
                additional_cond=rep_context,
                running_batch_size=batch_size
            )
            print("Reps Sampling Done!")
        else:
            print("Using provided fixed rep.")
            sampled_rep = fixed_rep
            
        # Now sample from PCDM conditioning on the rep
        print("Sampling Molecules conditioning on Reps ...")
        self.pcdm_sampler.eval()
        chain = self.pcdm_sampler.sample_chain(n_samples, n_nodes, node_mask, edge_mask, context, keep_frames=keep_frames, rep=sampled_rep)
        print("Molecules Sampling Done!")

        return chain