
from torch.utils.data import Subset
from qm9.models import DistributionNodes
from models.rdm.models.diffusion.ddim import DDIMSampler as BaseDDIMSampler
import torch
from models.encoders import get_global_representation
import numpy as np
from qm9 import dataset
from omegaconf import OmegaConf
from tqdm import tqdm
from models.sde.sde_sampling import get_pc_sampler, AncestralSamplingPredictor, LangevinCorrector
from models.sde.sde_lib import VPSDE
from models.rdm.models.diffusion.ddpm import DDPM as baseDDPM
from build_geom_dataset import GeomDrugsDataset
from models.encoders import initialize_encoder

class BaseRepSampler(torch.nn.Module):
    
    def __init__(self, sampler_type):
        super().__init__()
        print(f"Using {sampler_type} for REP Sampling!")
        
    @torch.no_grad()
    def sample(
        self,
        device,
        nodesxsample,
        additional_cond=None,
        running_batch_size=1000,
        dtype=torch.float32,
    ):
        nodesxsample = torch.tensor([nodesxsample]) if type(nodesxsample) is int else nodesxsample
        nodesxsample = nodesxsample.to(device)
        if additional_cond is not None:
            additional_cond = additional_cond.to(device)
        batch_size = nodesxsample.shape[0]
        
        sampled_rep_all = []
        num_full_batches = batch_size // running_batch_size
        
        if batch_size == running_batch_size:
            iter_ = range(num_full_batches)
        else:
            iter_ = tqdm(range(num_full_batches))
        for i in iter_:
            start_idx = i * running_batch_size
            end_idx = start_idx + running_batch_size
            running_nodesxsample = nodesxsample[start_idx:end_idx]
            if additional_cond is not None:
                running_additional_cond = additional_cond[start_idx:end_idx]
            else:
                running_additional_cond = None
            assert running_nodesxsample.shape[0] == running_batch_size
            
            sampled_rep = self.sample_(
                device=device,
                nodesxsample=running_nodesxsample,
                additional_cond=running_additional_cond,
                dtype=dtype,
            )
            
            sampled_rep = self.rep_normalization(sampled_rep)
            
            # Append the sampled representations to the list
            sampled_rep_all.append(sampled_rep)
        
        # We do not drop last
        if batch_size % running_batch_size != 0:
            remaining_nodesxsample = nodesxsample[num_full_batches * running_batch_size:]
            if additional_cond is not None:
                remaining_additional_cond = additional_cond[num_full_batches * running_batch_size:]
            else:
                remaining_additional_cond = None
            
            sampled_rep = self.sample_(
                device=device,
                nodesxsample=remaining_nodesxsample,
                additional_cond=remaining_additional_cond,
                dtype=dtype
            )
            
            sampled_rep = self.rep_normalization(sampled_rep)
            
            # Append the remaining sampled representations to the list
            sampled_rep_all.append(sampled_rep)
        
        # Concatenate all the sampled representations into a single tensor
        sampled_rep_all = torch.cat(sampled_rep_all, dim=0)
        assert sampled_rep_all.shape[0] == batch_size
        return sampled_rep_all
    
    @torch.no_grad()
    def sample_(
        self,
        *args,
        **kwargs
    ):
        pass
    
    def rep_normalization(
        self,
        sampled_rep
    ):
        assert len(sampled_rep.shape) == 2
        sampled_rep_std = torch.std(sampled_rep, dim=1, keepdim=True)
        sampled_rep_mean = torch.mean(sampled_rep, dim=1, keepdim=True) 
        sampled_rep = (sampled_rep - sampled_rep_mean) / sampled_rep_std
        return sampled_rep
    

    
class DDIMSampler(BaseRepSampler):
    def __init__(self, rdm_model, eta, step_num):
        super().__init__("DDIM Sampler")
        self.rdm_model = rdm_model
        self.ddim_sampler = BaseDDIMSampler(rdm_model)
        self.eta = eta
        self.step_num = step_num
        
    @torch.no_grad()
    def sample_(
        self,
        device,
        dtype,
        nodesxsample,
        additional_cond=None,
        **kwargs
    ):
        batch_size = nodesxsample.shape[0]
        if additional_cond is not None:
            assert len(additional_cond.shape) == 2
            if type(nodesxsample) is int:
                assert additional_cond.shape[0] == 1
            else:
                assert additional_cond.shape[0] == nodesxsample.shape[0]
        
        shape = [self.ddim_sampler.model.model.diffusion_model.in_channels,
                        self.ddim_sampler.model.model.diffusion_model.image_size,
                        self.ddim_sampler.model.model.diffusion_model.image_size]
            
        if additional_cond is None:
            cond = nodesxsample
            cond = self.ddim_sampler.model.get_learned_conditioning(cond)
            
            cond = cond.unsqueeze(0) if len(cond.shape) == 1 else cond
        else:
            cond = [nodesxsample, additional_cond.to(device)]
            cond = self.ddim_sampler.model.get_learned_conditioning(cond)

        sampled_rep, _ = self.ddim_sampler.sample(self.step_num, conditioning=cond, batch_size=batch_size,
                                            shape=shape, eta=self.eta, verbose=False)
        
        sampled_rep = sampled_rep.squeeze(-1).squeeze(-1)
        assert len(sampled_rep.shape) == 2

        
        return sampled_rep
        
class GtSampler(BaseRepSampler):
    def __init__(self, encoder, raw_dataset, collate_fn, debug=False):
        super().__init__("Gt Sampler")
        self.encoder = encoder
        self.raw_dataset = raw_dataset
        self.raw_dataset.perm = None
        self.geom = False
        if isinstance(raw_dataset, GeomDrugsDataset):
            self.calculate_geom_num_atoms()
            self.geom = True
        self.len_dataset = len(raw_dataset)
        self.collate_fn = collate_fn
        
        self.debug = debug
        
        if self.debug:
            print("Warning: Using debug mode!!")
        
    def calculate_geom_num_atoms(self):
        num_atoms = []
        for data in tqdm(self.raw_dataset, desc="Precalculating num atom information for GtSampler"):
            num_atom = data["positions"].shape[0]
            num_atoms.append(num_atom)
        self.num_atoms = torch.tensor(num_atoms, dtype=torch.long)
    @torch.no_grad()
    def sample_(
        self,
        nodesxsample,
        device,
        dtype,
        additional_cond=None,
    ):
        # if additional_cond is not None:
            # assert 0, "not implemented." # Careful with this.
            
        if self.debug:
            existing_node_size = self.raw_dataset[0]["positions"].shape[0]
            nodesxsample = torch.ones(nodesxsample.shape, dtype=nodesxsample.dtype, device=nodesxsample.device) * existing_node_size
            
        datas = []
        for n_node in nodesxsample:
            if self.geom:
                num_atoms = self.num_atoms.to(device)
            else:
                num_atoms = self.raw_dataset.data["num_atoms"].to(device)
            
            mask = num_atoms == n_node
            indice = np.random.choice(torch.arange(self.len_dataset).to(device)[mask].cpu())
            datas.append(self.raw_dataset[indice])
            if self.geom:
                assert self.raw_dataset[indice]["positions"].shape[0] == n_node
            else:
                assert self.raw_dataset[indice]["num_atoms"] == n_node
        
        processed_data = self.collate_fn(datas)
        z = processed_data["charges"].to(device).to(dtype)
        pos = processed_data["positions"].to(device).to(dtype)
        node_mask = processed_data["atom_mask"].to(device).to(dtype)


        rep = get_global_representation(node_mask, self.encoder, pos, z, training_encoder=False, device=device)
        
        
        
                
        return rep
    

class DDPM_VPSDE(VPSDE):
    '''
        Initialize all the schedules in VPSDE using the schedules of a well-trained DDPM
    '''
    def __init__(self, DDPM):
        assert isinstance(DDPM, baseDDPM) 
        N = DDPM.num_timesteps
        beta_min = DDPM.linear_start * N
        beta_max = DDPM.linear_end * N
        super().__init__(N=N, beta_min=beta_min, beta_max=beta_max)
        
        
        # However, the schedules may be calculated in a different manner.
        if not torch.allclose(self.discrete_betas.cpu(), DDPM.betas.cpu()):
            print(
                f"Different in betas when initializing VPSDE! We use betas in the trained model."
            )
            assert self.discrete_betas.shape == DDPM.betas.shape, "Betas' shape not matched"
            self.discrete_betas = DDPM.betas.cpu()
            self.alphas = 1. - self.discrete_betas
            self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
            self.alphas_cumprod_prev = np.append(1., self.alphas_cumprod[:-1])
            self.alphas_cumprod_prev = torch.tensor(self.alphas_cumprod_prev, dtype=self.alphas_cumprod.dtype, device=self.alphas_cumprod.device)
            self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
            self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
            
    def sde(self):
        assert 0
    def marginal_prob(self):
        assert 0
    def prior_logp(self):
        assert 0
    def discretize(self):
        assert 0

class PCSampler(BaseRepSampler):
    def __init__(self, DDPM, n_steps, inv_temp, snr=0.01):
        super().__init__("PC Sampler")
        self.sde = DDPM_VPSDE(DDPM)
        self.DDPM = DDPM
        self.predictor = AncestralSamplingPredictor
        self.corrector = LangevinCorrector
        self.inverse_scaler = lambda x: x
        self.snr = snr
        self.n_steps = n_steps
        self.probability_flow = False
        self.continuous = False
        self.denoise = False
        self.eps = 1e-5
        self.inv_temp = inv_temp
        
        
    def sample_(
        self,
        device,
        nodesxsample,
        dtype,
        additional_cond=None,
        **kwargs
    ):
        batch_size = nodesxsample.shape[0]
        model = self.DDPM
        shape = [batch_size, model.model.diffusion_model.in_channels, model.model.diffusion_model.image_size, model.model.diffusion_model.image_size]
        self.pc_sampler = get_pc_sampler(
            sde=self.sde,
            shape=shape,
            predictor=self.predictor,
            corrector=self.corrector,
            inverse_scaler=self.inverse_scaler,
            snr=self.snr,
            n_steps=self.n_steps,
            probability_flow=self.probability_flow,
            continuous=self.continuous,
            denoise=self.denoise,
            eps=self.eps,
            device=device
            )
        
        sampled_rep = self.pc_sampler(model, nodesxsample=nodesxsample, inv_temp=self.inv_temp, additional_cond=additional_cond)[0]
        sampled_rep = sampled_rep.squeeze(-1).squeeze(-1)
        
        return sampled_rep
    
    
def initilize_rep_sampler(rep_sampler_args, device, dataset_args=None, debug=False):
    # Initialize the representation sampler
    if rep_sampler_args.sampler == "GtSampler":
        # Params
        assert rep_sampler_args.Gt_dataset is not None and rep_sampler_args.Gt_dataset == "train", "Please use train split, since the histogram of nodes are calculated using training dataset."
        assert rep_sampler_args.encoder_path is not None
        assert dataset_args is not None

        try:
            which_split = rep_sampler_args.get("which_split", "edm")
        except:
            which_split = "edm"
        datasets, collate_fn = dataset.retrieve_dataloaders(dataset_args, return_raw_datasets_and_collate_fn=True, which_split=which_split)
        # Set up for encoder
        encoder = initialize_encoder(encoder_type=rep_sampler_args.encoder_type, device=device, encoder_ckpt_path=rep_sampler_args.encoder_path)
        for param in encoder.parameters():
            param.requires_grad = False
        encoder.eval()
        
        rep_sampler = GtSampler(encoder=encoder, raw_dataset=datasets[rep_sampler_args.Gt_dataset], collate_fn=collate_fn, debug=debug)
    elif rep_sampler_args.sampler == "DDIMSampler":
        # Params
        assert rep_sampler_args.eta is not None
        assert rep_sampler_args.step_num is not None
        assert rep_sampler_args.rdm_ckpt is not None
        
        
        from models.util import misc
        rdm_model, rdm_train_model_args = misc.initialize_and_load_rdm_model(rep_sampler_args.rdm_ckpt, device)
        rdm_model.eval()
        rep_sampler = DDIMSampler(rdm_model, eta=rep_sampler_args.eta, step_num=rep_sampler_args.step_num)
    elif rep_sampler_args.sampler == "PCSampler":
        assert rep_sampler_args.rdm_ckpt is not None
        assert rep_sampler_args.inv_temp is not None
        assert rep_sampler_args.n_steps is not None
        assert rep_sampler_args.snr is not None
        
        
        from models.util import misc
        rdm_model, rdm_train_model_args = misc.initialize_and_load_rdm_model(rep_sampler_args.rdm_ckpt, device)
        rdm_model.eval()
        rep_sampler = PCSampler(rdm_model, inv_temp=rep_sampler_args.inv_temp, n_steps=rep_sampler_args.n_steps, snr=rep_sampler_args.snr)
    else:
        raise ValueError(f"No sampler named {rep_sampler_args.sampler}")
    
    return rep_sampler



