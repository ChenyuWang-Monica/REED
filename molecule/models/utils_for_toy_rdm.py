from models.rdm.models.diffusion.ddpm import RDM
from omegaconf.listconfig import ListConfig
import numpy as np
from qm9 import dataset
from omegaconf import OmegaConf
from models.rep_samplers import BaseRepSampler, DDPM_VPSDE
from models.sde.sde_sampling import get_pc_sampler, AncestralSamplingPredictor, LangevinCorrector
from models.rep_samplers import *
import math
import sys
from typing import Iterable
import models.util.misc as misc
import models.util.lr_sched as lr_sched
from omegaconf.listconfig import ListConfig
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# Training
def train_one_epoch_toy(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    args=None):
    model.train()
    
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 200

    accum_iter = args.accum_iter

    optimizer.zero_grad()
    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        data = data.to(device)
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0 and args.cosine_lr:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        x_flatten = h_integer_flatten = None
        condition = None
        loss, loss_dict = model.forward(data, condition=condition)


        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# RDM
class RDM_toy(RDM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def get_input(self, data, condition=None, force_c_encode=False):
        if condition is not None:
            assert type(self.cond_stage_key) is list or type(self.cond_stage_key) is ListConfig
        else:
            assert self.cond_stage_key == "node_num"

        
        rep = data
        rep = rep.unsqueeze(-1).unsqueeze(-1)
        assert self.input_scale == 1., "Not using scale."
        rep = rep * self.input_scale
        
        assert self.model.conditioning_key is not None
        assert self.cond_stage_trainable
        
        node_num = torch.ones(data.shape[0], device=data.device, dtype=torch.long)
        
        if condition is None:
            c = node_num
        else:
            assert type(condition) is not list, "Not supporting multi conditioning now."
            c = [node_num, condition]
        out = [rep, c]

        return out
    def forward(self, data, condition=None, *args, **kwargs):
        x, c = self.get_input(data)
        t = torch.randint(0, self.num_timesteps, (x.shape[0],)).cuda().long()
        assert self.model.conditioning_key is not None
        assert self.cond_stage_trainable
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)

        loss, loss_dict = self.p_losses(x, c, t, *args, **kwargs)
        if self.use_ema:
            self.model_ema(self.model)
        return loss, loss_dict
    
    


# Samplers
class BaseRepSampler_toy(BaseRepSampler):
    
    def __init__(self, sampler_type):
        super().__init__(sampler_type)
        
    def rep_normalization(
        self,
        sampled_rep
    ):
        return sampled_rep
    
    
class GtSampler_toy(BaseRepSampler_toy):
    def __init__(self, encoder, raw_dataset, collate_fn):
        super().__init__(self.__class__)
        self.encoder = encoder
        self.raw_dataset = raw_dataset
        self.collate_fn = collate_fn
        self.len_dataset = len(raw_dataset)
        self.raw_dataset.perm = torch.arange(self.len_dataset)
        

    @torch.no_grad()
    def sample_(
        self,
        nodesxsample,
        device,
        batch_size,
        additional_cond=None
    ):
        datas = []
        for n_node in nodesxsample:
            mask = torch.ones((self.raw_dataset.data.shape[0],), dtype=torch.bool, device=nodesxsample.device)
            indice = np.random.choice(torch.arange(self.len_dataset).to(device)[mask].cpu())
            datas.append(self.raw_dataset[indice].unsqueeze(0))
        
        return torch.cat(datas, dim=0)

class PCSampler_toy(BaseRepSampler_toy):
    def __init__(self, DDPM, n_steps, inv_temp,):
        super().__init__(self.__class__)
        self.sde = DDPM_VPSDE(DDPM)
        self.DDPM = DDPM
        self.predictor = AncestralSamplingPredictor
        self.corrector = LangevinCorrector
        self.inverse_scaler = lambda x: x
        self.snr = 0.01
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
        batch_size,
        additional_cond=None,
        **kwargs
    ):
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
        
        sampled_rep = self.pc_sampler(model, nodesxsample=nodesxsample, inv_temp=self.inv_temp)[0].squeeze()
        
        return sampled_rep
    
    
def initilize_rep_sampler_toy(rep_sampler_args, device, dataset_args=None):
    # Initialize the representation sampler
    if rep_sampler_args.sampler == "GtSampler":
        # Params
        assert rep_sampler_args.Gt_dataset is not None
        assert rep_sampler_args.encoder_path is not None
        assert dataset_args is not None
        
        encoder_args = OmegaConf.load(rep_sampler_args.encoder_config_path).encoder_args

        
        datasets, collate_fn = retrieve_dataloaders_toy(dataset_args, return_raw_datasets_and_collate_fn=True)
        # Set up for encoder
        from models.torchmdnet.models.model import load_model
        encoder = load_model(filepath=rep_sampler_args.encoder_path, device=device, args=encoder_args, only_representation=True)
        encoder = encoder.to(device)
        encoder.eval()
        rep_sampler = GtSampler_toy(encoder=encoder, raw_dataset=datasets[rep_sampler_args.Gt_dataset], collate_fn=collate_fn)
    elif rep_sampler_args.sampler == "DDIMSampler":
        # Params
        assert 0
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
        
        
        from models.util import misc
        rdm_model, rdm_train_model_args = misc.initialize_and_load_rdm_model(rep_sampler_args.rdm_ckpt, device)
        rdm_model.eval()
        rep_sampler = PCSampler_toy(rdm_model, inv_temp=rep_sampler_args.inv_temp, n_steps=rep_sampler_args.n_steps)
    else:
        raise ValueError(f"No sampler named {rep_sampler_args.sampler}")
    
    return rep_sampler


# dataset


def retrieve_dataloaders_toy(cfg, return_raw_datasets_and_collate_fn=False):
    class MixtureOfGaussians(Dataset):
        def __init__(self, n_samples, means, covariances, priors):
            self.n_samples = n_samples
            self.means = means
            self.covariances = covariances
            self.priors = priors
            self.data = self._generate_samples()

        def _generate_samples(self):
            data = []
            for mean, cov, prior in zip(self.means, self.covariances, self.priors):
                n_samples = int(self.n_samples * prior)
                samples = np.random.multivariate_normal(mean, cov, n_samples)
                data.append(samples)
            return np.vstack(data)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return torch.tensor(self.data[idx], dtype=torch.float)
        
        
    # Example configuration and parameters
    n_samples = 30000
    radius = 0.5  # Radius of the circle where centers are located
    num_centers = 4  # Number of centers
    angles = np.linspace(0, 2 * np.pi, num_centers, endpoint=False)
    means = np.array([[radius * np.cos(angle), radius * np.sin(angle)] for angle in angles])
    covariances = [np.array([[0.015, 0], [0, 0.015]]) for _ in range(num_centers)]
    priors = [1 / num_centers for _ in range(num_centers)]

    # Create the dataset
    datasets = {
        "train": MixtureOfGaussians(n_samples, means, covariances, priors),
        "valid": MixtureOfGaussians(n_samples // 10, means, covariances, priors),
        "test": MixtureOfGaussians(n_samples // 10, means, covariances, priors)
    }

    # Create distributed samplers
    sampler_train = DistributedSampler(datasets["train"], num_replicas=cfg['world_size'], rank=cfg['rank'], shuffle=True)
    sampler_valid = DistributedSampler(datasets["valid"], num_replicas=cfg['world_size'], rank=cfg['rank'], shuffle=False)
    sampler_test = DistributedSampler(datasets["test"], num_replicas=cfg['world_size'], rank=cfg['rank'], shuffle=False)

    samplers = [sampler_train, sampler_valid, sampler_test]

    # Create dataloaders
    dataloaders = {
        split: DataLoader(dataset,
                          batch_size=cfg.batch_size,
                          num_workers=cfg.num_workers,
                          sampler=samplers[i])
        for i, (split, dataset) in enumerate(datasets.items())
    }
    if return_raw_datasets_and_collate_fn:
        return datasets, None

    return dataloaders, None
    
