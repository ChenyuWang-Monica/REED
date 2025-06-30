import torch
import numpy as np
import torch.nn.functional as F

IMAGE_ENCODERS = ['dinov2', 'mocov3', 'clip', 'mae', 'jepa']

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))


def sum_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))


class SILoss:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            encoders=[],
            enc_names=[],
            loss_weights={"dinov2": 1.0, "t5": 1.0},
            time_schedule="constant",
            cutoffs=[0.0, 1.0],
            accelerator=None,
            latents_scale=None,
            latents_bias=None,
    ):
        self.prediction = prediction  # Defines the target to predict (currently supports 'v' for velocity prediction in diffusion models).
        self.weighting = weighting  # Governs how time steps are sampled (e.g., uniform or lognormal).
        self.path_type = path_type  # Governs the type of interpolation in the latent space (e.g., linear or cosine).
        self.encoders = encoders
        self.enc_names = enc_names
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias
        self.loss_weights = loss_weights
        self.time_schedule = time_schedule
        self.cutoffs = cutoffs
        assert len(loss_weights) == len(enc_names), "Loss weights must be provided for each encoder."

    def interpolant(self, t):
        # This controls how noise and image contributions change over the diffusion process.
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t = 1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t = np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def encoder_weight(
            self,
            base_weight: float,
            current_step: int,
            total_steps: int,
            schedule: str = "linear",
            focus: str = "text",
            transition_point: float = 0.5,
            sharpness: float = 10,
    ) -> float:
        """
        Compute the training-progress-sensitive weight for an encoder.

        Args:
            base_weight (float): The base weight for this encoder.
            current_step (int): Current training step.
            total_steps (int): Total training steps.
            schedule (str): Weighting schedule ("linear", "cosine", "sigmoid").
            focus (str): Encoder type to prioritize initially ("text" or "image").
            transition_point (float): Point (fraction of total_steps) where the focus shifts.
            sharpness (float): Controls the sharpness of the transition (for sigmoid).

        Returns:
            float: Training-progress-sensitive weight for the encoder.

        @sharut: TODO: Call in __call__ method to compute encoder weights.
        """
        progress = current_step / total_steps  # Training progress (0.0 to 1.0)

        if schedule == "linear":
            if focus == "text":
                scale = 1 - progress  # More focus on text early, less later
            elif focus == "image":
                scale = progress  # More focus on image as training progresses
        elif schedule == "cosine":
            from math import pi, cos
            if focus == "text":
                scale = 0.5 * (1 + cos(pi * progress))  # Text focus decays smoothly
            elif focus == "image":
                scale = 0.5 * (1 - cos(pi * progress))  # Image focus grows smoothly
        elif schedule == "sigmoid":
            from math import exp
            x = (progress - transition_point) * sharpness
            if focus == "text":
                scale = 1 / (1 + exp(x))  # Text focus fades sigmoidally
            elif focus == "image":
                scale = 1 - 1 / (1 + exp(x))  # Image focus grows sigmoidally
        else:
            raise ValueError("Invalid schedule. Choose from 'linear', 'cosine', 'sigmoid'.")

        return base_weight * scale

    def time_weight(self, t: torch.Tensor, base_weight: float = 1.0, schedule: str = "constant", cutoffs: list = [0.0, 1.0]
                    ) -> torch.Tensor:
        """
        Compute time-sensitive weights for a batch of time steps.

        Args:
            t (torch.Tensor): normalized time steps for each batch element (shape: [B]).
            total_steps (int): Total diffusion time steps.
            base_weight (float): Base weight for this encoder.
            schedule (str): Weighting schedule ("linear", "cosine", "sigmoid").

        Returns:
            torch.Tensor: Time-sensitive weights (shape: [B]).
        """
        if schedule == "linear":
            scale = 1 - t
        elif schedule == "cosine":
            from math import pi
            scale = 0.5 * (1 + torch.cos(pi * t))
        elif schedule == "sigmoid":
            sharpness, midpoint = 10, 0.5
            scale = 1 / (1 + torch.exp((t - midpoint) * sharpness))
        elif schedule == "constant":
            scale = torch.ones_like(t)
        elif schedule == 'loglinear': # TODO!
            scale = 1 - torch.log(t + 1)
        elif schedule == 'cutoff':
            # set the scale to 0 for timesteps outside the cutoffs
            scale = torch.ones_like(t)
            scale[t < cutoffs[0]] = 0
            scale[t > cutoffs[1]] = 0
        else:
            raise ValueError("Invalid schedule. Choose from 'linear', 'cosine', 'sigmoid'.")
        return base_weight * scale

    def __call__(self, model, images, model_kwargs=None, zs=None, **kwargs):
        if model_kwargs == None:
            model_kwargs = {}

        # sample timesteps
        if self.weighting == "uniform":
            time_input = torch.rand((images.shape[0], 1, 1, 1))
        elif self.weighting == "lognormal":
            # sample timestep according to log-normal distribution of sigmas following EDM (Elucidated Diffusion Models) framework.
            # This places more emphasis on later steps (higher noise levels).
            rnd_normal = torch.randn((images.shape[0], 1, 1, 1))
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
            elif self.path_type == "cosine":
                time_input = 2 / np.pi * torch.atan(sigma)

        time_input = time_input.to(device=images.device, dtype=images.dtype)

        noises = torch.randn_like(images)

        # Combines the image and noise using interpolants (alpha_t and sigma_t):
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)
        model_input = alpha_t * images + sigma_t * noises  # bs, c, h, w (eg bs, 4, 32, 32, for moco features)

        if self.prediction == 'v':
            model_target = d_alpha_t * images + d_sigma_t * noises
        else:
            raise NotImplementedError()  # TODO: add x or eps prediction
        
        model_kwargs['inference'] = False

        model_output, zs_tilde = model(model_input, time_input.flatten(), **model_kwargs)
        denoising_loss = mean_flat((model_output - model_target) ** 2)
        
        # projection loss (regularization) for image embeddings
        proj_loss = 0.
        loss_accumulators = {
            "image": {"loss": 0., "count": 0},
            "text": {"loss": 0., "count": 0}
        }
        bsz = zs[0].shape[0]

        if kwargs.get("save_projloss", False):
            loss_saver = {
                "image": torch.zeros(bsz, device=images.device),
                "text": torch.zeros(bsz, device=images.device),
                "time": time_input.flatten()
            }

        # zs_tilde is latent from the diffusion model, zs is the pretrained encoder output
        for i, (z, z_tilde, enc_name) in enumerate(zip(zs, zs_tilde, self.enc_names)):
            # t=0 corresponds to clean image
            wts = self.time_weight(time_input, self.loss_weights.get(enc_name, 1.0), self.time_schedule, self.cutoffs)
            z_tilde = torch.nn.functional.normalize(z_tilde, dim=-1)
            z = torch.nn.functional.normalize(z, dim=-1)

            key = "image" if enc_name in IMAGE_ENCODERS or len(self.enc_names)==1 else "text"
            if z.ndim == 2:
                assert key == "text", "Only text encoders should have 2D embeddings."
                assert z_tilde.ndim == 2, "Pooling to 2D to align with text embeddings."
                z = z.unsqueeze(1)  # [B, 1, D]
                z_tilde = z_tilde.unsqueeze(1) # [B, 1, D]

            # only update the projector
            if self.loss_weights.get(enc_name, 1.0) == 0.0:
                wts = torch.ones_like(wts)

            curr_loss = -(z * z_tilde).sum(dim=-1).mean(dim=-1)  # [B]
            weighted_loss = (curr_loss * wts).mean()  # time sensitive weighting
            proj_loss += weighted_loss
            loss_accumulators[key]["loss"] += curr_loss.mean()
            loss_accumulators[key]["count"] += 1

            if kwargs.get("save_projloss", False):
                loss_saver[key] += curr_loss
        img_proj_loss = loss_accumulators["image"]["loss"] / max(1, loss_accumulators["image"]["count"])
        text_proj_loss = loss_accumulators["text"]["loss"] / max(1, loss_accumulators["text"]["count"])

        if kwargs.get("save_projloss", False):
            return {"denoising_loss": denoising_loss, "proj_loss": proj_loss, "img_proj_loss": img_proj_loss,
                    "text_proj_loss": text_proj_loss, "loss_saver": loss_saver}
        else:
            return {"denoising_loss": denoising_loss, "proj_loss": proj_loss, "img_proj_loss": img_proj_loss,
                    "text_proj_loss": text_proj_loss}
