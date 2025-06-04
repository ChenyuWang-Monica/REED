import math
import sys
from typing import Iterable

import torch

import models.util.misc as misc
import models.util.lr_sched as lr_sched
from models.encoders import flatten_x_h
from omegaconf.listconfig import ListConfig
def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    args=None, property_norms=None, dtype=None):
    model.train()
    
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 200

    accum_iter = args.accum_iter

    optimizer.zero_grad()
    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0 and args.cosine_lr:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        node_mask, x, h_integer = (
            data['atom_mask'].to(device).to(dtype), 
            data['positions'].to(device).to(dtype), 
            data['charges'].to(device).to(dtype)
        )
        
            

        if len(args.conditioning) > 0:
            assert len(args.conditioning) == 1
            condition_key = args.conditioning[0]
            condition = data[condition_key].to(device)
            condition = condition.unsqueeze(-1)
            assert condition.shape == (node_mask.shape[0], 1)
            assert property_norms is not None
            
            condition = (condition - property_norms[condition_key]["mean"]) / property_norms[condition_key]["mad"]
        else:
            condition = None
            
        loss, loss_dict = model(node_mask, x, h_integer, condition=condition)


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