import sys
sys.path.append(".")
from os.path import join
import torch
import pickle
from qm9 import dataset
from qm9.utils import compute_mean_mad
from qm9.sampling import sample
from qm9.property_prediction.main_qm9_prop import test
from qm9.property_prediction import main_qm9_prop
from qm9.sampling import sample, sample_sweep_conditional
import qm9.visualizer as vis
from models.rep_samplers import *
from eval_src.eval_utils import prepare_model_and_dataset_info, analyze_all_metrics
import logging
logger = logging.getLogger(name=__name__)

def get_classifier(dir_path='', device='cpu'):
    with open(join(dir_path, 'args.pickle'), 'rb') as f:
        args_classifier = pickle.load(f)
    args_classifier.device = device
    args_classifier.model_name = 'egnn'
    classifier = main_qm9_prop.get_model(args_classifier)
    classifier_state_dict = torch.load(join(dir_path, 'best_checkpoint.npy'), map_location=torch.device('cpu'))
    classifier.load_state_dict(classifier_state_dict)

    return classifier

def get_dataloader(args_gen):
    dataloaders, charge_scale = dataset.retrieve_dataloaders(args_gen)
    return dataloaders


def get_SelfConditionDiffusionDataloader(eval_args):
    class SelfConditionDiffusionDataloader:
        def __init__(self, model, nodes_dist, prop_dist, device,
                    batch_size, iterations, dataset_info, args_gen, unkown_labels=False):
            self.model = model
            self.nodes_dist = nodes_dist
            self.prop_dist = prop_dist
            self.batch_size = batch_size
            self.iterations = iterations
            self.device = device
            self.unkown_labels = unkown_labels
            self.dataset_info = dataset_info
            self.i = 0
            self.args_gen = args_gen

        def __iter__(self):
            return self

        def sample(self):
            nodesxsample = self.nodes_dist.sample(self.batch_size)
            context = self.prop_dist.sample_batch(nodesxsample)
            context = context.to(self.device)
            one_hot, charges, x, node_mask = sample(self.args_gen, self.device, self.model,
                                                    self.dataset_info, prop_dist=None, nodesxsample=nodesxsample,
                                                    rep_context=context)

            node_mask = node_mask.squeeze(2)
            context = context.squeeze(1)

            # edge_mask
            bs, n_nodes = node_mask.size()
            edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
            diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
            diag_mask = diag_mask.to(self.device)
            edge_mask *= diag_mask
            edge_mask = edge_mask.view(bs * n_nodes * n_nodes, 1)

            prop_key = self.prop_dist.properties[0]
            if self.unkown_labels:
                context[:] = self.prop_dist.normalizer[prop_key]['mean']
            else:
                context = context * self.prop_dist.normalizer[prop_key]['mad'] + self.prop_dist.normalizer[prop_key]['mean']
            data = {
                'positions': x.detach(),
                'atom_mask': node_mask.detach(),
                'edge_mask': edge_mask.detach(),
                'one_hot': one_hot.detach(),
                prop_key: context.detach()
            }
            return data

        def __next__(self):
            if self.i <= self.iterations:
                self.i += 1
                return self.sample()
            else:
                self.i = 0
                raise StopIteration

        def __len__(self):
            return self.iterations

    self_conditioned_sampler, pcdm_args, nodes_dist, prop_dist, dataset_info = prepare_model_and_dataset_info(eval_args, eval_args.device)
    
    sc_diffusion_dataloader = SelfConditionDiffusionDataloader(
        model=self_conditioned_sampler,
        nodes_dist=nodes_dist,
        prop_dist=prop_dist,
        device=eval_args.device,
        batch_size=eval_args.batch_size,
        iterations=eval_args.iterations,
        dataset_info=dataset_info,
        args_gen=pcdm_args
    )
    return sc_diffusion_dataloader
    

    
import hydra
def main_quantitative(args):
    # Get classifier
    #if args.task == "numnodes":
    #    class_dir = args.classifiers_path[:-6] + "numnodes_%s" % args.property
    #else:
    class_dir = args.classifiers_path
    classifier = get_classifier(class_dir).to(args.device)

    # Get generator and dataloader used to train the generator and evalute the classifier
    dataset_args = {
        "dataset": "qm9_second_half",
        "conditioning": [args.property],
        "include_charges": True,
        "world_size": 1,
        "rank": 0,
        "filter_n_atoms": None,
        "remove_h": False,
        "batch_size": 128,
        "num_workers": 4,
        "datadir": "./data"
    }
    dataset_args = OmegaConf.create(dataset_args)

    dataloaders = get_dataloader(dataset_args)
    property_norms = compute_mean_mad(dataloaders, dataset_args.conditioning, dataset_args.dataset)

    # Create a dataloader with the generator
    mean, mad = property_norms[args.property]['mean'], property_norms[args.property]['mad']

    if args.task == 'pcdm':
        diffusion_dataloader = get_SelfConditionDiffusionDataloader(eval_args=args)
        
        print("PCDM: We evaluate the classifier on our generated samples")
        loss, molecules = test(classifier, 0, diffusion_dataloader, mean, mad, args.property, args.device, 1, args.debug_break, output_molecules=True)
        print("Loss classifier on Generated samples: %.4f" % loss)
        
        metrics = analyze_all_metrics(molecules, diffusion_dataloader.dataset_info)
        metrics["loss"] = loss
        return metrics
        
        
    elif args.task == 'qm9_second_half':
        print("qm9_second_half: We evaluate the classifier on QM9")
        loss = test(classifier, 0, dataloaders['train'], mean, mad, args.property, args.device, args.log_interval,
                    args.debug_break)
        print("Loss classifier on qm9_second_half: %.4f" % loss)
        return None
    else:
        ValueError(f"No {args.task}.")





@hydra.main(config_path="../configs", config_name="eval_conditional.yaml", version_base='1.3')
def main(args):
    OmegaConf.set_struct(args, False)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device.type

    metrics = main_quantitative(args)
    
    if args.task == "pcdm":
        for k, val in metrics.items():
            logger.info(f"{k}: {val}")
    

if __name__ == "__main__":
    main()