import sys
sys.path.append(".")

# Rdkit import should be first, do not move it
from qm9 import visualizer as qm9_visualizer
from models.rep_samplers import *
from eval_src.eval_utils import prepare_model_and_dataset_info
from pathlib import Path
import hydra
import logging
from eval_src.eval_analyze import molecule_sampling_ddp, molecule_sampling, sample_loop
import datetime
from qm9.sampling import sample_sweep_conditional

def molecule_sampling_sweep(args, device, model, prop_dist, dataset_info, n_frames, start_value=None, end_value=None):
    one_hot, charges, x, node_mask, property_values = sample_sweep_conditional(args, device, model, dataset_info, prop_dist, n_frames=n_frames, return_property_values=True, start_value=start_value, end_value=end_value)
    molecules = {'one_hot': one_hot, 'x': x, 'node_mask': node_mask}

    return molecules, property_values


def sample_visualize(molecules, target_path, dataset_info, id_from=0, name="unconditional"):
    qm9_visualizer.save_xyz_file(
            target_path, molecules["one_hot"], None, molecules["x"],
            dataset_info, id_from, name=name, node_mask=molecules["node_mask"])

    qm9_visualizer.visualize(
        target_path, dataset_info,
        spheres_3d=True, max_num=len(molecules["one_hot"])
        )
    

def molecule_sampling_prop(eval_args, device, self_conditioned_sampler, pcdm_args, nodes_dist, prop_dist, dataset_info):
    molecules, prop = sample_loop(
        pcdm_args, 
        device, 
        self_conditioned_sampler,
        nodes_dist, 
        prop_dist, 
        dataset_info, 
        n_samples=eval_args.n_samples,
        batch_size=eval_args.batch_size_gen, 
        save_molecules=eval_args.save_molecules,
        pcdm_model_path=eval_args.pcdm_model_path,
        return_prop=True
        )
    return molecules, prop
    
@hydra.main(config_path="../configs", config_name="eval_visualize_samples.yaml", version_base='1.3')
def main(args):
    OmegaConf.set_struct(args, False)
    # Initialize Sampler and dataset info. 
    device = "cuda"
    self_conditioned_sampler, pcdm_args, nodes_dist, prop_dist, dataset_info = prepare_model_and_dataset_info(args, device, only_dataset_info=(args.saved_molecules_path is not None)) # If one load saved molecules, then only dataset info is needed.
    
    
    
    
    
    if args.sweep:
        molecules, property_values = molecule_sampling_sweep(
            pcdm_args, 
            device=device,
            model=self_conditioned_sampler,
            prop_dist=prop_dist,
            dataset_info=dataset_info,
            n_frames=args.n_samples,
            start_value=args.start_value,
            end_value=args.end_value
        )
        
        name = "conditional_sweep"
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        target_path = Path(f"./eval_src/visualize_results/{name}_{current_time}/")
        target_path.mkdir(parents=True, exist_ok=True)        
        
        # Define the path for saving the property values as a text file
        property_file_path = target_path / "property_values.log"

        # Save property values to the text file
        with open(property_file_path, 'w') as file:
            for value in property_values:
                file.write(f"{value.item()}\n")
        
        sample_visualize(
            molecules,
            target_path=str(target_path) + "/",
            dataset_info=dataset_info,
            name=name
        )
    else:
        if args.property is None:
            molecules = molecule_sampling(args, device, self_conditioned_sampler, pcdm_args, nodes_dist, prop_dist, dataset_info)
            name = "unconditional_random" 
        else:
            molecules, prop = molecule_sampling_prop(args, device, self_conditioned_sampler, pcdm_args, nodes_dist, prop_dist, dataset_info)
            name = "conditional_random"
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        target_path = Path(f"./eval_src/visualize_results/{name}_{current_time}/")
        target_path.mkdir(parents=True, exist_ok=True)        
        
        
        sample_visualize(
            molecules,
            target_path=str(target_path) + "/",
            dataset_info=dataset_info,
            name=name
        )
        
        if args.property is not None:
            save_prop_path = target_path / "prop.pickle"
            import pickle
            with open(save_prop_path, "wb") as f:
                pickle.dump(prop, f)


if __name__ == "__main__":
    main()
