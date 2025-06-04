import argparse
from pathlib import Path
from functools import partial

import torch
import numpy as np
import lightning as L

import scriptutil as util
from flowmodels.fm import Integrator, MolecularCFM
from flowmodels.semla import EquiInvDynamics, SemlaGenerator

from data.datasets import GeometricDataset
from data.datamodules import GeometricInterpolantDM
from data.interpolate import GeometricInterpolant, GeometricNoiseSampler
from flowmodels.encoders import initialize_encoder
from flowmodels.rep_samplers import *
from train import DEFAULT_CATEGORICAL_STRATEGY


# Default script arguments
DEFAULT_DATASET_SPLIT = "test"
DEFAULT_N_MOLECULES = 1000
DEFAULT_N_REPLICATES = 3
DEFAULT_BATCH_COST = 8192
DEFAULT_BUCKET_COST_SCALE = "linear"
DEFAULT_INTEGRATION_STEPS = 100
DEFAULT_CAT_SAMPLING_NOISE_LEVEL = 1
DEFAULT_ODE_SAMPLING_STRATEGY = "log"


def load_model(args, vocab):
    checkpoint = torch.load(args.ckpt_path)
    hparams = checkpoint["hyper_parameters"]

    hparams["compile_model"] = False
    hparams["integration-steps"] = args.integration_steps
    hparams["sampling_strategy"] = args.ode_sampling_strategy

    n_bond_types = util.get_n_bond_types(hparams["integration-type-strategy"])

    # Set default arch to semla if nothing has been saved
    if hparams.get("architecture") is None:
        hparams["architecture"] = "semla"

    if hparams["architecture"] == "semla":
        dynamics = EquiInvDynamics(
            hparams["d_model"],
            hparams["d_message"],
            hparams["n_coord_sets"],
            hparams["n_layers"],
            n_attn_heads=hparams["n_attn_heads"],
            d_message_hidden=hparams["d_message_hidden"],
            d_edge=hparams["d_edge"],
            self_cond=hparams["self_cond"],
            coord_norm=hparams["coord_norm"],
            d_rep=hparams["d_rep"],
            cond_type=hparams["cond_type"],
            rep_alignment=hparams["rep_alignment"],
            align_depth=hparams["align_depth"]
        )
        egnn_gen = SemlaGenerator(
            hparams["d_model"],
            dynamics,
            vocab.size,
            hparams["n_atom_feats"],
            d_edge=hparams["d_edge"],
            n_edge_types=n_bond_types,
            self_cond=hparams["self_cond"],
            size_emb=hparams["size_emb"],
            max_atoms=hparams["max_atoms"],
            rep_alignment=hparams["rep_alignment"]
        )

    elif hparams["architecture"] == "eqgat":
        from semlaflow.models.eqgat import EqgatGenerator

        egnn_gen = EqgatGenerator(
            hparams["d_model"],
            hparams["n_layers"],
            hparams["n_equi_feats"],
            vocab.size,
            hparams["n_atom_feats"],
            hparams["d_edge"],
            hparams["n_edge_types"],
            d_rep=hparams["d_rep"]
        )

    elif hparams["architecture"] == "egnn":
        from semlaflow.models.egnn import VanillaEgnnGenerator

        n_layers = args.n_layers if hparams.get("n_layers") is None else hparams["n_layers"]
        if n_layers is None:
            raise ValueError("No hparam for n_layers was saved, use script arg to provide n_layers")

        egnn_gen = VanillaEgnnGenerator(
            hparams["d_model"],
            n_layers,
            vocab.size,
            hparams["n_atom_feats"],
            d_edge=hparams["d_edge"],
            n_edge_types=n_bond_types,
            d_rep=hparams["d_rep"]
        )

    else:
        raise ValueError(f"Unknown architecture hyperparameter.")

    type_mask_index = vocab.indices_from_tokens(["<MASK>"])[0] if hparams["train-type-interpolation"] == "mask" else None
    bond_mask_index = None

    integrator = Integrator(
        args.integration_steps,
        type_strategy=hparams["integration-type-strategy"],
        bond_strategy=hparams["integration-bond-strategy"],
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
        cat_noise_level=args.cat_sampling_noise_level
    )
    # Set up for encoder
    # encoder = initialize_encoder(encoder_type=args.encoder_type,
    #                              encoder_ckpt_path=args.encoder_path)
    # if args.finetune_encoder:
    #     raise NotImplementedError
    # else:
    #     for param in encoder.parameters():
    #         param.requires_grad = False
    #     encoder.eval()

    # rep_sampler = initilize_rep_sampler(args, args)

    print(hparams['rep_condition'])
    hparams['rep_condition'] = False
    fm_model = MolecularCFM.load_from_checkpoint(
        args.ckpt_path,
        gen=egnn_gen,
        vocab=vocab,
        integrator=integrator,
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
        encoder=None, #encoder,
        rdm=None, #rep_sampler,
        finetune_encoder=args.finetune_encoder,
        **hparams,
        strict=False
    )
    return fm_model


def build_dm(args, hparams, vocab):
    if args.dataset == "qm9":
        coord_std = util.QM9_COORDS_STD_DEV
        bucket_limits = util.QM9_BUCKET_LIMITS

    elif args.dataset == "geom-drugs":
        coord_std = util.GEOM_COORDS_STD_DEV
        bucket_limits = util.GEOM_DRUGS_BUCKET_LIMITS

    else:
        raise ValueError(f"Unknown dataset {args.dataset}")
 
    n_bond_types = 5
    transform = partial(util.mol_transform, vocab=vocab, n_bonds=n_bond_types, coord_std=coord_std)

    if args.dataset_split == "train":
        dataset_path = Path(args.data_path) / "train.smol"
    elif args.dataset_split == "val":
        dataset_path = Path(args.data_path) / "val.smol"
    elif args.dataset_split == "test":
        dataset_path = Path(args.data_path) / "test.smol"

    dataset = GeometricDataset.load(dataset_path, transform=transform)
    dataset = dataset.sample(args.n_molecules, replacement=True)

    type_mask_index = vocab.indices_from_tokens(["<MASK>"])[0] if hparams["val-type-interpolation"] == "mask" else None
    bond_mask_index = None

    prior_sampler = GeometricNoiseSampler(
        vocab.size,
        n_bond_types,
        coord_noise="gaussian",
        type_noise=hparams["val-prior-type-noise"],
        bond_noise=hparams["val-prior-bond-noise"],
        scale_ot=hparams["val-prior-noise-scale-ot"],
        zero_com=True,
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index
    )
    eval_interpolant = GeometricInterpolant(
        prior_sampler,
        coord_interpolation="linear",
        type_interpolation=hparams["val-type-interpolation"],
        bond_interpolation=hparams["val-bond-interpolation"],
        equivariant_ot=False,
        batch_ot=False
    )
    dm = GeometricInterpolantDM(
        None,
        None,
        dataset,
        args.batch_cost,
        test_interpolant=eval_interpolant,
        bucket_limits=bucket_limits,
        bucket_cost_scale=args.bucket_cost_scale,
        pad_to_bucket=False
    )
    return dm


def dm_from_ckpt(args, vocab):
    checkpoint = torch.load(args.ckpt_path)
    hparams = checkpoint["hyper_parameters"]
    dm = build_dm(args, hparams, vocab)
    return dm


def evaluate(args, model, dm, metrics, stab_metrics):
    results_list = []
    for replicate_index in range(args.n_replicates):
        print(f"Running replicate {replicate_index + 1} out of {args.n_replicates}")
        # molecules, _, stabilities = util.generate_molecules(
        molecules, _ = util.generate_molecules(
            model,
            dm,
            args.integration_steps,
            args.ode_sampling_strategy,
            stabilities=False  # True
        )

        print("Calculating metrics...")
        results = util.calc_metrics_(molecules, metrics, stab_metrics=stab_metrics, mol_stabs=stabilities)
        results_list.append(results)
        print(results)

        # Save molecule visualizations
        print("Saving molecule visualizations...")
        
        # Create output directory for this replicate
        output_dir = f"./molecule_images/new_replicate_{replicate_index+1}"
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)


        if args.save_images:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            from rdkit.Chem import Draw
            from pymol import cmd
            import tempfile
            # Save images for each molecule in this replicate
            for i, mol in enumerate(molecules):
                if mol is not None:
                    try:
                        file_prefix=f"molecule_{i}"
                        size = (300, 300)
                        if not mol.GetNumConformers():
                            mol = Chem.AddHs(mol)
                            AllChem.EmbedMolecule(mol)
                            AllChem.MMFFOptimizeMolecule(mol)
                        # Save the molecule to a temporary file
                        temp_file = tempfile.NamedTemporaryFile(suffix='.mol', delete=False)
                        temp_filename = temp_file.name
                        temp_file.close()

                        with open(temp_filename, 'w') as f:
                            f.write(Chem.MolToMolBlock(mol))

                        # Initialize PyMOL
                        cmd.reinitialize()

                        # Load the molecule
                        cmd.load(temp_filename, 'mol')

                        # Set up the display
                        cmd.bg_color('black')
                        cmd.set('stick_radius', 0.15)
                        cmd.color('white', 'elem C')
                        cmd.color('red', 'elem O')
                        cmd.color('blue', 'elem N')
                        cmd.color('yellow', 'elem S')
                        cmd.color('cyan', 'elem F')
                        cmd.color('green', 'elem Cl')
                        cmd.color('orange', 'elem P')
                        cmd.color('purple', 'elem I')

                        # Set the view size
                        cmd.viewport(size[0], size[1])

                        # Save different views
                        rotations = [
                            (0, 0, 0),       # Front view
                            (90, 0, 0),      # Top view
                            (0, 90, 0),      # Side view
                            (45, 45, 0),     # Angled view
                        ]

                        view_names = ['front', 'top', 'side', 'angle']

                        for i, (rot_x, rot_y, rot_z) in enumerate(rotations):
                            view_name = view_names[i]

                            # Reset rotation
                            cmd.reset()

                            # Apply rotation
                            cmd.rotate('x', rot_x)
                            cmd.rotate('y', rot_y)
                            cmd.rotate('z', rot_z)

                            # Center and zoom
                            cmd.center('mol')
                            cmd.zoom('mol')

                            # Save the image
                            output_path = os.path.join(output_dir, f"{file_prefix}_3d_{view_name}.png")
                            cmd.png(output_path, width=size[0], height=size[1], dpi=300, ray=1)

                        # Clean up
                        os.unlink(temp_filename)
                    except Exception as e:
                        print(f"Failed to save visualization for molecule {i}: {e}")

    results_dict = {key: [] for key in results_list[0].keys()}
    for results in results_list:
        for metric, value in results.items():
            results_dict[metric].append(value.item())

    mean_results = {metric: np.mean(values) for metric, values in results_dict.items()}
    std_results = {metric: np.std(values) for metric, values in results_dict.items()}

    return mean_results, std_results, results_dict


def main(args):
    print(f"Running evaluation script for {args.n_replicates} replicates with {args.n_molecules} molecules each...")
    print(f"Using model stored at {args.ckpt_path}")

    if args.n_replicates < 1:
        raise ValueError("n_replicates must be at least 1.")

    L.seed_everything(12345)
    util.disable_lib_stdout()
    util.configure_fs()

    print("Building model vocab...")
    vocab = util.build_vocab()
    print("Vocab complete.")

    print("Loading datamodule...")
    dm = dm_from_ckpt(args, vocab)
    print("Datamodule complete.")

    print(f"Loading model...")
    model = load_model(args, vocab)
    print("Model complete.")

    print("Initialising metrics...")
    metrics, stab_metrics = util.init_metrics(args.data_path, model)
    print("Metrics complete.")

    print("Running evaluation...")
    avg_results, std_results, list_results = evaluate(args, model, dm, metrics, stab_metrics)
    print("Evaluation complete.")

    util.print_results(avg_results, std_results=std_results)

    print("All replicate results...")
    print(f"{'Metric':<22}Result")
    print("-" * 30)

    for metric, results_list in list_results.items():
        print(f"{metric:<22}{results_list}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, default='./data/geom/smol')
    parser.add_argument("--dataset", type=str, default='geom-drugs')

    parser.add_argument("--batch_cost", type=int, default=DEFAULT_BATCH_COST)
    parser.add_argument("--dataset_split", type=str, default=DEFAULT_DATASET_SPLIT)
    parser.add_argument("--n_molecules", type=int, default=DEFAULT_N_MOLECULES)
    parser.add_argument("--n_replicates", type=int, default=DEFAULT_N_REPLICATES)
    parser.add_argument("--integration_steps", type=int, default=DEFAULT_INTEGRATION_STEPS)
    parser.add_argument("--cat_sampling_noise_level", type=int, default=DEFAULT_CAT_SAMPLING_NOISE_LEVEL)
    parser.add_argument("--ode_sampling_strategy", type=str, default=DEFAULT_ODE_SAMPLING_STRATEGY)
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--encoder_type", type=str, default='unimol_global')  # 'unimol' or 'frad'
    parser.add_argument("--encoder_path", type=str, default='./checkpoints/unimol_global.pt')
    parser.add_argument("--finetune_encoder", action="store_true")
    parser.add_argument("--noise_sigma", type=float, default=0.0)  # larger noise sigma is better for DRUGS

    parser.add_argument("--sampler", type=str, default='GtSampler')  # ["PCSampler", "GtSampler", "DDIMSampler"]
    #     For DDIMSampler and PCSampler
    parser.add_argument("--rdm_ckpt", type=str,
                        default='./checkpoints/rdm_ckpts/unimol-huge-checkpoint-77.pth')  # resume from checkpoint
    # './checkpoints/rdm_ckpts/rdm_diffusion_finetuned.pth'
    #     For DDIMSampler
    parser.add_argument("--step_num", type=int, default=250)
    parser.add_argument("--eta", type=float, default=1.0)
    #     For PCSampler
    parser.add_argument("--inv_temp", type=float, default=1.0)
    parser.add_argument("--n_steps", type=int, default=10)  # 5
    parser.add_argument("--snr", type=float, default=0.05)
    #     For GtSampler
    parser.add_argument("--Gt_dataset", type=str, default="train")  # ["train", "test", "valid"]

    parser.add_argument("--bucket_cost_scale", type=str, default=DEFAULT_BUCKET_COST_SCALE)

    # Allow overridding for EGNN arch since some flowmodels were not saved with a value for n_layers
    parser.add_argument("--n_layers", type=int, default=None)

    parser.add_argument("--categorical_strategy", type=str, default=DEFAULT_CATEGORICAL_STRATEGY)

    args = parser.parse_args()
    main(args)
