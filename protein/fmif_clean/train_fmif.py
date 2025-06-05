import argparse
import random
import string
import datetime
from datetime import date
import os
import shutil
import torch
import pyrosetta

pyrosetta.init(extra_options="-out:level 100")
# from pyrosetta.rosetta.core.pack.task import TaskFactory
# from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover
# from pyrosetta.rosetta.protocols.relax import FastRelax
# from pyrosetta.rosetta.core.pack.task.operation import RestrictToRepacking
from pyrosetta import *
from biotite.sequence.io import fasta
import pandas as pd

ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
runid = ''.join(random.choice(string.ascii_letters) for i in range(10)) + '_' + str(
    datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))


def cal_rmsd(S_sp, S, name, the_folding_model, mask_for_loss, pdb_path, runid, base_path):
    with torch.no_grad():
        results_list = []
        sc_output_dir = os.path.join(base_path, 'sc_tmp', runid)
        # os.makedirs(sc_output_dir, exist_ok=True)

        for _it, ssp in enumerate(S_sp):
            if os.path.exists(sc_output_dir):
                shutil.rmtree(sc_output_dir)
            os.makedirs(sc_output_dir, exist_ok=False)
            the_name = name[_it]
            the_pdb_path = os.path.join(pdb_path, the_name.split('_')[0][1:3], f"{the_name}.pdb")
            true_detok_seq = "".join([ALPHABET[x] for _ix, x in enumerate(S[_it]) if mask_for_loss[_it][_ix] == 1])
            true_pose = pyrosetta.pose_from_file(the_pdb_path)

            os.makedirs(os.path.join(sc_output_dir, 'fmif_seqs'), exist_ok=False)
            codesign_fasta = fasta.FastaFile()
            detok_seq = "".join([ALPHABET[x] for _ix, x in enumerate(ssp) if mask_for_loss[_it][_ix] == 1])
            codesign_fasta['codesign_seq_1'] = detok_seq
            codesign_fasta_path = os.path.join(sc_output_dir, 'fmif_seqs', 'codesign.fa')
            codesign_fasta.write(codesign_fasta_path)

            folded_dir = os.path.join(sc_output_dir, 'folded')
            # if os.path.exists(folded_dir):
            #     shutil.rmtree(folded_dir)
            os.makedirs(folded_dir, exist_ok=False)

            folded_output = the_folding_model.fold_fasta(codesign_fasta_path, folded_dir)
            gen_folded_pdb_path = os.path.join(folded_dir, 'folded_codesign_seq_1.pdb')
            pose = pyrosetta.pose_from_file(gen_folded_pdb_path)

            print(the_name, mask_for_loss[_it].sum().item(), true_pose.total_residue(), pose.total_residue())
            if the_name == '5naf_B':
                print(mask_for_loss[_it])

            gen_true_bbrmsd = pyrosetta.rosetta.core.scoring.bb_rmsd(true_pose, pose)
            seq_revovery = (S_sp[_it] == S[_it]).float().mean().item()
            plddt = folded_output['plddt'].loc[0]
            resultdf = pd.DataFrame(columns=['gen_true_bb_rmsd', 'seq_recovery', 'plddt'])
            resultdf.loc[0] = [gen_true_bbrmsd, seq_revovery, plddt]
            resultdf['seq'] = detok_seq
            resultdf['true_seq'] = true_detok_seq
            resultdf['name'] = the_name
            resultdf['num'] = _it
            results_list.append(resultdf)

    return results_list


def main(args):
    import time, os
    import numpy as np
    import torch
    import queue
    import os.path
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    # try global import
    # from fmif.utils import worker_init_fn, get_pdbs, loader_pdb, build_training_clusters, PDB_dataset, StructureDataset, StructureLoader, set_seed
    # from fmif.model_utils import featurize, loss_smoothed, loss_nll, ProteinMPNNFMIF
    # from fmif.fm_utils import Interpolant, fm_model_step
    from utils import worker_init_fn, get_pdbs, loader_pdb, build_training_clusters, PDB_dataset, StructureDataset, \
        StructureLoader, set_seed
    from model_utils import featurize, loss_smoothed, loss_nll, ProteinMPNNFMIF, loss_repr
    from fm_utils import Interpolant, fm_model_step
    from tqdm import tqdm
    import wandb
    from types import SimpleNamespace
    import sys
    # sys.path.append(os.path.expanduser('~/your/path/to/folder'))
    from multiflow.models import folding_model

    scaler = torch.cuda.amp.GradScaler()

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    path_for_outputs = os.path.join(args.output_path)
    base_folder = time.strftime(path_for_outputs, time.localtime())
    pdb_base_path = os.path.join(args.base_path, 'pmpnn/pdb_2021aug02/pdb_processed')

    base_folder = os.path.join(base_folder, args.name, runid)
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    if base_folder[-1] != '/':
        base_folder += '/'
    subfolders = ['model_weights']
    for subfolder in subfolders:
        if not os.path.exists(base_folder + subfolder):
            os.makedirs(base_folder + subfolder)

    PATH = args.previous_checkpoint

    logfile = base_folder + 'log.txt'
    if not PATH:
        with open(logfile, 'w') as f:
            f.write('Epoch\tTrain\tValidation\n')

    assert torch.cuda.is_available(), "CUDA is not available"
    # set random seed
    set_seed(args.seed, use_cuda=True)

    # wandb
    if not args.debug:
        
        wandb.init(project='pmpnn', name=args.name + runid, dir=base_folder, config=args)
        curr_date = date.today().strftime("%B %d, %Y")
        wandb.config.update({"curr_date": curr_date}, allow_val_change=True)
        wandb.config.update(args, allow_val_change=True)
    else:
        with open(logfile, 'a') as f:
            f.write("Debug mode, not logging to wandb\n")

    with open(logfile, 'a') as f:
        f.write(f"Run ID: {runid}\n")
        f.write(f"Arguments: {args}\n")

    data_path = os.path.join(args.base_path,
                             'pmpnn/raw/pdb_2021aug02')  # 'pmpnn/raw/pdb_2021aug02', or 'pmpnn/raw/pdb_2021aug02_sample'
    params = {
        "LIST": f"{data_path}/list_filtered.csv",
        # "LIST"    : f"{data_path}/list.csv",
        "VAL": f"{data_path}/valid_clusters.txt",
        "TEST": f"{data_path}/test_clusters.txt",
        "DIR": f"{data_path}",
        "DATCUT": "2030-Jan-01",
        "RESCUT": args.rescut,  # resolution cutoff for PDBs
        "HOMO": 0.70,  # min seq.id. to detect homo chains
        "REPR_DIR": args.repr_dir,
        "MAXLEN": args.max_protein_length
    }

    LOAD_PARAM = {'batch_size': 1,
                  'shuffle': True,
                  'pin_memory': False,
                  'num_workers': 4,
                  # 'prefetch_factor': 2,
                  # 'persistent_workers': True,
                  # 'multiprocessing_context': 'spawn'
                  }

    REPR_DIMS = {'single': 384,
                 'pair': 128,
                 'structure': 768}

    if args.debug:
        args.num_examples_per_epoch = 50
        args.max_protein_length = 1000
        args.batch_size = 1000

    train, valid, test = build_training_clusters(params, args.debug)
    print(len(train), len(valid), len(test))  # 23346 1464 1539

    import pickle
    with open(f"{data_path}/cluster_seq_dict_removeX.pkl", "rb") as f:
        cluster_seq_dict_removeX = pickle.load(f)

    # keep items in train that have the same key in cluster_seq_dict_removeX
    cids = set(list(cluster_seq_dict_removeX.keys()))
    train = {k: v for k, v in train.items() if k in cids}
    valid = {k: v for k, v in valid.items() if k in cids}
    test = {k: v for k, v in test.items() if k in cids}
    print(len(train), len(valid), len(test))  # 23079 1455 1533

    train_set = PDB_dataset(list(train.keys()), loader_pdb, train, params, load_repr=args.load_repr)
    train_loader = torch.utils.data.DataLoader(train_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)
    valid_set = PDB_dataset(list(valid.keys()), loader_pdb, valid, params, load_repr=args.load_repr)
    valid_loader = torch.utils.data.DataLoader(valid_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)
    test_set = PDB_dataset(list(test.keys()), loader_pdb, test, params, load_repr=args.load_repr)
    test_loader = torch.utils.data.DataLoader(test_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)

    model = ProteinMPNNFMIF(node_features=args.hidden_dim,
                            edge_features=args.hidden_dim,
                            hidden_dim=args.hidden_dim,
                            num_encoder_layers=args.num_encoder_layers,
                            num_decoder_layers=args.num_decoder_layers,
                            k_neighbors=args.num_neighbors,
                            dropout=args.dropout,
                            augment_eps=args.backbone_noise,
                            update_edge=args.update_edge,
                            align_depth=args.align_depth,
                            learnable_node=args.learnable_node,
                            single_dim=REPR_DIMS['single'],
                            pair_dim=REPR_DIMS['pair'],
                            structure_dim=REPR_DIMS['structure'],
                            mdlm_parameterization=args.mdlm_parameterization, )
    model.to(device)

    noise_interpolant = Interpolant(args)
    noise_interpolant.set_device(device)

    if args.calculate_rmsd:
        folding_cfg = {
            'seq_per_sample': 1,
            'folding_model': 'esmf',
            'own_device': False,
            'pmpnn_path': './ProteinMPNN/',
            'pt_hub_dir': os.path.join(args.base_path, '.cache/torch/'),
            'colabfold_path': os.path.join(args.base_path, 'colabfold-conda/bin/colabfold_batch')  # for AF2
        }
        folding_cfg = SimpleNamespace(**folding_cfg)
        the_folding_model = folding_model.FoldingModel(folding_cfg)

    if PATH:
        checkpoint = torch.load(PATH)
        total_step = checkpoint['step']  # write total_step from the checkpoint
        epoch = checkpoint['epoch']  # write epoch from the checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        total_step = 0
        epoch = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if PATH:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    start_time = time.time()
    pdb_dict_train = get_pdbs(train_loader, 1, args.max_protein_length, args.num_examples_per_epoch, args.load_repr)
    pdb_dict_valid = get_pdbs(valid_loader, 1, args.max_protein_length, args.num_examples_per_epoch, args.load_repr)
    pdb_dict_test = get_pdbs(test_loader, 1, args.max_protein_length, args.num_examples_per_epoch, args.load_repr)
    
    dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=args.max_protein_length)
    dataset_valid = StructureDataset(pdb_dict_valid, truncate=None, max_length=args.max_protein_length)
    dataset_test = StructureDataset(pdb_dict_test, truncate=None, max_length=args.max_protein_length)
    loader_train = StructureLoader(dataset_train, batch_size=args.batch_size)
    loader_valid = StructureLoader(dataset_valid, batch_size=args.batch_size)
    loader_test = StructureLoader(dataset_test, batch_size=args.batch_size)
    print(len(loader_train), len(loader_valid), len(loader_test))
    end_time = time.time()
    print(end_time - start_time)

    reload_c = 1
    for e in range(args.num_epochs):
        t0 = time.time()
        e = epoch + e
        # print("epoch" + str(e))
        model.train()
        train_sum, train_weights, train_acc = 0., 0., 0.
        train_single, train_pair, train_structure = 0., 0., 0.
        train_single_masked, train_pair_masked, train_structure_masked = 0., 0., 0.
        train_masked_sum, train_masked_weights, train_masked_acc = 0., 0., 0.
        
        if args.repa_weight_decay == "constant":
            _repa_weight_decay = 1.0
        elif args.repa_weight_decay == "linear":
            _repa_weight_decay = max(1.0 - e / args.repa_epoch, 0.)
        elif args.repa_weight_decay == "cosine":
            _repa_weight_decay = max((1.0 + np.cos(np.pi * e / args.repa_epoch)) / 2, 0.)
        else:
            raise NotImplementedError

        top_epoch = args.diffusion_warm_up_epoch + args.start_diffusion_epoch
        if e <= args.start_diffusion_epoch:
            _diffusion_loss_decay = 0.0
        elif args.start_diffusion_epoch < e < top_epoch:
            _diffusion_loss_decay = (e - args.start_diffusion_epoch) / args.diffusion_warm_up_epoch
        else:
            if args.diffusion_decay == "constant":
                _diffusion_loss_decay = 1.0
            elif args.diffusion_decay == "linear":
                _diffusion_loss_decay = 1.0 - (e - top_epoch) / (args.num_epochs - top_epoch)
            elif args.diffusion_decay == "cosine":
                _diffusion_loss_decay = (1.0 + np.cos(np.pi * (e - top_epoch) / args.num_epochs - top_epoch)) / 2
            else:
                raise NotImplementedError

        for _, batch in enumerate(loader_train):
            start_batch = time.time()
            X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all, repr_single, repr_pair, repr_structure = featurize(
                batch, device, REPR_DIMS, args.load_repr)
            noisy_batch = noise_interpolant.corrupt_batch((X, S, mask, chain_M, residue_idx, chain_encoding_all))
            elapsed_featurize = time.time() - start_batch
            optimizer.zero_grad()
            mask_for_loss = mask * chain_M
            mask_for_residue = noisy_batch['is_masked']
            t = noisy_batch['t']  # t=1 is clean data
            # mask_repr_loss = mask_for_residue if args.mask_repr_loss else mask_for_loss
            if torch.any(torch.sum(noisy_batch['mask'] * noisy_batch['chain_M'], dim=-1) < 1):
                import warnings
                warnings.warn("Encouter empty batch")
                continue
            _detach_repr = False if args.repr_weight > 0.0 else True
            _repr_weight = (args.repr_weight * _repa_weight_decay) if args.repr_weight > 0.0 else 1.0
            if args.mdlm_parameterization: assert args.mixed_precision == False, "MDLM parameterization does not support mixed precision training"
            if args.mixed_precision:
                with torch.cuda.amp.autocast():
                    log_probs, zs_single, zs_pair, zs_structure, E_idx = fm_model_step(model, noisy_batch,
                                                                                       return_rep=True,
                                                                                       detach_repr=_detach_repr)
                    _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss, t=t, t_schedule=args.t_schedule)
                    loss_av_smoothed *= _diffusion_loss_decay
                    if args.load_repr:
                        loss_single, loss_pair, loss_structure, loss_single_masked, loss_pair_masked, loss_structure_masked = loss_repr(
                            repr_single, repr_pair, repr_structure,
                            zs_single, zs_pair, zs_structure, E_idx,
                            mask_for_loss, mask_for_residue, repr_norm=args.repr_norm, repr_noise=args.repr_noise)
                        
                        if args.mask_repr_loss:
                            loss_av_smoothed += (loss_single_masked * args.repa_coeff[0] + loss_pair_masked *
                                                 args.repa_coeff[1] + loss_structure_masked * args.repa_coeff[
                                                     2]) * _repr_weight
                        else:
                            loss_av_smoothed += (loss_single * args.repa_coeff[0] + loss_pair * args.repa_coeff[
                                1] + loss_structure * args.repa_coeff[2]) * _repr_weight

                scaler.scale(loss_av_smoothed).backward()
                if args.gradient_norm > 0.0:
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                log_probs, zs_single, zs_pair, zs_structure, E_idx = fm_model_step(model, noisy_batch,
                                                                                   return_rep=True,
                                                                                   detach_repr=_detach_repr)
                
                _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss, t=t, t_schedule=args.t_schedule)
                loss_av_smoothed *= _diffusion_loss_decay
                if args.load_repr:
                    loss_single, loss_pair, loss_structure, loss_single_masked, loss_pair_masked, loss_structure_masked = loss_repr(
                        repr_single, repr_pair, repr_structure,
                        zs_single, zs_pair, zs_structure, E_idx,
                        mask_for_loss, mask_for_residue, repr_norm=args.repr_norm, repr_noise=args.repr_noise)
                    
                    if args.mask_repr_loss:
                        loss_av_smoothed += (loss_single_masked * args.repa_coeff[0] + loss_pair_masked *
                                             args.repa_coeff[1] + loss_structure_masked * args.repa_coeff[
                                                 2]) * _repr_weight
                    else:
                        loss_av_smoothed += (loss_single * args.repa_coeff[0] + loss_pair * args.repa_coeff[
                            1] + loss_structure * args.repa_coeff[2]) * _repr_weight

                loss_av_smoothed.backward()

                if args.gradient_norm > 0.0:
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)
                optimizer.step()

            loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
            train_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
            train_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
            train_weights += torch.sum(mask_for_loss).cpu().data.numpy()

            train_masked_sum += torch.sum(loss * mask_for_residue).cpu().data.numpy()
            train_masked_acc += torch.sum(true_false * mask_for_residue).cpu().data.numpy()
            train_masked_weights += torch.sum(mask_for_residue).cpu().data.numpy()
            
            if args.load_repr:
                train_single += loss_single.item()
                train_pair += loss_pair.item()
                train_structure += loss_structure.item()
                train_single_masked += loss_single_masked.item()
                train_pair_masked += loss_pair_masked.item()
                train_structure_masked += loss_structure_masked.item()
            total_step += 1

        model.eval()
        with torch.no_grad():
            validation_sum, validation_weights, validation_acc = 0., 0., 0.
            validation_single, validation_pair, validation_structure = 0., 0., 0.
            validation_single_masked, validation_pair_masked, validation_structure_masked = 0., 0., 0.
            validation_masked_sum, validation_masked_weights, validation_masked_acc = 0., 0., 0.
            for _, batch in enumerate(loader_valid):
                X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all, repr_single, repr_pair, repr_structure = featurize(
                    batch, device, REPR_DIMS, args.load_repr)
                noisy_batch = noise_interpolant.corrupt_batch((X, S, mask, chain_M, residue_idx, chain_encoding_all))
                log_probs, zs_single, zs_pair, zs_structure, E_idx = model(X, noisy_batch['S_t'], mask, chain_M,
                                                                           residue_idx,
                                                                           chain_encoding_all, return_rep=True,
                                                                           detach_repr=False)
                mask_for_loss = mask * chain_M
                mask_for_residue = noisy_batch['is_masked']
                loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
                if args.load_repr:
                    loss_single, loss_pair, loss_structure, loss_single_masked, loss_pair_masked, loss_structure_masked = loss_repr(
                        repr_single, repr_pair, repr_structure,
                        zs_single, zs_pair, zs_structure, E_idx,
                        mask_for_loss, mask_for_residue, repr_norm=args.repr_norm)
                validation_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                validation_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                validation_weights += torch.sum(mask_for_loss).cpu().data.numpy()
                validation_masked_sum += torch.sum(loss * mask_for_residue).cpu().data.numpy()
                validation_masked_acc += torch.sum(true_false * mask_for_residue).cpu().data.numpy()
                validation_masked_weights += torch.sum(mask_for_residue).cpu().data.numpy()
                if args.load_repr:
                    validation_single += loss_single.item()
                    validation_pair += loss_pair.item()
                    validation_structure += loss_structure.item()
                    validation_single_masked += loss_single_masked.item()
                    validation_pair_masked += loss_pair_masked.item()
                    validation_structure_masked += loss_structure_masked.item()

            test_sum, test_weights, test_acc = 0., 0., 0.
            test_single, test_pair, test_structure = 0., 0., 0.
            test_single_masked, test_pair_masked, test_structure_masked = 0., 0., 0.
            test_masked_sum, test_masked_weights, test_masked_acc = 0., 0., 0.
            for _, batch in enumerate(loader_test):
                X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all, repr_single, repr_pair, repr_structure = featurize(
                    batch, device, REPR_DIMS, args.load_repr)
                noisy_batch = noise_interpolant.corrupt_batch((X, S, mask, chain_M, residue_idx, chain_encoding_all))
                log_probs, zs_single, zs_pair, zs_structure, E_idx = model(X, noisy_batch['S_t'], mask, chain_M,
                                                                           residue_idx,
                                                                           chain_encoding_all, return_rep=True,
                                                                           detach_repr=False)
                mask_for_loss = mask * chain_M
                mask_for_residue = noisy_batch['is_masked']
                loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
                if args.load_repr:
                    loss_single, loss_pair, loss_structure, loss_single_masked, loss_pair_masked, loss_structure_masked = loss_repr(
                        repr_single, repr_pair, repr_structure,
                        zs_single, zs_pair, zs_structure, E_idx,
                        mask_for_loss, mask_for_residue, repr_norm=args.repr_norm)
                test_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                test_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                test_weights += torch.sum(mask_for_loss).cpu().data.numpy()
                test_masked_sum += torch.sum(loss * mask_for_residue).cpu().data.numpy()
                test_masked_acc += torch.sum(true_false * mask_for_residue).cpu().data.numpy()
                test_masked_weights += torch.sum(mask_for_residue).cpu().data.numpy()
                if args.load_repr:
                    test_single += loss_single.item()
                    test_pair += loss_pair.item()
                    test_structure += loss_structure.item()
                    test_single_masked += loss_single_masked.item()
                    test_pair_masked += loss_pair_masked.item()
                    test_structure_masked += loss_structure_masked.item()

        validation_sp_accuracy_ = '-'
        test_sp_accuracy_ = '-'
        if args.calculate_rmsd:
            validation_avg_rmsd_ = '-'
            validation_mid_rmsd_ = '-'
            validation_rmsd_rate_ = '-'
            validation_avg_plddt_ = '-'
            validation_mid_plddt_ = '-'
            validation_plddt_rate_ = '-'
            test_avg_rmsd_ = '-'
            test_mid_rmsd_ = '-'
            test_rmsd_rate_ = '-'
            test_avg_plddt_ = '-'
            test_mid_plddt_ = '-'
            test_plddt_rate_ = '-'

        if (e + 1) % args.eval_every_n_epochs == 0:
            with torch.no_grad():
                print(len(loader_valid))
                valid_sp_acc, valid_sp_weights = 0., 0.
                valid_results_merge = []
                for _, batch in tqdm(enumerate(loader_valid)):
                    X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all, repr_single, repr_pair, repr_structure = featurize(
                        batch, device, REPR_DIMS, args.load_repr)
                    S_sp, _, _ = noise_interpolant.sample(model, X, mask, chain_M, residue_idx, chain_encoding_all)
                    true_false_sp = (S_sp == S).float()
                    mask_for_loss = mask * chain_M
                    valid_sp_acc += torch.sum(true_false_sp * mask_for_loss).cpu().data.numpy()
                    valid_sp_weights += torch.sum(mask_for_loss).cpu().data.numpy()
                    names = [b['name'] for b in batch]
                    if args.calculate_rmsd:
                        results_list = cal_rmsd(S_sp, S, names, the_folding_model, mask_for_loss, pdb_base_path,
                                                runid, args.base_path)
                        valid_results_merge.extend(results_list)
                validation_sp_accuracy = valid_sp_acc / valid_sp_weights
                validation_sp_accuracy_ = np.format_float_positional(np.float32(validation_sp_accuracy),
                                                                     unique=False, precision=3)
                if args.calculate_rmsd:
                    valid_results_merge = pd.concat(valid_results_merge)
                    validation_avg_rmsd = valid_results_merge['gen_true_bb_rmsd'].mean()
                    validation_mid_rmsd = valid_results_merge['gen_true_bb_rmsd'].median()
                    validation_rmsd_rate = valid_results_merge['gen_true_bb_rmsd'].apply(
                        lambda x: 1 if x < 2 else 0).mean()
                    validation_avg_rmsd_ = np.format_float_positional(validation_avg_rmsd, unique=False,
                                                                      precision=3)
                    validation_mid_rmsd_ = np.format_float_positional(validation_mid_rmsd, unique=False,
                                                                      precision=3)
                    validation_rmsd_rate_ = np.format_float_positional(validation_rmsd_rate, unique=False,
                                                                       precision=3)
                    validation_avg_plddt = valid_results_merge['plddt'].mean()
                    validation_mid_plddt = valid_results_merge['plddt'].median()
                    validation_plddt_rate = valid_results_merge['plddt'].apply(lambda x: 1 if x > 80 else 0).mean()
                    validation_avg_plddt_ = np.format_float_positional(validation_avg_plddt, unique=False,
                                                                       precision=3)
                    validation_mid_plddt_ = np.format_float_positional(validation_mid_plddt, unique=False,
                                                                       precision=3)
                    validation_plddt_rate_ = np.format_float_positional(validation_plddt_rate, unique=False,
                                                                        precision=3)

                print(len(loader_test))
                test_sp_acc, test_sp_weights = 0., 0.
                test_results_merge = []
                for _, batch in tqdm(enumerate(loader_test)):
                    X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all, repr_single, repr_pair, repr_structure = featurize(
                        batch, device, REPR_DIMS, args.load_repr)
                    S_sp, _, _ = noise_interpolant.sample(model, X, mask, chain_M, residue_idx, chain_encoding_all)
                    true_false_sp = (S_sp == S).float()
                    mask_for_loss = mask * chain_M
                    test_sp_acc += torch.sum(true_false_sp * mask_for_loss).cpu().data.numpy()
                    test_sp_weights += torch.sum(mask_for_loss).cpu().data.numpy()
                    names = [b['name'] for b in batch]
                    if args.calculate_rmsd:
                        results_list = cal_rmsd(S_sp, S, names, the_folding_model, mask_for_loss, pdb_base_path,
                                                runid, args.base_path)
                        test_results_merge.extend(results_list)
                test_sp_accuracy = test_sp_acc / test_sp_weights
                test_sp_accuracy_ = np.format_float_positional(np.float32(test_sp_accuracy), unique=False,
                                                               precision=3)
                if args.calculate_rmsd:
                    test_results_merge = pd.concat(test_results_merge)
                    test_avg_rmsd = test_results_merge['gen_true_bb_rmsd'].mean()
                    test_mid_rmsd = test_results_merge['gen_true_bb_rmsd'].median()
                    test_rmsd_rate = test_results_merge['gen_true_bb_rmsd'].apply(
                        lambda x: 1 if x < 2 else 0).mean()
                    test_avg_rmsd_ = np.format_float_positional(test_avg_rmsd, unique=False, precision=3)
                    test_mid_rmsd_ = np.format_float_positional(test_mid_rmsd, unique=False, precision=3)
                    test_rmsd_rate_ = np.format_float_positional(test_rmsd_rate, unique=False, precision=3)
                    test_avg_plddt = test_results_merge['plddt'].mean()
                    test_mid_plddt = test_results_merge['plddt'].median()
                    test_plddt_rate = test_results_merge['plddt'].apply(lambda x: 1 if x > 80 else 0).mean()
                    test_avg_plddt_ = np.format_float_positional(test_avg_plddt, unique=False, precision=3)
                    test_mid_plddt_ = np.format_float_positional(test_mid_plddt, unique=False, precision=3)
                    test_plddt_rate_ = np.format_float_positional(test_plddt_rate, unique=False, precision=3)

        train_loss = train_sum / train_weights
        train_accuracy = train_acc / train_weights
        train_perplexity = np.exp(train_loss)
        train_masked_loss = train_masked_sum / train_masked_weights
        train_masked_accuracy = train_masked_acc / train_masked_weights
        train_single = train_single / len(loader_train)
        train_pair = train_pair / len(loader_train)
        train_structure = train_structure / len(loader_train)
        train_single_masked = train_single_masked / len(loader_train)
        train_pair_masked = train_pair_masked / len(loader_train)
        train_structure_masked = train_structure_masked / len(loader_train)

        validation_loss = validation_sum / validation_weights
        validation_accuracy = validation_acc / validation_weights
        validation_perplexity = np.exp(validation_loss)
        validation_masked_loss = validation_masked_sum / validation_masked_weights
        validation_masked_accuracy = validation_masked_acc / validation_masked_weights
        validation_single = validation_single / len(loader_valid)
        validation_pair = validation_pair / len(loader_valid)
        validation_structure = validation_structure / len(loader_valid)
        validation_single_masked = validation_single_masked / len(loader_valid)
        validation_pair_masked = validation_pair_masked / len(loader_valid)
        validation_structure_masked = validation_structure_masked / len(loader_valid)

        test_loss = test_sum / test_weights
        test_accuracy = test_acc / test_weights
        test_perplexity = np.exp(test_loss)
        test_masked_loss = test_masked_sum / test_masked_weights
        test_masked_accuracy = test_masked_acc / test_masked_weights
        test_single = test_single / len(loader_test)
        test_pair = test_pair / len(loader_test)
        test_structure = test_structure / len(loader_test)
        test_single_masked = test_single_masked / len(loader_test)
        test_pair_masked = test_pair_masked / len(loader_test)
        test_structure_masked = test_structure_masked / len(loader_test)

        train_perplexity_ = np.format_float_positional(np.float32(train_perplexity), unique=False, precision=3)
        validation_perplexity_ = np.format_float_positional(np.float32(validation_perplexity), unique=False,
                                                            precision=3)
        test_perplexity_ = np.format_float_positional(np.float32(test_perplexity), unique=False, precision=3)
        train_accuracy_ = np.format_float_positional(np.float32(train_accuracy), unique=False, precision=3)
        validation_accuracy_ = np.format_float_positional(np.float32(validation_accuracy), unique=False,
                                                          precision=3)
        test_accuracy_ = np.format_float_positional(np.float32(test_accuracy), unique=False, precision=3)
        train_single_ = np.format_float_positional(np.float32(train_single), unique=False, precision=3)
        train_pair_ = np.format_float_positional(np.float32(train_pair), unique=False, precision=3)
        train_structure_ = np.format_float_positional(np.float32(train_structure), unique=False, precision=3)
        validation_single_ = np.format_float_positional(np.float32(validation_single), unique=False, precision=3)
        validation_pair_ = np.format_float_positional(np.float32(validation_pair), unique=False, precision=3)
        validation_structure_ = np.format_float_positional(np.float32(validation_structure), unique=False,
                                                           precision=3)
        test_single_ = np.format_float_positional(np.float32(test_single), unique=False, precision=3)
        test_pair_ = np.format_float_positional(np.float32(test_pair), unique=False, precision=3)
        test_structure_ = np.format_float_positional(np.float32(test_structure), unique=False, precision=3)
        train_single_masked_ = np.format_float_positional(np.float32(train_single_masked), unique=False, precision=3)
        train_pair_masked_ = np.format_float_positional(np.float32(train_pair_masked), unique=False, precision=3)
        train_structure_masked_ = np.format_float_positional(np.float32(train_structure_masked), unique=False,
                                                             precision=3)
        validation_single_masked_ = np.format_float_positional(np.float32(validation_single_masked), unique=False,
                                                               precision=3)
        validation_pair_masked_ = np.format_float_positional(np.float32(validation_pair_masked), unique=False,
                                                             precision=3)
        validation_structure_masked_ = np.format_float_positional(np.float32(validation_structure_masked), unique=False,
                                                                  precision=3)
        test_single_masked_ = np.format_float_positional(np.float32(test_single_masked), unique=False, precision=3)
        test_pair_masked_ = np.format_float_positional(np.float32(test_pair_masked), unique=False, precision=3)
        test_structure_masked_ = np.format_float_positional(np.float32(test_structure_masked), unique=False,
                                                            precision=3)

        train_masked_loss_ = np.format_float_positional(np.float32(train_masked_loss), unique=False, precision=3)
        train_masked_accuracy_ = np.format_float_positional(np.float32(train_masked_accuracy), unique=False,
                                                            precision=3)
        validation_masked_loss_ = np.format_float_positional(np.float32(validation_masked_loss), unique=False,
                                                             precision=3)
        validation_masked_accuracy_ = np.format_float_positional(np.float32(validation_masked_accuracy), unique=False,
                                                                 precision=3)
        test_masked_loss_ = np.format_float_positional(np.float32(test_masked_loss), unique=False, precision=3)
        test_masked_accuracy_ = np.format_float_positional(np.float32(test_masked_accuracy), unique=False, precision=3)

        t1 = time.time()
        dt = np.format_float_positional(np.float32(t1 - t0), unique=False, precision=1)
        if args.calculate_rmsd:
            with open(logfile, 'a') as f:
                f.write(
                    f'epoch: {e + 1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, test: {test_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}, test_acc: {test_accuracy_}, valid_sp_acc: {validation_sp_accuracy_}, test_sp_acc: {test_sp_accuracy_}, ' +
                    f"valid_avg_rmsd: {validation_avg_rmsd_}, valid_mid_rmsd: {validation_mid_rmsd_}, valid_rmsd_rate: {validation_rmsd_rate_}, valid_avg_plddt: {validation_avg_plddt_}, valid_mid_plddt: {validation_mid_plddt_}, valid_plddt_rate: {validation_plddt_rate_}, " +
                    f"test_avg_rmsd: {test_avg_rmsd_}, test_mid_rmsd: {test_mid_rmsd_}, test_rmsd_rate: {test_rmsd_rate_}, test_avg_plddt: {test_avg_plddt_}, test_mid_plddt: {test_mid_plddt_}, test_plddt_rate: {test_plddt_rate_}\n")
            print(
                f'epoch: {e + 1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, test: {test_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}, test_acc: {test_accuracy_}, valid_sp_acc: {validation_sp_accuracy_}, test_sp_acc: {test_sp_accuracy_}, ' +
                f"valid_avg_rmsd: {validation_avg_rmsd_}, valid_mid_rmsd: {validation_mid_rmsd_}, valid_rmsd_rate: {validation_rmsd_rate_}, valid_avg_plddt: {validation_avg_plddt_}, valid_mid_plddt: {validation_mid_plddt_}, valid_plddt_rate: {validation_plddt_rate_}, " +
                f"test_avg_rmsd: {test_avg_rmsd_}, test_mid_rmsd: {test_mid_rmsd_}, test_rmsd_rate: {test_rmsd_rate_}, test_avg_plddt: {test_avg_plddt_}, test_mid_plddt: {test_mid_plddt_}, test_plddt_rate: {test_plddt_rate_}")

            if not args.debug:
                wandb.log({"train_perplexity": train_perplexity, "valid_perplexity": validation_perplexity,
                           "test_perplexity": test_perplexity,
                           "train_accuracy": train_accuracy, "valid_accuracy": validation_accuracy,
                           "test_accuracy": test_accuracy},
                          step=total_step, commit=False)
                if (e + 1) % args.eval_every_n_epochs == 0:
                    wandb.log({"valid_sp_accuracy": validation_sp_accuracy, "test_sp_accuracy": test_sp_accuracy,
                               "valid_avg_rmsd": validation_avg_rmsd, "valid_mid_rmsd": validation_mid_rmsd,
                               "valid_rmsd_rate": validation_rmsd_rate,
                               "valid_avg_plddt": validation_avg_plddt, "valid_mid_plddt": validation_mid_plddt,
                               "valid_plddt_rate": validation_plddt_rate,
                               "test_avg_rmsd": test_avg_rmsd, "test_mid_rmsd": test_mid_rmsd,
                               "test_rmsd_rate": test_rmsd_rate,
                               "test_avg_plddt": test_avg_plddt, "test_mid_plddt": test_mid_plddt,
                               "test_plddt_rate": test_plddt_rate},
                              step=total_step)
        else:
            with open(logfile, 'a') as f:
                f.write(
                    f'epoch: {e + 1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, test: {test_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}, test_acc: {test_accuracy_}, valid_sp_acc: {validation_sp_accuracy_}, test_sp_acc: {test_sp_accuracy_}, ' +
                    f'train_masked_loss: {train_masked_loss_}, train_masked_acc: {train_masked_accuracy_}, ' +
                    f'valid_masked_loss: {validation_masked_loss_}, valid_masked_acc: {validation_masked_accuracy_}, ' +
                    f'test_masked_loss: {test_masked_loss_}, test_masked_acc: {test_masked_accuracy_}, ' +
                    f'train_single: {train_single_}, train_pair: {train_pair_}, train_structure: {train_structure_}, ' +
                    f'valid_single: {validation_single_}, valid_pair: {validation_pair_}, valid_structure: {validation_structure_}, ' +
                    f'test_single: {test_single_}, test_pair: {test_pair_}, test_structure: {test_structure_}, ' +
                    f'train_single_masked: {train_single_masked_}, train_pair_masked: {train_pair_masked_}, train_structure_masked: {train_structure_masked_}, ' +
                    f'valid_single_masked: {validation_single_masked_}, valid_pair_masked: {validation_pair_masked_}, valid_structure_masked: {validation_structure_masked_}, ' +
                    f'test_single_masked: {test_single_masked_}, test_pair_masked: {test_pair_masked_}, test_structure_masked: {test_structure_masked_}\n')
            print(
                f'epoch: {e + 1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, test: {test_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}, test_acc: {test_accuracy_}, valid_sp_acc: {validation_sp_accuracy_}, test_sp_acc: {test_sp_accuracy_}, ' +
                f'train_masked_loss: {train_masked_loss_}, train_masked_acc: {train_masked_accuracy_}, ' +
                f'valid_masked_loss: {validation_masked_loss_}, valid_masked_acc: {validation_masked_accuracy_}, ' +
                f'test_masked_loss: {test_masked_loss_}, test_masked_acc: {test_masked_accuracy_}, ' +
                f'train_single: {train_single_}, train_pair: {train_pair_}, train_structure: {train_structure_}, ' +
                f'valid_single: {validation_single_}, valid_pair: {validation_pair_}, valid_structure: {validation_structure_}, ' +
                f'test_single: {test_single_}, test_pair: {test_pair_}, test_structure: {test_structure_}' +
                f'train_single_masked: {train_single_masked_}, train_pair_masked: {train_pair_masked_}, train_structure_masked: {train_structure_masked_}, ' +
                f'valid_single_masked: {validation_single_masked_}, valid_pair_masked: {validation_pair_masked_}, valid_structure_masked: {validation_structure_masked_}, ' +
                f'test_single_masked: {test_single_masked_}, test_pair_masked: {test_pair_masked_}, test_structure_masked: {test_structure_masked_}')

            if not args.debug:
                wandb.log({"train_perplexity": train_perplexity, "valid_perplexity": validation_perplexity,
                           "test_perplexity": test_perplexity,
                           "train_accuracy": train_accuracy, "valid_accuracy": validation_accuracy,
                           "test_accuracy": test_accuracy},
                          step=total_step, commit=False)
                wandb.log({"train_masked_loss": train_masked_loss, "train_masked_accuracy": train_masked_accuracy,
                           "valid_masked_loss": validation_masked_loss,
                           "valid_masked_accuracy": validation_masked_accuracy,
                           "test_masked_loss": test_masked_loss, "test_masked_accuracy": test_masked_accuracy},
                          step=total_step, commit=False)
                wandb.log(
                    {"train_single": train_single, "train_pair": train_pair, "train_structure": train_structure,
                     "valid_single": validation_single, "valid_pair": validation_pair,
                     "valid_structure": validation_structure,
                     "test_single": test_single, "test_pair": test_pair, "test_structure": test_structure},
                    step=total_step, commit=False)
                wandb.log(
                    {"train_single_masked": train_single_masked, "train_pair_masked": train_pair_masked,
                     "train_structure_masked": train_structure_masked,
                     "valid_single_masked": validation_single_masked, "valid_pair_masked": validation_pair_masked,
                     "valid_structure_masked": validation_structure_masked,
                     "test_single_masked": test_single_masked, "test_pair_masked": test_pair_masked,
                     "test_structure_masked": test_structure_masked}, step=total_step, commit=False)
                if (e + 1) % args.eval_every_n_epochs == 0:
                    wandb.log({"valid_sp_accuracy": validation_sp_accuracy, "test_sp_accuracy": test_sp_accuracy},
                              step=total_step)

        checkpoint_filename_last = base_folder + 'model_weights/epoch_last.pt'.format(e + 1, total_step)
        torch.save({
            'epoch': e + 1,
            'step': total_step,
            'num_edges': args.num_neighbors,
            'noise_level': args.backbone_noise,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_filename_last)

        if (e + 1) % args.save_model_every_n_epochs == 0:
            checkpoint_filename = base_folder + 'model_weights/epoch{}_step{}.pt'.format(e + 1, total_step)
            torch.save({
                'epoch': e + 1,
                'step': total_step,
                'num_edges': args.num_neighbors,
                'noise_level': args.backbone_noise,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_filename)

    if not args.debug:
        wandb.finish()


if __name__ == "__main__":
    # import multiprocessing
    # multiprocessing.set_start_method('spawn')
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--base_path", type=str, required=True,
                           help="base path for data and model")
    argparser.add_argument("--name", type=str, default="", help="name for the experiment")
    argparser.add_argument("--output_path", type=str, default="outputs", help="output path")
    argparser.add_argument("--repr_dir", type=str, required=True, help="path for saved AlphaFold3 representations")
    argparser.add_argument("--previous_checkpoint", type=str, default="",
                           help="path for previous model weights, e.g. file.pt")
    argparser.add_argument("--num_epochs", type=int, default=400, help="number of epochs to train for")
    argparser.add_argument("--save_model_every_n_epochs", type=int, default=50,
                           help="save model weights every n epochs")
    argparser.add_argument("--reload_data_every_n_epochs", type=int, default=2,
                           help="reload training data every n epochs")
    argparser.add_argument("--num_examples_per_epoch", type=int, default=100000,  # 100000
                           help="number of training example to load for one epoch")
    argparser.add_argument("--batch_size", type=int, default=20000, help="number of tokens for one batch")
    argparser.add_argument("--max_protein_length", type=int, default=256,  # 10000
                           help="maximum length of the protein complext")
    argparser.add_argument("--hidden_dim", type=int, default=128, help="hidden model dimension")
    argparser.add_argument("--num_encoder_layers", type=int, default=3, help="number of encoder layers")
    argparser.add_argument("--num_decoder_layers", type=int, default=3, help="number of decoder layers")
    argparser.add_argument("--num_neighbors", type=int, default=30, help="number of neighbors for the sparse graph")
    argparser.add_argument("--dropout", type=float, default=0.1, help="dropout level; 0.0 means no dropout")
    argparser.add_argument("--backbone_noise", type=float, default=0.1,
                           help="amount of noise added to backbone during training")
    argparser.add_argument("--rescut", type=float, default=3.5, help="PDB resolution cutoff")
    argparser.add_argument("--debug", action='store_true', default=False, help="minimal data loading for debugging")
    argparser.add_argument("--gradient_norm", type=float, default=-1.0,
                           help="clip gradient norm, set to negative to omit clipping")
    argparser.add_argument("--mixed_precision", action='store_false', default=True, help="train with mixed precision")
    argparser.add_argument("--min_t", type=float, default=1e-2)
    argparser.add_argument("--schedule", type=str, default='linear')  # other schedule is not implemented
    argparser.add_argument("--schedule_exp_rate", type=float, default=-3)
    argparser.add_argument("--temp", type=float, default=0.1)
    argparser.add_argument("--noise", type=float, default=1.0)
    argparser.add_argument("--interpolant_type", type=str, default='masking')
    argparser.add_argument("--do_purity", action='store_false', default=True)
    argparser.add_argument("--num_timesteps", type=int, default=500)
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--eval_every_n_epochs", type=int, default=50)
    argparser.add_argument("--calculate_rmsd", action="store_true", default=False, help="whether to calculate rmsd")

    # alignment related hyper-parameters
    argparser.add_argument("--align_depth", type=int, default=1, help="the depth of representation to align")
    argparser.add_argument("--update_edge", action="store_true", default=False,
                           help="whether to update edge representation")
    argparser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate")
    argparser.add_argument("--learnable_node", action="store_true", default=False,
                           help="whether use learnable node initialization")

    # representation loss related hyper-parameters
    argparser.add_argument("--load_repr", action='store_false', default=True, help="whether to load representation")
    argparser.add_argument("--repr_weight", type=float, default=1.0, help="weight for representation loss")
    argparser.add_argument("--repr_norm", action='store_true', default=False,
                           help="normalization of the representations")
    argparser.add_argument("--repa_coeff", type=float, nargs='+',
                           default=[1.0, 1.0, 1.0])  # alignment loss coefficients for different image/text encoders
    argparser.add_argument("--mdlm_parameterization", action='store_true', default=False,
                           help="use mdlm parameterization")
    argparser.add_argument("--mask_repr_loss", action='store_true', default=False,
                           help="mask the representation loss by residue")
    argparser.add_argument("--t_schedule", action='store_true', default=False, help="use time schedule")
    argparser.add_argument("--repr_noise", type=float, default=0.0, help="scale of noise added to normalized representation")
    argparser.add_argument("--repa_weight_decay", type=str, default="constant", help="scheduler for repa loss weight w.r.t. training epoch")
    argparser.add_argument("--repa_epoch", type=int, default=100, help="total number of epochs with repa loss")
    argparser.add_argument("--start_diffusion_epoch", type=int, default=0, help="before this epoch only use repa loss as a pretrain stage")
    argparser.add_argument("--diffusion_warm_up_epoch", type=int, default=50, help="warm up diffusion loss weight")
    argparser.add_argument("--diffusion_decay", type=str, default="constant", help="diffusion loss decay")
    args = argparser.parse_args()
    main(args)
