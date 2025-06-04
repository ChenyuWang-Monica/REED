import os
import os.path as osp
import pathlib
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from hydra.utils import get_original_cwd

from midi.datasets.dataset_utils import mol_to_torch_geometric, remove_hydrogens, Statistics
from midi.datasets.dataset_utils import load_pickle, save_pickle
from midi.datasets.abstract_dataset import AbstractDatasetInfos, AbstractDataModule, AbstractAdaptiveDataModule
from midi.metrics.metrics_utils import compute_all_statistics
from midi.utils import PlaceHolder


def files_exist(files) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


full_atom_encoder = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}


class QM9Dataset(InMemoryDataset):
    raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
               'molnet_publish/qm9.zip')
    raw_url2 = 'https://ndownloader.figshare.com/files/3195404'
    processed_url = 'https://data.pyg.org/datasets/qm9_v3.zip'

    def __init__(self, split, root, remove_h: bool, transform=None, pre_transform=None, pre_filter=None, which_split="edm"):
        # NOTE: add split option
        assert which_split in ["edm", "midi"], f"Need which_split to be chosen from ['edm', 'midi'], detected {which_split}"
        self.which_split = which_split
        
        self.split = split
        if self.split == 'train':
            self.file_idx = 0
        elif self.split == 'val':
            self.file_idx = 1
        else:
            self.file_idx = 2
        self.remove_h = remove_h

        self.atom_encoder = full_atom_encoder
        if remove_h:
            self.atom_encoder = {k: v - 1 for k, v in self.atom_encoder.items() if k != 'H'}

        super().__init__(root, transform, pre_transform, pre_filter)
        
        try:
            # Attempt to load processed paths
            self.data, self.slices = torch.load(self.processed_paths[0])

            self.statistics = Statistics(
                num_nodes=load_pickle(self.processed_paths[1]),
                atom_types=torch.from_numpy(np.load(self.processed_paths[2])).float(),
                bond_types=torch.from_numpy(np.load(self.processed_paths[3])).float(),
                charge_types=torch.from_numpy(np.load(self.processed_paths[4])).float(),
                valencies=load_pickle(self.processed_paths[5]),
                bond_lengths=load_pickle(self.processed_paths[6]),
                bond_angles=torch.from_numpy(np.load(self.processed_paths[7], allow_pickle=True)).float()
            )
            self.smiles = load_pickle(self.processed_paths[8])

        except (FileNotFoundError, EOFError, OSError) as e:
            # If there's any issue loading the processed data, re-call download and process
            print(f"Failed to load processed data: {e}. Downloading and processing again.")
            self.download()
            self.process()

            # Retry loading after processing
            self.data, self.slices = torch.load(self.processed_paths[0])

            self.statistics = Statistics(
                num_nodes=load_pickle(self.processed_paths[1]),
                atom_types=torch.from_numpy(np.load(self.processed_paths[2])).float(),
                bond_types=torch.from_numpy(np.load(self.processed_paths[3])).float(),
                charge_types=torch.from_numpy(np.load(self.processed_paths[4])).float(),
                valencies=load_pickle(self.processed_paths[5]),
                bond_lengths=load_pickle(self.processed_paths[6]),
                bond_angles=torch.from_numpy(np.load(self.processed_paths[7], allow_pickle=True)).float()
            )
            self.smiles = load_pickle(self.processed_paths[8])

    @property
    def raw_file_names(self):
        return ['gdb9.sdf', 'gdb9.sdf.csv', 'uncharacterized.txt']

    @property
    def split_file_name(self):
        return [f'train_{self.which_split}.csv', f'val_{self.which_split}.csv', f'test_{self.which_split}.csv']

    @property
    def split_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        splitting."""
        files = to_list(self.split_file_name)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        h = 'noh' if self.remove_h else 'h'
        if self.split == 'train':
            return [f'train_{self.which_split}_{h}.pt', f'train_{self.which_split}_n_{h}.pickle', f'train_{self.which_split}_atom_types_{h}.npy', f'train_{self.which_split}_bond_types_{h}.npy',
                    f'train_{self.which_split}_charges_{h}.npy', f'train_{self.which_split}_valency_{h}.pickle', f'train_{self.which_split}_bond_lengths_{h}.pickle',
                    f'train_{self.which_split}_angles_{h}.npy', f'train_{self.which_split}_smiles.pickle']
        elif self.split == 'val':
            return [f'val_{self.which_split}_{h}.pt', f'val_{self.which_split}_n_{h}.pickle', f'val_{self.which_split}_atom_types_{h}.npy', f'val_{self.which_split}_bond_types_{h}.npy',
                    f'val_{self.which_split}_charges_{h}.npy', f'val_{self.which_split}_valency_{h}.pickle', f'val_{self.which_split}_bond_lengths_{h}.pickle',
                    f'val_{self.which_split}_angles_{h}.npy', f'val_{self.which_split}_smiles.pickle']
        else:
            return [f'test_{self.which_split}_{h}.pt', f'test_{self.which_split}_n_{h}.pickle', f'test_{self.which_split}_atom_types_{h}.npy', f'test_{self.which_split}_bond_types_{h}.npy',
                    f'test_{self.which_split}_charges_{h}.npy', f'test_{self.which_split}_valency_{h}.pickle', f'test_{self.which_split}_bond_lengths_{h}.pickle',
                    f'test_{self.which_split}_angles_{h}.npy', f'test_{self.which_split}_smiles.pickle']

    def download(self):
        """
        Download raw qm9 files. Taken from PyG QM9 class
        """
        try:
            import rdkit  # noqa
            file_path = download_url(self.raw_url, self.raw_dir)
            extract_zip(file_path, self.raw_dir)
            os.unlink(file_path)

            file_path = download_url(self.raw_url2, self.raw_dir)
            os.rename(osp.join(self.raw_dir, '3195404'),
                      osp.join(self.raw_dir, 'uncharacterized.txt'))
        except ImportError:
            path = download_url(self.processed_url, self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.unlink(path)

        if files_exist(self.split_paths):
            return

        dataset = pd.read_csv(self.raw_paths[1])

        # NOTE: MiDi Split
        if self.which_split == "midi":
            n_samples = len(dataset)
            n_train = 100000
            n_test = int(0.1 * n_samples)
            n_val = n_samples - (n_train + n_test)

            # Shuffle dataset with df.sample, then split
            train, val, test = np.split(dataset.sample(frac=1, random_state=42), [n_train, n_val + n_train])
            
            # Extract the row indices for each split
            train_indices = train.index.to_numpy()
            val_indices = val.index.to_numpy()
            test_indices = test.index.to_numpy()

            # Save indices to splits dictionary
            splits = {
                "train": train_indices,
                "valid": val_indices,
                "test": test_indices
            }

            # Save the splits dictionary to a file
            save_path = "./eval_src/midi_metrics/midi_qm9_splits.npy"
            np.save(save_path, splits)
        
        # NOTE: EDM Split
        else:
            def is_int(str):
                try:
                    int(str)
                    return True
                except:
                    return False
            excluded_strings = []
            with open(self.raw_paths[2]) as f:
                lines = f.readlines()
                excluded_strings = [line.split()[0]
                                    for line in lines if len(line.split()) > 0]

            excluded_idxs = [int(idx) - 1 for idx in excluded_strings if is_int(idx)]

            assert len(excluded_idxs) == 3054, 'There should be exactly 3054 excluded atoms. Found {}'.format(
                len(excluded_idxs))

            # Now, create a list of indices
            Ngdb9 = 133885
            Nexcluded = 3054

            included_idxs = np.array(
                sorted(list(set(range(Ngdb9)) - set(excluded_idxs))))

            # Now generate random permutations to assign molecules to training/validation/test sets.
            Nmols = Ngdb9 - Nexcluded

            Ntrain = 100000
            Ntest = int(0.1*Nmols)
            Nvalid = Nmols - (Ntrain + Ntest)

            # Generate random permutation
            np.random.seed(0)
            data_perm = np.random.permutation(Nmols)

            # Now use the permutations to generate the indices of the dataset splits.
            # train, valid, test, extra = np.split(included_idxs[data_perm], [Ntrain, Ntrain+Nvalid, Ntrain+Nvalid+Ntest])

            train, valid, test, extra = np.split(
                data_perm, [Ntrain, Ntrain+Nvalid, Ntrain+Nvalid+Ntest])

            assert(len(extra) == 0), 'Split was inexact {} {} {} {}'.format(
                len(train), len(valid), len(test), len(extra))
            
            train = included_idxs[train]
            valid = included_idxs[valid]
            test = included_idxs[test]
            
            train, val, test = dataset.iloc[train], dataset.iloc[valid], dataset.iloc[test]
        

        # Save
        train.to_csv(os.path.join(self.raw_dir, f'train_{self.which_split}.csv'))
        val.to_csv(os.path.join(self.raw_dir, f'val_{self.which_split}.csv'))
        test.to_csv(os.path.join(self.raw_dir, f'test_{self.which_split}.csv'))

    def process(self):
        RDLogger.DisableLog('rdApp.*')

        target_df = pd.read_csv(self.split_paths[self.file_idx], index_col=0)
        target_df.drop(columns=['mol_id'], inplace=True)

        with open(self.raw_paths[-1], 'r') as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False, sanitize=False)
        data_list = []
        all_smiles = []
        num_errors = 0
        a = target_df.index.tolist()
        a.sort()
        for i, mol in enumerate(tqdm(suppl)):
            if i in skip or i not in target_df.index:
                continue
            smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
            if smiles is None:
                num_errors += 1
            else:
                all_smiles.append(smiles)

            data = mol_to_torch_geometric(mol, full_atom_encoder, smiles)
            if self.remove_h:
                data = remove_hydrogens(data)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])

        statistics = compute_all_statistics(data_list, self.atom_encoder, charges_dic={-1: 0, 0: 1, 1: 2})

        save_pickle(statistics.num_nodes, self.processed_paths[1])
        np.save(self.processed_paths[2], statistics.atom_types)
        np.save(self.processed_paths[3], statistics.bond_types)
        np.save(self.processed_paths[4], statistics.charge_types)
        save_pickle(statistics.valencies, self.processed_paths[5])
        save_pickle(statistics.bond_lengths, self.processed_paths[6])
        np.save(self.processed_paths[7], statistics.bond_angles)
        print("Number of molecules that could not be mapped to smiles: ", num_errors)
        save_pickle(set(all_smiles), self.processed_paths[8])
        torch.save(self.collate(data_list), self.processed_paths[0])


class QM9DataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(get_original_cwd()).parents[0]
        root_path = os.path.join(base_path, self.datadir)

        train_dataset = QM9Dataset(split='train', root=root_path, remove_h=cfg.dataset.remove_h, which_split=cfg.which_split)
        val_dataset = QM9Dataset(split='val', root=root_path, remove_h=cfg.dataset.remove_h, which_split=cfg.which_split)
        test_dataset = QM9Dataset(split='test', root=root_path, remove_h=cfg.dataset.remove_h, which_split=cfg.which_split)
        self.statistics = {'train': train_dataset.statistics, 'val': val_dataset.statistics,
                           'test': test_dataset.statistics}
        self.remove_h = cfg.dataset.remove_h
        super().__init__(cfg, train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset)


class QM9infos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg):
        self.remove_h = cfg.dataset.remove_h
        self.statistics = datamodule.statistics
        self.name = 'qm9'
        self.atom_encoder = full_atom_encoder
        self.collapse_charges = torch.Tensor([-1, 0, 1]).int()
        if self.remove_h:
            self.atom_encoder = {k: v - 1 for k, v in self.atom_encoder.items() if k != 'H'}
        super().complete_infos(datamodule.statistics, self.atom_encoder)
        self.input_dims = PlaceHolder(X=self.num_atom_types, charges=3, E=5, y=1, pos=3)
        self.output_dims = PlaceHolder(X=self.num_atom_types, charges=3, E=5, y=0, pos=3)

    def to_one_hot(self, X, charges, E, node_mask):
        X = F.one_hot(X, num_classes=self.num_atom_types).float()
        E = F.one_hot(E, num_classes=5).float()
        charges = F.one_hot(charges + 1, num_classes=3).float()
        placeholder = PlaceHolder(X=X, charges=charges, E=E,  y=None, pos=None)
        pl = placeholder.mask(node_mask)
        return pl.X, pl.charges, pl.E

    def one_hot_charges(self, charges):
        return F.one_hot((charges + 1).long(), num_classes=3).float()


