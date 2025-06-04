from rdkit import Chem
import numpy as np
from qm9.bond_analyze import get_bond_order, geom_predictor
from . import dataset
import torch
from configs.datasets_config import get_dataset_info
import pickle
import os
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule, UFFHasAllMoleculeParams
import warnings
from typeguard import typechecked
import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import pandas as pd
from qm9 import bond_analyze
import logging
from posebusters import PoseBusters
import glob
import os
import random
import logging
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from pathlib import Path
from rdkit import Chem
from torchtyping import TensorType
from typeguard import typechecked
import subprocess
import pandas as pd
from tqdm import tqdm
import tempfile
@typechecked
def save_xyz_file(
    path: str,
    positions: TensorType["batch_num_nodes", 3],
    one_hot_max: torch.Tensor,  # TODO: incorporate charges within saved XYZ file
    dataset_info: Dict[str, Any],
    id_from: int = 0,
    name: str = "molecule",
    batch_index: Optional[TensorType["batch_num_nodes"]] = None,
):
    try:
        os.makedirs(path)
    except OSError:
        pass


    for batch_i in torch.unique(batch_index):
        current_batch_index = (batch_index == batch_i)
        num_atoms = int(torch.sum(current_batch_index).item())
        f = open(path + name + "_" + "%03d.xyz" % (batch_i + id_from), "w")
        f.write("%d\n\n" % num_atoms)
        atom_ids = one_hot_max[current_batch_index]
        batch_pos = positions[current_batch_index]
        for atom_i in range(num_atoms):
            atom = dataset_info["atom_decoder"][atom_ids[atom_i].item()]
            
            f.write("%s %.9f %.9f %.9f\n" % (atom, batch_pos[atom_i, 0], batch_pos[atom_i, 1], batch_pos[atom_i, 2]))
        f.close()


@typechecked
def write_xyz_file(
    positions: TensorType["num_nodes", 3],
    atom_types: TensorType["num_nodes"],
    filename: str
):
    out = f"{len(positions)}\n\n"
    assert len(positions) == len(atom_types)
    for i in range(len(positions)):
        out += f"{atom_types[i]} {positions[i, 0]:.3f} {positions[i, 1]:.3f} {positions[i, 2]:.3f}\n"
    with open(filename, "w") as f:
        f.write(out)


@typechecked
def write_sdf_file(sdf_path: Path, molecules: List[Chem.Mol], verbose: bool = True):
    w = Chem.SDWriter(str(sdf_path))
    for m in molecules:
        if m is not None:
            w.write(m)
    if verbose:
        logging.info(f"Wrote generated molecules to SDF file {sdf_path}")


@typechecked
def load_molecule_xyz(
    file: str,
    dataset_info: Dict[str, Any]
) -> Tuple[
    TensorType["num_nodes", 3],
    TensorType["num_nodes", "num_atom_types"]
]:
    with open(file, encoding="utf8") as f:
        num_atoms = int(f.readline())
        one_hot = torch.zeros(num_atoms, len(dataset_info["atom_decoder"]))
        positions = torch.zeros(num_atoms, 3)
        f.readline()
        atoms = f.readlines()
        for i in range(num_atoms):
            atom = atoms[i].split(" ")
            atom_type = atom[0]
            one_hot[i, dataset_info["atom_encoder"][atom_type]] = 1
            position = torch.Tensor([float(e) for e in atom[1:]])
            positions[i, :] = position
        return positions, one_hot


@typechecked
def load_files_with_ext(path: str, ext: str, shuffle: bool = True) -> List[str]:
    files = glob.glob(path + f"/*.{ext}")
    if shuffle:
        random.shuffle(files)
    return files


def convert_xyz_to_sdf(input_xyz_filepath: str) -> Optional[str]:
    """Convert an XYZ file to an SDF file using OpenBabel.

    :param input_xyz_filepath: Input XYZ file path.
    :return: Output SDF file path.
    """
    output_sdf_filepath = input_xyz_filepath.replace(".xyz", ".sdf")
    if not os.path.exists(output_sdf_filepath):
        subprocess.run(
            [
                "obabel",
                input_xyz_filepath,
                "-O",
                output_sdf_filepath,
            ],
            check=True,
        )
    return output_sdf_filepath if os.path.exists(output_sdf_filepath) else None

def create_molecule_table(input_molecule_dir: str) -> pd.DataFrame:
    """Create a molecule table from the inference results of a trained model checkpoint.

    :param input_molecule_dir: Directory containing the generated molecules of a trained model checkpoint.
    :return: Molecule table as a Pandas DataFrame.
    """
    inference_xyz_results = [str(item) for item in Path(input_molecule_dir).rglob("*.xyz")]
    inference_sdf_results = [str(item) for item in Path(input_molecule_dir).rglob("*.sdf")]
    if not inference_sdf_results or len(inference_sdf_results) != len(inference_xyz_results):
        logging.info(f"Converting xyz file into intermediate sdf file with obabel.")
        inference_sdf_results = [
            convert_xyz_to_sdf(item) for item in tqdm(
                inference_xyz_results, desc="Converting XYZ input files to SDF files"
            )
        ]
    mol_table = pd.DataFrame(
        {
            "mol_pred": [item for item in inference_sdf_results if item is not None],
            "mol_true": None,
            "mol_cond": None,
        }
    )
    return mol_table

def compute_qm9_smiles(dataset_name, remove_h):
    '''

    :param dataset_name: qm9 or qm9_second_half
    :return:
    '''
    print("\tConverting QM9 dataset to SMILES ...")

    class StaticArgs:
        def __init__(self, dataset, remove_h):
            self.dataset = dataset
            self.batch_size = 1
            self.num_workers = 1
            self.filter_n_atoms = None
            self.datadir = './data/'
            self.remove_h = remove_h
            self.include_charges = True
            import torch.distributed as dist
            try:
                self.rank = dist.get_rank()
                self.world_size = dist.get_world_size()
            except:
                print("Not using distributed learning.")
                self.rank = 0
                self.world_size = 1
    args_dataset = StaticArgs(dataset_name, remove_h)
    dataloaders, charge_scale = dataset.retrieve_dataloaders(args_dataset)
    dataset_info = get_dataset_info(args_dataset.dataset, args_dataset.remove_h)
    n_types = 4 if remove_h else 5
    mols_smiles = []
    for i, data in enumerate(dataloaders['train']):
        positions = data['positions'][0].view(-1, 3).numpy()
        one_hot = data['one_hot'][0].view(-1, n_types).type(torch.float32)
        atom_type = torch.argmax(one_hot, dim=1).numpy()

        mol = build_molecule(torch.tensor(positions), torch.tensor(atom_type), dataset_info)
        mol = mol2smiles(mol)
        if mol is not None:
            mols_smiles.append(mol)
        if i % 1000 == 0:
            print("\tConverting QM9 dataset to SMILES {0:.2%}".format(float(i)/len(dataloaders['train'])))
    return mols_smiles


def retrieve_qm9_smiles(dataset_info):
    dataset_name = dataset_info['name']
    if dataset_info['with_h']:
        pickle_name = dataset_name
    else:
        pickle_name = dataset_name + '_noH'

    file_name = 'qm9/temp/%s_smiles.pickle' % pickle_name
    try:
        with open(file_name, 'rb') as f:
            qm9_smiles = pickle.load(f)
        return qm9_smiles
    except OSError:
        try:
            os.makedirs('qm9/temp')
        except:
            pass
        qm9_smiles = compute_qm9_smiles(dataset_name, remove_h=not dataset_info['with_h'])
        with open(file_name, 'wb') as f:
            pickle.dump(qm9_smiles, f)
        return qm9_smiles


#### New implementation ####

bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                 Chem.rdchem.BondType.AROMATIC]


class BasicMolecularMetrics(object):
    def __init__(self, dataset_info, dataset_smiles_list=None):
        self.atom_decoder = dataset_info['atom_decoder']
        self.dataset_smiles_list = dataset_smiles_list
        self.dataset_info = dataset_info

        # Retrieve dataset smiles only for qm9 currently.
        if dataset_smiles_list is None and 'qm9' in dataset_info['name']:
            self.dataset_smiles_list = retrieve_qm9_smiles(
                self.dataset_info)

    def compute_validity(self, generated):
        """ generated: list of couples (positions, atom_types)"""
        valid = []

        for graph in generated:
            mol = build_molecule(*graph, self.dataset_info)
            smiles = mol2smiles(mol)
            if smiles is not None:
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
                largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                smiles = mol2smiles(largest_mol)
                valid.append(smiles)
        return valid, len(valid) / len(generated)

    def compute_uniqueness(self, valid):
        """ valid: list of SMILES strings."""
        return list(set(valid)), len(set(valid)) / len(valid)

    def compute_novelty(self, unique):
        num_novel = 0
        novel = []
        for smiles in unique:
            if smiles not in self.dataset_smiles_list:
                novel.append(smiles)
                num_novel += 1
        return novel, num_novel / len(unique)
    
    def compute_relaxed_validity(self, generated):
        """ generated: list of couples (positions, atom_types)"""
        valid = []

        for graph in generated:
            mol = build_molecule_with_partial_charges(*graph, self.dataset_info)
            smiles = mol2smiles(mol)
            if smiles is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
                    largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                    smiles = mol2smiles(largest_mol)
                    valid.append(smiles)
                except:
                    assert 0, "Non-None smiles reports errors when using Chem.rdmolops.GetMolFrags...?"
        return valid, len(valid) / len(generated)

    

    def evaluate(self, generated, verbose=False):
        """ generated: list of pairs (positions: n x 3, atom_types: n [int])
            the positions and atom types should already be masked. """
        
        
        
        valid, validity = self.compute_validity(generated)
        if verbose: print(f"Validity over {len(generated)} molecules: {validity * 100 :.2f}%")
        if validity > 0:
            unique, uniqueness = self.compute_uniqueness(valid)
            if verbose: print(f"Uniqueness over {len(valid)} valid molecules: {uniqueness * 100 :.2f}%")

            if self.dataset_smiles_list is not None:
                _, novelty = self.compute_novelty(unique)
                if verbose: print(f"Novelty over {len(unique)} unique valid molecules: {novelty * 100 :.2f}%")
            else:
                novelty = 0.0
        else:
            novelty = 0.0
            uniqueness = 0.0
            unique = None
        return [validity, uniqueness, novelty], unique
    
    def get_unique_and_valid_index_and_smiles(self, generated):
        """ generated: list of couples (positions, atom_types)"""
        valid_mask = torch.zeros(len(generated), dtype=torch.bool, device=generated[0][0].device)
        valid = []
        for i, graph in enumerate(generated):
            mol = build_molecule(*graph, self.dataset_info)
            smiles = mol2smiles(mol)
            if smiles is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
                    largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                    smiles = mol2smiles(largest_mol)
                    valid_mask[i] = True
                    valid.append(smiles)
                except:
                    assert 0, "Non-None smiles reports errors when using Chem.rdmolops.GetMolFrags."
                    
        unique = set(valid)
        import copy
        unique_origin = copy.deepcopy(unique)
        unique_num = len(unique)
        unique_and_valid_mask = torch.zeros(len(generated), dtype=torch.bool, device=generated[0][0].device)
            
        for i in range(len(generated)):
            if valid_mask[i]: 
                if valid[valid_mask[:i].sum()] in unique:
                    unique_and_valid_mask[i] = True
                    unique = unique - set([valid[valid_mask[:i].sum()]])
        assert unique_and_valid_mask.sum() == unique_num, f"unique_and_valid_mask.sum() {unique_and_valid_mask.sum()}, unique_num {unique_num}"
        return unique_and_valid_mask, unique_origin
    
    
    def compute_posebusters(self, generated):
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                timestamp = datetime.datetime.now().strftime("%m%d%Y_%H_%M_%S")
                bust_results_filepath = Path(temp_dir) / f"buster_{timestamp}.csv"
                # Compute intermediate xyz files.
                save_xyz_file(
                        path=str(temp_dir) + "/",
                        positions=torch.cat([pos for pos, _ in generated], dim=0),
                        one_hot_max=torch.cat([charge for _, charge in generated], dim=0),
                        id_from=0,
                        name=timestamp,
                        batch_index=torch.repeat_interleave(torch.arange(len(generated)), torch.tensor([len(pos) for pos, _ in generated])),
                        dataset_info=self.dataset_info
                    )
                logging.info(f"Created intermediate xyz file to {str(temp_dir) + '/'} for computing posebusters.")
                mol_table = create_molecule_table(temp_dir)
                buster = PoseBusters(config="mol", top_n=None)
                bust_results = buster.bust_table(mol_table, full_report=True)
                bust_results.to_csv(bust_results_filepath, index=False)
                logging.info(f"PoseBusters results saved to {bust_results_filepath}.")
                pb_results = pd.read_csv(bust_results_filepath)
                pb_results["valid"] = (
                    pb_results["mol_pred_loaded"].astype(bool)
                    & pb_results["sanitization"].astype(bool)
                    & pb_results["all_atoms_connected"].astype(bool)
                    & pb_results["bond_lengths"].astype(bool)
                    & pb_results["bond_angles"].astype(bool)
                    & pb_results["internal_steric_clash"].astype(bool)
                    & pb_results["aromatic_ring_flatness"].astype(bool)
                    & pb_results["double_bond_flatness"].astype(bool)
                    & pb_results["internal_energy"].astype(bool)
                    & pb_results["passes_valence_checks"].astype(bool)
                    & pb_results["passes_kekulization"].astype(bool)
                )
                pb_result = pb_results["valid"].mean()
                return pb_result
        except Exception as e:
            logging.error(f"Met exception {e} when computing posebusters.")
            return None
        
    def compute_posebusters_edm(self, generated, add_hydrogens, sanitize, relax_iter, largest_frag, output_dir):
        """ generated: list of pairs (positions: n x 3, atom_types: n [int])
            the positions and atom types should already be masked.
            To save .sdf files for calculating pd metrics.
        """
            
        molecules = []
        for graph in generated:
            positions, atom_types = graph
            mol = build_molecule(
                positions,
                atom_types,
                dataset_info=self.dataset_info,
                add_coords=True
            )
            mol = process_molecule(
                rdmol=mol,
                add_hydrogens=add_hydrogens,
                sanitize=sanitize,
                relax_iter=relax_iter,
                largest_frag=largest_frag
            )
            if mol is not None:
                molecules.append(mol)
        Path(output_dir).mkdir(exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%m%d%Y_%H_%M_%S")
        sdf_save_path = Path(output_dir, f"{timestamp}_mol.sdf")
        write_sdf_file(sdf_save_path, molecules)
        pb_results_filepath = str(sdf_save_path).replace(".sdf", ".csv")
        
        # OS
        try:
            os.system(f"bust {sdf_save_path} --outfmt csv --output {pb_results_filepath} --full-report")
            # Now, evaluate and report PoseBusters results
            pb_results = pd.read_csv(pb_results_filepath)
            pb_results["valid"] = (
                pb_results["mol_pred_loaded"].astype(bool)
                & pb_results["sanitization"].astype(bool)
                & pb_results["all_atoms_connected"].astype(bool)
                & pb_results["bond_lengths"].astype(bool)
                & pb_results["bond_angles"].astype(bool)
                & pb_results["internal_steric_clash"].astype(bool)
                & pb_results["aromatic_ring_flatness"].astype(bool)
                & pb_results["double_bond_flatness"].astype(bool)
                & pb_results["internal_energy"].astype(bool)
                & pb_results["passes_valence_checks"].astype(bool)
                & pb_results["passes_kekulization"].astype(bool)
            )
            return pb_results["valid"].mean()
        except:
            print(f"Error Meet when calling bust to analyse file {sdf_save_path} or when reading from {pb_results_filepath}")
            return None
        
    def compute_stability(self, generated):
        
        molecule_stable = 0
        nr_stable_bonds = 0
        n_atoms = 0
        n_samples = len(generated)
        
        for mol in generated:
            pos, atom_type = mol
            validity_results = check_stability(pos, atom_type, self.dataset_info)

            molecule_stable += int(validity_results[0])
            nr_stable_bonds += int(validity_results[1])
            n_atoms += int(validity_results[2])
            
        # Validity
        fraction_mol_stable = molecule_stable / float(n_samples)
        fraction_atm_stable = nr_stable_bonds / float(n_atoms)
        validity_dict = {
            'mol_stable': fraction_mol_stable,
            'atm_stable': fraction_atm_stable,
        }
        
        return validity_dict
    
    
    def compute_3d_metrics(self):
        pass
def compute_canonic_smiles_all(dataset_info, generated):
    """ generated: list of couples (positions, atom_types)"""
    valid = []
    for graph in generated:
        mol = build_molecule(*graph, dataset_info)
        smiles = mol2smiles(mol)
        if smiles is not None:
            try:
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
                largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                smiles = mol2smiles(largest_mol)
                valid.append(smiles)
            except:
                assert 0, "Non-None smiles reports errors when using Chem.rdmolops.GetMolFrags."
        else:
            valid.append(None)
            
    valid = [Chem.MolToSmiles(Chem.MolFromSmiles(smile)) if smile is not None else None for smile in valid]
    return valid
############################
# Validity and bond analysis
def check_stability(positions, atom_type, dataset_info, debug=False):
    assert len(positions.shape) == 2
    assert positions.shape[1] == 3
    atom_decoder = dataset_info['atom_decoder']
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    nr_bonds = np.zeros(len(x), dtype='int')

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            atom1, atom2 = atom_decoder[atom_type[i]], atom_decoder[atom_type[j]]
            pair = sorted([atom_type[i], atom_type[j]])
            if dataset_info['name'] == 'qm9' or dataset_info['name'] == 'qm9_second_half' or dataset_info['name'] == 'qm9_first_half':
                order = bond_analyze.get_bond_order(atom1, atom2, dist)
            elif dataset_info['name'] == 'geom':
                order = bond_analyze.geom_predictor(
                    (atom_decoder[pair[0]], atom_decoder[pair[1]]), dist)
            nr_bonds[i] += order
            nr_bonds[j] += order
    nr_stable_bonds = 0
    for atom_type_i, nr_bonds_i in zip(atom_type, nr_bonds):
        possible_bonds = bond_analyze.allowed_bonds[atom_decoder[atom_type_i]]
        if type(possible_bonds) == int:
            is_stable = possible_bonds == nr_bonds_i
        else:
            is_stable = nr_bonds_i in possible_bonds
        if not is_stable and debug:
            print("Invalid bonds for molecule %s with %d bonds" % (atom_decoder[atom_type_i], nr_bonds_i))
        nr_stable_bonds += int(is_stable)

    molecule_stable = nr_stable_bonds == len(x)
    return molecule_stable, nr_stable_bonds, len(x)



def preprocess_generated_molecules(mol_list_with_paddling):
    '''
    Preprocess the generated molecules to create a list of molecules with x and h representations, without  padding.        
    Output: processed_list: [(pos, atom_type) for each molecule]
    '''
    
    one_hot = mol_list_with_paddling['one_hot']
    x = mol_list_with_paddling['x']
    node_mask = mol_list_with_paddling['node_mask']
    assert isinstance(node_mask, torch.Tensor)

    atomsxmol = [torch.sum(m) for m in node_mask]
    n_samples = len(x)

    processed_list = []

    for i in range(n_samples):
        atom_type = one_hot[i].argmax(1).cpu().detach()
        pos = x[i].cpu().detach()

        atom_type = atom_type[0:int(atomsxmol[i])]
        pos = pos[0:int(atomsxmol[i])]
        processed_list.append((pos, atom_type))
        
    return processed_list

@typechecked
def process_molecule(
    rdmol: Chem.Mol,
    add_hydrogens: bool = False,
    sanitize: bool = False,
    relax_iter: int = 0,
    largest_frag: bool = False
) -> Optional[Chem.Mol]:
    """
    Apply filters to an RDKit molecule. Makes a copy first.
    Args:
        rdmol: RDKit molecule
        add_hydrogens: whether to add hydrogen atoms to the generated molecule
        sanitize: whether to sanitize molecules
        relax_iter: maximum number of UFF optimization iterations
        largest_frag: whether to filter out the largest fragment in a set of disjoint molecules
    Returns:
        RDKit molecule or `None` if it does not pass the filters
    """
    # create a copy
    mol = Chem.Mol(rdmol)

    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            warnings.warn('Sanitization failed. Returning None.')
            return None

    if add_hydrogens:
        mol = Chem.AddHs(mol, addCoords=(len(mol.GetConformers()) > 0))

    if largest_frag:
        mol_frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
        if sanitize:
            # sanitize the updated molecule
            try:
                Chem.SanitizeMol(mol)
            except ValueError:
                return None

    if relax_iter > 0:
        if not UFFHasAllMoleculeParams(mol):
            warnings.warn('UFF parameters not available for all atoms. '
                          'Returning None.')
            return None

        try:
            uff_relax(mol, relax_iter)
            if sanitize:
                # sanitize the updated molecule
                Chem.SanitizeMol(mol)
        except (RuntimeError, ValueError) as e:
            return None

    return mol


@typechecked
def uff_relax(mol: Chem.Mol, max_iter: int = 200) -> bool:
    """
    Uses RDKit's universal force field (UFF) implementation to optimize a
    molecule.
    """
    convergence_status = UFFOptimizeMolecule(mol, maxIters=max_iter)
    more_iterations_required = convergence_status == 1
    if more_iterations_required:
        warnings.warn(f'Maximum number of FF iterations reached. '
                      f'Returning molecule after {max_iter} relaxation steps.')
    return more_iterations_required


def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)


def build_molecule(positions, atom_types, dataset_info, add_coords=False):
    atom_decoder = dataset_info["atom_decoder"]
    X, A, E = build_xae_molecule(positions, atom_types, dataset_info)
    mol = Chem.RWMol()
    for atom in X:
        a = Chem.Atom(atom_decoder[atom.item()])
        mol.AddAtom(a)

    all_bonds = torch.nonzero(A)
    for bond in all_bonds:
        mol.AddBond(bond[0].item(), bond[1].item(), bond_dict[E[bond[0], bond[1]].item()])
        
        
    if add_coords:
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(i, (positions[i, 0].item(),
                                     positions[i, 1].item(),
                                     positions[i, 2].item()))
        mol.AddConformer(conf)

    return mol

@typechecked
def write_sdf_file(sdf_path: Path, molecules: List[Chem.Mol], verbose: bool = True):
    w = Chem.SDWriter(str(sdf_path))
    for m in molecules:
        if m is not None:
            w.write(m)


ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}

import re
def check_valency(mol):
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find('#')
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r'\d+', e_sub)))
        return False, atomid_valence

def build_molecule_with_partial_charges(positions, atom_types, dataset_info):
    atom_decoder = dataset_info["atom_decoder"]
    X, A, E = build_xae_molecule(positions, atom_types, dataset_info)
    mol = Chem.RWMol()
    for atom in X:
        a = Chem.Atom(atom_decoder[atom.item()])
        mol.AddAtom(a)

    all_bonds = torch.nonzero(A)

    for i, bond in enumerate(all_bonds):
        if bond[0].item() != bond[1].item():
            mol.AddBond(bond[0].item(), bond[1].item(), bond_dict[E[bond[0], bond[1]].item()])
            # add formal charge to atom: e.g. [O+], [N+], [S+]
            # not support [O-], [N-], [S-], [NH+] etc.
            flag, atomid_valence = check_valency(mol)
            if flag:
                continue
            else:
                assert len(atomid_valence) == 2
                idx = atomid_valence[0]
                v = atomid_valence[1]
                an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
                    mol.GetAtomWithIdx(idx).SetFormalCharge(1)
                    # print("Formal charge added")
    return mol



def build_xae_molecule(positions, atom_types, dataset_info):
    """ Returns a triplet (X, A, E): atom_types, adjacency matrix, edge_types
        args:
        positions: N x 3  (already masked to keep final number nodes)
        atom_types: N
        returns:
        X: N         (int)
        A: N x N     (bool)                  (binary adjacency matrix)
        E: N x N     (int)  (bond type, 0 if no bond) such that A = E.bool()
    """
    atom_decoder = dataset_info['atom_decoder']
    n = positions.shape[0]
    X = atom_types
    A = torch.zeros((n, n), dtype=torch.bool)
    E = torch.zeros((n, n), dtype=torch.int)

    pos = positions.unsqueeze(0)
    dists = torch.cdist(pos, pos, p=2).squeeze(0)
    for i in range(n):
        for j in range(i):
            pair = sorted([atom_types[i], atom_types[j]])
            if dataset_info['name'] == 'qm9' or dataset_info['name'] == 'qm9_second_half' or dataset_info['name'] == 'qm9_first_half':
                order = get_bond_order(atom_decoder[pair[0]], atom_decoder[pair[1]], dists[i, j])
            elif dataset_info['name'] == 'geom':
                order = geom_predictor((atom_decoder[pair[0]], atom_decoder[pair[1]]), dists[i, j], limit_bonds_to_one=True)
            # TODO: a batched version of get_bond_order to avoid the for loop
            if order > 0:
                # Warning: the graph should be DIRECTED
                A[i, j] = 1
                E[i, j] = order
    return X, A, E

if __name__ == '__main__':
    smiles_mol = 'C1CCC1'
    print("Smiles mol %s" % smiles_mol)
    chem_mol = Chem.MolFromSmiles(smiles_mol)
    block_mol = Chem.MolToMolBlock(chem_mol)
    print("Block mol:")
    print(block_mol)

