# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os, sys
from pathlib import Path
import numpy as np
from unicore.data import (
    Dictionary,
    NestedDictionaryDataset,
    AppendTokenDataset,
    PrependTokenDataset,
    RightPadDataset,
    EpochShuffleDataset,
    TokenizeDataset,
    RightPadDataset2D,
    FromNumpyDataset,
    RawArrayDataset,
)
from unimol.data import (
    KeyDataset,
    ConformerSampleDataset,
    DistanceDataset,
    EdgeTypeDataset,
    MaskPointsDataset,
    RemoveHydrogenDataset,
    AtomTypeDataset,
    NormalizeDataset,
    CroppingDataset,
    RightPadDatasetCoord,
    Add2DConformerDataset,
    LMDBDataset,
    TTADataset,
)
from unicore.data import BaseWrapperDataset

from unicore.tasks import UnicoreTask, register_task
from unicore import checkpoint_utils
import torch



logger = logging.getLogger(__name__)

class ComposedDataset:
    def __init__(self, dataset_list):
        self.dataset_list = dataset_list
        
        print(f"ComposedDataset: {len(self.dataset_list)} datasets")
        for i, dataset in enumerate(self.dataset_list):
            print(f"Dataset {i} {dataset.__class__.__name__}: {len(dataset)}")

    def __len__(self):
        return sum([len(dataset) for dataset in self.dataset_list])
    def __getitem__(self, idx):
        for dataset in self.dataset_list:
            if idx < len(dataset):
                return dataset[idx]
            else:
                idx -= len(dataset)
        raise IndexError

class DrugDataset:
    def __init__(self, db_path, split):
        self.db_path = db_path
        assert os.path.isfile(self.db_path), "{} not found".format(self.db_path)
        
        
        sequential = False
        include_charges = True
        device = "cpu"
        remove_h = False
        filter_size = None
        val_proportion, test_proportion = 0.1, 0.1
        assert Path(db_path).parent.parent.parent.exists(), "Please ensure that the working dir is the 3rd parent of the data file."
        sys.path.append(str(Path(db_path).parent.parent.parent))
        
        import build_geom_dataset
        from configs.datasets_config import get_dataset_info
        dataset_info = get_dataset_info("geom", remove_h)

        # Retrieve QM9 dataloaders
        print("Loading geom data ...")
        idx = {
            "train": 0,
            "valid": 1,
            "test": 2
        }
        split_idx = idx[split]
        split_data = build_geom_dataset.load_split_data(db_path,
                                                        val_proportion=val_proportion,
                                                        test_proportion=test_proportion,
                                                        filter_size=filter_size)
        print("Geom data loaded.")
        transform = build_geom_dataset.GeomDrugsTransform(dataset_info,
                                                          include_charges=False,
                                                          device=device,
                                                          sequential=sequential)
        dataset = build_geom_dataset.GeomDrugsDataset(split_data[split_idx],
                                                        transform=transform, debug=False)
        self.raw_dataset = dataset

        atom_encoder = dataset_info["atom_encoder"]
        assert len(set([k for k, v in atom_encoder.items()])) == len([k for k, v in atom_encoder.items()])
        self.one_hot_idx2atom_type_str = {v: k for k, v in atom_encoder.items()}
        
        del split_data

    def __len__(self):
        return len(self.raw_dataset)

    # @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        # A shallow wrap for data
        smi = "empty"
        data = self.raw_dataset[idx]
        one_hot_idx = torch.argmax(data["one_hot"].to(torch.float32), dim=-1).cpu().numpy()
        atoms = np.vectorize(lambda x: self.one_hot_idx2atom_type_str[x])(one_hot_idx)
        coordinates = [data["positions"].cpu().numpy()]
        return {"smi": smi, "atoms": atoms, "coordinates": coordinates}




@register_task("unimol_drug_more_global")
class UniMolDrugMoreGlobalTask(UnicoreTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data",
            help="colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner",
        )
        parser.add_argument(
            "--data_more",
            help="colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner",
        )

        parser.add_argument(
            "--mask-prob",
            default=0.15,
            type=float,
            help="probability of replacing a token with mask",
        )
        parser.add_argument(
            "--leave-unmasked-prob",
            default=0.05,
            type=float,
            help="probability that a masked token is unmasked",
        )
        parser.add_argument(
            "--random-token-prob",
            default=0.05,
            type=float,
            help="probability of replacing a token with a random token",
        )
        parser.add_argument(
            "--noise-type",
            default="uniform",
            choices=["trunc_normal", "uniform", "normal", "none"],
            help="noise type in coordinate noise",
        )
        parser.add_argument(
            "--noise",
            default=1.0,
            type=float,
            help="coordinate noise for masked atoms",
        )
        parser.add_argument(
            "--remove-hydrogen",
            action="store_true",
            help="remove hydrogen atoms",
        )
        parser.add_argument(
            "--remove-polar-hydrogen",
            action="store_true",
            help="remove polar hydrogen atoms",
        )
        parser.add_argument(
            "--max-atoms",
            type=int,
            default=256,
            help="selected maximum number of atoms in a molecule",
        )
        parser.add_argument(
            "--dict-name",
            default="dict.txt",
            help="dictionary file",
        )
        parser.add_argument(
            "--only-polar",
            default=1,
            type=int,
            help="1: only polar hydrogen ; -1: all hydrogen ; 0: remove all hydrogen ",
        )
        parser.add_argument(
            "--conf-size",
            default=10,
            type=int,
            help="number of conformers generated with each molecule",
        )
        
        parser.add_argument(
            "--finetune-mol-model",
            default=None,
            type=str,
            help="",
        )
        

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed
        # add mask token
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)
        if self.args.only_polar > 0:
            self.args.remove_polar_hydrogen = True
        elif args.only_polar < 0:
            self.args.remove_polar_hydrogen = False
        else:
            self.args.remove_hydrogen = True

    @classmethod
    def setup_task(cls, args, **kwargs):
        dictionary = Dictionary.load(os.path.join(args.data, args.dict_name))
        logger.info("dictionary: {} types".format(len(dictionary)))
        return cls(args, dictionary)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        
        split_path = os.path.join(self.args.data_more, split + ".lmdb")
        raw_dataset_more = LMDBDataset(split_path)
        raw_dataset_more = Add2DConformerDataset(
                    raw_dataset_more, "smi", "atoms", "coordinates"
                )
        raw_dataset_drug = DrugDataset(Path(self.args.data) / "geom_drugs_30.npy", split)
        
        
        raw_dataset = ComposedDataset([raw_dataset_more, raw_dataset_drug])

        

        def one_dataset(raw_dataset, coord_seed, mask_seed):
            if self.args.mode =='train':
                smi_dataset = KeyDataset(raw_dataset, "smi")
                dataset = ConformerSampleDataset(
                    raw_dataset, coord_seed, "atoms", "coordinates"
                )
                dataset = AtomTypeDataset(raw_dataset, dataset)
            elif self.args.mode == 'infer':
                dataset = TTADataset(
                    raw_dataset, self.args.seed, "atoms", "coordinates", self.args.conf_size
                )
                dataset = AtomTypeDataset(dataset, dataset)
                smi_dataset = KeyDataset(dataset, "smi")
            dataset = RemoveHydrogenDataset(
                dataset,
                "atoms",
                "coordinates",
                self.args.remove_hydrogen,
                self.args.remove_polar_hydrogen,
            )
            dataset = CroppingDataset(
                dataset, self.seed, "atoms", "coordinates", self.args.max_atoms
            )
            dataset = NormalizeDataset(dataset, "coordinates", normalize_coord=True)
            token_dataset = KeyDataset(dataset, "atoms")
            token_dataset = TokenizeDataset(
                token_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
            )
            coord_dataset = KeyDataset(dataset, "coordinates")
            expand_dataset = MaskPointsDataset(
                token_dataset,
                coord_dataset,
                self.dictionary,
                pad_idx=self.dictionary.pad(),
                mask_idx=self.mask_idx,
                noise_type=self.args.noise_type,
                noise=self.args.noise,
                seed=mask_seed,
                mask_prob=self.args.mask_prob,
                leave_unmasked_prob=self.args.leave_unmasked_prob,
                random_token_prob=self.args.random_token_prob,
            )
            expand_dataset_difficult = MaskPointsDataset(
                token_dataset,
                coord_dataset,
                self.dictionary,
                pad_idx=self.dictionary.pad(),
                mask_idx=self.mask_idx,
                noise_type=self.args.noise_type,
                noise=self.args.noise * 2,
                seed=mask_seed,
                mask_prob=self.args.mask_prob * 2,
                leave_unmasked_prob=self.args.leave_unmasked_prob,
                random_token_prob=self.args.random_token_prob,
            )
            
            clean_dataset = MaskPointsDataset(
                token_dataset,
                coord_dataset,
                self.dictionary,
                pad_idx=self.dictionary.pad(),
                mask_idx=self.mask_idx,
                noise_type=self.args.noise_type,
                noise=0.,
                seed=mask_seed,
                mask_prob=0.,
                leave_unmasked_prob=0.,
                random_token_prob=0.,
            )

            def PrependAndAppend(dataset, pre_token, app_token):
                dataset = PrependTokenDataset(dataset, pre_token)
                return AppendTokenDataset(dataset, app_token)

            encoder_token_dataset = KeyDataset(expand_dataset, "atoms")
            encoder_target_dataset = KeyDataset(expand_dataset, "targets")
            encoder_coord_dataset = KeyDataset(expand_dataset, "coordinates")
            
            difficult_encoder_token_dataset = KeyDataset(expand_dataset_difficult, "atoms")
            difficult_encoder_target_dataset = KeyDataset(expand_dataset_difficult, "targets")
            difficult_encoder_coord_dataset = KeyDataset(expand_dataset_difficult, "coordinates")
            
            clean_encoder_token_dataset = KeyDataset(clean_dataset, "atoms")
            clean_encoder_target_dataset = KeyDataset(clean_dataset, "targets")
            clean_encoder_coord_dataset = KeyDataset(clean_dataset, "coordinates")
            
            src_dataset = PrependAndAppend(
                encoder_token_dataset, self.dictionary.bos(), self.dictionary.eos()
            )
            difficult_src_dataset = PrependAndAppend(
                difficult_encoder_token_dataset, self.dictionary.bos(), self.dictionary.eos()
            )
            clean_src_dataset = PrependAndAppend(
                clean_encoder_token_dataset, self.dictionary.bos(), self.dictionary.eos()
            )
            
            tgt_dataset = PrependAndAppend(
                encoder_target_dataset, self.dictionary.pad(), self.dictionary.pad()
            )
            difficult_tgt_dataset = PrependAndAppend(
                difficult_encoder_target_dataset, self.dictionary.pad(), self.dictionary.pad()
            )
            
            encoder_coord_dataset = PrependAndAppend(encoder_coord_dataset, 0.0, 0.0)
            difficult_encoder_coord_dataset = PrependAndAppend(difficult_encoder_coord_dataset, 0.0, 0.0)
            clean_encoder_coord_dataset = PrependAndAppend(clean_encoder_coord_dataset, 0.0, 0.0)
            
            encoder_distance_dataset = DistanceDataset(encoder_coord_dataset)
            difficult_encoder_distance_dataset = DistanceDataset(difficult_encoder_coord_dataset)
            clean_encoder_distance_dataset = DistanceDataset(clean_encoder_coord_dataset)
            
            edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
            difficult_edge_type = EdgeTypeDataset(difficult_src_dataset, len(self.dictionary))
            clean_edge_type = EdgeTypeDataset(clean_src_dataset, len(self.dictionary))
            
            coord_dataset = FromNumpyDataset(coord_dataset)
            coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
            
            distance_dataset = DistanceDataset(coord_dataset)
            
            return (
                {
                "src_tokens": RightPadDataset(
                    src_dataset,
                    pad_idx=self.dictionary.pad(),
                ),
                "src_coord": RightPadDatasetCoord(
                    encoder_coord_dataset,
                    pad_idx=0,
                ),
                "src_distance": RightPadDataset2D(
                    encoder_distance_dataset,
                    pad_idx=0,
                ),
                "src_edge_type": RightPadDataset2D(
                    edge_type,
                    pad_idx=0,
                ),
                }, 
                # NOTE: This is for difficult (second-stage) dataset
                {
                "src_tokens": RightPadDataset(
                    difficult_src_dataset,
                    pad_idx=self.dictionary.pad(),
                ),
                "src_coord": RightPadDatasetCoord(
                    difficult_encoder_coord_dataset,
                    pad_idx=0,
                ),
                "src_distance": RightPadDataset2D(
                    difficult_encoder_distance_dataset,
                    pad_idx=0,
                ),
                "src_edge_type": RightPadDataset2D(
                    difficult_edge_type,
                    pad_idx=0,
                ),
                },
                # NOTE: This is for clean structure
                {
                "src_tokens": RightPadDataset(
                    clean_src_dataset,
                    pad_idx=self.dictionary.pad(),
                ),
                "src_coord": RightPadDatasetCoord(
                    clean_encoder_coord_dataset,
                    pad_idx=0,
                ),
                "src_distance": RightPadDataset2D(
                    clean_encoder_distance_dataset,
                    pad_idx=0,
                ),
                "src_edge_type": RightPadDataset2D(
                    clean_edge_type,
                    pad_idx=0,
                ),
                },
                
                {
                "tokens_target": RightPadDataset(
                    tgt_dataset, pad_idx=self.dictionary.pad()
                ),
                "distance_target": RightPadDataset2D(distance_dataset, pad_idx=0),
                "coord_target": RightPadDatasetCoord(coord_dataset, pad_idx=0),
                "smi_name": RawArrayDataset(smi_dataset),
                },
                # NOTE: This is for difficult (second-stage) target dataset
                {
                "tokens_target": RightPadDataset(
                    difficult_tgt_dataset, pad_idx=self.dictionary.pad()
                ),
                "distance_target": RightPadDataset2D(distance_dataset, pad_idx=0),
                "coord_target": RightPadDatasetCoord(coord_dataset, pad_idx=0),
                "smi_name": RawArrayDataset(smi_dataset),
                }
            )

        net_input, difficult_net_input, clean_net_input, target, difficult_target = one_dataset(raw_dataset, self.args.seed, self.args.seed)
        dataset = {"net_input": net_input, "target": target, "difficult_net_input": difficult_net_input, "difficult_target": difficult_target, "clean_net_input": clean_net_input}
        dataset = NestedDictionaryDataset(dataset)
        if split in ["train", "train.small"]:
            dataset = EpochShuffleDataset(dataset, len(dataset), self.args.seed)
        self.datasets[split] = dataset

    def build_model(self, args):
        from unicore import models

        model = models.build_model(args, self)
        
        # # NOTE: We modify here to load weights
        if args.finetune_mol_model is not None:
            print("load pretrain model weight from...", args.finetune_mol_model)
            state = checkpoint_utils.load_checkpoint_to_cpu(
                args.finetune_mol_model,
            )
            load_profile = model.load_state_dict({key.replace('unimol.', ''): value for key, value in state["model"].items()}, strict=False)
            print(load_profile)
        
        return model

    def disable_shuffling(self) -> bool:
        return True
    
