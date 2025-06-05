# REED - Protein Sequence Design

## Installation

The codebase is modified from [MultiFlow](https://github.com/jasonkyuyim/multiflow), where detailed instructions on installation can be found.

## Precomputing AlphaFold3 Embeddings

We use a modified version of AlphaFold3 code to obtain protein representations. Details can be found in the REED-AlphaFold3 branch.

## Training Discrete Diffusion Models

To reproduce the results in the main paper, run:

```
python fmif/train_fmif.py --base_path /your/path/to/dataset/ --repr_dir /your/path/to/alphafold/repr --num_epochs 200 --eval_every_n_epochs 10 \
--update_edge --learnable_node --repr_weight=0.2 --repr_norm --repa_coeff 0.5 2.0 1.0 --learning_rate 1e-3 \
--start_diffusion_epoch 0 --diffusion_warm_up_epoch 50 --diffusion_decay constant --repa_weight_decay cosine --repa_epoch 200 
```

Replace the paths with your paths to the dataset (set in the Installation section) and the AlphaFold3 embeddings (set in the Precomputing AlphaFold3 Embeddings section). For additional hyperparameters as well as their usages and default values, see fmif/train_fmif.py for more details. You are welcome to try out different settings.
