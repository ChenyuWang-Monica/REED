# AlphaFold3 - REED

This branch is the helper repo to obtain AlphaFold3 representations for the protein sequence design experiments in the REED paper (Learning Diffusion Models with Flexible Representation Guidance). This version is largely built on the original [AlphaFold3](https://github.com/google-deepmind/alphafold3) repo with Jax. Modifications include:

- Representation extraction: we extract te single and pair features in the last Pairformer layer as node and edge representations for sequence, and take the attention output latents in the last hidden layer in the diffusion head as structural embeddings.
- Fixed bugs in the original repo, including gpu specification and other minor issues.

## Installation

Please refer to [AlphaFold3 Installation](https://github.com/google-deepmind/alphafold3/blob/main/docs/installation.md) for detailed instructions on installations. Inference can be conducted on a single A100 80G gpu.

## Usage

CD to /your/path/to/AlphaFold3 to continue all the followings. Here, the path is where you place the AlphaFold3 folder, which should contain a subfolder named alphafold3 with source code files.

### Generate input json files

First, use the following command to convert the PDB data in the training set to AlphaFold3 acceptable json formats. 

```
python alphafold3/save_json.py
```

Remember to change the raw data and save path to yours. We manually split all the data into 24 folders to enable parallel AlphaFold3 inference on different gpus, but you can cancel the folderids to save all the json files in one directory.

To avoid MSA computing, we explicitly set both unpairedMSA and pairedMSA to "", so AlphaFold3 will run in the MSA-free mode. See [AlphaFold3 Input](https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md) for more details. The generated jsons should contain data with the same format as follows.

```
{
  "name": "<NAME_OF_THE_PROTEIN>",
  "modelSeeds": [0],
  "sequences": [
    {"protein": {
    "id": "A",
    "sequence": "<SEQUENCE_OF_THE_PROTEIN>",
    "unpairedMsa": "",
    "pairedMsa": ""
    }}
  ],
  "dialect": "alphafold3",
  "version": 1
}
```

### Extract embeddings

Use the following command to run AlphaFold3 inference and save embeddings:
```
docker run -it \
    --volume /your/path/to/data/pmpnn/raw/seq_json:/root/af_input \
    --volume /your/path/to/output:/root/af_output \
    --volume <MODEL_PARAMETERS_DIR>:/root/models \
    --volume <DB_DIR>:/root/public_databases \
    --gpus device=_GPU_ID \
    -e NVIDIA_VISIBLE_DEVICES=_GPU_ID \
    -e CUDA_VISIBLE_DEVICES=_GPU_ID \
    --privileged \
    alphafold3 \
    python run_alphafold.py \
    --input_dir=/root/af_input/_FOLDER_ID \
    --model_dir=/root/models \
    --output_dir=/root/af_output \
    --gpu_device=0 \  # do not change this
    --run_data_pipeline=True \
    --num_recycles=10 \
    --num_diffusion_steps=200 \
    --num_diffusion_samples=1 \
    --flash_attention_implementation=xla
```

Specify the _GPU_ID (e.g., 1) and _FOLDER_ID (from 0 to 23), as well as all the paths. /your/path/to/data/pmpnn/raw/seq_json is your path to the generated input json files, /your/path/to/output is the path where you want to store the AlphaFold3 embeddings, <MODEL_PARAMETERS_DIR> is the path to your model weights, and <DB_DIR> is the path to your databases. See [AlphaFold3 Installation](https://github.com/google-deepmind/alphafold3/blob/main/docs/installation.md) for more details.
