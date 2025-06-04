# REED - Molecule Generation

This folder is the pytorch implementation of the molecule generation tasks in REED. This branch is largely based on the [GeoRCG](https://arxiv.org/abs/2410.03655) codebase, which supports both EDM and SemlaFlow base models on both QM9 and GEOM-DRUG datasets. In our paper, we focus on the state-of-the-art SemlaFlow base model and use the more challenging GEOM-DRUG dataset to illustrate the effectiveness of REED. We leverage the representation alignment method instead of the representation conditioning method in GeoRCG.

## Dependencies

To get started, you'll need the following dependencies:

- `torch==2.4.1`
- `torch-geometric==2.6.0`
- `pyg-lib==0.4.0+pt24cu121`
- `torch-scatter==2.1.2+pt24cu121`
- `torch-sparse==0.6.18+pt24cu121`
- `torch-spline-conv==1.2.2+pt24cu121`
- `torch-cluster==1.6.3+pt24cu121`
- `hydra-core==1.3.2`
- `networkx==3.1`
- `posebusters==0.3.1`
- `unicore==0.0.1` (only when you use unimol as the encoder.)

## Usage

### Pre-trained Encoder

For the QM9 dataset, we leverage [Frad](https://github.com/fengshikun/Frad) as the geometric encoder. You can download the pre-trained weights [here](https://drive.google.com/file/d/1O6f6FzYogBS2Mp4XsdAAEN4arLtLH38G/view?usp=share_link).  

For the GEOM-DRUG dataset, we pre-trained [Unimol](https://openreview.net/forum?id=6K2RM6wVqKu) using their official codebase on GEOM-DRUG dataset itself (only use the training dataset to avoid data leak). If you opt to use Unimol as the encoder, ensure you install [uni-core](https://github.com/dptech-corp/Uni-Core). A checkpoint of the finetuned Unimol encoder on GEOM-DRUG dataset is available in ./checkpoints/unimol_global.pt.

### Dataset

The QM9 dataset is automatically downloaded upon execution. For the GEOM-DRUG dataset, please follow the instructions in the `README.md` provided in the [EDM GitHub repository](https://github.com/ehoogeboom/e3_diffusion_for_molecules). Notice that the EDM code is integrated into this codebase, which should simplify the setup process.

### Training the Molecule Generator with REED

To train the molecule generator with SemlaFlow (https://github.com/rssrwn/semla-flow/) base model, use the following commands:

```bash
python semlaflow/train.py  --num_inference_steps 100  --encoder_type 'unimol_global' --encoder_path './checkpoints/unimol_global.pt' --rep_alignment --repa_loss_weight 0.2 --align_depth 4 --diffusion_loss_max_epoch 30
```

You can try different configurations for the model.

We note that for the GEOM-DRUG dataset, training was conducted on two Nvidia A100 GPUs with a batch size of 64 using `torch.distributed.run`. If you lack access to resources, you can reduce the batch size to match your hardware capabilities (but may lead to different model performance). 

### Evaluation

For unconditional evaluation, you can use the following command:

```bash
python semlaflow/evaluate.py --ckpt_path /your/path/to/checkpoint.pt
```

#### Visualization

For visualization, simply add the `--save_images` flag.


