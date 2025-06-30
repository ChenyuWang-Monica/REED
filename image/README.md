# REED - Image Generation
This codebase is developed on top of [REPA (Yu et.al, 2024)](https://github.com/sihyun-yu/REPA).

## 1. Environment setup

```bash
conda create -n reed python=3.9 -y
conda activate reed
pip install -r requirements.txt
```

## 2. Data Processing

We use [ImageNet](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data) for class conditional image generation, and follow the preprocessing code and guidelines used in [REPA](https://github.com/sihyun-yu/REPA). You can place the data that you want and can specifiy it via `--data-dir` arguments in training scripts. Please refer to REPA's [preprocessing guide](https://github.com/sihyun-yu/REPA/tree/master/preprocessing).


## 3. Image captioning and VLM embedding
We provide code and guidelines in `vlm_captioning/`.


## 4. Training

To train the SiT-XL/2 model for 1M steps, use the following command:
```
accelerate launch train.py \
    --report-to=wandb \
    --allow-tf32 \
    --mixed-precision=fp16 \
    --seed=0 \
    --model=SiT-XL/2 \
    --enc-type=dinov2-vit-b \
    --proj-coeff=0.5 \
    --encoder-depth=8 \
    --encoder-depth-text=16 \
    --output-dir=[YOUR_OUTPUT_DIR] \
    --exp-name=sitxl-dinov2-b-enc8-textenc16-qwenvl-7b-layer15-0.5-warmup50k \
    --data-dir=[YOUR_DATA_PATH] \
    --text-embeds-dir=text_embeds_Qwen2-VL-7B-Instruct_layer_15 \
    --repa-coef 1.0 0.5 \
    --diffusion-warm-up-steps=50000 \
    --repa-weight-decay=constant \
    --max-train-steps=1000000
```

The model checkpoint of SiT-XL trained for 200 epochs (1M steps) is provided in [this link](https://www.dropbox.com/scl/fi/a3vnv1qc4ap8573zzgxhx/last.pt?rlkey=urr4txqqh8ar33plraeq5diko&st=ey6leca8&dl=0).

## 5. Evaluation

You can generate images (and the .npz file can be used for [ADM evaluation](https://github.com/openai/guided-diffusion/tree/main/evaluations) suite) through the following script:

```
torchrun --nnodes=1 --nproc_per_node=4 generate.py \
    --model SiT-XL/2 \
    --num-fid-samples 50000 \
    --ckpt YOUR_CHECKPOINT_PATH \
    --path-type=linear \
    --encoder-depth=8 \
    --projector-embed-dims=768 \
    --per-proc-batch-size=128 \
    --mode=sde --num-steps=250 \
    --cfg-scale=1.0 \
    --sample-dir YOUR_SAMPLE_SAVING_PATH
```
