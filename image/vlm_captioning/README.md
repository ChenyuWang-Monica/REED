## Guidelines: Image captioning and VLM embedding

### 1. Environment setup
To use Qwen-2-VL:
```
pip install -U transformers
```
To use Qwen-2.5-VL:
```
pip install git+https://github.com/huggingface/transformers@f3f6c86582611976e72be054675e2bf0abb5f775
pip install accelerate
pip install qwen-vl-utils
pip install 'vllm>0.7.2'
```
You can find more instructions on downloading and using the models through the official repos of Qwen: [Qwen2-VL - a Qwen Collection](https://huggingface.co/collections/Qwen/qwen2-vl-66cee7455501d7126940800d), [Qwen2.5-VL - a Qwen Collection](https://huggingface.co/collections/Qwen/qwen25-vl-6795ffac22b334a837c0f9a5).

### 2. Image Captioning
We generate and save the caption for each image in the ImageNet dataset using Qwen2-VL 2B model with the following command:
```
accelerate launch captioning.py \
    --data-dir=[YOUR_DATA_PATH] \
    --model-name-or-path="Qwen2-VL-2B-Instruct"
```

### 3. Save VLM Embedding
We calculate and save the VLM embedding for each image-caption pair using Qwen2-VL 7B model with the following command. The embeddings corresponding to layer 0, 1, 15, 27 of the Qwen2-VL 7B model are saved in separate folders.
```
accelerate launch captioning_embedding.py \
    --data-dir=[YOUR_DATA_PATH] \
    --model-name-or-path="Qwen2-VL-7B-Instruct"
```
