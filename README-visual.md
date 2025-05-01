# Reproduce Visual Exp on Imagenet1k

## 1. Environment Setup
Run following commands to setup a env using Miniconda.

```Bash
# Python=3.9
# CUDA 11.8
# Ubuntu 22.04 + x86_64
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
~/miniconda3/bin/conda init bash
source ~/.bashrc
conda create -n csr python=3.9 -y
conda activate csr
```
Then install all dependencies and packages:
```Bash
git clone https://github.com/Hadlay-Zhang/CSR_Adaptive_Rep.git
cd CSR_Adaptive_Rep
bash setup.sh # all dependencies and packages installed in this script
```

## 2. Dataset Preparation
Download the ffcv pre-processed formatted ImageNet-1K data (at least 750GB storage needed to download and merge data):
```Bash
huggingface-cli download "HadlayZ/ImageNet-1K-ffcv" --local-dir data_imagenet_ffcv/ --repo-type dataset
```
Due to huggingface single-file limitation, train set is splitted and stored (val set is complete, no need for merging). After downloading, run following commands to merge:
```Bash
cd data_imagenet_ffcv
cat train_chunk_* > train_500_0.50_90.ffcv
# rm -rf train_chunk_*   # delete chunks to free storage if needed
cd ../
```

## 3. Get Pre-trained Embeddings
For training and evaluation simplicity, we precompute image embeddings using models from [Timm](https://github.com/huggingface/pytorch-image-models). By default, [resnet50d.ra4_e3600_r224_in1k](https://huggingface.co/timm/resnet50d.ra4_e3600_r224_in1k) is adopted as pre-trained visual backbone. We also provide embeds extracted by **FF2048 backbones** (same backbone weights with MRL), and embeds by **SoTA backbones** at [Dataset Link](https://huggingface.co/datasets/W1nd-navigator/CSR-precompute-embeds).

To extract pre-trained resnet-50 embeddings, run:
```Bash
cd inference
python pretrained_inference.py --train_data_ffcv ../data_imagenet_ffcv/train_500_0.50_90.ffcv --eval_data_ffcv ../data_imagenet_ffcv/val_500_0.50_90.ffcv --model_name resnet50d.ra4_e3600_r224_in1k
cd ../
```
Then stack embeds together:
```Bash
python stack_emb.py
```

## 4. Train Contrastvie Sparse Representation on Imagenet1K
```Bash
python main_visual.py --use_ddp --batch-size 4096 --lr 4e-4 --use_CL --topk 8 --auxk 512 --hidden-size 8192
```

## 5. Get CSR Embeddings for Evaluation
```Bash
python csr_inference.py --model_name resnet50d.ra4_e3600_r224_in1k --topk 8 --hidden-size 512 --csr_ckpt ../ckpt/CSR_topk_8/checkpoint_9.pth
```

## 6. KNN Evaluation
[FAISS](https://github.com/facebookresearch/faiss) supports GPU-usage for acceleration. 
(Note: At least 40GB GPU memory needed to use GPU acceleration)
```Bash
cd retrieval
# Get FAISS index
python faiss_nn_gpus.py --topk 8 --gpus <number of GPUs> # 0 for CPU, n > 0 for n-GPUs 
# Evaluate Top1 accuracy
python compute_metrics.py --topk 8 --prec 'float32'
```