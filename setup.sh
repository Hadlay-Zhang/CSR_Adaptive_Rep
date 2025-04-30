# Python3.9
# CUDA 11.8
# Ubuntu 22.04

# Create conda env
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
~/miniconda3/bin/conda init bash
source ~/.bashrc
conda create -n csr python=3.9 -y
conda activate csr

git clone https://github.com/neilwen987/CSR_Adaptive_Rep.git
cd CSR_Adaptive_Rep

# dependencies
conda install -c conda-forge gcc gxx pkg-config -y
conda install pytorch==2.1.2 torchvision==0.16.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install faiss-gpu==1.7.2
conda install -c conda-forge opencv libjpeg-turbo -y

# build ffcv
git clone https://github.com/libffcv/ffcv.git
cd ffcv 
pip install -e .
cd ../

# others
conda install -c conda-forge numpy pandas scipy matplotlib scikit-learn -y
pip install -r new-requirements.txt

sudo apt install -y libgl1-mesa-glx

# CUDA-related
# export CUDA_PATH=/usr/local/cuda
# export CFLAGS="-I$CUDA_PATH/include"
# export LDFLAGS="-L$CUDA_PATH/lib64"
# export LIBRARY_PATH=$CUDA_PATH/lib64:$LIBRARY_PATH
# export PATH=$CUDA_PATH/bin:$PATH
pip install cupy-cuda11x

# HF transfer
export HF_HUB_ENABLE_HF_TRANSFER=1