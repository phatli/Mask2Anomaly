FROM phatli/base:2004

USER user

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113


RUN cd /home/user && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb && \
    sudo dpkg -i cuda-keyring_1.1-1_all.deb && \
    sudo apt-get update && \
    sudo apt-get install -y --no-install-recommends cuda-toolkit-11-3

RUN sudo apt-get update && \
    export CUDA_HOME=/usr/local/cuda && \
    sudo apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    ninja-build && \
    cd /home/user && \
    git clone https://github.com/facebookresearch/detectron2.git && \
    cd detectron2 && \
    pip install --no-cache-dir -e . && \
    pip install -U --no-cache-dir opencv-python

RUN pip install --no-cache-dir cython scipy shapely timm h5py submitit scikit-image


COPY --chown=user:user ./entrypoint.sh /entrypoint.sh
RUN sudo chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/zsh"]