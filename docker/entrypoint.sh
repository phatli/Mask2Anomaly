#!/bin/zsh

cd /home/user/Mask2Anomaly
export CUDA_HOME=/usr/local/cuda
cd mask2former/modeling/pixel_decoder/ops
pip install -e .

exec "$@"