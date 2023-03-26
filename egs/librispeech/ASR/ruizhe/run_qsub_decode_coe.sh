#!/usr/bin/env bash
#$ -wd /exp/rhuang/icefall_latest/egs/librispeech/ASR
#$ -V
#$ -N decode
#$ -j y -o ruizhe/log/log-$JOB_NAME-$JOB_ID.out
#$ -M ruizhe@jhu.edu
#$ -m e
#$ -l mem_free=20G,h_rt=600:00:00,gpu=1
#$ -q gpu.q@@rtx

# #$ -q gpu.q@@v100

# #$ -l ram_free=300G,mem_free=300G,gpu=0,hostname=b*

# hostname=b19
# hostname=!c04*&!b*&!octopod*
hostname
nvidia-smi

export PATH="/exp/rhuang/mambaforge/envs/icefall2/bin/":$PATH
which python
export PATH="/exp/rhuang/gcc-7.2.0/mybuild/bin":$PATH
export PATH=/exp/rhuang/cuda-10.2/bin/:$PATH
which nvcc
export LD_LIBRARY_PATH=/exp/rhuang/cuda-10.2/lib64/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/exp/rhuang/gcc-7.2.0/mybuild/lib64:$LD_LIBRARY_PATH

# k2
K2_ROOT=/exp/rhuang/k2/
export PYTHONPATH=$K2_ROOT/k2/python:$PYTHONPATH # for `import k2`
export PYTHONPATH=$K2_ROOT/build_debug_cuda10.2/lib:$PYTHONPATH # for `import _k2`

# kaldifeat
export PYTHONPATH=/exp/rhuang/kaldifeat/build/lib:/exp/rhuang/kaldifeat/kaldifeat/python:$PYTHONPATH

# icefall
# export PYTHONPATH=/exp/rhuang/icefall/:$PYTHONPATH
export PYTHONPATH=/exp/rhuang/icefall_latest/:$PYTHONPATH
# export PYTHONPATH=/exp/rhuang/icefall/icefall/transformer_lm/:$PYTHONPATH

# To verify SGE_HGR_gpu and CUDA_VISIBLE_DEVICES match for GPU jobs.
env | grep SGE_HGR_gpu
env | grep CUDA_VISIBLE_DEVICES
echo "hostname: `hostname`"

# libri100
# python tdnn_lstm_ctc/decode.py \
#     --exp-dir "tdnn_lstm_ctc/exp/exp_libri_100" \
#     --lang-dir "data/lang_phone" \
#     --lm-dir "data/lm" \
#     --epoch 29 \
#     --avg 5 \
#     --max-duration 100

# python zipformer_ctc/decode.py \
#     --exp-dir "zipformer_ctc/exp/exp_libri_100" \
#     --lang-dir "data/lang_bpe_500" \
#     --epoch 30 \
#     --avg 10 \
#     --max-duration 100 \
#     --nbest-scale 1.2 \
#     --hp-scale 1.0 \
#     --decoding-method nbest-rescoring-LG

# python zipformer_mmi/decode.py \
#     --exp-dir "zipformer_mmi/exp/exp_libri_100" \
#     --lang-dir "data/lang_bpe_500" \
#     --epoch 30 \
#     --avg 10 \
#     --max-duration 30 \
#     --nbest-scale 1.2 \
#     --hp-scale 1.0 \
#     --decoding-method nbest-rescoring-LG

python zipformer_mmi_hmm/decode.py \
    --exp-dir "zipformer_mmi_hmm/exp/exp_libri_100_ml" \
    --lang-dir "data/lang_bpe_500" \
    --epoch 30 \
    --avg 10 \
    --max-duration 30 \
    --nbest-scale 1.2 \
    --hp-scale 1.0 \
    --decoding-method nbest-rescoring-LG \
    --sil-modeling false
