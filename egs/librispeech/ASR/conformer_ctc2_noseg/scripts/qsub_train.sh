#!/usr/bin/env bash
#$ -wd /exp/rhuang/meta/icefall/egs/librispeech/ASR/
#$ -V
#$ -N train_conformer
#$ -j y -o /exp/rhuang/meta/icefall/egs/librispeech/ASR/log/log-$JOB_NAME-$JOB_ID.out
#$ -M ruizhe@jhu.edu
#$ -m e
#$ -l mem_free=32G,h_rt=600:00:00,gpu=4,hostname=!r7n07*
#$ -q gpu.q@@v100

# #$ -q gpu.q@@v100
# #$ -q gpu.q@@rtx

# #$ -l ram_free=300G,mem_free=300G,gpu=0,hostname=b*

# hostname=b19
# hostname=!c04*&!b*&!octopod*
# hostname
# nvidia-smi

# conda activate /home/hltcoe/rhuang/mambaforge/envs/aligner5
export PATH="/home/hltcoe/rhuang/mambaforge/envs/aligner5/bin/":$PATH
module load cuda11.7/toolkit
module load cudnn/8.5.0.96_cuda11.x
module load nccl/2.13.4-1_cuda11.7
module load gcc/7.2.0
module load intel/mkl/64/2019/5.281

which python
nvcc --version
nvidia-smi
date

# k2
K2_ROOT=/exp/rhuang/meta/k2/
export PYTHONPATH=$K2_ROOT/k2/python:$PYTHONPATH # for `import k2`
export PYTHONPATH=$K2_ROOT/temp.linux-x86_64-cpython-310/lib:$PYTHONPATH # for `import _k2`
export PYTHONPATH=/exp/rhuang/meta/icefall:$PYTHONPATH

# # torchaudio recipe
# cd /exp/rhuang/meta/audio
# cd examples/asr/librispeech_conformer_ctc

# To verify SGE_HGR_gpu and CUDA_VISIBLE_DEVICES match for GPU jobs.
env | grep SGE_HGR_gpu
env | grep CUDA_VISIBLE_DEVICES
echo "hostname: `hostname`"
echo "current path:" `pwd`

# export PYTHONPATH=/exp/rhuang/meta/audio/examples/asr/librispeech_conformer_ctc2:$PYTHONPATH

exp_dir=/exp/rhuang/meta/icefall/egs/librispeech/ASR/conformer_ctc2_noseg/exp/exp_seed

mkdir -p $exp_dir

echo
echo "exp_dir:" $exp_dir
echo

####################################
# train
####################################

# if false; then
#    echo "True"
# else
#    echo "False"
# fi

if true; then
    ./conformer_ctc2_noseg/train.py \
      --exp-dir $exp_dir \
      --lang-dir data/lang_bpe_500 \
      --full-libri 1 \
      --max-duration 400 \
      --concatenate-cuts 0 \
      --world-size 4 \
      --bucketing-sampler 1 \
      --start-epoch 1 \
      --num-epochs 30 \
      --att-rate 0
fi



####################################
# tensorboard
####################################
# tensorboard dev upload --logdir /exp/rhuang/meta/icefall/egs/librispeech/ASR/$exp_dir/tensorboard --description `pwd`
# wandb sync $exp_dir/tensorboard

# https://github.com/k2-fsa/icefall/issues/1298
# python -c "import wandb; wandb.init(project='icefall-asr-gigaspeech-zipformer-2023-10-20')"
# wandb sync zipformer/exp/tensorboard -p icefall-asr-gigaspeech-zipformer-2023-10-20

# https://stackoverflow.com/questions/37987839/how-can-i-run-tensorboard-on-a-remote-server
# ssh -L 16006:127.0.0.1:6006 rhuang@test1.hltcoe.jhu.edu
# tensorboard --logdir $exp_dir/tensorboard --port 6006
# http://localhost:16006 

