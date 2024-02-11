#!/usr/bin/env bash
#$ -wd /exp/rhuang/meta/icefall/egs/librispeech/ASR/
#$ -V
#$ -N train_posterior_dropout
#$ -j y -o /exp/rhuang/meta/icefall/egs/librispeech/ASR/log/log-$JOB_NAME-$JOB_ID.out
#$ -M ruizhe@jhu.edu
#$ -m e
#$ -l mem_free=20G,h_rt=600:00:00,gpu=4
#$ -q gpu.q@@v100

# #$ -q gpu.q@@v100
# #$ -q gpu.q@@rtx

# #$ -l ram_free=300G,mem_free=300G,gpu=0,hostname=b*

# hostname=b19
# hostname=!c04*&!b*&!octopod*
# hostname
# nvidia-smi

# conda activate aligner5
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

exp_dir=zipformer_early_ctc/exp/exp-test

echo
echo "exp_dir:" $exp_dir
echo

# echo 
# echo "max_frame_dropout_rate = 0.2, nei"
# echo "changed_ratio = 0.8"
# echo

####################################
# train transducer+ctc
####################################
python zipformer_early_ctc/train.py \
  --world-size 4 \
  --num-epochs 40 \
  --start-epoch 1 \
  --use-fp16 true \
  --master-port 12535 \
  --causal 0 \
  --full-libri true \
  --use-transducer true \
  --use-ctc true \
  --ctc-loss-scale 0.2 \
  --exp-dir $exp_dir \
  --max-duration 1000


####################################
# train ctc
####################################
# python posterior_dropout/train.py \
#   --world-size 4 \
#   --num-epochs 40 \
#   --start-epoch 1 \
#   --use-fp16 true \
#   --master-port 12535 \
#   --causal 0 \
#   --full-libri true \
#   --use-transducer false \
#   --use-ctc true \
#   --ctc-loss-scale 0.2 \
#   --exp-dir $exp_dir \
#   --max-duration 800 # \
#   # --start-epoch 35

# CTC:
# https://tensorboard.dev/experiment/guzomYumRDWyRoDtrnDxCg/#scalars

####################################
# train transducer
####################################
python posterior_dropout/train.py \
  --world-size 4 \
  --num-epochs 40 \
  --start-epoch 1 \
  --use-fp16 true \
  --master-port 12535 \
  --causal 0 \
  --full-libri true \
  --use-transducer true \
  --use-ctc false \
  --ctc-loss-scale 0.2 \
  --exp-dir $exp_dir \
  --max-duration 800 \
  --start-epoch 37

# Transducer:
# https://tensorboard.dev/experiment/C87OKiEzRVqBFA4RaBS7Ew/
# Transducer pos-0.3-0.8 libri100
# https://tensorboard.dev/experiment/iM14ORI9TqWExRCqk3UAQw/

####################################
# tensorboard
####################################
# tensorboard dev upload --logdir /exp/rhuang/meta/icefall/egs/librispeech/ASR/$exp_dir/tensorboard --description `pwd`
# tensorboard dev upload --logdir $exp_dir/tensorboard --description `pwd`/$exp_dir/
