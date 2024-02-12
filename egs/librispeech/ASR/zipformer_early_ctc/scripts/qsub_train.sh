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

# export CUDA_VISIBLE_DEVICES="0"
# for m in ctc-decoding 1best nbest nbest-rescoring whole-lattice-rescoring; do
#   ./zipformer_early_ctc/ctc_decode.py \
#       --epoch 40 \
#       --avg 16 \
#       --exp-dir $exp_dir \
#       --use-transducer 1 \
#       --use-ctc 1 \
#       --max-duration 300 \
#       --causal 0 \
#       --num-paths 100 \
#       --nbest-scale 1.0 \
#       --hlg-scale 0.6 \
#       --decoding-method $m
# done

######## exp_dir=zipformer_early_ctc/exp/exp-test-4 ########
# ctc-decoding 4.22/10.02
# 1best 3.24/7.39
# nbest 3.24/7.39
# nbest-rescoring  2.96/6.86
# whole-lattice-rescoring 2.95/6.83
######## exp_dir=zipformer_early_ctc/exp/exp-test-5 ########
# ctc-decoding 6.43/14.58
# 1best 4.33/9.98
# nbest 4.33/9.99
# nbest-rescoring  3.86/9.1
# whole-lattice-rescoring 3.85/8.94


# export CUDA_VISIBLE_DEVICES="1"
# for m in greedy_search modified_beam_search fast_beam_search; do
#   ./zipformer_early_ctc/decode.py \
#     --epoch 40 \
#     --avg 16 \
#     --use-averaged-model 1 \
#     --use-transducer 1 \
#     --use-ctc 1 \
#     --exp-dir $exp_dir \
#     --max-duration 600 \
#     --decoding-method $m
# done

######## exp_dir=zipformer_early_ctc/exp/exp-test-4 ########
# greedy_search 2.17/4.93
# modified_beam_search 2.13/4.87
# fast_beam_search 2.14/4.86
######## exp_dir=zipformer_early_ctc/exp/exp-test-5 ########
# greedy_search 2.25/5.14
# modified_beam_search 2.22/5.08
# fast_beam_search 2.25/5.07

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

# # https://www.xmodulo.com/skip-existing-files-scp.html
# scp -r \
#   rhuang25@login.rockfish.jhu.edu:/scratch4/skhudan1/rhuang25/icefall/egs/librispeech/ASR/zipformer_early_ctc/exp/exp-test-4/tensorboard \
#   zipformer_early_ctc/exp/exp-test/tensorboard/exp-test-4
# scp -r \
#   rhuang25@login.rockfish.jhu.edu:/scratch4/skhudan1/rhuang25/icefall/egs/librispeech/ASR/zipformer_early_ctc/exp/exp-test-5/tensorboard \
#   zipformer_early_ctc/exp/exp-test/tensorboard/exp-test-5
# rsync -avz --progress zipformer_early_ctc/exp/exp-test/tensorboard/exp-test-4 rhuang25@login.rockfish.jhu.edu:/scratch4/skhudan1/rhuang25/icefall/egs/librispeech/ASR/zipformer_early_ctc/exp/exp-test-4/tensorboard
# rsync -avz --progress zipformer_early_ctc/exp/exp-test/tensorboard/exp-test-5 rhuang25@login.rockfish.jhu.edu:/scratch4/skhudan1/rhuang25/icefall/egs/librispeech/ASR/zipformer_early_ctc/exp/exp-test-5/tensorboard
# scp \
#   rhuang25@login.rockfish.jhu.edu:/scratch4/skhudan1/rhuang25/icefall/egs/librispeech/ASR/zipformer_early_ctc/exp/exp-test-4/tensorboard/* \
#   zipformer_early_ctc/exp/exp-test/tensorboard/exp-test-4/.
# scp \
#   rhuang25@login.rockfish.jhu.edu:/scratch4/skhudan1/rhuang25/icefall/egs/librispeech/ASR/zipformer_early_ctc/exp/exp-test-5/tensorboard/* \
#   zipformer_early_ctc/exp/exp-test/tensorboard/exp-test-5/.


# source activate aligner5
# tensorboard --logdir $exp_dir/tensorboard --port 6006 --window_title $exp_dir
# ssh -L 16006:127.0.0.1:6006 rhuang25@login.rockfish.jhu.edu
# ssh -L 16006:127.0.0.1:6006 rhuang@test2.hltcoe.jhu.edu
# http://localhost:16006 
