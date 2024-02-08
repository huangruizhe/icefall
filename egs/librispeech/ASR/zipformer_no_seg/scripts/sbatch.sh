#!/bin/bash
#SBATCH --job-name=train_librispeech
#SBATCH --output=log/slurm-%A.out # stdout file
#SBATCH --error=log/slurm-%A.err  # stderr file
#SBATCH --time=72:0:0
#SBATCH --partition=a100 
#SBATCH --nodes=1 
#SBATCH --gres=gpu:4 
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH -A skhudan1_gpu
#SBATCH --qos=qos_gpu
#SBATCH --export=ALL
#SBATCH --mail-type=end
#SBATCH --mail-user=ruizhe@jhu.edu


ml mamba || exit 1
ml ffmpeg/4.2.2 git-lfs/2.11.0 ffmpeg/4.2.2 cmake/3.18.4 openblas/0.3.10 || exit 1
ml cuda/11.8.0 cudnn/8.0.4.30-11.1-linux-x64 || exit 1

echo "Start: `date`"
hostname
nvidia-smi
echo CUDA_VISIBLE_DEVICES:"$CUDA_VISIBLE_DEVICES"

source activate aligner5
# export PATH="/home/rhuang25/.conda/envs/aligner5/bin/":$PATH

export PYTHONPATH=/scratch4/skhudan1/rhuang25/k2/k2/python:$PYTHONPATH # for `import k2`
export PYTHONPATH=/scratch4/skhudan1/rhuang25/k2/build_debug/lib:$PYTHONPATH # for `import _k2`
export PYTHONPATH=/scratch4/skhudan1/rhuang25/icefall:$PYTHONPATH
export PYTHONPATH=/scratch4/skhudan1/rhuang25/kaldifeat/kaldifeat/python:$PYTHONPATH
export PYTHONPATH=/scratch4/skhudan1/rhuang25/kaldifeat/build/lib:$PYTHONPATH
# export LD_LIBRARY_PATH=/scratch4/skhudan1/rhuang25/libs:$LD_LIBRARY_PATH  # libcuda.so.1
export LC_ALL=C

# for debugging
# export K2_DISABLE_CHECKS=0
# export K2_SYNC_KERNELS=1
# export CUDA_LAUNCH_BLOCKING=1

cd /home/rhuang25/work/icefall/egs/librispeech/ASR/

####################################
# train ctc with rnnt aux loss
####################################

# exp_dir=zipformer_no_seg/exp-ctc-rnnt
exp_dir=zipformer_no_seg/exp-test
# exp_dir=zipformer_no_seg/exp-test2  # train ctc branch only
# exp_dir=zipformer_no_seg/exp-no-seg1

echo
echo "exp_dir:" $exp_dir
echo

python zipformer_no_seg/train_concat_libri.py \
  --world-size 4 \
  --num-epochs 40 \
  --start-epoch 1 \
  --use-fp16 true \
  --master-port 12535 \
  --causal false \
  --full-libri false \
  --use-transducer false \
  --use-ctc true \
  --ctc-loss-scale 1.0 \
  --exp-dir $exp_dir \
  --max-duration 400 --num-workers 3 \
  --start-epoch 15 --ctc-beam-size 4

echo "Done: `date`"


# # Decode
# for m in ctc-decoding 1best nbest nbest-rescoring whole-lattice-rescoring; do
#   ./zipformer/ctc_decode.py \
#       --epoch 40 \
#       --avg 16 \
#       --exp-dir $exp_dir \
#       --use-transducer 1 \
#       --use-ctc 1 \
#       --max-duration 500 \
#       --causal 0 \
#       --num-paths 100 \
#       --nbest-scale 1.0 \
#       --hlg-scale 0.6 \
#       --decoding-method $m
# done


# https://stackoverflow.com/questions/37987839/how-can-i-run-tensorboard-on-a-remote-server
# source activate aligner5
# tensorboard --logdir $exp_dir/tensorboard --port 6006 --window_title $exp_dir
# ssh -L 16006:127.0.0.1:6006 rhuang25@login.rockfish.jhu.edu
# http://localhost:16006 

# tensorboard --logdir /home/rhuang25/work/icefall/egs/librispeech/ASR/zipformer_no_seg/exp-test/tensorboard/cmp/ --port 6006

#########################
# Get a seed model first on a small subset
#########################

# exp_dir=zipformer_no_seg/exp-seed
# python zipformer_no_seg/train_seed.py \
#   --world-size 4 \
#   --num-epochs 20 \
#   --start-epoch 1 \
#   --use-fp16 true \
#   --master-port 12535 \
#   --causal false \
#   --full-libri false \
#   --use-transducer false \
#   --use-ctc true \
#   --ctc-loss-scale 1.0 \
#   --exp-dir $exp_dir \
#   --max-duration 200 --num-workers 3

# diff zipformer_no_seg/exp-seed/{best-valid-loss.pt,epoch-14.pt}

# python zipformer_no_seg/export.py \
#   --exp-dir $exp_dir \
#   --use-transducer false \
#   --use-ctc true \
#   --epoch 20 \
#   --avg 7

# ln -s $(realpath zipformer_no_seg/exp-seed/pretrained.pt) zipformer_no_seg/exp-no-seg1/epoch-1.pt
# ln -s $(realpath zipformer_no_seg/exp-seed/pretrained.pt) zipformer_no_seg/exp-test/epoch-1.pt

# Decode
# for m in ctc-decoding 1best; do
#   ./zipformer_no_seg/ctc_decode.py \
#       --epoch 20 \
#       --avg 7 \
#       --exp-dir $exp_dir \
#       --use-transducer 0 \
#       --use-ctc 1 \
#       --max-duration 500 \
#       --causal 0 \
#       --num-paths 100 \
#       --nbest-scale 1.0 \
#       --hlg-scale 0.6 \
#       --decoding-method $m
# done

# --epoch 20 --avg 7
# --epoch 1 --avg 1 --use-averaged-model 1
# --epoch 1 --avg 1 --use-averaged-model 0

###### (epoch-14.pt - epoch-13.pt)
# epoch14-avg1  test-clean test-other
# ctc-decoding  15.44      32.56
# 1best         10.8       25.22

###### (epoch-14.pt)
# epoch14-avg1-use-averaged-model0  test-clean test-other
# ctc-decoding  22.37      41.54
# 1best         14.94      32.6

###### (epoch-20.pt - epoch-13.pt)
# epoch20-avg7  test-clean test-other
# ctc-decoding  13.29      29.01
# 1best         9.95       23.26


# #### zero_grad experiments: ####
#
# exp_dir=zipformer_zero_grad/exp-test

# echo
# echo "exp_dir:" $exp_dir
# echo

# python zipformer_zero_grad/train.py \
#   --world-size 4 \
#   --num-epochs 40 \
#   --start-epoch 1 \
#   --use-fp16 true \
#   --master-port 12535 \
#   --causal false \
#   --full-libri true \
#   --use-transducer false \
#   --use-ctc true \
#   --ctc-loss-scale 1.0 \
#   --exp-dir $exp_dir \
#   --max-duration 1200 --num-workers 3 



exp_dir=zipformer_test/exp-test

echo
echo "exp_dir:" $exp_dir
echo

python zipformer_test/train.py \
  --world-size 4 \
  --num-epochs 40 \
  --start-epoch 1 \
  --use-fp16 true \
  --master-port 12535 \
  --causal false \
  --full-libri false \
  --use-transducer false \
  --use-ctc true \
  --ctc-loss-scale 1.0 \
  --exp-dir $exp_dir \
  --max-duration 400 --num-workers 3 \
  --start-epoch 15 --ctc-beam-size 4