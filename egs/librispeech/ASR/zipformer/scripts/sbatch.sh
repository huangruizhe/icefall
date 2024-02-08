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

exp_dir=zipformer/exp-ctc-rnnt

echo
echo "exp_dir:" $exp_dir
echo

python zipformer/train.py \
  --world-size 4 \
  --num-epochs 40 \
  --start-epoch 1 \
  --use-fp16 true \
  --master-port 12535 \
  --causal false \
  --full-libri true \
  --use-transducer true \
  --use-ctc true \
  --ctc-loss-scale 0.2 \
  --exp-dir $exp_dir \
  --max-duration 1200 # \
  # --start-epoch 35

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

# It seems increasing max-duration from 1000 to 1200 has slight improvement? check decoding logs in $exp_dir

# https://stackoverflow.com/questions/37987839/how-can-i-run-tensorboard-on-a-remote-server
# source activate aligner5
# tensorboard --logdir $exp_dir/tensorboard --port 6006
# ssh -L 16006:127.0.0.1:6006 rhuang25@login.rockfish.jhu.edu
# http://localhost:16006 
