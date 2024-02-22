#!/usr/bin/env bash
#$ -wd /export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR
#$ -V
#$ -N decode
#$ -j y -o ruizhe_contextual/log/$JOB_NAME-$JOB_ID.out
#$ -M ruizhe@jhu.edu
#$ -m e
#$ -l ram_free=16G,mem_free=16G,gpu=1,hostname=!b*
#$ -q g.q

# &!octopod*
# &!c18*&!c04*&!c07*

#### Activate dev environments and call programs
# mamba activate /home/rhuang/mambaforge/envs/efrat
mamba activate /home/rhuang/mambaforge/envs/efrat2
export PYTHONPATH=/export/fs04/a12/rhuang/k2/k2/python:$PYTHONPATH # for `import k2`
export PYTHONPATH=/export/fs04/a12/rhuang/k2/build/temp.linux-x86_64-cpython-38/lib/:$PYTHONPATH # for `import _k2`
# export PYTHONPATH=/export/fs04/a12/rhuang/icefall_align/:$PYTHONPATH
# export PYTHONPATH=/export/fs04/a12/rhuang/icefall/:$PYTHONPATH
export PYTHONPATH=/export/fs04/a12/rhuang/icefall_align2/:$PYTHONPATH

echo "python: `which python`"

#### Assign a free-GPU to your program (make sure -n matches the requested number of GPUs above)
source /home/gqin2/scripts/acquire-gpu 1

# ngpus=4 # num GPUs for multiple GPUs training within a single node; should match those in $free_gpu
# free_gpu= # comma-separated available GPU ids, eg., "0" or "0,1"; automatically assigned if on CLSP grid
# [ -z "$free_gpu" ] && [[ $(hostname -f) == *.clsp.jhu.edu ]] && free_gpu=$(free-gpu -n $ngpus) || \
# echo "Unable to get $ngpus GPUs"
# [ -z "$free_gpu" ] && echo "$0: please specify --free-gpu" && exit 1;
# [ $(echo $free_gpu | sed 's/,/ /g' | awk '{print NF}') -ne "$ngpus" ] && \
#  echo "number of GPU ids in --free-gpu=$free_gpu does not match --ngpus=$ngpus" && exit 1;
# export CUDA_VISIBLE_DEVICES="$free_gpu"
echo $CUDA_VISIBLE_DEVICES

#### Test running qsub
hostname
python3 -c "import torch; print(torch.__version__)"
nvidia-smi

# https://github.com/huangruizhe/icefall/blob/contextual/egs/librispeech/ASR/ruizhe_contextual/run_qsub_decode_clsp.sh

####################################
# modified_beam_search
####################################

# qlogin -l "hostname=c*|octo*,gpu=1,mem_free=8G,ram_free=8G" -q i.q -now no

# ln -sf /export/fs04/a12/rhuang/deep_smoothing/data_librispeech/icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11/exp/pretrained.pt \
#   $exp_dir/epoch-1.pt

exp_dir=pruned_transducer_stateless7_context/exp/exp_libri_full_c-1_stage2/temp
./pruned_transducer_stateless7/decode_training_set.py \
   --epoch 30 \
   --avg 1 \
   --use-averaged-model false \
   --exp-dir $exp_dir \
   --feedforward-dims  "1024,1024,2048,2048,1024" \
   --max-duration 600 \
   --decoding-method modified_beam_search --part 1/8


# exp_dir=/exp/rhuang/icefall_latest/egs/librispeech/ASR/pruned_transducer_stateless7_context/exp/exp_libri_full_c-1_stage2/temp


./pruned_transducer_stateless7/decode_training_set.py \
   --epoch 1 \
   --avg 1 \
   --use-averaged-model false \
   --exp-dir $exp_dir \
   --feedforward-dims  "1024,1024,2048,2048,1024" \
   --max-duration 600 \
   --decoding-method modified_beam_search --part 1/8