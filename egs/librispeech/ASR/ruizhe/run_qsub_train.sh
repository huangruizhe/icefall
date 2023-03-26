#!/usr/bin/env bash
#$ -wd /export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR
#$ -V
#$ -N train
#$ -j y -o ruizhe/log/$JOB_NAME-$JOB_ID.out
#$ -M ruizhe@jhu.edu
#$ -m e
#$ -l ram_free=16G,mem_free=16G,gpu=4,hostname=!b*
#$ -q 4gpu.q

# &!octopod*

#### Activate dev environments and call programs
mamba activate /home/rhuang/mambaforge/envs/efrat
export PYTHONPATH=/export/fs04/a12/rhuang/k2/k2/python:$PYTHONPATH # for `import k2`
export PYTHONPATH=/export/fs04/a12/rhuang/k2/build/temp.linux-x86_64-cpython-38/lib/:$PYTHONPATH # for `import _k2`
# export PYTHONPATH=/export/fs04/a12/rhuang/icefall_align/:$PYTHONPATH
# export PYTHONPATH=/export/fs04/a12/rhuang/icefall/:$PYTHONPATH
export PYTHONPATH=/export/fs04/a12/rhuang/icefall_align2/:$PYTHONPATH

echo "python: `which python`"

#### Assign a free-GPU to your program (make sure -n matches the requested number of GPUs above)
# source /home/gqin2/scripts/acquire-gpu 4

ngpus=4 # num GPUs for multiple GPUs training within a single node; should match those in $free_gpu
free_gpu= # comma-separated available GPU ids, eg., "0" or "0,1"; automatically assigned if on CLSP grid
[ -z "$free_gpu" ] && [[ $(hostname -f) == *.clsp.jhu.edu ]] && free_gpu=$(free-gpu -n $ngpus) || \
echo "Unable to get $ngpus GPUs"
[ -z "$free_gpu" ] && echo "$0: please specify --free-gpu" && exit 1;
[ $(echo $free_gpu | sed 's/,/ /g' | awk '{print NF}') -ne "$ngpus" ] && \
 echo "number of GPU ids in --free-gpu=$free_gpu does not match --ngpus=$ngpus" && exit 1;
export CUDA_VISIBLE_DEVICES="$free_gpu"
echo $CUDA_VISIBLE_DEVICES

#### Test running qsub
python3 -c "import torch; print(torch.__version__)"

#### Your script
# export CUDA_VISIBLE_DEVICES="0,1,2,3"
# python3 tdnn_lstm_ctc/train.py --world-size 4
# python3 tdnn_lstm_ctc/train.py --world-size 4 --master-port 12355
# python3 tdnn_lstm_ctc2/train.py --world-size 4

# python3 tdnn_lstm_ctc/train_bpe.py --world-size 4 

# full librispeech
# python tdnn_lstm_ctc/train.py \
#     --world-size 4 \
#     --full-libri true \
#     --max-duration 200 \
#     --num-epochs 30 \
#     --exp-dir "tdnn_lstm_ctc/exp/exp_libri_full"

# libri100
python tdnn_lstm_ctc/train.py \
    --world-size 4 \
    --full-libri false \
    --max-duration 200 \
    --num-epochs 30 \
    --valid-interval 100 \
    --exp-dir "tdnn_lstm_ctc/exp/exp_libri_100"
