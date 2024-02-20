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

##################################################
# Standard training recipe
##################################################
if true; then
    exp_dir=/exp/rhuang/meta/icefall/egs/librispeech/ASR/conformer_ctc2_noseg/exp/exp_ctc
    ./conformer_ctc2_noseg/train.py \
      --exp-dir $exp_dir \
      --lang-dir data/lang_bpe_500 \
      --full-libri 1 \
      --max-duration 600 \
      --concatenate-cuts 0 \
      --world-size 4 \
      --bucketing-sampler 1 \
      --start-epoch 1 \
      --num-epochs 30 \
      --att-rate 0
    
    for method in ctc-greedy-search ctc-decoding 1best nbest-oracle; do
      python3 ./conformer_ctc2_noseg/decode.py \
      --exp-dir $exp_dir \
      --use-averaged-model True --epoch 30 --avg 8 --max-duration 400 --method $method
    done

    ###### --epoch 30 --avg 8 ######
    # ctc-greedy-search 3.29/8.15
    # ctc-decoding      3.29/8.15
    # 1best             3.1/7.13
    # nbest-oracle      1.48/3.66
fi

##################################################
# Get a seed model first on a small subset
##################################################
# python -c """
# import lhotse
# cs = lhotse.load_manifest('data/fbank/librispeech_cuts_train-clean-100.jsonl.gz')
# cs = cs.to_eager()
# cs = cs.sample(n_cuts=10000)
# cs.describe()
# cs.to_file('data/fbank/librispeech_cuts_train-clean-100-35h.jsonl.gz')
# """
if false; then
    exp_dir=/exp/rhuang/meta/icefall/egs/librispeech/ASR/conformer_ctc2_noseg/exp/exp_seed
    ./conformer_ctc2_noseg/train_seed.py \
      --exp-dir $exp_dir \
      --lang-dir data/lang_bpe_500 \
      --full-libri 0 \
      --max-duration 200 \
      --concatenate-cuts 0 \
      --world-size 4 \
      --bucketing-sampler 1 \
      --start-epoch 1 \
      --num-epochs 30 \
      --att-rate 0
    
    for method in ctc-greedy-search ctc-decoding 1best nbest-oracle; do
      python3 ./conformer_ctc2_noseg/decode.py \
      --exp-dir $exp_dir \
      --use-averaged-model True --epoch 30 --avg 8 --max-duration 400 --method $method
    done

    ###### --epoch 30 --avg 8 ######
    # ctc-greedy-search 14.79/30.62
    # ctc-decoding      14.79/30.62
    # 1best             10.8/23.63
    # nbest-oracle      6.98/16.87

    ###### --use-averaged-model true --epoch 14 --avg 1 ######
    # ctc-greedy-search 15.24/31.92
    # ctc-decoding      15.24/31.92
    # 1best             10.5/23.74
    # nbest-oracle      6.42/16.5

    ###### --use-averaged-model false --epoch 14 --avg 1 ######
    # ctc-greedy-search 18.95/37.63
    # ctc-decoding      18.95/37.63
    # 1best             12.55/28.0
    # nbest-oracle      7.84/20.1

    ###### --use-averaged-model false --epoch 888 --avg 1 (best-valid-loss.ppt) ######
    # ctc-greedy-search 23.82/43.37
    # ctc-decoding      23.82/43.37
    # 1best             15.16/32.63
    # nbest-oracle      9.12/23.09
fi

##################################################
# Get a SMALL seed model first on a small subset
##################################################

# Number of model parameters: 9929502

if false; then
    exp_dir=/exp/rhuang/meta/icefall/egs/librispeech/ASR/conformer_ctc2_noseg/exp/exp_seed_small_model
    ./conformer_ctc2_noseg/train_seed_small.py \
      --exp-dir $exp_dir \
      --lang-dir data/lang_bpe_500 \
      --full-libri 0 \
      --num-decoder-layers 0 \
      --max-duration 200 \
      --concatenate-cuts 0 \
      --world-size 4 \
      --bucketing-sampler 1 \
      --start-epoch 1 \
      --num-epochs 30 \
      --att-rate 0
    
    for method in ctc-greedy-search ctc-decoding 1best nbest-oracle; do
      python3 ./conformer_ctc2_noseg/decode_small.py \
        --exp-dir $exp_dir \
        --use-averaged-model True --epoch 30 --avg 8 --max-duration 800 --method $method \
        --num-decoder-layers 0
    done
    
    # --epoch 10 --avg 3 
    # ctc-greedy-search 16.96/35.67
    # ctc-decoding      16.96/35.67
    # 1best             10.66/25.62
    # nbest-oracle      5.97/17.44

    # --epoch 30 --avg 8
    # ctc-greedy-search 14.18/31.9
    # ctc-decoding      14.18/31.9
    # 1best             9.64/23.49
    # nbest-oracle      5.91/16.39

    # decode with factor transducer and long text ground truth
    python3 ./conformer_ctc2_noseg/decode_small.py \
      --exp-dir $exp_dir \
      --use-averaged-model True --epoch 30 --avg 8 --max-duration 1000 --method 'ctc-decoding' \
      --num-decoder-layers 0 --oracle True
    
    # train/eval mode, --epoch 10 --avg 3 
    # train-100  3.62% [104587 / 2891987, 2093 ins, 101879 del, 615 sub ] / 3.40% [98435 / 2891987, 2027 ins, 95545 del, 863 sub ]
    # train-full 
    # test-clean 2.89% [1519 / 52576, 32 ins, 1465 del, 22 sub ] / 2.72% [1432 / 52576, 15 ins, 1404 del, 13 sub ]
    # test-other 2.99% [1566 / 52343, 118 ins, 1392 del, 56 sub ] / 2.77% [1450 / 52343, 168 ins, 1188 del, 94 sub ]

    # train/eval mode, --epoch 30 --avg 8
    # train-100  3.52% [103319 / 2935692, 2608 ins, 98989 del, 1722 sub ] / 3.37% [98836 / 2935692, 3266 ins, 93108 del, 2462 sub ]
    # train-full 4.63% [1305687 / 28178989, 46629 ins, 1211074 del, 47984 sub ] / 4.44% [1252477 / 28178989, 59116 ins, 1120030 del, 73331 sub ]
    # test-clean 3.01% [1581 / 52576, 52 ins, 1451 del, 78 sub ] / 3.22% [1695 / 52576, 122 ins, 1409 del, 164 sub ]
    # test-other 3.78% [1979 / 52343, 308 ins, 1398 del, 273 sub ] / 3.26% [1705 / 52343, 306 ins, 1207 del, 192 sub ]
fi


##################################################
# Get weak alignment
##################################################

if false; then
    exp_dir=/exp/rhuang/meta/icefall/egs/librispeech/ASR/conformer_ctc2_noseg/exp/exp_seed_small_model

    python3 ./conformer_ctc2_noseg/decode_small_alignment.py \
        --num-decoder-layers 0 \
        --exp-dir $exp_dir \
        --use-averaged-model True --epoch 30 --avg 8 --max-duration 1000 \
        --world-size 1
fi


##################################################
# Train with train_concat_libri
##################################################

if false; then
    exp_dir=/exp/rhuang/meta/icefall/egs/librispeech/ASR/conformer_ctc2_noseg/exp/exp_concat_libri
    ln -s /exp/rhuang/meta/icefall/egs/librispeech/ASR/conformer_ctc2_noseg/exp/exp_seed/epoch-14.pt $exp_dir/.
    ./conformer_ctc2_noseg/train_concat_libri.py \
      --exp-dir $exp_dir \
      --lang-dir data/lang_bpe_500 \
      --full-libri 0 \
      --max-duration 200 \
      --concatenate-cuts 0 \
      --world-size 1 \
      --bucketing-sampler 1 \
      --start-epoch 15 \
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
# tensorboard --logdir $exp_dir/tensorboard --port 6006 --window_title $exp_dir
# http://localhost:16006 

