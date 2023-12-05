#!/usr/bin/env bash
#$ -wd /exp/rhuang/meta/audio_ruizhe/whisper/
#$ -V
#$ -N decode_whisper
#$ -j y -o log/log-$JOB_NAME-$JOB_ID.out
#$ -M ruizhe@jhu.edu
#$ -m e
#$ -l mem_free=20G,h_rt=600:00:00,gpu=1,hostname=!r7n07*
#$ -q gpu.q@@rtx

# #$ -q gpu.q@@v100
# #$ -q gpu.q@@rtx

# #$ -l ram_free=300G,mem_free=300G,gpu=0,hostname=b*

# hostname=b19
# hostname=!c04*&!b*&!octopod*
# hostname
# nvidia-smi

# conda activate whisper
export PATH="/home/hltcoe/rhuang/miniconda3/envs/whisper/bin/":$PATH
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
export PYTHONPATH=$K2_ROOT/build/temp.linux-x86_64-cpython-310/lib:$PYTHONPATH # for `import _k2`
export PYTHONPATH=/exp/rhuang/meta/icefall:$PYTHONPATH

# To verify SGE_HGR_gpu and CUDA_VISIBLE_DEVICES match for GPU jobs.
env | grep SGE_HGR_gpu
env | grep CUDA_VISIBLE_DEVICES
echo "hostname: `hostname`"
echo "current path:" `pwd`


# iteration=11
# exp_dir=/exp/rhuang/meta/audio_ruizhe/ec21/data/earnings21_json/alignment_sp/segments/iteration_${iteration}
# # cuts=/exp/rhuang/meta/audio_ruizhe/ec21/data/earnings21_json/alignment_sp/segments/cuts.jsonl.gz
# # cuts=/exp/rhuang/meta/audio_ruizhe/ec21/data/earnings21_json/alignment_sp/segments/cuts_wav.jsonl.gz
# cuts=/exp/rhuang/meta/audio_ruizhe/ec21/data/earnings21_json/alignment_sp/segments/cuts_wav.iter$( expr $iteration - 1 ).jsonl.gz
# mkdir -p $exp_dir

# exp_dir=/exp/rhuang/meta/audio_ruizhe/ec21/data/earnings21_json/exp_decode_whisper/
# cuts=/exp/rhuang/meta/audio_ruizhe/ec21/data/earnings21_json/alignment_sp/segments/cuts_wav.iter10-5.jsonl.gz
# exp_dir=/exp/rhuang/meta/audio_ruizhe/ec21/data/earnings21_json/exp_decode_whisper_cuts_wav.iter10-6/
# cuts=/exp/rhuang/meta/audio_ruizhe/ec21/data/earnings21_json/alignment_sp/segments/cuts_wav.iter10-6.feats.jsonl.gz

# exp_dir=/exp/rhuang/meta/audio_ruizhe/ec21/data/earnings21_updated/alignment/segments/decode_cuts_wav-5.wer.iter4/
# cuts=/exp/rhuang/meta/audio_ruizhe/ec21/data/earnings21_updated/alignment/segments/cuts_wav-5.wer.iter4.wer.jsonl.gz

exp_dir=/exp/rhuang/meta/audio_ruizhe/ec21/data/earnings21_updated4/timed/raw/segments/exp_decode_whisper_ec21_cuts.iter6/
cuts=/exp/rhuang/meta/audio_ruizhe/ec21/data/earnings21_updated4/timed/raw/segments/ec21_cuts.iter6.wer.updated.feats.jsonl.gz


####################################
# decode whisper
####################################

size="tiny"
# size="base"
# size="small"
# size="medium"
# size="large"

echo
echo "cuts:    $cuts"
echo "exp_dir: $exp_dir"
echo "whisper model size: $size"
echo

# python /exp/rhuang/meta/audio_ruizhe/ec21/decode_whisper.py \
python /exp/rhuang/meta/audio_ruizhe/whisper/decode_whisper_ec21.py \
  --d "cuda" \
  --l "en" \
  --n $size \
  --c $cuts \
  --out ${exp_dir}/${size} # --whole-recording

# --whole-recording