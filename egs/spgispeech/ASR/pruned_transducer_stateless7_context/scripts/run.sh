# conda activate aligner5
conda activate aligner5
export PATH="/home/hltcoe/rhuang/mambaforge/envs/aligner5/bin/":$PATH
module load cuda11.7/toolkit
module load cudnn/8.5.0.96_cuda11.x
module load nccl/2.13.4-1_cuda11.7
module load gcc/7.2.0
module load intel/mkl/64/2019/5.281

# k2
K2_ROOT=/exp/rhuang/meta/k2/
export PYTHONPATH=$K2_ROOT/k2/python:$PYTHONPATH # for `import k2`
export PYTHONPATH=$K2_ROOT/temp.linux-x86_64-cpython-310/lib:$PYTHONPATH # for `import _k2`
export PYTHONPATH=/exp/rhuang/meta/icefall:$PYTHONPATH

# generate_contexts
cd /exp/rhuang/meta/icefall/egs/spgispeech/ASR/pruned_transducer_stateless7_context
python ./scripts/generate_contexts.py \
  --context-dir /exp/rhuang/icefall_latest/egs/spgispeech/ASR/data/rare_words/ \
  --lang-dir ../data/lang_bpe_500 \
  --output-dir ../data/uniphore_contexts/ \
  --manifest-dir /exp/draj/jsalt2023/icefall/egs/librispeech/ASR/data/manifests/ \
  --n-distractors 500

# compute_rare_wer for whisper results
cd /exp/rhuang/meta/icefall/egs/spgispeech/ASR/pruned_transducer_stateless7_context
size=large
per_utt_dir=/exp/rhuang/meta/audio_ruizhe/whisper/exp/uniphore/whisper_$size/
for part in "healthcare" "banking" "insurance"; do
  python ./scripts/compute_rare_wer.py \
    --context-dir /exp/rhuang/icefall_latest/egs/spgispeech/ASR/data/rare_words/ \
    --lang-dir ../data/lang_bpe_500 \
    --per-utt $per_utt_dir/recogs-$part-whisper_$size-ruizhe.txt
done
