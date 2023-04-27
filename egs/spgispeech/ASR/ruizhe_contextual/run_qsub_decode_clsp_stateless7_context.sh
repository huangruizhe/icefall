#!/usr/bin/env bash
#$ -wd /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR
#$ -V
#$ -N decode
#$ -j y -o ruizhe_contextual/log/$JOB_NAME-$JOB_ID.out
#$ -M ruizhe@jhu.edu
#$ -m e
#$ -l ram_free=16G,mem_free=16G,gpu=1,hostname=!b*&!c18*&!c04*
#$ -q g.q

# &!octopod*

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

####################################
# modified_beam_search
####################################

n_distractors=100
exp_dir=pruned_transducer_stateless7_context/exp/exp_libri_full_c-1_stage2_6k
exp_dir=pruned_transducer_stateless7_context/exp/exp_libri_full_c-1_stage2
exp_dir=pruned_transducer_stateless7_context/exp/exp_libri_full_c-1_stage1
context_suffix="_slides"
# context_suffix="_0.0"
# context_suffix="_100recall"

epochs=16
avgs=1
use_averaged_model=$([ "$avgs" = 1 ] && echo "false" || echo "true")

stage=2
stop_stage=$stage
echo "Stage: $stage"

# download model from coe:
# cd /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR
# mkdir -p $exp_dir
#
# # scp -r rhuang@test1.hltcoe.jhu.edu:/exp/rhuang/icefall_latest/egs/spgispeech/ASR/$exp_dir/epoch-4.pt \
# #   /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/${exp_dir}/epoch-99.pt
#
# # https://www.xmodulo.com/skip-existing-files-scp.html
# rsync -av --ignore-existing --progress \
#   rhuang@test1.hltcoe.jhu.edu:/exp/rhuang/icefall_latest/egs/spgispeech/ASR/$exp_dir/* \
#   /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/$exp_dir


# No biasing at all
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  # greedy_search fast_beam_search
  # python -m pdb -c continue
  ./pruned_transducer_stateless7/decode.py \
      --epoch 1 \
      --use-averaged-model false \
      --exp-dir $exp_dir \
      --bpe-model "data/lang_bpe_500/bpe.model" \
      --max-duration 200 \
      --decoding-method "modified_beam_search" \
      --beam-size 4
fi

# Results:
# /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/log/decode-3629147.out
#   10.39_47315     9.53_39474      62.09_2124      57.78_245       33.62_1229      15.08_308

# No biasing at all + RNNLM + LODR
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  rnnlm_dir="/export/fs04/a12/rhuang/contextualizedASR/lm/LM/my-rnnlm-exp-1024-3-tied"   # --rnn-lm-num-layers 3 --lm-epoch 7 --lm-avg 5 --lm-scale 0.15 --ngram-lm-scale -0.1
  # rnnlm_dir="/export/fs04/a12/rhuang/contextualizedASR/lm/LM/my-rnnlm-exp-1024-2-tied"  # --rnn-lm-num-layers 2 --lm-epoch 14 --lm-avg 5 --lm-scale 0.1 --ngram-lm-scale -0.05
  lm_type=rnn
  lang_dir="/export/fs04/a12/rhuang/contextualizedASR/lm/LM/my-ngram-exp/bpe"
  rnn_lm_scale=0.2
  ngram_lm_scale=-0.1
  tokens_ngram_order=2
  python pruned_transducer_stateless7/decode.py \
      --epoch 1 \
      --avg 1 \
      --use-averaged-model false \
      --exp-dir $exp_dir \
      --bpe-model "data/lang_bpe_500/bpe.model" \
      --max-duration 200 \
      --lang-dir $lang_dir \
      --decoding-method modified_beam_search_LODR \
      --beam 4 \
      --max-contexts 4 \
      --use-shallow-fusion true \
      --lm-type $lm_type \
      --lm-exp-dir $rnnlm_dir \
      --lm-epoch 7 \
      --lm-avg 5 \
      --lm-scale $rnn_lm_scale \
      --rnn-lm-embedding-dim 1024 \
      --rnn-lm-hidden-dim 1024 \
      --rnn-lm-num-layers 3 \
      --rnn-lm-tie-weights true \
      --tokens-ngram $tokens_ngram_order \
      --ngram-lm-scale $ngram_lm_scale 
fi

# Results:
# 1024-3 (0.15-0.1): /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/log/decode-3629150.out
#           10.20_46421     9.34_38691      60.86_2082      57.31_243       32.36_1183      15.03_307
# 1024-3 (0.2-0.1):  /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/log/decode-3629176.out /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/log/decode-3629312.out
#           10.17_46319     9.32_38595      60.68_2076      57.31_243       32.03_1171      15.12_309
# 1024-2 (0.1-0.05): /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/log/decode-3629152.out
#           10.24_46635     9.38_38848      61.50_2104      58.25_247       32.85_1201      15.08_308


# Use biasing
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  for m in modified_beam_search ; do
    for epoch in $epochs; do
      for avg in $avgs; do
        # python -m pdb -c continue
        # python pruned_transducer_stateless7_context/decode.py \
        python pruned_transducer_stateless7_context/decode_ec53.py \
            --epoch $epoch \
            --avg $avg \
            --use-averaged-model $use_averaged_model \
            --exp-dir $exp_dir \
            --bpe-model "data/lang_bpe_500/bpe.model" \
            --max-duration 200 \
            --decoding-method $m \
            --beam-size 4 \
            --context-dir "data/rare_words" \
            --n-distractors $n_distractors \
            --keep-ratio 1.0 --is-bi-context-encoder true --slides "/export/fs04/a12/rhuang/contextualizedASR/data/ec53_kaldi_heuristics2/context${context_suffix}" --is-predefined true
        # --context-dir "data/rare_words"
        # --slides "/export/fs04/a12/rhuang/contextualizedASR/data/ec53_kaldi_heuristics2/context${context_suffix}" --is-predefined true
        # --is-full-context true
        # --n-distractors 0
        # --no-encoder-biasing true --no-decoder-biasing true
        # --is-predefined true
        # --is-pretrained-context-encoder true
        # --no-wfst-lm-biasing false --biased-lm-scale 7
        # --is-predefined true --no-wfst-lm-biasing false --biased-lm-scale 10 --no-encoder-biasing true --no-decoder-biasing true
        # --is-bi-context-encoder false
        #
        # lm-biasing (cheating+distractors): --no-encoder-biasing true --no-decoder-biasing true --no-wfst-lm-biasing false --biased-lm-scale 11 --n-distractors 100
        # lm-biasing (slides): --no-encoder-biasing true --no-decoder-biasing true --no-wfst-lm-biasing false --biased-lm-scale 10 --slides "/export/fs04/a12/rhuang/contextualizedASR/data/ec53_kaldi_heuristics2/context${context_suffix}" --is-predefined true
        #
        # neural-biasing (slides): --max-duration 10 --slides "/export/fs04/a12/rhuang/contextualizedASR/data/ec53_kaldi_heuristics2/context${context_suffix}" --is-predefined true
      done
    done
  done
fi

# Results (LM biasing):
# Baseline (modified beam search): 
#     10.39_47315     9.53_39474      62.09_2124      57.78_245       33.62_1229      15.08_308
# --biased-lm-scale 0: /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/log/decode-3629328.out
#     10.39_47315     9.53_39472      62.09_2124      57.78_245       33.62_1229      15.08_308
# --biased-lm-scale 5: /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/log/decode-3629239.out
#     10.25_46664     9.44_39079      59.28_2028      54.25_230       30.14_1102      14.19_290
# --biased-lm-scale 6: /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/log/decode-3629240.out
#     10.24_46630     9.44_39083      58.96_2017      53.54_227       29.70_1086      14.19_290
# --biased-lm-scale 7: /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/log/decode-3629241.out
#     10.23_46594     9.44_39076      58.29_1994      53.07_225       29.38_1074      14.10_288
# --biased-lm-scale 8: /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/log/decode-3629242.out
#     10.25_46657     9.46_39159      58.08_1987      53.54_227       29.21_1068      14.10_288
# --biased-lm-scale 9: /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/log/decode-3629243.out
#     10.26_46695     9.47_39215      57.82_1978      52.36_222       29.02_1061      14.10_288
# --biased-lm-scale 10: /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/log/decode-3629244.out <-- best
#     10.28_46820     9.50_39342      57.50_1967      52.12_221       28.97_1059      14.05_287
# --biased-lm-scale 11: /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/log/decode-3629303.out
#     10.32_47007     9.54_39515      57.47_1966      52.12_221       28.99_1060      14.05_287
# --biased-lm-scale 12: /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/log/decode-3629304.out
#     10.35_47141     9.57_39626      57.26_1959      52.59_223       28.91_1057      14.15_289


# Results (Neural biasing):
# epoch 16: /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/log/decode-3630354.out
#     15.90_72375     14.73_60975     73.72_2522      66.98_284       45.32_1657      26.77_547


# Neural biasing
# slides (removed common words, cuts100)
# beam_size_4     8.65    5.04    best for ec53

# lm biasing only for deocde_ec53.py: 3616861 10.53/6.73
# neural biasing (epoch-4.pt): 3616863   10.65/6.7
# neural biasing + lm biasing (epoch-4.pt): 3616864  10.69/6.82
# neural biasing (epoch-4.pt) + rnnlm + lodr: 3616865 10.52/6.77
# neural biasing + lm biasing (epoch-4.pt) + rnnlm + lodr: 3616866  10.57/6.95

# decode_ec53 with predefined biasing list oracles:
# lm biasing7 only for deocde_ec53.py: 3616868                        10.82/6.88 10.67_48604     9.75_40373      64.78_2216      59.20_251       33.83_1237      14.83_303
# lm biasing7 only 100recall  3616924                                 x
# lm biasing10 only 100recall 3616925                                 10.96/7.04 10.80_49177     9.95_41177      59.66_2041      55.90_237       32.11_1174      14.54_297
# lm biasing7 only 0.0        3616922                                 10.71/6.85 10.56_48099     9.71_40215      59.34_2030      54.01_229       31.43_1149      13.61_278
# lm biasing10 only 0.0       3616902                                 10.7/6.87  10.55_48035     9.73_40292      57.29_1960      52.59_223       29.76_1088      13.41_274
# neural biasing (epoch-4.pt): 3616871/3616904                        11.25/7.03 11.05_50332     10.10_41804     66.76_2284      60.61_257       35.50_1298      15.86_324
# neural biasing (epoch-4.pt) + only known words: 3616919             11.25/7.04 11.06_50349     10.10_41820     66.21_2265      61.56_261       35.61_1302      15.76_322
# neural biasing (epoch-4.pt) + rd_100_distractors: 3616907           11.07/6.95 10.92_49741     9.96_41223      67.41_2306      62.26_264       37.14_1358      16.05_328
# neural biasing (epoch-4.pt) + rd_500_distractors: 3616908           11.06/6.98 10.93_49757     9.95_41177      68.05_2328      61.56_261       37.83_1383      15.76_322
# neural biasing (epoch-4.pt) + rd_-1_distractors:  3616909           11.07/6.97 10.93_49761     9.96_41229      67.52_2310      62.26_264       37.55_1373      15.22_311
# neural biasing (epoch-4.pt) + rd_100_distractors + keep0.5: 3616910 11.13/6.98 10.99_50031     10.01_41445     67.87_2322      61.79_262       37.69_1378      15.86_324
# neural biasing (epoch-4.pt) + rd_500_distractors + keep0.5: 3616911 11.13/6.98 10.99_50022     10.00_41384     68.17_2332      62.50_265       38.29_1400      16.54_338
# neural biasing (epoch-4.pt) + rd_-1_distractors  + keep0.5: 3616912 11.12/6.99 10.97_49949     9.99_41348      68.52_2344      62.97_267       37.77_1381      15.96_326
# neural biasing + 0.0: 3616927                                       10.9/6.87  10.72_48795     9.93_41131      54.31_1858      51.18_217       26.75_978       13.36_273
# neural biasing + 10.0: 3617061                                      11.09/6.96 10.92_49737     10.04_41590     61.59_2107      57.31_243       32.71_1196      14.44_295
# neural biasing + 100recall: 3616926                                 11.14/6.99 10.92_49743     10.05_41624     59.54_2037      58.96_250       31.84_1164      15.03_307
# neural biasing + lm biasing7 (epoch-4.pt): 3616869/3616915          11.42/7.18 11.08_50430     10.14_41966     66.24_2266      60.14_255       34.16_1249      15.71_321
# neural biasing + lm biasing4 (epoch-4.pt): 3616870/3616916          11.28/7.05 11.07_50419     10.13_41951     66.56_2277      60.85_258       34.00_1243      15.71_321
# neural biasing + lm biasing7 (epoch-4.pt) + 0.0: 3616928/3616877    10.83/6.87 10.64_48425     9.95_41191      46.51_1591      43.16_183       23.55_861       11.60_237
# neural biasing + lm biasing4 (epoch-4.pt) + 0.0: 3616929            10.83/6.85 10.64_48467     9.92_41093      49.25_1685      45.99_195       24.70_903       12.29_251
# neural biasing + lm biasing7 (epoch-4.pt) + 10.0: 3617062           11.12/7.0  10.94_49834     10.12_41906     57.82_1978      54.01_229       30.55_1117      13.56_277
# neural biasing + lm biasing4 (epoch-4.pt) + 10.0: 3617063           11.1/6.96  10.93_49756     10.08_41725     59.57_2038      55.90_237       31.21_1141      14.10_288
# neural biasing + lm biasing7 (epoch-4.pt) + 100recall: 3616930      11.26/7.08 11.03_50209     10.21_42287     55.42_1896      54.95_233       29.16_1066      14.78_302
# neural biasing + lm biasing4 (epoch-4.pt) + 100recall: 3616931      11.14/6.99 10.92_49710     10.08_41722     57.50_1967      56.84_241       30.33_1109      14.15_289
# neural biasing (epoch-4.pt) + rnnlm + lodr: 3616872/3616917         10.9/7.04  10.76_48997     9.80_40579      65.27_2233      61.79_262       33.94_1241      15.91_325
# neural biasing + lm biasing7 (epoch-4.pt)+rnn+lodr: 3616873/3616920 11.09/7.19 10.94_49792     9.98_41335      64.51_2207      59.20_251       33.01_1207      16.30_333
# neural biasing + lm biasing4 (epoch-4.pt)+rnn+lodr: 3616874/3616921 x

# 8.89_227        7.49_172        60.87_14        66.67_2 48.00_12        11.11_2
# 8.62_220        7.28_167        47.83_11        66.67_2 48.00_12        16.67_3

####################################
# LODR
####################################

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  rnnlm_dir="/export/fs04/a12/rhuang/contextualizedASR/lm/LM/my-rnnlm-exp-1024-3-tied"
  lm_type="rnn"

  lang_dir="/export/fs04/a12/rhuang/contextualizedASR/lm/LM/my-ngram-exp/bpe"
  rnn_lm_scale=0.2
  ngram_lm_scale=-0.1
  tokens_ngram_order=2
  for m in modified_beam_search_LODR ; do
    for epoch in $epochs; do
      for avg in $avgs; do
        # python pruned_transducer_stateless2_context/decode.py \
        python pruned_transducer_stateless2_context/decode_ec53.py \
            --epoch $epoch \
            --avg $avg \
            --exp-dir $exp_dir \
            --bpe-model "/export/fs04/a12/rhuang/icefall_align/egs/spgispeech/ASR/tmp/icefall-asr-spgispeech-pruned-transducer-stateless2/data/lang_bpe_500/bpe.model" \
            --max-duration 300 \
            --lang-dir $lang_dir \
            --decoding-method $m \
            --beam-size 4 \
            --max-contexts 4 \
            --use-shallow-fusion true \
            --lm-type $lm_type \
            --lm-exp-dir $rnnlm_dir \
            --lm-epoch 7 \
            --lm-avg 5 \
            --lm-scale $rnn_lm_scale \
            --rnn-lm-embedding-dim 1024 \
            --rnn-lm-hidden-dim 1024 \
            --rnn-lm-num-layers 3 \
            --rnn-lm-tie-weights true \
            --tokens-ngram $tokens_ngram_order \
            --ngram-lm-scale $ngram_lm_scale \
            --context-dir "data/rare_words" \
            --n-distractors $n_distractors \
            --keep-ratio 1.0 --no-wfst-lm-biasing false --biased-lm-scale 4 --slides "/export/fs04/a12/rhuang/contextualizedASR/data/ec53_kaldi_heuristics2/context${context_suffix}" --is-predefined true
        # --is-predefined true
        # --no-encoder-biasing true --no-decoder-biasing true
        # --no-wfst-lm-biasing false --biased-lm-scale 10
        #
        # all: --is-predefined true --n-distractors 500 --no-wfst-lm-biasing false --biased-lm-scale 7
        # lm-biasing (slides) + lodr: --no-encoder-biasing true --no-decoder-biasing true --no-wfst-lm-biasing false --biased-lm-scale 7 --slides "/export/fs04/a12/rhuang/contextualizedASR/data/ec53_kaldi_heuristics2/context${context_suffix}" --is-predefined true
      done
    done
  done
fi

# biased_lm_scale 4  3616834 10.5/6.79  10.39_47294     9.46_39173      63.81_2183      58.25_247       32.88_1202      14.29_292
# biased_lm_scale 5  3616849 10.51/6.8  10.39_47309     9.47_39205      63.64_2177      56.84_241       32.60_1192      14.24_291
# biased_lm_scale 6  3616835 10.52/6.82 10.41_47400     9.50_39320      63.46_2171      56.60_240       32.19_1177      14.19_290  # /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/log/decode-3616835.out
# biased_lm_scale 7  3616839 10.57/6.86 10.46_47610     9.54_39511      63.49_2172      56.60_240       32.17_1176      14.34_293
# biased_lm_scale 8  3616840 10.63/6.92 10.52_47891     9.61_39795      63.26_2164      56.60_240       31.87_1165      14.49_296
# biased_lm_scale 9  3616836 10.7/6.98  10.59_48205     9.68_40085      62.85_2150      56.13_238       31.89_1166      14.73_301
# biased_lm_scale 10 3616837 10.78/7.06 10.67_48563     9.76_40420      62.61_2142      56.37_239       31.87_1165      14.73_301


####################################
# Export averaged models 
# https://icefall.readthedocs.io/en/latest/model-export/export-model-state-dict.html#how-to-export
####################################

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  python pruned_transducer_stateless7_context/export.py \
    --exp-dir $exp_dir \
    --bpe-model data/lang_bpe_500/bpe.model \
    --epoch $epochs \
    --avg $avgs

  mv $exp_dir/pretrained.pt $exp_dir/stage1.pt
fi
