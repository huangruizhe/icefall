cd /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR

# Look for the cuts/segmentatio that has good quality
grep 47026 /export/fs04/a12/rhuang/icefall_align/egs/spgispeech/ASR/tmp/icefall-asr-spgispeech-pruned-transducer-stateless2/exp*/*/wer/scoring_kaldi/wer
# /export/fs04/a12/rhuang/icefall_align/egs/spgispeech/ASR/tmp/icefall-asr-spgispeech-pruned-transducer-stateless2/exp20220108_50.0/modified_beam_search_rnnlm_shallow_fusion_biased/wer/scoring_kaldi/wer:%WER 10.33 [ 47026 / 455331, 11073 ins, 8801 del, 27152 sub ]

less /export/fs04/a12/rhuang/icefall_align/egs/spgispeech/ASR/tmp/icefall-asr-spgispeech-pruned-transducer-stateless2/exp20220108_50.0/modified_beam_search_rnnlm_shallow_fusion_biased/log-decode-epoch-30-avg-15-modified_beam_search_rnnlm_shallow_fusion_biased-beam-size-10-3-ngram-lm-scale-0.01-rnnlm-lm-scale-0.1-biased-lm-scale-9.0-use-averaged-model-2023-01-09-07-23-11
# /export/fs04/a12/rhuang/contextualizedASR/data/ec53_kaldi_heuristics11_temp/cuts_no_feat_20230109050816_merged.jsonl.gz

####################################
# prepare kaldi dir: text
####################################

# BAD!!
# cuts="/export/fs04/a12/rhuang/contextualizedASR/data/ec53_kaldi_sp_gentle/20220129/cuts3.jsonl.gz" 
# cuts="/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/manifests/cuts_ec53_norm.jsonl.gz"

# GOOD
cuts="/export/fs04/a12/rhuang/contextualizedASR/data/ec53_kaldi_heuristics11_temp/cuts_no_feat_20230109050816_merged.jsonl.gz"
cuts="/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/manifests/cuts_ec53_norm.jsonl.gz"

cuts="/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/manifests/cuts_val.jsonl.gz"
cuts="/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/manifests/cuts_val.jsonl.gz"

mkdir -p data/kaldi
python -c '''
from lhotse import CutSet
cuts = CutSet.from_file("/export/fs04/a12/rhuang/contextualizedASR/data/ec53_kaldi_heuristics11_temp/cuts_no_feat_20230109050816_merged.jsonl.gz")
for cut in cuts:
  uid = cut.supervisions[0].id
  text = cut.supervisions[0].text
  print(f"{uid}\t{text}")
''' | tr -s " " | sort > data/kaldi/text
wc data/kaldi/text

cat data/kaldi/text | awk '{print $1" aaa"}' > data/kaldi/utt2spk

# Also refer to:
# /export/fs04/a12/rhuang/contextualizedASR/spgi/run.sh

####################################
# prepare ref
####################################

# Refer to:
# /export/fs04/a12/rhuang/contextualizedASR/lm/ngram.sh

# Can we reuse some ref files produced before?
grep "TSM_2020_Q1_20200416_00-56-37-680_00-57-41-850_461" /export/fs04/a12/rhuang/icefall_align/egs/spgispeech/ASR/tmp/icefall-asr-spgispeech-pruned-transducer-stateless2/exp*/*/wer/ref.txt
# /export/fs04/a12/rhuang/icefall_align/egs/spgispeech/ASR/tmp/icefall-asr-spgispeech-pruned-transducer-stateless2/exp20230124/modified_beam_search/wer/ref.txt:TSM_2020_Q1_20200416_00-56-37-680_00-57-41-850_461 and the 7 nanometer was pretty tight at the beginning of the year . charlie we cannot hear you clearly
# /export/fs04/a12/rhuang/icefall_align/egs/spgispeech/ASR/tmp/icefall-asr-spgispeech-pruned-transducer-stateless2/exp20230129/modified_beam_search/wer/ref.txt:TSM_2020_Q1_20200416_00-56-37-680_00-57-41-850_461 and the 7 nanometer was pretty tight at the beginning of the year . charlie we cannot hear you clearly
# /export/fs04/a12/rhuang/icefall_align/egs/spgispeech/ASR/tmp/icefall-asr-spgispeech-pruned-transducer-stateless2/exp20220108_50.0/modified_beam_search_rnnlm_shallow_fusion_biased/wer/ref*
cp /export/fs04/a12/rhuang/icefall_align/egs/spgispeech/ASR/tmp/icefall-asr-spgispeech-pruned-transducer-stateless2/exp20220108_50.0/modified_beam_search_rnnlm_shallow_fusion_biased/wer/ref* /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/kaldi/ref/.

# normalize ref
# https://unix.stackexchange.com/questions/14838/sed-one-liner-to-delete-everything-between-a-pair-of-brackets
# https://stackoverflow.com/questions/27825977/using-sed-to-delete-a-string-between-parentheses
mkdir -p data/kaldi/ref/
mamba activate whisper
# /export/fs04/a12/rhuang/contextualizedASR/data/ec53_kaldi_heuristics/text
cat /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/kaldi/text | \
    sed -e 's/\[[^][]*\]//g' | \
    python /export/fs04/a12/rhuang/contextualizedASR/earnings21/whisper_normalizer.py \
    --mode "kaldi_rm_sym" | sort \
> data/kaldi/ref/ref.txt

# skip entity tagging (or go to "ec53/prepare_kaldi_dir.sh" to see how to do tagging)
cp /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/kaldi/ref/ref.* $wer_dir/.

# entity tagging (go to local/ner_spacy.sh and modify the paths there)
qsub /export/fs04/a12/rhuang/contextualizedASR/local/ner_spacy.sh
# /export/fs04/a12/rhuang/contextualizedASR/log-ner-3616650.out

wer_dir="/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/kaldi/ref/"
python local/ner_spacy.py \
  --text "$wer_dir/ref.txt" \
  --out "$wer_dir/ref.ner" \
  --entities "$wer_dir/ref.entities" \
  --raw "$wer_dir/ref.ner.raw"

####################################
# prepare context/biasing list
####################################

# Refer to: /export/fs04/a12/rhuang/contextualizedASR/lm/ngram.sh

### get real contexts from slides

mamba activate whisper
export PYTHONPATH=/export/fs04/a12/rhuang/icefall/:$PYTHONPATH
export PYTHONPATH=/export/fs04/a12/rhuang/icefall_align/egs/spgispeech/ASR/:$PYTHONPATH
export PYTHONPATH=/export/fs04/a12/rhuang/contextualizedASR/:$PYTHONPATH

# python /export/fs04/a12/rhuang/contextualizedASR/local/pdf/pdf2context.py \
#   --pdf "/export/fs04/a12/rhuang/contextualizedASR/data/ec53_json/BAYZF_2018_Q4_20190227_1.pdf"

# python /export/fs04/a12/rhuang/contextualizedASR/local/pdf/pdf2context.py \
#   --pdf "/export/fs04/a12/rhuang/contextualizedASR/data/ec53_json/BAYZF_2018_Q4_20190227_original.pdf" |\
#   awk '{print $0, 1.0}' \
#   > /export/fs04/a12/rhuang/contextualizedASR/lm/LM/my-ngram-exp/entities.weighted.real.txt

output_dir="/export/fs04/a12/rhuang/contextualizedASR/data/ec53_kaldi_heuristics2/context"
mkdir -p ${output_dir}
data_dir="/export/fs04/a12/rhuang/contextualizedASR/data/ec53_json/"
for f in ${data_dir}/*.mp3; do
    fbase=$(basename $f)
    fbase=${fbase%.mp3}  # remove suffix
    echo "`date` Processing: $f ($fbase)"

    python /export/fs04/a12/rhuang/contextualizedASR/local/pdf/pdf2context.py \
      --pdf "${data_dir}/${fbase}"'*.pdf' |\
      awk '{print $0, 1.0}' \
      > ${output_dir}/${fbase}.txt
done
ls -1 ${output_dir}/*.txt | wc -l
wc ${output_dir}/*.txt


### get oracle entity context

decode="/export/fs04/a12/rhuang/icefall_align/egs/spgispeech/ASR/tmp/icefall-asr-spgispeech-pruned-transducer-stateless2/exp20220103_2/modified_beam_search_rnnlm_shallow_fusion_biased/wer"
ref_ner=$decode/ref.ner
per_utt=$decode/scoring_kaldi/wer_details/per_utt

distractor_ratio=0.0
outdir="/export/fs04/a12/rhuang/contextualizedASR/data/ec53_kaldi_heuristics2/context_${distractor_ratio}/"
mkdir -p $outdir
python /export/fs04/a12/rhuang/contextualizedASR/local/collect_entities_oracle.py \
  --ref_ner $ref_ner \
  --per_utt $per_utt \
  --distractor_ratio $distractor_ratio \
  --real_contexts "/export/fs04/a12/rhuang/contextualizedASR/data/ec53_kaldi_heuristics2/context/" \
  --out $outdir

wc /export/fs04/a12/rhuang/contextualizedASR/data/ec53_kaldi_heuristics2/context/*.txt
wc $outdir/*.txt

# count OOVs
comm -23 <(sort $output_dir/entities.1g.txt) <(sort $output_dir/vocab.txt) | wc -lah


####################################
# eval wer
####################################

eval_wer () {
  icefall_hyp=$1

  # cuts="data/ec53_manifests/cuts_ec53_trimmed2.jsonl.gz"
  # kaldi_data="/export/fs04/a12/rhuang/contextualizedASR/data/ec53_kaldi_heuristics"
  # cuts="/export/fs04/a12/rhuang/contextualizedASR/data/ec53_kaldi_sp_gentle/cuts.jsonl.gz"
  # kaldi_data="/export/fs04/a12/rhuang/contextualizedASR/data/ec53_kaldi_sp_gentle/"
  cuts="/export/fs04/a12/rhuang/contextualizedASR/data/ec53_kaldi_heuristics11_temp/cuts_no_feat_20230109050816_merged.jsonl.gz" 
  cuts="/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/manifests/cuts_ec53_norm.jsonl.gz"
  kaldi_data="/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/kaldi/"

  python /export/fs04/a12/rhuang/contextualizedASR/local/recogs_to_text.py \
    --input $icefall_hyp \
    --out ${icefall_hyp%.*}.text \
    --cuts $cuts

  wer_dir=$(dirname $icefall_hyp)/wer
  mkdir -p $wer_dir

  cat ${icefall_hyp%.*}.text | \
    python /export/fs04/a12/rhuang/contextualizedASR/earnings21/whisper_normalizer.py \
    --mode "kaldi_rm_sym" | sort \
  > $wer_dir/hyp.txt

  python /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/pruned_transducer_stateless7_context/score.py \
    --refs /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/rare_words/ref_norm/biasing_list_ec53.txt \
    --hyps $wer_dir/hyp.txt

  # cat $kaldi_data/text | \
  #   sed -e 's/\[[^][]*\]//g' | \
  #   python /export/fs04/a12/rhuang/contextualizedASR/earnings21/whisper_normalizer.py \
  #   --mode "kaldi_rm_sym" | sort \
  # > $wer_dir/ref.txt
  # qsub /export/fs04/a12/rhuang/contextualizedASR/local/ner_spacy.sh
  # or:
  # cp /export/fs04/a12/rhuang/icefall_align/egs/spgispeech/ASR/tmp/icefall-asr-spgispeech-pruned-transducer-stateless2/exp20220108/modified_beam_search/wer/ref.* $wer_dir/.
  cp /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/kaldi/ref/ref.* $wer_dir/.

  # compute WER
  ref=$(realpath $wer_dir/ref.txt)
  datadir=$(realpath ${kaldi_data})
  hyp=$(realpath $wer_dir/hyp.txt)
  decode=$(dirname $hyp)
  wc -l $ref $hyp

  cd /export/fs04/a12/rhuang/kws/kws_exp/shay/s5c/
  # . /export/fs04/a12/rhuang/kws/kws_exp/shay/s5c/path.sh
  bash /export/fs04/a12/rhuang/espnet/egs2/spgispeech/asr1/local/score_kaldi_light.sh \
    $ref $hyp $datadir $decode
  cd -

  # export PYTHONPATH=/export/fs04/a12/rhuang/contextualizedASR/:$PYTHONPATH
  python /export/fs04/a12/rhuang/contextualizedASR/local/check_ner2.py \
    --special_symbol "'***'" \
    --per_utt $decode/scoring_kaldi/wer_details/per_utt \
    --ref_ner $decode/ref.ner #--biasing_list "/export/fs04/a12/rhuang/contextualizedASR/data/ec53_kaldi_heuristics2/context/"
}

mamba activate whisper
export PYTHONPATH=/export/fs04/a12/rhuang/contextualizedASR/:$PYTHONPATH

recogs=/export/fs04/a12/rhuang/icefall_align/egs/spgispeech/ASR/tmp/icefall-asr-spgispeech-pruned-transducer-stateless2/exp/modified_beam_search/recogs-ec53-epoch-999-avg-1-modified_beam_search-beam-size-4.txt
recogs=/export/fs04/a12/rhuang/icefall_align/egs/spgispeech/ASR/tmp/icefall-asr-spgispeech-pruned-transducer-stateless2/exp/modified_beam_search_LODR/recogs-ec53-epoch-999-avg-1-modified_beam_search_LODR-beam-size-4-rnnlm-lm-scale-0.2-LODR-2gram-scale--0.1.txt
recogs=pruned_transducer_stateless2_context/exp/exp_libri_full_c-1_stage2_6k/modified_beam_search/recogs-ec53-epoch-99-avg-1-modified_beam_search-beam-size-4.txt
eval_wer $recogs

# modified beam search (baseline)
# [20230108] /export/fs04/a12/rhuang/icefall_align/egs/spgispeech/ASR/tmp/icefall-asr-spgispeech-pruned-transducer-stateless2/exp20220108/modified_beam_search/log-decode-epoch-30-avg-15-modified_beam_search-beam-size-20-3-ngram-lm-scale-0.01-use-averaged-model-2023-01-09-05-11-29 
# *beam_size=4:  /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/log/decode-3616646.out
# beam_size=20: /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/log/decode-3616651.out
# 10.70_48705     9.74_40326      66.53_2276      60.61_257       36.32_1328      15.17_310

# modified beam search + rnnlm only
# 0.1-0.0:  /export/fs04/a12/rhuang/icefall_align/egs/spgispeech/ASR/tmp/icefall-asr-spgispeech-pruned-transducer-stateless2/exp/modified_beam_search_LODR/recogs-ec53-epoch-999-avg-1-modified_beam_search_LODR-beam-size-4-rnnlm-lm-scale-0.1-LODR-2gram-scale--0.0.txt
# 10.56_48096     9.59_39716      66.18_2264      60.61_257       35.80_1309      14.78_302

# modified beam search + rnnlm + lodr
# [20230108] /export/fs04/a12/rhuang/icefall_align/egs/spgispeech/ASR/tmp/icefall-asr-spgispeech-pruned-transducer-stateless2/exp20220108/modified_beam_search_rnnlm_LODR/log-decode-epoch-30-avg-15-modified_beam_search_rnnlm_LODR-beam-size-4-2-ngram-lm-scale--0.05-rnnlm-lm-scale-0.2-LODR-use-averaged-model-2023-01-09-17-15-46
# 0.4-0.16: /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/log/decode-3616647.out
# *0.2-0.1:  /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/log/decode-3616664.out 
# 10.44_47519     9.49_39277      65.07_2226      59.91_254       34.71_1269      14.68_300

# modified beam search + lm biasing
# 'biased_lm_scale': 4.0
# /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/log/decode-3616831.out
# 10.63_48410     9.70_40179      64.86_2219      58.73_249       34.08_1246      14.59_298
# 'biased_lm_scale': 7.0
# /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/log/decode-3616848.out
# 10.68_48634     9.77_40436      64.07_2192      57.78_245       33.12_1211      14.49_296

# modified beam search + lm biasing + rnnlm + lodr
# rnn 0.2-0.1, biased_lm_scale 6.0
# /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/log/decode-3616835.out
# 10.41_47400     9.50_39320      63.46_2171      56.60_240       32.19_1177      14.19_290 

# modified beam search + neural biasing

# modified beam search + neural biasing + rnnlm + lodr

# modified beam search + neural biasing + lm biasing + rnnlm + lodr



# For spgi test set
# Also ref: egs/spgispeech/ASR/ruizhe_contextual/run.sh
eval_wer () {
  cd /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR
  icefall_hyp=$1

  cuts="/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/manifests/cuts_val.jsonl.gz" 
  kaldi_data="/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/kaldi_spgi_val/"

  python /export/fs04/a12/rhuang/contextualizedASR/local/recogs_to_text.py \
    --input $icefall_hyp \
    --out ${icefall_hyp%.*}.text \
    --cuts $cuts --use-uid

  wer_dir=$(dirname $icefall_hyp)/wer
  mkdir -p $wer_dir

  cp ${icefall_hyp%.*}.text $wer_dir/hyp.txt

  python /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/pruned_transducer_stateless7_context/score.py \
    --refs /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/rare_words/ref/biasing_list_val.txt \
    --hyps $wer_dir/hyp.txt # --lenient

  cp /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/kaldi_spgi_val/ref/ref.* $wer_dir/.

  # compute WER
  ref=$(realpath $wer_dir/ref.txt)
  datadir=$(realpath ${kaldi_data})
  hyp=$(realpath $wer_dir/hyp.txt)
  decode=$(dirname $hyp)
  wc -l $ref $hyp

  cd /export/fs04/a12/rhuang/kws/kws_exp/shay/s5c/
  # . /export/fs04/a12/rhuang/kws/kws_exp/shay/s5c/path.sh
  bash /export/fs04/a12/rhuang/espnet/egs2/spgispeech/asr1/local/score_kaldi_light.sh \
    $ref $hyp $datadir $decode
  cd -

  # export PYTHONPATH=/export/fs04/a12/rhuang/contextualizedASR/:$PYTHONPATH
  python /export/fs04/a12/rhuang/contextualizedASR/local/check_ner2.py \
    --special_symbol "'***'" \
    --per_utt $decode/scoring_kaldi/wer_details/per_utt \
    --ref_ner $decode/ref.ner #--biasing_list "/export/fs04/a12/rhuang/contextualizedASR/data/ec53_kaldi_heuristics2/context/"
}