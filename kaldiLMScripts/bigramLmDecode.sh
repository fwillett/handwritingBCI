#!/bin/bash
set -e;
export LC_NUMERIC=C;

acoustic_scale=1.79;
beam=65;
max_active=5000000;
help_message="
Usage: ${0##*/} [options] lmDir inputMatrix outputPrefix

Options:
  --acoustic_scale   : (type = float, default = $acoustic_scale)
                        List of acoustic scale factors, separated by spaces.
                        The first is used with a word LM and the second with
                        a character LM.
  --beam              : (type = float, default = $beam)
                        Decoding beam.
  --max_active        : (type = integer, default = $max_active)
                        Max. number of tokens during Viterbi decoding
                        (a.k.a. histogram prunning).
";
source ./kaldiLMScripts/parseOptions.inc.sh || exit 1;
[ $# -ne 3 ] && echo "$help_message" >&2 && exit 1;
wdir="$1";
inpm="$2";
opfx="$3";

olat="${opfx}lat_dec.ark";
owrd="${opfx}words_dec.ark";
oali="${opfx}align_dec.ark";
    
# generate lattices
{
  date "+%F %T - Started latgen-lazylm-faster-mapped" && \
  latgen-lazylm-faster-mapped \
    --acoustic-scale="$acoustic_scale" \
    --allow-partial="true" \
    --beam="$beam" \
    --max-active="$max_active" \
    --min-active="20" \
   "$wdir/model" "$wdir/HCL.fst" "$wdir/G.fst" "ark,t:$inpm" \
   "ark,t:$olat" "ark,t:$owrd" "ark,t:$oali" && \
  date "+%F %T - Finished latgen-lazylm-faster-mapped";
} ||
{ echo "ERROR: Failed latgen-lazylm-faster-mapped, see \$log" && exit 1; }

# n best lists
olat="${opfx}lat_dec.ark";
obest="${opfx}best_dec.ark";

oali="${opfx}best_ali.ark";
owrd="${opfx}best_words.ark";
olm="${opfx}best_lmscore.ark";
oac="${opfx}best_acscore.ark";

lattice-to-nbest --acoustic-scale=1.0 --n=128 "ark:$olat" "ark:$obest"
nbest-to-linear "ark:$obest" "ark,t:$oali" "ark,t:$owrd" "ark,t:$olm" "ark,t:$oac"

ali-to-phones "$wdir/model" "ark:$oali" ark,t:- 2> /dev/null |
./kaldiLMScripts/int2sym.pl -f 2- "$wdir/chars.txt" |
./kaldiLMScripts/remove_transcript_dummy_boundaries.sh > "${opfx}transcript.txt";