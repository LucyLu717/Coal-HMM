cd script
CHROMO="hg38.chr10"
cat /dev/null > ../outputs/viterbi_$CHROMO.output
python3 ../src/coal-hmm.py --rerun -out ../outputs/viterbi_$CHROMO.output -c $CHROMO
# python3 ../src/coal-hmm.py -f ../outputs/multiple_alignments_$CHROMO.output -out ../outputs/viterbi_$CHROMO.output -c $CHROMO
cd -