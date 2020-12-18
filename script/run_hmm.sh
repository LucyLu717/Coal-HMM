cd script
cat /dev/null > ../outputs/viterbi.output
python3 ../src/coal-hmm.py -f ../outputs/multiple_alignments.output -out ../outputs/viterbi.output
cd -