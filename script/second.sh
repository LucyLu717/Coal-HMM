cd script
CHROMO="hg38.chr10"
for SPECIES in "gorGor6" "panTro6" "ponAbe3"
do
    python3 ../src/process_data.py -f ../data/hg38.$SPECIES.synNet.maf --align -c $CHROMO > ../outputs/output-${SPECIES:0:3}-$CHROMO.output
done
cd -