cd script
python3 ../src/process_data.py -f ../data/hg38.gorGor6.synNet.maf --chromo > ../outputs/chromo.output
python3 ../src/process_data.py -f ../data/hg38.panTro6.synNet.maf --chromo >> ../outputs/chromo.output
python3 ../src/process_data.py -f ../data/hg38.ponAbe3.synNet.maf --chromo >> ../outputs/chromo.output
cd -