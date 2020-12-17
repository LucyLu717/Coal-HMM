python3 process_data.py -f data/hg38.gorGor6.synNet.maf --chromo > chromo.output
python3 process_data.py -f data/hg38.panTro6.synNet.maf --chromo >> chromo.output
python3 process_data.py -f data/hg38.ponAbe3.synNet.maf --chromo >> chromo.output
python3 process_data.py -f data/hg38.gorGor6.synNet.maf --align > output-gor.output
python3 process_data.py -f data/hg38.panTro6.synNet.maf --align > output-pan.output
python3 process_data.py -f data/hg38.ponAbe3.synNet.maf --align > output-pon.output