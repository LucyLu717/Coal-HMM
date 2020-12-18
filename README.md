# Coal-HMM
A re-implementation of Coal-HMM model proposed by [Hobolth et al](https://doi.org/10.1371/journal.pgen.0030007) (2007, Genomic Relationships and Speciation Time of Human, Chimpanzee, and Gorilla Inferred from a Coalescent Hidden Markov Model). Code to compuate model parameters was adapted from [Julien Y. Dutheil, et al](https://doi.org/10.1534/genetics.109.103010). Pairwise alignment data were downloaded from [UCSC Genome Browser](https://hgdownload.soe.ucsc.edu/downloads.html).

## Usage
Put data under `data` folder. In current directory, run `first.sh`. This gives alignment positions. 

For first time users, run `multiple_alignments.py` and save processed multiple alignment sequences to `multiple_alignments.output` for later use. 

Then run `run_hmm.sh` to run Viterbi and compute posterior probabilities on input data.
