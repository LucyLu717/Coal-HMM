# Coal-HMM
A re-implementation of Coal-HMM model proposed by [Hobolth et al](https://doi.org/10.1371/journal.pgen.0030007) (2007, Genomic Relationships and Speciation Time of Human, Chimpanzee, and Gorilla Inferred from a Coalescent Hidden Markov Model). Code to compute model parameters was adapted from [Julien Y. Dutheil, et al](https://doi.org/10.1534/genetics.109.103010). Pairwise alignment data were downloaded from [UCSC Genome Browser](https://hgdownload.soe.ucsc.edu/downloads.html).

## Usage
Put data under `data` folder. 

In current directory, run `first.sh`. This gives information about chromosomes that have sufficiently long pairwise alignments. `second.sh` gives alignment positions based on chromosome provided.

For first time users, either add `--rerun` to `coal_hmm.py` or run `multiple_alignments.py` and save processed multiple alignment sequences for later use. 

Run `run_hmm.sh` to run Viterbi and compute posterior probabilities on input data.
