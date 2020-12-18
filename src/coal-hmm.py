#!/usr/bin/env python3

"""
Script for computing Viterbi intervals and respective posterior probabilities
   in a given alignment using Coal-HMM model.

Arguments:
    -f: file containing the multiple alignments
    -out: file to output intervals to

Output:
    Viterbi intervals
    Graphs of posterior probabilities per alignment 

Example Usage:
    python coal-hmm.py -f multiple_alignments.output -out viterbi.output
"""

import argparse
import numpy as np
import warnings

from emission_probs import likelihood
from multiple_alignments import get_multiple_alignments
from params import get_params
from utility import STATES, get_element
from plot import plot
from viterbi import viterbi, find_intervals
from fr_br import forward_backward
from divergent_sites import divergent_sites

warnings.filterwarnings("ignore")

"""
Reads the file and outputs the alignments to analyze.
Arguments:
	filename: name of the file
Returns:
	seqs: an array of multiple alignments with positions on the chromosome
"""


def read_seqs(filename):
    with open(filename, "r") as f:
        seqs = [(s.split("\n")[0], s.split("\n")[1:]) for s in f.read().split("\n\n")]
    return seqs


def main():
    parser = argparse.ArgumentParser(description="Arg parser for Coal-HMM")
    parser.add_argument("-f", action="store", dest="f", type=str, required=False)
    parser.add_argument("-out", action="store", dest="out", type=str, required=True)
    parser.add_argument(
        "-c", action="store", dest="chromosome", type=str, required=True
    )
    parser.add_argument("--rerun", action="store_true", required=False)
    parser.add_argument("--outgroup", action="store_true", required=False)

    args = parser.parse_args()
    intervals_file = args.out

    if args.rerun:
        sequences = get_multiple_alignments(args.chromosome)
    else:
        sequences = read_seqs(args.f)

    for align_index, alignments in enumerate(sequences):
        (s, u, v1, v2), init_ps, (a, b, c, a_t, b_t, c_t), mu = get_params(
            len(alignments[1][0])
        )
        print("Parameters: ", s, u, v1, v2, init_ps, (a, b, c, a_t, b_t, c_t))

        transition_probabilities = {
            "HC1": {
                "HC1": np.log(1 - 3 * s),
                "HC2": np.log(s),
                "HG": np.log(s),
                "CG": np.log(s),
            },
            "HC2": {
                "HC1": np.log(u),
                "HC2": np.log(1 - u - 2 * v1),
                "HG": np.log(v1),
                "CG": np.log(v1),
            },
            "HG": {
                "HC1": np.log(u),
                "HC2": np.log(v1),
                "HG": np.log(1 - u - v1 - v2),
                "CG": np.log(v2),
            },
            "CG": {
                "HC1": np.log(u),
                "HC2": np.log(v1),
                "HG": np.log(v2),
                "CG": np.log(1 - u - v1 - v2),
            },
        }
        initial_probabilities = dict(zip(STATES, init_ps))

        emission_probabilities = {}
        for i in range(4):
            st = STATES[i]
            if i == 0:
                emission_probabilities[st] = likelihood(i, a, b, c, mu)
            else:
                emission_probabilities[st] = likelihood(i, a_t, b_t, c_t, mu)

        # Viterbi
        sequence, p = viterbi(
            alignments[1],
            transition_probabilities,
            emission_probabilities,
            initial_probabilities,
        )
        intervals = find_intervals(sequence)
        # Write intervals
        with open(intervals_file, "a") as f:
            for i in range(4):
                f.write("%s\n" % (STATES[i]))
                f.write(
                    "\n".join(
                        [("(%d,%d)" % (start, end)) for (start, end) in intervals[i]]
                    )
                )
                f.write("\n")
        print("{} Viterbi probability: {:.2f}".format(alignments[0], p))

        # Compute posterior probs
        F, likelihood_f, B, likelihood_b, R, R_combined = forward_backward(
            alignments[1],
            transition_probabilities,
            emission_probabilities,
            initial_probabilities,
        )
        # Save posterior probs
        # np.savetxt(
        #     "posteriors.{}.csv".format(align_index), R, delimiter=",", fmt="%.4e"
        # )
        print("{} Forward likelihood: {:.2f}".format(alignments[0], likelihood_f))

        # get genealogy from divergent sites
        divergent_info = divergent_sites(alignments[1], args.outgroup)
        # Plot
        for i in range(4):
            plot(
                R[i],
                R_combined[i - 1],
                intervals,
                i,
                align_index,
                [int(i) for i in alignments[0][1:-1].split(",")],
                args.chromosome,
                divergent_info,
            )


if __name__ == "__main__":
    main()
