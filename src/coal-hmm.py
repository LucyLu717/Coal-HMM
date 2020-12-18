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
import emission_probs
import multiple_alignments
from math import exp
import params
import warnings

warnings.filterwarnings("ignore")

STATES = ["HC1", "HC2", "HG", "CG"]  # Hidden states
FIGS_DIR = "../figs/"

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


"""
Arguments:
	a, b: log probabilities
Returns:
	log(exp(a) + exp(b)) 
"""


def sumLogProb(a, b):
    if a > b:
        return a + np.log1p(exp(b - a))
    else:
        return b + np.log1p(exp(a - b))


"""
A helper function that returns a column at position i in array seq.
"""


def get_element(i, seq):
    return tuple([seq[j][i].upper() for j in range(4)])


""" Outputs the Viterbi decoding of a given observation.
Arguments:
	seq: observed alignments of emitted states (list of emissions)
	trans_probs: transition log-probabilities (dictionary of dictionaries)
	emiss_probs: emission log-probabilities (dictionary of dictionaries)
	init_probs: initial log-probabilities for each hidden state (dictionary)
Returns:
	l: list of most likely hidden states at each position (list of hidden
           states)
	p: log-probability of the returned hidden state sequence
"""


def viterbi(seq, trans_probs, emiss_probs, init_probs):
    n_col = len(seq[0])
    matrix = {}  # a matrix in the form of a dictionary with STATES as keys
    traceback = {}  # a matrix of backpointers
    # initialize
    for st in STATES:
        matrix[st] = np.zeros(shape=(n_col))
        traceback[st] = [None] * n_col
        matrix[st][0] = init_probs[st] + emiss_probs[st][get_element(0, seq)]

    # fill the matrix
    for pos in range(1, n_col):
        for st in STATES:
            probs = []
            for prev_st in STATES:
                probs.append(matrix[prev_st][pos - 1] + trans_probs[prev_st][st])
            max_pos = np.argmax(probs)
            max_p = probs[max_pos]
            traceback[st][pos] = STATES[max_pos]
            matrix[st][pos] = max_p + emiss_probs[st][get_element(pos, seq)]

    final_p = [matrix[st][-1] for st in STATES]
    max_pos = np.argmax(final_p)
    max_p = final_p[max_pos]
    l = []

    # traceback
    l.append(STATES[max_pos])
    state = STATES[max_pos]

    for pos in range(n_col - 1, 0, -1):
        state = traceback[state][pos]
        l.append(state)
    return l[::-1], max_p


""" Returns a list of non-overlapping intervals from Viterbi.
Arguments:
	sequence: list of hidden states
Returns:
	intervals: list of tuples (i, j), 1 <= i <= j <= len(sequence).
"""


def find_intervals(sequence):
    intervals = [[] for _ in range(4)]
    st = sequence[0]
    start = 0
    pos = 1
    while pos < len(sequence):
        if sequence[pos] != st:
            index = STATES.index(st)
            intervals[index].append((start, pos - 1))
            start = pos
            st = sequence[pos]
        pos += 1
    index = STATES.index(st)
    intervals[index].append((start, pos - 1))
    return intervals


""" Outputs the forward probabilities of a given observation.
Arguments:
	seq: observed sequence of emitted states (list of emissions)
	trans_probs: transition log-probabilities (dictionary of dictionaries)
	emiss_probs: emission log-probabilities (dictionary of dictionaries)
	init_probs: initial log-probabilities for each hidden state (dictionary)
Returns:
	F: matrix of forward probabilities
    likelihood_f: P(obs) calculated using the forward algorithm
"""


def forward(seq, trans_probs, emiss_probs, init_probs):
    n_col = len(seq[0])
    matrix = {}  # a matrix in the form of a dictionary with keys h and l
    # initialize
    for st in STATES:
        matrix[st] = np.zeros(shape=(n_col))
        matrix[st][0] = init_probs[st] + emiss_probs[st][get_element(0, seq)]

    # fill the matrix
    for pos in range(1, n_col):
        for st in STATES:
            probs = []
            for prev in STATES:
                probs.append(matrix[prev][pos - 1] + trans_probs[prev][st])
            prob_sum = sumLogProb(
                sumLogProb(probs[0], probs[1]), sumLogProb(probs[2], probs[3])
            )
            matrix[st][pos] = prob_sum + emiss_probs[st][get_element(pos, seq)]

    return (
        matrix,
        sumLogProb(
            sumLogProb(matrix[STATES[0]][-1], matrix[STATES[1]][-1]),
            sumLogProb(matrix[STATES[2]][-1], matrix[STATES[3]][-1]),
        ),
    )


""" Outputs the backward probabilities of a given observation.
Arguments:
	seq: observed sequence of emitted states (list of emissions)
	trans_probs: transition log-probabilities (dictionary of dictionaries)
	emiss_probs: emission log-probabilities (dictionary of dictionaries)
	init_probs: initial log-probabilities for each hidden state (dictionary)
Returns:
	F: matrix of backward probabilities
    likelihood_f: P(obs) calculated using the backward algorithm
"""


def backward(seq, trans_probs, emiss_probs, init_probs):
    n_col = len(seq[0])
    matrix = {}  # a matrix in the form of a dictionary with keys h and l
    # initialize
    for st in STATES:
        matrix[st] = np.zeros(shape=(n_col))
        matrix[st][-1] = 1

    # fill the matrix
    for pos in range(n_col - 2, -1, -1):
        for st in STATES:
            probs = []
            for prev in STATES:
                probs.append(
                    matrix[prev][pos + 1]
                    + emiss_probs[prev][get_element(pos + 1, seq)]
                    + trans_probs[st][prev]
                )
            prob_sum = sumLogProb(
                sumLogProb(probs[0], probs[1]), sumLogProb(probs[2], probs[3])
            )
            matrix[st][pos] = prob_sum
    probs = []
    for st in STATES:
        probs.append(
            matrix[st][0] + emiss_probs[st][get_element(0, seq)] + init_probs[st]
        )
    final = sumLogProb(sumLogProb(probs[0], probs[1]), sumLogProb(probs[2], probs[3]))
    return matrix, final


""" Outputs the forward and backward probabilities of a given observation.
Arguments:
	seq: observed sequence of emitted states (list of emissions)
	trans_probs: transition log-probabilities (dictionary of dictionaries)
	emiss_probs: emission log-probabilities (dictionary of dictionaries)
	init_probs: initial log-probabilities for each hidden state (dictionary)
Returns:
	F: matrix of forward probabilities
    likelihood_f: P(obs) calculated using the forward algorithm
	B: matrix of backward probabilities
    likelihood_b: P(obs) calculated using the backward algorithm
	R: matrix of posterior probabilities
"""


def forward_backward(seq, trans_probs, emiss_probs, init_probs):
    F, likelihood_f = forward(seq, trans_probs, emiss_probs, init_probs)
    B, likelihood_b = backward(seq, trans_probs, emiss_probs, init_probs)

    n_col = len(seq[0])
    matrix = {}

    # initialize
    for st in STATES:
        matrix[st] = np.zeros(shape=(n_col))

    # fill the matrix
    for pos in range(n_col):
        sum_probs = []
        for st in STATES:
            sum_probs.append(F[st][pos] + B[st][pos])
        denominator = sumLogProb(
            sumLogProb(sum_probs[0], sum_probs[1]),
            sumLogProb(sum_probs[2], sum_probs[3]),
        )
        for st in STATES:
            numerator = F[st][pos] + B[st][pos]
            matrix[st][pos] = exp(numerator - denominator)

    R = np.concatenate(
        (
            [matrix[STATES[0]]],
            [matrix[STATES[1]]],
            [matrix[STATES[2]]],
            [matrix[STATES[3]]],
        ),
        axis=0,
    )
    R_combined = np.concatenate(
        (
            [matrix[STATES[0]] + matrix[STATES[1]]],
            [matrix[STATES[2]]],
            [matrix[STATES[3]]],
        ),
        axis=0,
    )
    return F, likelihood_f, B, likelihood_b, R, R_combined


""" Plots graph with marginal posterier probabilities."""


def plot(R_st, Rcom_st, intervals, index, align_index, positions, chr, length=10000):
    import matplotlib.pyplot as plt

    # Draw posterior probs with all four states
    bins = np.linspace(positions[0], positions[1], len(R_st), dtype=int)
    plt.figure(str(align_index) + "1", figsize=(80, 10))
    plt.plot(bins, R_st, label="{} probability".format(STATES[index]))
    plt.ticklabel_format(axis="x", style="plain")
    plt.tight_layout()
    plt.legend(loc="best", fontsize="xx-small")
    plt.xlabel("sequence position")
    plt.ylabel("marginal posterior probability at {}".format(align_index))
    if index == 3:
        plt.savefig(
            FIGS_DIR
            + "{} {}-{} full posterior probability {}.png".format(
                chr, positions[0], positions[1], align_index
            )
        )

    # Draw posterior probs with HC1 and HC2 combined
    if index >= 1:
        combined_states = ["", "HC", "HG", "CG"]
        bins = np.linspace(positions[0], positions[1], len(R_st), dtype=int)
        plt.figure(str(align_index) + "2", figsize=(80, 10))
        plt.plot(bins, Rcom_st, label="{} probability".format(combined_states[index]))
        plt.ticklabel_format(axis="x", style="plain")
        plt.tight_layout()
        plt.legend(loc="best", fontsize="xx-small")
        plt.xlabel("sequence position")
        plt.ylabel("marginal posterior probability at {}".format(align_index))
        if index == 3:
            plt.savefig(
                FIGS_DIR
                + "{} {}-{} HC combined posterior probability {}.png".format(
                    chr, positions[0], positions[1], align_index
                )
            )

        # Draw HC combined with limited length
        Rcom_st = Rcom_st[:length]
        bins = np.linspace(positions[0], positions[0] + length, len(Rcom_st), dtype=int)
        plt.figure(str(align_index) + "3", figsize=(20, 10))
        plt.subplot(211)
        plt.plot(bins, Rcom_st, label="{} probability".format(combined_states[index]))
        plt.ticklabel_format(axis="x", style="plain", useOffset=False)
        plt.tight_layout()
        plt.legend(loc="best", fontsize="xx-small")
        plt.xlabel("sequence position")
        plt.ylabel("marginal posterior probability at {}".format(align_index))

        # Draw corresponding viterbi intervals for limited length (combined HC)
        plt.subplot(212)
        viterbi = [None] * len(Rcom_st)
        for pos, _ in enumerate(viterbi):
            for a, b in intervals[index]:
                if a <= pos + 1 <= b:
                    viterbi[pos] = index - 1
                    break
        if index == 1:
            for pos, _ in enumerate(viterbi):
                for a, b in intervals[0]:
                    if a <= pos + 1 <= b:
                        viterbi[pos] = 0
                        break

        plt.plot(bins, viterbi, label="{} intervals".format(combined_states[index]))
        plt.ticklabel_format(axis="x", style="plain", useOffset=False)
        plt.tight_layout()
        plt.legend(loc="best", fontsize="xx-small")
        plt.xlabel("Viterbi intervals")

        if index == 3:
            plt.savefig(
                FIGS_DIR
                + "{} {}-{} HC combined posterior probability with first {}kb {}.png".format(
                    chr, positions[0], positions[1], length // 1000, align_index
                )
            )


def main():
    parser = argparse.ArgumentParser(description="Arg parser for Coal-HMM")
    parser.add_argument("-f", action="store", dest="f", type=str, required=False)
    parser.add_argument("-out", action="store", dest="out", type=str, required=True)
    parser.add_argument(
        "-c", action="store", dest="chromosome", type=str, required=True
    )
    parser.add_argument("--rerun", action="store_true", required=False)

    args = parser.parse_args()
    intervals_file = args.out

    if args.rerun:
        sequences = multiple_alignments.get_multiple_alignments(args.chromosome)
    else:
        sequences = read_seqs(args.f)

    for align_index, alignments in enumerate(sequences):
        (s, u, v1, v2), init_ps, (a, b, c, a_t, b_t, c_t), mu = params.get_params(
            len(alignments[1][0])
        )
        print(s, u, v1, v2, init_ps, (a, b, c, a_t, b_t, c_t))

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
                emission_probabilities[st] = emission_probs.likelihood(i, a, b, c, mu)
            else:
                emission_probabilities[st] = emission_probs.likelihood(
                    i, a_t, b_t, c_t, mu
                )

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
            )


if __name__ == "__main__":
    main()
