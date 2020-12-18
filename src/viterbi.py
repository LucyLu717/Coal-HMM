from utility import STATES, get_element
import numpy as np

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
