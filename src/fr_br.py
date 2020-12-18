from utility import STATES, get_element
import numpy as np
from math import exp

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
