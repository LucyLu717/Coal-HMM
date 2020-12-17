#!/usr/bin/env python3

""" Construct emission probabilities for Coal-HMM based on the Jukes-Cantor 
    substitution model using Felsenstein's algorithm.
"""

import numpy as np
import math
import itertools

BASE = "ACGT"
BASES = [BASE] * 4
POSSIBILITIES = [e for e in itertools.product(*BASES)]


class Node:
    """ Initializes a node with given parameters.

    Arguments:
        name: name of node (only relevant for leaves)
        left: left child (Node)
        right: right child (Node)
        branch_length: length of branch that leads to this node (float)
        branch_id: id of branch that leads to this node (int)
        probs: probability of observed bases beneath this node
                [list of 4 probs for 'ACGT'] (initialized to None)
    """

    def __init__(self, name, left, right, branch_length, branch_id):
        self.name = name
        self.left = left
        self.right = right
        self.branch_length = branch_length
        self.branch_id = branch_id
        self.probs = [None for _ in range(4)]


""" Evaluates P(b|a, t) under the Jukes-Cantor model

Arguments:
    b: descendant base (string)
    a: ancestral base (string)
    t: branch length (float)
    u: mutation rate (float, defaults to 1)
Returns:
    prob: float probability P(b|a, t)
"""


def jcm(b, a, t, u):
    e_term = math.exp(-4 * u * t / 3)
    if b == a:
        return (1 + 3 * e_term) / 4
    else:
        return (1 - e_term) / 4


""" Constructs the ordering of the post-order traversal of ```index```
    topology from the pset.
Arguments:
    index: which topology to use
Returns:
    list of Nodes corresponding to post-order traversal of the topology
    branch_probs: 6x4x4 matrices, indexed as:
                  branch_probs[branch_id][a][b] = P(b | a, t_branch_id)
"""


def initialize_topology(index, a, b, c, mu):
    bases = BASE
    branch_lengths = np.array(
        [
            [a, a, a + b, c, b, 0],
            [a, a, a + b, c, b, 0],
            [a, a + b, a, c, b, 0],
            [a + b, a, a, c, b, 0],
        ],
        dtype=float,
    )
    names = [0, 1, 2, 3]  # "human", "chimp", "gorilla", "orangutan"
    branches = [0, 1, 2, 3]
    leaves = [
        Node(s, None, None, bl, i)
        for (s, i, bl) in zip(names, branches, branch_lengths[index, :])
    ]
    ordering = None
    branch_probs = [np.zeros((4, 4), dtype=float) for _ in range(6)]

    if index == 0 or index == 1:
        hum_chimp = Node(None, leaves[0], leaves[1], branch_lengths[index, -2], 4)
        gr_or = Node(None, leaves[2], leaves[3], branch_lengths[index, -1], 5)
        root = Node("root", hum_chimp, gr_or, None, None)
        ordering = [
            leaves[0],
            leaves[1],
            hum_chimp,
            leaves[2],
            leaves[3],
            gr_or,
            root,
        ]
    elif index == 2:
        human_gor = Node(None, leaves[0], leaves[2], branch_lengths[index, -2], 4)
        hg_or = Node(None, human_gor, leaves[3], branch_lengths[index, -1], 5)
        root = Node("root", leaves[1], hg_or, None, None)
        ordering = [
            leaves[0],
            leaves[2],
            human_gor,
            leaves[3],
            hg_or,
            leaves[1],
            root,
        ]
    elif index == 3:
        chimp_gor = Node(None, leaves[1], leaves[2], branch_lengths[index, -2], 4)
        cg_or = Node(None, chimp_gor, leaves[3], branch_lengths[index, -1], 5)
        root = Node("root", leaves[0], cg_or, None, None)
        ordering = [
            leaves[1],
            leaves[2],
            chimp_gor,
            leaves[3],
            cg_or,
            leaves[0],
            root,
        ]
    else:
        raise ValueError(
            "Unrecognizable index {} given to initialize_topology".format(index)
        )

    """ Assign 6x4x4 branch_probs values: branch_probs[branch_id][ancestor_base][descendant_base] """
    for bid in range(len(branch_probs)):
        for anc in range(4):
            for des in range(4):
                t = branch_lengths[index][bid]
                branch_probs[bid][anc][des] = jcm(bases[des], bases[anc], t, mu)
    return ordering, branch_probs


""" Computes the likelihood of the data given the topology specified by ordering

Arguments:
    data: sequence data (dict: name of sequence owner -> sequence)
    seqlen: length of sequences
    ordering: postorder traversal of our topology
    bp: branch probabilities for the given branches: 6x4x4 matrix indexed as
        branch_probs[branch_id][a][b] = P(b | a, t_branch_id)
Returns:
    total_log_prob: log likelihood of the topology given the sequence data
"""


def likelihood(index, a, b, c, mu):
    ordering, branch_probs = initialize_topology(index, a, b, c, mu)
    emission_probs = {}
    for element in POSSIBILITIES:
        probs = {}
        for node in ordering:
            probs[node] = {}
            if node.name != None and node.name != "root":  # leaves
                base = element[node.name]
                for b in BASE:
                    if b == base:
                        probs[node][b] = 1
                    else:
                        probs[node][b] = 0
            else:
                for anc in range(4):
                    left_sum = 0
                    right_sum = 0
                    for des in range(4):
                        left_sum += (
                            probs[node.left][BASE[des]]
                            * branch_probs[node.left.branch_id][anc][des]
                        )
                        right_sum += (
                            probs[node.right][BASE[des]]
                            * branch_probs[node.right.branch_id][anc][des]
                        )
                    probs[node][BASE[anc]] = left_sum * right_sum
                if node.name == "root":
                    sum_root = 0
                    for anc in BASE:
                        sum_root += probs[node][anc] / 4
                    emission_probs[element] = np.log(sum_root)
    return emission_probs
