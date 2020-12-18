STATES = ["HC1", "HC2", "HG", "CG"]  # Hidden states
FIGS_DIR = "../figs/"
OUTPUT_DIR = "../outputs/"
colors = ["b", "c", "g", "r"]
combined_states = ["", "HC", "HG", "CG"]
combined_colors = ["", "b", "g", "r"]
"""
A helper function that returns a column at position i in array seq.
"""


def get_element(i, seq):
    return tuple([seq[j][i].upper() for j in range(4)])
