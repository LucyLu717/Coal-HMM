from utility import get_element
from collections import defaultdict


"""
A helper function that determines genealogy at a divergent site [element].
"""


def divergent_site(element, outgroup=False):
    H = element[0]
    C = element[1]
    G = element[2]
    O = element[3]

    if H == C and H != G:
        if outgroup:
            if G == O:
                return "HC"
            else:
                return ""
        return "HC"
    if H == G and H != C:
        if outgroup:
            if C == O:
                return "HG"
            else:
                return ""
        return "HG"
    if C == G and H != C:
        if outgroup:
            if H == O:
                return "CG"
            else:
                return ""
        return "CG"
    return ""


"""
Determines genealogy at divergent sites in the given alignment sequence.
Returns:
	divergent_info: genealogy -> a list of relative positions starting from 0
"""


def divergent_sites(seq, outgroup=False):
    divergent_info = defaultdict(list)
    for i in range(len(seq[0])):
        site = get_element(i, seq)
        divergent_info[divergent_site(site, outgroup)].append(i)
    return divergent_info
