import argparse

OUTPUT_DIR = "../outputs/"


"""
Reads output files from first.sh.
Arguments:
    input_file: base position pairs of a certain pairwise alignment 
Returns:
	pos: an array of base position pairs on the human genome from the 
        pairwise alignment given
"""


def read_input(input_file):
    counter = 0
    pos = []
    with open(input_file) as f:
        for line in f:
            line = line.strip()
            counter += 1
            if counter > 3:
                p = [int(n.strip()) for n in line[1:-1].split(",")]
                pos.append((p[0], p[1]))
    return pos


"""
Combine two pairwise alignments into a three-way multiple alignment
Arguments:
    this, other: base position pairs of a certain pairwise alignment 
Returns:
	new_alignments: an array of new base position pairs on the human genome 
        for the new alignment
    dict_this, dict_other: new positions -> original positions
"""


def combine_alignments1(this, other):
    new_alignments = []
    dict_this = {}
    dict_other = {}

    for start1, end1 in this:
        for start2, end2 in other:
            new_alignment = ()
            if start1 <= start2:
                if end2 <= end1:
                    new_alignment = (start2, end2)
                elif start2 <= end1:
                    new_alignment = (start2, end1)
            else:
                if end1 <= end2:
                    new_alignment = (start1, end1)
                elif start1 <= end2:
                    new_alignment = (start1, end2)
            if len(new_alignment) != 0:
                new_alignments.append(new_alignment)
                dict_this[new_alignment] = (start1, end1)
                dict_other[new_alignment] = (start2, end2)
    return new_alignments, dict_this, dict_other


"""
Combine a three-way multiple alignment and the other pairwise alignment into 
    a four-way multiple alignment
Arguments:
    new, third: base position pairs of a certain pairwise alignment 
Returns:
	new_alignments: an array of new base position pairs on the human genome 
        for the new alignment
    dict_this, dict_other, dict_third: new positions -> original positions
"""


def combine_alignments2(new, third, dict_this, dict_other):
    new_alignments = []
    dict_third = {}

    for start1, end1 in new:
        for start2, end2 in third:
            new_alignment = ()
            if start1 <= start2:
                if end2 <= end1:
                    new_alignment = (start2, end2)
                elif start2 <= end1:
                    new_alignment = (start2, end1)
            else:
                if end1 <= end2:
                    new_alignment = (start1, end1)
                elif start1 <= end2:
                    new_alignment = (start1, end2)
            if len(new_alignment) != 0:
                new_alignments.append(new_alignment)
                if new_alignment not in dict_this:
                    dict_this[new_alignment] = (start1, end1)
                    dict_other[new_alignment] = (start1, end1)
                dict_third[new_alignment] = (start2, end2)
    return new_alignments, dict_this, dict_other, dict_third


def get_positions(log=False):
    pos_lists = [None] * 3
    for i, species in enumerate(["gor", "pan", "pon"]):
        inputs = read_input(OUTPUT_DIR + "output-" + species + ".output")
        pos_lists[i] = inputs
    new_alignments, dict_this, dict_other = combine_alignments1(
        pos_lists[0], pos_lists[1]
    )
    final_alignments, dict_this, dict_other, dict_third = combine_alignments2(
        new_alignments, pos_lists[2], dict_this, dict_other
    )
    if log:
        print(len(final_alignments), len(dict_this), len(dict_other), len(dict_third))
        print(final_alignments)
        print(["gor", "pan", "pon"])
        print(dict_this)
        print(dict_other)
        print(dict_third)
    return final_alignments, dict_this, dict_other, dict_third


def main():
    parser = argparse.ArgumentParser(
        description="Process pairwise alignment data from UCSC"
    )
    parser.add_argument("--log", action="store_true", required=False, default=False)
    args = parser.parse_args()
    get_positions(args.log)


if __name__ == "__main__":
    main()
