"""
    Usage: python3 multiple_alignments.py
"""


import process_data
import combine_alignments
import logging
import argparse
from utility import OUTPUT_DIR


def write_result(chromosome, multiple_alignments):
    file_name = OUTPUT_DIR + "multiple_alignments_" + chromosome + ".output"
    with open(file_name, "w") as f:
        count = 0
        for pos, alignments in multiple_alignments:
            count += 1
            f.write(pos)
            f.write("\n")
            for a in alignments:
                f.write(a)
                f.write("\n")
            if count < 3:
                f.write("\n")


"""
Reads output files from first.sh.
Arguments:
    input_file: base position pairs of a certain pairwise alignment 
Returns:
	pos_dict: base position pair on human genome 
        -> position pair on the genome of other species
"""


def get_positions(input_file):
    counter = 0
    pos_dict = {}
    with open(input_file) as f:
        for line in f:
            line = line.strip()
            counter += 1
            if counter > 3:
                p = [int(n.strip()) for n in line[1:-1].split(",")]
                pos[(p[0], p[1])] = (p[2], p[3])
    return pos_dict


"""
Remove gaps in alignment sequence.
Arguments:
    with_gaps: alignment sequence to remove gaps from
    others: an array of alignment sequences of other species
Returns:
	with_gaps, others: with gaps (-) removed
"""


def remove_gaps_multiple_helper(with_gaps, others):
    i = 0
    while i < len(with_gaps):
        if with_gaps[i] == "-":
            with_gaps = with_gaps[:i] + with_gaps[i + 1 :]
            for j in range(len(others)):
                others[j] = others[j][:i] + others[j][i + 1 :]
        else:
            i += 1

    return with_gaps, others


"""
Remove gaps in all alignment sequences.
"""


def remove_gaps_in_multiple(alignments):
    human = alignments[0]
    other1 = alignments[1]
    other2 = alignments[2]
    other3 = alignments[3]
    other1, [human, other2, other3] = remove_gaps_multiple_helper(
        other1, [human, other2, other3]
    )
    assert len(human) == len(other1) == len(other2) == len(other3)
    other2, [human, other1, other3] = remove_gaps_multiple_helper(
        other2, [human, other1, other3]
    )
    assert len(human) == len(other1) == len(other2) == len(other3)
    other3, [human, other1, other2] = remove_gaps_multiple_helper(
        other3, [human, other1, other2]
    )
    assert len(human) == len(other1) == len(other2) == len(other3)
    return [human, other1, other2, other3]


"""
Get subsequences based on multiple alignment positions.
Arguments:
    final_pos: position pair in final multiple alignment
    human_pos: original position pair on human genome
    human_a, other_a: sequence of human and one other species
Returns:
	human, others: alignment sequences with gaps removed in human
"""


def get_combined_pairwise_alignments(final_pos, human_pos, human_a, other_a):
    # non-gaps in human == final_pos[1] - final_pos[0]
    # total span == length for other_a == length for human_a
    base_offset = final_pos[0] - human_pos[0]
    bases = final_pos[1] - final_pos[0]

    start_offset = 0
    end_offset = 0
    for i, b in enumerate(human_a):
        if b != "-":
            if base_offset != 0:
                base_offset -= 1
                if base_offset == 0:
                    c = i
                    while human_a[c + 1] == "-":
                        c += 1
                    start_offset = c + 1
            else:
                bases -= 1
                if bases == 0:
                    end_offset = i + 1
                    break
    return remove_gaps_multiple_helper(
        human_a[start_offset:end_offset], [other_a[start_offset:end_offset]]
    )  # remove gaps in human, correct because final_pos excludes gaps and is the same across three pairwise alignments


def get_multiple_alignments(chromosome, debug=False):
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    SPECIES = ["gorGor6", "panTro6", "ponAbe3"]

    # get multiple alignments positions
    (
        final_alignments,
        dict_this,
        dict_other,
        dict_third,
    ) = combine_alignments.get_positions(chromosome)

    alignments = [None] * 3

    logging.info("getting pairwise alignments")
    for i, species in enumerate(SPECIES):
        logging.debug(species)
        input_file = "../data/hg38.{}.synNet.maf".format(species)
        _, alignments[i] = process_data.get_alignments(input_file, species, chromosome)

    logging.info("combining pairwise alignment sequences")
    """
        dict_this and dict_other need to go two levels, dict_third needs one
        from final_alignments
    """
    multiple_alignments = []
    for alignment in final_alignments:
        if alignment[1] - alignment[0] >= 100000:
            logging.debug("Current alignment: " + str(alignment))
            # alignment := final human start and end base position
            # from dict := human start and end in each pairwise alignment
            result = []

            # gor
            human_gor = dict_this[alignment]  # human position before aligning
            if human_gor in dict_this:
                human_gor = dict_this[human_gor]

            human_gor_a, gor_a = get_combined_pairwise_alignments(
                alignment,
                human_gor,
                alignments[0]["human"][human_gor],
                alignments[0][SPECIES[0]][human_gor],
            )
            result.append(human_gor_a)
            result.append(gor_a[0])

            # pan
            human_pan = dict_other[alignment]
            if human_pan in dict_other:
                human_pan = dict_other[human_pan]
            human_pan_a, pan_a = get_combined_pairwise_alignments(
                alignment,
                human_pan,
                alignments[1]["human"][human_pan],
                alignments[1][SPECIES[1]][human_pan],
            )
            assert (
                len(human_gor_a) == len(human_pan_a) == len(gor_a[0]) == len(pan_a[0])
            )
            result.append(pan_a[0])

            # pon
            human_pon = dict_third[alignment]

            human_pon_a, pon_a = get_combined_pairwise_alignments(
                alignment,
                human_pon,
                alignments[2]["human"][human_pon],
                alignments[2][SPECIES[2]][human_pon],
            )
            assert len(human_gor_a) == len(human_pon_a) == len(pon_a[0])
            result.append(pon_a[0])

            multiple_alignments.append(
                (str(alignment), remove_gaps_in_multiple(result))
            )
    write_result(chromosome, multiple_alignments)
    return multiple_alignments


if __name__ == "__main__":
    get_multiple_alignments("hg38.chr7", True)
