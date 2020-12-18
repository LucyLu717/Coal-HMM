import argparse


def get_info_from_line(line):
    parts = line.strip().split(" ")
    filtered_parts = []
    for p in parts:
        if p != "":
            filtered_parts.append(p)
    return filtered_parts


def get_alignments(fasta_file, other, c, log=False):
    got_human = False
    counter = 0
    alignments = {}
    lengths = {}
    seq_info = {}

    length = 0
    seq_info = {"human": [], other: []}
    alignments = {"human": {}, other: {}}

    with open(fasta_file) as f:
        for line in f:
            if len(line.strip()) < 1:  # skip new line
                continue
            parts = get_info_from_line(line)
            if parts[0] == "s":
                chromo = parts[1]
                if chromo == c or got_human:
                    start_pos = int(parts[2])
                    seqlen = int(parts[3])  # doesn't contain gaps
                    chromo_len = int(parts[5])
                    if lengths.get(chromo, 0) == 0:
                        length = chromo_len
                    seq = parts[6]
                    if seqlen >= 100000:
                        counter += 1

                        if got_human:
                            alignments[other][
                                (seq_info["human"][-1][0], seq_info["human"][-1][1])
                            ] = seq
                            seq_info[other].append(
                                (
                                    seq_info["human"][-1][0],
                                    seq_info["human"][-1][1],
                                    start_pos,
                                    start_pos + seqlen,
                                )
                            )
                            got_human = False
                        else:
                            alignments["human"][(start_pos, start_pos + seqlen)] = seq
                            seq_info["human"].append((start_pos, start_pos + seqlen))
                            got_human = True
    if log:
        print("Sequences: ", counter)
        print("Chromosome length: ", length)
    return seq_info, alignments


def get_chromosomes(fasta_file):
    human = set()
    other = set()
    got_human = False
    counter = 0
    with open(fasta_file) as f:
        for line in f:
            if len(line.strip()) < 1:  # skip new line
                got_human = False
                continue
            if line[0] == "s":
                phrase = line.split(" ")[1].split(".")
                chromo = phrase[1]
                species = phrase[0]
                seq = line.split(" ")[-1]
                if len(seq) >= 200000:
                    counter += 1
                    if got_human:
                        other.add(chromo)
                    else:
                        human.add(chromo)
                        assert species == "hg38"
                        got_human = True
    print("Sequences: ", counter)
    print("human chromosomes: ", human)
    print("other chromosomes: ", other)


def main():
    parser = argparse.ArgumentParser(
        description="Process pairwise alignment data from UCSC"
    )
    parser.add_argument("-f", action="store", dest="f", type=str, required=True)
    parser.add_argument(
        "--chromo", action="store_true", required=False
    )  # find which chromosomes have long pairwise alignments
    parser.add_argument(
        "--align", action="store_true", required=False
    )  # find base positions in pairwise alignments
    parser.add_argument(
        "-c", action="store", dest="chromosome", type=str, required=False
    )
    parser.add_argument("--log", action="store_true", required=False, default=False)

    args = parser.parse_args()
    input_file = args.f
    species = input_file.split("hg38.")[1][:7]
    print(species)

    if args.chromo:
        get_chromosomes(input_file)
    if args.align:
        CHROMOSOME = args.chromosome
        seq_info, _ = get_alignments(input_file, species, CHROMOSOME, args.log)
        print(CHROMOSOME)
        for sp, li in seq_info.items():
            li.sort(key=lambda tu: tu[0])
            if sp == species:
                print(len(li))
                for it in li:
                    print(it)


if __name__ == "__main__":
    main()
