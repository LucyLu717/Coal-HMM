""" Plots graph with marginal posterier probabilities."""

from utility import STATES, FIGS_DIR, colors, combined_colors, combined_states
import numpy as np
import matplotlib.pyplot as plt


def plot(
    R_st,
    Rcom_st,
    intervals,
    index,
    align_index,
    positions,
    chr,
    divergent_info,
    length=10000,
):

    # Draw posterior probs with all four states
    bins = np.linspace(positions[0], positions[1], len(R_st), dtype=int)
    plt.figure(str(align_index) + "1", figsize=(80, 10))
    plt.xlim(positions[0], positions[1])
    plt.plot(
        bins, R_st, label="{} probability".format(STATES[index]), color=colors[index]
    )
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
        bins = np.linspace(positions[0], positions[1], len(R_st), dtype=int)
        plt.figure(str(align_index) + "2", figsize=(80, 10))
        plt.subplot(211)
        plt.xlim(positions[0], positions[1])
        plt.plot(
            bins,
            Rcom_st,
            label="{} probability".format(combined_states[index]),
            color=combined_colors[index],
        )
        plt.ticklabel_format(axis="x", style="plain")
        plt.tight_layout()
        plt.legend(loc="best", fontsize="xx-small")
        plt.xlabel("sequence position")
        plt.ylabel("marginal posterior probability at {}".format(align_index))

        # draw divergent sites genealogy info
        plt.subplot(212)
        divergents = divergent_info[combined_states[index]]
        divergents = [d + positions[0] for d in divergents]
        plt.xlim(positions[0], positions[1])
        plt.vlines(divergents, ymin=0, ymax=1, color=combined_colors[index])
        plt.ticklabel_format(axis="x", style="plain", useOffset=False)
        plt.tight_layout()
        plt.xlabel("Divergent sites")

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
        plt.subplot(311)
        plt.xlim(positions[0], positions[0] + length)
        plt.plot(
            bins,
            Rcom_st,
            label="{} probability".format(combined_states[index]),
            color=combined_colors[index],
        )
        plt.ticklabel_format(axis="x", style="plain", useOffset=False)
        plt.tight_layout()
        plt.legend(loc="best", fontsize="xx-small")
        plt.xlabel("sequence position")
        plt.ylabel("marginal posterior probability at {}".format(align_index))

        # Draw corresponding viterbi intervals for limited length (combined HC)
        plt.subplot(312)
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
        plt.xlim(positions[0], positions[0] + length)
        plt.plot(
            bins,
            viterbi,
            label="{} intervals".format(combined_states[index]),
            color=combined_colors[index],
        )
        plt.ticklabel_format(axis="x", style="plain", useOffset=False)
        plt.tight_layout()
        plt.legend(loc="best", fontsize="xx-small")
        plt.xlabel("Viterbi intervals")

        plt.subplot(313)
        plt.xlim(positions[0], positions[0] + length)
        divergents = divergent_info[combined_states[index]]
        divergents = [d + positions[0] for d in divergents if d < length]
        plt.vlines(divergents, ymin=0, ymax=1, color=combined_colors[index])
        plt.ticklabel_format(axis="x", style="plain", useOffset=False)
        plt.tight_layout()
        plt.xlabel("Divergent sites")

        if index == 3:
            plt.savefig(
                FIGS_DIR
                + "{} {}-{} HC combined posterior probability with first {}kb {}.png".format(
                    chr, positions[0], positions[1], length // 1000, align_index
                )
            )
