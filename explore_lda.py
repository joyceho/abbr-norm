import pandas as pd
import seaborn as sns
import matplotlib
import gensim
from gensim import models as gsm

import pareto
import lda_util as lu
import noodle_util as nu
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("notefile", help="raw notes file")
    parser.add_argument("outfile", help="raw notes file")
    parser.add_argument("--thr", type=float, default=0.74,
                        help="threshold frequency")
    parser.add_argument("--lb", type=int, default=10,
                        help="lower bound on number of topics")
    parser.add_argument("--ub", type=int, default=100,
                        help="upper bound on number of topics")
    parser.add_argument("--step", type=int, default=5,
                        help="step size to explore")

    args = parser.parse_args()
    aboveThr = args.thr
    progNoteDF = pd.read_csv(args.notefile)
    progScoreDF = lu.compute_lda_scores(progNoteDF, 'procNote', args.lb,
                                        args.ub, args.step,
                                        nAbove=aboveThr)
    # find the pareto front, or the points that are not dominated by others.
    aggPF = pareto.eps_sort([list(progScoreDF.itertuples(False))], [1, 6])
    print(pd.DataFrame(aggPF, columns=progScoreDF.columns.get_values()))
    progScoreDF.to_csv(args.outfile, index=False)


if __name__ == "__main__":
    main()