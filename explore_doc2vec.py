import pandas as pd
import seaborn as sns
import matplotlib
import gensim
from gensim import models as gsm

import pareto
import doc2vec_util as dvu
import noodle_util as nu
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("notefile", help="raw notes file")
    parser.add_argument("outfile", help="scoring output file")
    parser.add_argument("--lb", type=int, default=10,
                        help="lower bound on number of topics")
    parser.add_argument("--ub", type=int, default=100,
                        help="upper bound on number of topics")
    parser.add_argument("--step", type=int, default=5,
                        help="step size to explore")

    args = parser.parse_args()
    progNoteDF = pd.read_csv(args.notefile)
    freqDist = dvu.tokenize_and_count(progNoteDF, 'procNote')
    # Preprocess the notes to tag each text
    tagNotes = dvu.tagNotes(progNoteDF, 'procNote', freqDist)
    progRankScoreMap = dvu.compute_rank_score(tagNotes, args.lb,
                                              args.ub, args.step)
    scoreDF = dvu.construct_top_earmarks(progRankScoreMap, [10, 25, 50, 100])
    scoreDF.to_csv(args.outfile, index=False)


if __name__ == "__main__":
    main()
