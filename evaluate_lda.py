import argparse
import pandas as pd
from tqdm import tqdm

import noodle_util as nu
import lda_util as ldau


BASE_FEAT_MAP = {'e': ['elix'], 's': ['pol', 'subj'], 'v': ['neg', 'neu', 'pos', 'compound']}


def evaluate_topic_pred(noteDF, noteCol, nTopicList,
                        patCIDF, aboveThr=0.75):
    # prepare the corpus
    corpus, wordDict, noteText = ldau.prepare_lda_corpus(noteDF, noteCol,
                                 nAbove=aboveThr)
    # for each # of topics, evaluate the transform
    noteEvalDF = pd.DataFrame(columns=['k', 'feat', 'auc'])
    for k in tqdm(nTopicList, desc="k-loop", leave=False):
        # perform the transform
        docTopics = ldau.lda_transform(corpus, wordDict, k)
        xFeat = nu.combineTopicSentiment(patCIDF, noteDF, noteCol, docTopics, k)
        tmpDF = nu.eval_diff_features(xFeat, {'t': ["topic_" + str(i) for i in range(k)],
                                              **BASE_FEAT_MAP})
        tmpDF['k'] = k
        noteEvalDF = noteEvalDF.append(tmpDF, ignore_index=True, sort=True)
    return noteEvalDF


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("notefile", help="raw notes file")
    parser.add_argument("elixfile", help="elixhauser file")
    parser.add_argument("outfile", help="raw notes file")
    parser.add_argument("-k", type=int, nargs='+', help="number of topics", required=True)
    parser.add_argument("--thr", type=float, default=0.74,
                        help="threshold frequency")

    args = parser.parse_args()

    aboveThr = args.thr
    noteEvalDF = pd.DataFrame(columns=['txtCol', 'k', 'feat', 'auc'])
    patCIDF = pd.read_csv(args.elixfile)
    noteDF = pd.read_csv(args.notefile)

    # read the data frame
    for txtCol in ['note', 'procNote']:
        tmpDF = evaluate_topic_pred(noteDF, txtCol, args.k, patCIDF, aboveThr)
        tmpDF['txtCol'] = txtCol
        noteEvalDF = noteEvalDF.append(tmpDF, ignore_index=True, sort=True)

    noteEvalDF.to_csv(args.outfile)


if __name__ == "__main__":
    main()
