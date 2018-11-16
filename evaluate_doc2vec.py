import argparse
import pandas as pd
from tqdm import tqdm

import noodle_util as nu
import doc2vec_util as dvu


BASE_FEAT_MAP = {'e': ['elix'], 's': ['pol', 'subj'], 'v': ['neg', 'neu', 'pos', 'compound']}


def evaluate_doc2vec_pred(noteDF, noteCol, nTopicList,
                          patCIDF, epochs):

    freqDist = dvu.tokenize_and_count(noteDF, noteCol)
    tagNotes = dvu.tagNotes(noteDF, noteCol, freqDist)
    # for each # of word representation, evaluate the transform
    noteEvalDF = pd.DataFrame(columns=['k', 'feat', 'auc'])
    for k in tqdm(nTopicList, desc='k-loop', leave=False):
        # perform the transform
        docTopics = dvu.doc2vec_transform(k, tagNotes, tagNotes, epo=epochs)
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
    parser.add_argument("--epo", type=int, default=3000,
                        help="threshold frequency")

    args = parser.parse_args()

    noteEvalDF = pd.DataFrame(columns=['txtCol', 'k', 'feat', 'auc'])
    patCIDF = pd.read_csv(args.elixfile)
    noteDF = pd.read_csv(args.notefile)

    # read the data frame
    for txtCol in ['note', 'procNote']:
        tmpDF = evaluate_doc2vec_pred(noteDF, txtCol, args.k, patCIDF, args.epo)
        tmpDF['txtCol'] = txtCol
        noteEvalDF = noteEvalDF.append(tmpDF, ignore_index=True, sort=True)

    noteEvalDF.to_csv(args.outfile)


if __name__ == "__main__":
    main()
