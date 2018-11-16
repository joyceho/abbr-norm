import collections
from gensim.models import doc2vec as dv
from tqdm import trange
import pandas as pd
import numpy as np
import noodle_util as nu



def tokenize_and_count(noteDF, noteCol):
    noteText = nu.tokenize_and_stop(noteDF, noteCol)
    encCount = collections.Counter()
    for note in noteText:
        for word in set(note):
            encCount[word.strip()] += 1
    return encCount


def _check_word(word, freqDist, low, high):
    if word in freqDist:
        if freqDist[word] >= low and freqDist[word] <= high:
            return True 
    return False


def tagNotes(noteDF, noteCol, freqDist, low=4, highPer=0.74):
    highCount = highPer * noteDF.shape[0]
    taggedNotes = []
    for i, note in enumerate(nu.tokenize_and_stop(noteDF, noteCol)):
        clean_note = list(filter(lambda x: _check_word(x, freqDist, low, highCount),
                                 note))
        taggedNotes.append(dv.TaggedDocument(clean_note, [i]))
    return taggedNotes


def compute_rank_score(tagNotes, start, limit, step, 
                       minCount=4, epochs=1000):
    rankScoreMap = {}
    for vecSize in trange(start, limit, step):
        model = dv.Doc2Vec(vector_size=vecSize,
                           minCount=minCount,
                           epochs=epochs)
        model.build_vocab(tagNotes)
        ranks = []
        for doc_id in range(len(tagNotes)):
            inferred_vector = model.infer_vector(tagNotes[doc_id].words)
            sims = model.docvecs.most_similar([inferred_vector],
                                              topn=len(model.docvecs))
            rank = [docid for docid, sim in sims].index(doc_id)
            ranks.append(rank)
        rankScoreMap[vecSize] = collections.Counter(ranks)
    return rankScoreMap


def construct_top_earmarks(rankScoreMap, topKList, topX=100):
    # fill out the top 100
    topXEval = dict(zip(rankScoreMap.keys(), [{k:v for k, v in x.items() if k < topX} for x in rankScoreMap.values()]))
    topXEvalDF = pd.DataFrame(topXEval)
    topXEvalDF = topXEvalDF.fillna(0)
    tmpDF = []
    # then aggregate based on the groupings
    for k in topKList:
        summaryDF = topXEvalDF.head(k).agg(['mean', np.median, 'sum'])
        summaryDF['k'] = k
        tmpDF.append(summaryDF.reset_index())
    totalKEvalDF = pd.concat(tmpDF, sort=True, ignore_index=True)
    totalKEvalDF.sort_values(by=['index'], inplace=True)
    return totalKEvalDF


def doc2vec_transform(k, trainTagList, xTagList, epo=100):
    model = dv.Doc2Vec(vector_size=k, minCount=4, epochs=epo)
    model.build_vocab(trainTagList)
    docCorpusList = [model.infer_vector(xTagList[doc_id].words) for doc_id in range(len(xTagList))]
    docCorpusArr = np.vstack(docCorpusList)
    return docCorpusArr
