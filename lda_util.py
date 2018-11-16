import gensim
from gensim import corpora
from gensim import models as gsm
import noodle_util as nu
import pandas as pd
from sklearn import model_selection as sksm
import numpy as np
from tqdm import tqdm


def prepare_lda_corpus(noteDF, noteCol, nBelow=4, nAbove=0.8, stopWord=True):
    '''
    Prepare a corpus of notes to apply gensim's LDA on it
    '''
    # go from string -> list for the words
    noteText = nu.tokenize_and_stop(noteDF, noteCol, stopWord)
    # create a mapping from word -> integer
    wordDict = corpora.Dictionary(noteText)
    # filter words that appear in less than 3 documents or more than 80% of the documents
    wordDict.filter_extremes(no_below=nBelow, no_above=nAbove)
    # create a bag of words frequency representation for each note
    corpus = [wordDict.doc2bow(text) for text in noteText]
    return corpus, wordDict, noteText


def compute_lda_scores(noteDF, noteCol, startK, endK, stepK,
                       testSet=0.3, nsamples=10, nThr=3,
                       nBelow=4, nAbove=0.8, stopWord=True):
    '''
    Compute the Coherence and Perplexity of the model using a train/test split
    '''
    scoreDF = []
    for num_topics in tqdm(range(startK, endK, stepK), desc='k-loop'):
        for i in tqdm(range(nsamples), desc='sample-loop', leave=False):
            # first we will split into train/test split
            train, test = sksm.train_test_split(noteDF, test_size=testSet)
            # then prepare the training corpus
            trainCorpus, wordDict, trainText = prepare_lda_corpus(train, noteCol, nBelow, nAbove, stopWord)
            # then prepare the test corpus
            testText = nu.tokenize_and_stop(test, noteCol, stopWord)
            testCorpus = [wordDict.doc2bow(text) for text in testText]
            # build the model
            model = gsm.ldamulticore.LdaMulticore(corpus=trainCorpus, num_topics=num_topics, id2word=wordDict)
            perp = model.log_perplexity(testCorpus)
            coherencemodel = gsm.coherencemodel.CoherenceModel(model=model, texts=testText, dictionary=wordDict)
            scoreDF.append([num_topics, i, perp, coherencemodel.get_coherence()])
    # post-process the frame
    scoreDF = pd.DataFrame(scoreDF, columns=['k', 'i', 'ppx', 'chr'])
    scoreDF.dropna(inplace=True)
    # aggregate the multiple samples into one
    aggScoreDF = scoreDF.groupby('k').agg({'ppx': [np.mean, np.std, 'count'],
                                          'chr': [np.median, np.std]})
    # Flatten the aggregated indexes
    aggScoreDF.columns = pd.Index("_".join(i) for i in aggScoreDF.columns)
    aggScoreDF = aggScoreDF.reset_index()
    aggScoreDF['neg_chr'] = -aggScoreDF['chr_median']
    # drop those without at least 3 samples
    aggScoreDF = aggScoreDF[aggScoreDF['ppx_count'] > nThr]
    return aggScoreDF


def lda_transform(corpus, wordDict, k):
    '''
    Transform a corpus
    '''
    ldaModel = gsm.ldamulticore.LdaMulticore(corpus=corpus, num_topics=k, id2word=wordDict)
    # convert the notes to loadings on the topics
    docTopics = gensim.matutils.corpus2csc(ldaModel[corpus]).T.toarray()
    return docTopics


def train_save(noteDF, noteCol, topicList, outputFormat, aboveThr=0.8):
    corpus, wordDict, noteText = prepare_lda_corpus(noteDF, noteCol, nAbove=aboveThr)
    for k in topicList:
        ldaModel = gsm.ldamulticore.LdaMulticore(corpus=corpus, num_topics=k, id2word=wordDict)
        ldaModel.save(outputFormat.format(k))
