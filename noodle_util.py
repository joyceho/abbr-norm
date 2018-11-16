import pandas as pd
import nltk
import nltk.sentiment.vader as vader
import gensim
from gensim import corpora
from gensim import models as gsm
from textblob import TextBlob
import itertools

import numpy as np
from sklearn import linear_model as sklm
from sklearn import model_selection as skms


def merge_notes(x):
    '''
    Merge the notes by joining the strings
    '''
    return ','.join(str(v) for v in x)
    

def compress_measDF(measDF, grpByList, measCol, grpFunc):
    '''
    Compress a dataframe using the columns in the grpByList.
    Applies the grouping function to the measurement column.
    '''
    aggMap = {}
    for meas in measCol:
        aggMap[meas] = grpFunc
    encMeasDF = measDF.groupby(grpByList).agg(aggMap)
    return pd.DataFrame(encMeasDF).reset_index()


def create_cohort_measDF(patDF, measFile, measCol, encGrpFunc):
    '''
    Create a measurement data frame based on the defined cohort.
    '''
    measDF = pd.read_csv(measFile)
    # merge the two data frames
    patMeasDF = pd.merge(patDF, measDF, on=['pat_id', 'enc_id'])
    # drop any duplicates
    patMeasDF.drop_duplicates(inplace=True)
    # group them together using a defined function
    return compress_measDF(patMeasDF, ['pat_id', 'enc_id', 'label'],
                           measCol, encGrpFunc)


def tokenize_and_stop(noteDF, noteCol, stopWord=True):
    noteText = noteDF[noteCol].apply(lambda x: nltk.tokenize.word_tokenize(x))
    if stopWord:
        # remove stopwords from each set
        enStopWrd = set(nltk.corpus.stopwords.words('english'))
        noteText = noteText.apply(lambda x: [w for w in x if not w in enStopWrd])
        noteText = noteText.apply(lambda x: [w for w in x if w.isalpha()])
    return noteText 


def combineTopicSentiment(patDF, noteDF, noteCol, docTopics, nTopic):
    # standard sentiment analysis using textblob
    sid = vader.SentimentIntensityAnalyzer()
    return pd.concat([patDF,
                      pd.DataFrame(docTopics,
                                   columns=["topic_" + str(i) for i in range(nTopic)]), 
                      noteDF[noteCol].apply(analyze_sentiment),
                      noteDF[noteCol].apply(lambda x: analyze_vader_sent(x, sid))], axis=1)


# Evaluate the model using these columns
def eval_logr(df, cols, l1=False, rseed=10, nsplit=5, max_iter=300):
    if l1:
        logR = sklm.LogisticRegressionCV(penalty='l1',
                                         solver='liblinear', cv=5,
                                         max_iter=max_iter)
    else:
        logR = sklm.LogisticRegressionCV(penalty='l2', cv=5,
                                         max_iter=max_iter)
    return skms.cross_val_score(logR, df[cols],
                                df['label'], scoring='roc_auc',
                                cv=skms.StratifiedKFold(n_splits=nsplit, 
                                                        random_state=rseed))

# Evaluate different features based on the combinations
def eval_diff_features(totalDF, featureSets, l1=False):
    featEvalDF = pd.DataFrame(columns=["feat", "auc"])
    ft = sorted(featureSets.keys())
    for i in range(1, len(ft)+1):
        # get the possible combinations of features to concatenate
        for ftTuple in list(itertools.combinations(ft, i)):
            ftList = [featureSets[x] for x in ftTuple]
            aucArr = np.array(eval_logr(totalDF,
                                        list(itertools.chain(*ftList)),
                                        l1)).T
            aucDF = pd.DataFrame(aucArr, columns=['auc'])
            aucDF['feat'] = '+'.join(ftTuple)
            featEvalDF = featEvalDF.append(aucDF, sort=True)
    return featEvalDF


def analyze_sentiment(note):
    blob = TextBlob(note)
    return pd.Series({"pol": blob.sentiment.polarity,
                      "subj": blob.sentiment.subjectivity})

def analyze_vader_sent(note, sid):
    return pd.Series(sid.polarity_scores(note))

