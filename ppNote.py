import argparse
import pandas as pd
from gensim.parsing import preprocessing
from textblob import TextBlob, Word
import re
import json

def _replace_abbr(x, abbrDict):
    '''
    Replace an abbreviation using the dictionary
    Assumes x is a single word
    '''
    # check the word directly
    if x in abbrDict:
        return abbrDict[x]
    # check if it's ended by a period (end of sentence)
    if x.rstrip(".") in abbrDict:
        return abbrDict[x.rstrip(".")]
    if x.rstrip("'d") in abbrDict:
        return abbrDict[x.rstrip("'d")]
    return x


def clean_abbreviations(note, abbrDict, wordSep=' |,'):
    '''
    Given a note (collection of words) and abbreviation dictionary,
    replace any abbreviation with the qualified name.
    '''
    return ' '.join([_replace_abbr(word, abbrDict) for word in re.split(wordSep, note)])


def _word_std(word, pos):
    '''
    Standardize a word by verb tense and plural.
    '''
    if word.endswith("ing") or word.endswith("ed"):
        return word.lemmatize('v')
    if "NN" in pos:
        return word.lemmatize()
    elif "VB" in pos:
        return word.lemmatize('v')
    return word


def standardize_note(note, abbrDict):
    '''
    Standardize a note (collection of words).
    Assumes a first pass of abbreviation approximation
    has been performed.
    '''
    noteLw = note.lower()
    noteLw = noteLw.replace("'s", "")
    blob = TextBlob(noteLw)
    lematBlob = []  
    for k, (word, pos) in enumerate(blob.tags):
        # clean the word
        word = word.strip("/")
        word = word.strip(".")
        # check once more to see if there is something
        word = Word(_replace_abbr(word, abbrDict))
        if "." in word:
            splitMore = word.split('.')
            for spWord in splitMore:
                fixedWord = _replace_abbr(spWord, abbrDict)
                for w in fixedWord.split():
                    lematBlob.append(_word_std(Word(w), pos))
        else:
            for w in word.split():
                lematBlob.append(_word_std(Word(w), pos))
    return " ".join(filter(lambda x: x != None, lematBlob))


def clean_raw_notes(noteFile):
    patNoteDF = pd.read_csv(noteFile)
    patNoteDF['note'] = patNoteDF['note'].apply(lambda x: str(x).replace("[**MASKED PHI**]", ""))
    # clean up the whitespaces so that there are no multiple whitespaces
    patNoteDF['note'] = patNoteDF['note'].apply(lambda x: preprocessing.strip_multiple_whitespaces(x))
    return patNoteDF


def find_orig_diff(origText, modText, wordSep=' |,'):
    origTokens = set(re.split(wordSep, origText))
    modTokens = set(re.split(wordSep, modText))
    return list(origTokens.difference(modTokens))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("notefile", help="raw notes file")
    parser.add_argument("outfile", help="filename for cleaned notes")
    parser.add_argument("--abbrDict", default="data/nurse_abbr.json",
                         help="abbreviation dictionary to use")
    args = parser.parse_args()

    abbrDict = json.load(open(args.abbrDict, 'r'))
    cleanNotes = clean_raw_notes(args.notefile)
    cleanNotes['ppNote'] = cleanNotes['note'].apply(clean_abbreviations, abbrDict=abbrDict)
    cleanNotes['procNote'] = cleanNotes['ppNote'].apply(standardize_note, abbrDict=abbrDict)
    # clean the patients without any notes
    cleanNotes = cleanNotes[cleanNotes['procNote'] != '']
    cleanNotes['abbr'] = cleanNotes.apply(lambda row: find_orig_diff(row['note'], row['ppNote']), axis=1)
    cleanNotes.to_csv(args.outfile, index=False)


if __name__ == "__main__":
    main()
