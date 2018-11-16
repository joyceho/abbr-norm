# Abbreviation Normalization

Python script to normalize abbreviation

## Requirements
A pre-requisite is to have Python 3 installed. The easiest way is via the [Anaconda distribution](https://www.anaconda.com/download/). This module also assumes the following packages have been installed:
* scrapy to extract structured data from websites
* nltk (installed with Anaconda)
* numpy (installed with Anaconda)
* pandas (installed with Anaconda)
* TextBlob
* scikit-learn (installed with Anaconda)
* seaborn (installed with Anaconda)
* gensim (`conda install -c conda-forge gensim`)
* tqdm 

### Step 1: Scrape Medical Abbreviations
1. Scrape [nurselab](https://nurseslabs.com/medical-terminologies-abbreviations-listcheat-sheet) and 
[Tabers Dictionary](https://www.tabers.com/tabersonline/view/Tabers-Dictionary/767492/all/Medical_Abbreviations)
```
scrapy runspider scrape_nurselab_abbr.py -o data/nurselab_abbr.json
scrapy runspider scrape_tabers_abbr.py -o data/tabers_abbr.json
```
2. Setup the abbreviation lookup dictionary
```
python format_nursing_abbr.py data/nurse_abbr.json -i data/tabers_abbr.json data/nurselab_abbr.json
```


### Step 2: Clean the notes

Python script to replace abbreviations and to lemmatize the words using TextBlob and NLTK. This script assumes a comma separated file with all the notes in the column named 'note'. It will create a new column named 'ppNote' with just the abbreviation replacements and 'procNote' that contains both abbreviation normalization and lemmatization.
```
python ppNote.py data/<notefile>.csv data/clean_<notefile>.csv
```

### Step 3: Create the Elixhauser comorbidity score 
Create the Elixhauser score from a list of ICD9. Assumes there is a CSV file that contains the column icd9, where the entry in that column contains a list of ICD9 codes. An example is the entry ['250.00', '305.1', '041.85']. Creates a new column 'elix' that contains the score. 
```
python comorb_score.py data/<icd9file>.csv data/<elix_out>.csv
```

### Step 4: Find the Pareto Optimal Points for the two types of topic models
Explore the impact of the topics and finding 'optimal topics'
```
python explore_lda.py data/clean_<notefile>.csv data/<lda_k>.csv
python explore_doc2vec.py data/clean_<notefile>.csv data/<d2v_k>.csv
```

### Step 5: Evaluate the impact of topics + sentiment analysis

Evaluate the use of topics + sentiment analysis on a specific prediction task. A sequence of integers after the argument -k are the number of topics to try. The prediction task assumes that the labels are in the file specified as elix_out and under the column 'label'.
```
python evaluate_lda.py data/clean_<notefile>.csv data/<elix_out>.csv <resultfile>.csv -k 10 20
python evaluate_doc2vec.py data/clean_<notefile>.csv data/<elix_out>.csv <resultfile>.csv -k 10 20
```
