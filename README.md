# Cross-lingual-classification
Cross-lingual Embeddings and Text Classification

## Packages
nltk, sklearn, thulac, numpy, keras

And run following python code:
```
nltk.download('wordnet')
```

## How to run
Run main.py after following 'Inputs' or 'Middle outputs' are prepared.

## Inputs
Inputs for data pre-processing:

| File Name  | Folder | Description |
| ------------- | ------------- | ------------- |
| wiki.en.align.vec  | ./data/  | English aligned word embedding [FastText](https://fasttext.cc/docs/en/aligned-vectors.html)|
| wiki.zh.align.vec  | ./data/  | Chinese aligned word embedding [FastText](https://fasttext.cc/docs/en/aligned-vectors.html)|
| ... | ./data/sport  | Chinese news corpus [THUCTC](http://thuctc.thunlp.org/)|
| ... | ./data/politics  | Chinese news corpus [THUCTC](http://thuctc.thunlp.org/)|
| ... | ./data/science  | Chinese news corpus [THUCTC](http://thuctc.thunlp.org/)|
| ... | ./data/UM-Corpus  | Chinese news corpus [UM-corpus](http://nlp2ct.cis.umac.mo/um-corpus/)|
| chinese.train.1000 | ./data/  | Chinese training corpus [Reuters Corpora](https://trec.nist.gov/data/reuters/reuters.html)|
| chinese.dev | ./data/  | Chinese validation corpus [Reuters Corpora](https://trec.nist.gov/data/reuters/reuters.html)|
| chinese.test | ./data/  | Chinese testing corpus [Reuters Corpora](https://trec.nist.gov/data/reuters/reuters.html)|
| english.train.1000 | ./data/  | English training corpus [Reuters Corpora](https://trec.nist.gov/data/reuters/reuters.html)|
| english.dev | ./data/  | English validation corpus [Reuters Corpora](https://trec.nist.gov/data/reuters/reuters.html)|
| english.test | ./data/  | English testing corpus [Reuters Corpora](https://trec.nist.gov/data/reuters/reuters.html)|
| stopwords-zh.txt | ./data/  | Chinese stop words|

## Middle outputs
With middle outputs prepared, one can comment:

==========================Data preparation==========================

in main.py, and jump to:

==========================Model training==========================

directly.

| File Name  | Folder | Description |
| ------------- | ------------- | ------------- |
| X_c3  | ./mid/  | Chinese corpus after data pre-processing |
| y_c3  | ./mid/  | Chinese label after data pre-processing |
| X_e3  | ./mid/  | English corpus after data pre-processing |
| y_e3  | ./mid/  | English label after data pre-processing |

## Outputs
Outputs for running main.py:

| File Name  | Folder | Description |
| ------------- | ------------- | ------------- |
| .log  | ./output/  | Logs |
| .png  | ./output/  | Charts |
