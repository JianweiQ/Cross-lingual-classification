from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import re
import io
from collections import defaultdict
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


def fetch20newsgroup(cat):
    """ param: cat, a list of categories
    return: a list of strings"""
    print('fetch20newsgroup...')
    news = fetch_20newsgroups(categories=cat)
    return news


def fetchTHUCTC(folder):
    """param: folder of the news documents
    return: a list of strings and a list of labels (0,1,2)"""
    print('fetchTHUCTC...')
    #TODO
    return

def tokenizeEnglish(news):
    """param: news, a list of strings
    return: a list of lists of tokens (words)"""
    print('tokenizeEnglish...')

    class LemmaTokenizer(object):  # for stemming and lemmatizing
        def __init__(self):
            self.wnl = WordNetLemmatizer()

        def __call__(self, doc):
            tokenizer = RegexpTokenizer(r'(?u)\b\w\w+\b')  # remove punctuations
            doc = re.sub(r'\S*@\S*\s?', '', doc.lower())  # to lower case, remove emails
            doc = ''.join([i for i in doc if not i.isdigit()])  # remove digits/numbers
            return [self.wnl.lemmatize(t) for t in tokenizer.tokenize(doc)]

    # vectorizer = TfidfVectorizer(stop_words='english', max_df=0.3, min_df=5, tokenizer=LemmaTokenizer())
    # X = vectorizer.fit_transform(englishNews.data)
    # y = englishNews.target
    tokenizer = LemmaTokenizer()
    docs = [tokenizer(doc) for doc in news.data]
    return docs


def tokenizeChinese(news):
    """param: news, a list of strings
    return: a list of lists of tokens (words)"""
    print('tokenizeChinese...')
    #TODO


def featurizeChinese(docs, w2v):
    """param: docs, a list of lists of tokens (strings)
    return: a matrix, a row represents a doc's feature vector"""
    print('featurizeChinese...')
    #TODO


def loadWordVectors(fname):
    """param: fname, file name of word embeddings
    return: a dictionary storing the embeddings"""
    print('loadWordVectors...')
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    print(n,d) # n is #words, d is dimension
    #fin.readline()
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data


def featurizeEnglish(docs, w2v):
    """param: docs, a list of lists of tokens (strings)
    return: a matrix, a row represents a doc's feature vector"""
    print('featurizeEnglish...')

    class TfidfEmbeddingVectorizer(object):
        def __init__(self, word2vec, dim):
            self.word2vec = word2vec
            self.word2weight = None
            self.dim = dim

        def fit(self, X):
            """X is tokenized docs, list[list[string]]"""
            tfidf = TfidfVectorizer(analyzer=lambda x: x, stop_words='english', max_df=0.3, min_df=5)
            tfidf.fit(X)
            # if a word was never seen - it must be at least as infrequent
            # as any of the known words - so the default idf is the max of
            # known idf's
            max_idf = max(tfidf.idf_)
            self.word2weight = defaultdict(
                lambda: max_idf,
                [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

            return self

        def transform(self, X):
            """X is tokenized docs, list[list[string]]"""
            return np.array([
                                np.mean([self.word2vec[w] * self.word2weight[w]
                                         for w in words if w in self.word2vec] or
                                        [np.zeros(self.dim)], axis=0)
                                for words in X
                                ])

    vectorizer = TfidfEmbeddingVectorizer(w2v, 300)
    vectorizer.fit(docs)
    X = vectorizer.transform(docs)
    return X


def classifyMonolingual(X, y):
    """ classify documents with different classifiers"""
    print('classifyMonolingual...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    print("BernoulliNB:")
    clf = BernoulliNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("accuracy =",accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    # use this model to predict

    print("LinearSVC:")
    clf = LinearSVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("accuracy =",accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    print("ExtraTreesClassifier:")
    clf = ExtraTreesClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("accuracy =",accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    print("LogisticRegression:")
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("accuracy =",accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    print("MultinomialNB:")
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("accuracy =",accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))


def classifyCrosslingual(X, y, X2, y2):
    """param: X, y are feature vectors and targets of one language
    param: X2, y2 are feature vectors and targets of another language"""
    print("BernoulliNB:")
    clf = BernoulliNB()
    clf.fit(X, y) # fit the model on language 1
    y2_pred = clf.predict(X2) # use the model to predict on language 2
    print("accuracy =",accuracy_score(y2, y2_pred))
    print(confusion_matrix(y2, y2_pred))


    print("LinearSVC:")
    clf = LinearSVC()
    clf.fit(X, y) # fit the model on language 1
    y2_pred = clf.predict(X2) # use the model to predict on language 2
    print("accuracy =",accuracy_score(y2, y2_pred))
    print(confusion_matrix(y2, y2_pred))

    print("ExtraTreesClassifier:")
    clf = ExtraTreesClassifier()
    clf.fit(X, y) # fit the model on language 1
    y2_pred = clf.predict(X2) # use the model to predict on language 2
    print("accuracy =",accuracy_score(y2, y2_pred))
    print(confusion_matrix(y2, y2_pred))

    print("LogisticRegression:")
    clf = LogisticRegression()
    clf.fit(X, y) # fit the model on language 1
    y2_pred = clf.predict(X2) # use the model to predict on language 2
    print("accuracy =",accuracy_score(y2, y2_pred))
    print(confusion_matrix(y2, y2_pred))

    print("MultinomialNB:")
    clf = MultinomialNB()
    clf.fit(X, y) # fit the model on language 1
    y2_pred = clf.predict(X2) # use the model to predict on language 2
    print("accuracy =",accuracy_score(y2, y2_pred))
    print(confusion_matrix(y2, y2_pred))


if __name__ == '__main__':
    # apply word2vec to featurize, train and test on English corpus
    cat = ['rec.sport.baseball', 'talk.politics.misc', 'sci.electronics']  # 'comp.graphics'
    englishNews = fetch20newsgroup(cat)
    w2v_en = loadWordVectors('wiki.en.align.vec')
    X = featurizeEnglish(tokenizeEnglish(englishNews), w2v_en)
    y = englishNews.target
    classifyMonolingual(X,y)

    # apply word2vec to featurize, train and test on Chinese corpus
    if False:
        chineseNews, y2 = fetchTHUCTC('THUCNews')
        w2v_zh = loadWordVectors('wiki.zh.align.vec')
        X2 = featurizeChinese(tokenizeChinese(chineseNews), w2v_en)
        classifyMonolingual(X2, y2)

        # train on English corpus but test on Chinese corpus
        print('\n\ntrain on English corpus but test on Chinese corpus\n')
        classifyCrosslingual(X, y, X2, y2)
        # train on Chinese corpus but test on English corpus
        print('\n\ntrain on Chinese corpus but test on English corpus\n')
        classifyCrosslingual(X2, y2, X, y)
