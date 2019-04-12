import numpy as np
import re
import io
from collections import defaultdict
from sklearn.datasets import fetch_20newsgroups
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# import nltk
# nltk.download('wordnet')


def main():
    # English w2v
    cat = ['rec.sport.baseball', 'talk.politics.misc', 'sci.electronics'] 
    englishNews = fetch20newsgroup(cat, "all")
    englishWords = tokenizeEnglish(englishNews)
    w2v_en = loadWordVectors('data/wiki.en.align.vec', 10000)  # Yixin: can be slow, try 10000 first
    X = featurizeEnglish(englishWords, w2v_en)
    y = englishNews.target
    
    # English classifiers
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    classifiers(X_train, y_train, X_test, y_test)
    
    # Chinese w2v


def fetch20newsgroup(cat, subset):
    """ param: cat, a list of categories
    return: a list of strings"""
    print('fetch20newsgroup...')
    # if dataset.target[i] == :  # 07 auto 08 motorcycles
    # elif dataset.target[i] >= 11 and dataset.target[i] <= 14:  # science
    # elif dataset.target[i] == 9 or dataset.target[i] == 10:  # sport
    dataset = fetch_20newsgroups(subset=subset, categories=cat, shuffle=True, random_state=10,
                         remove=('headers', 'footers', 'quotes'))
    return dataset


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


def loadWordVectors(fname, size=float('inf')):
    """param: fname, file name of word embeddings
    size: for max loaded lines restriction
    return: a dictionary storing the embeddings"""
    print('loadWordVectors...')
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    print(n, d)  # n is #words, d is dimension
    # fin.readline()
    data = {}
    cur = 0
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.asarray(list(map(float, tokens[1:])))  # Yixin: added np.asarray(list())
        if (cur > size): break 
        else: cur += 1
    return data


def featurizeEnglish(docs, w2v):
    """param: docs, a list of lists of tokens (strings)
    w2v, a dictionary of word embeddings
    return: a matrix, a row represents a doc's feature vector"""
    print('featurizeEnglish...')
    vectorizer = TfidfEmbeddingVectorizer(w2v, 300)
    vectorizer.fit(docs)
    X = vectorizer.transform(docs)
    return X


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


class MeanEmbeddingVectorizer(object):

    def __init__(self, word2vec, dim=300):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
#         self.dim = len(word2vec.itervalues().next())
        self.dim = dim

    def fit(self, X):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


def classifiers(X_train, y_train, X_test, y_test):
    """param: X_train, y_train are feature vectors and targets of one language
    param: X_test, y_test are feature vectors and targets of another language"""

    def classifier(clf, clf_name):
        print(clf_name + ":")
        clf.fit(X_train, y_train)
        y2_pred = clf.predict(X_test)
        print("accuracy =", accuracy_score(y_test, y2_pred))
        print(confusion_matrix(y_test, y2_pred))
        
    classifier(LinearSVC(), "LinearSVC")
    classifier(ExtraTreesClassifier(), "ExtraTreesClassifier")
    classifier(LogisticRegression(), "LogisticRegression")
#     classifier(BernoulliNB(), "BernoulliNB")
#     classifier(MultinomialNB(), "MultinomialNB")


if __name__ == '__main__':
    main()
