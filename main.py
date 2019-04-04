import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from gensim.models import FastText
from gensim.models import Word2Vec


class MeanEmbeddingVectorizer(object):

    def __init__(self, word2vec, dim=100):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
#         self.dim = len(word2vec.itervalues().next())
        self.dim = dim

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


def tokens(doc):
    tokens = doc.rstrip().split(' ')
    return tokens


def fetch_20newsgroups_sub(subset, size=float('inf')):
    dataset = fetch_20newsgroups(subset=subset, shuffle=True, random_state=10,
                         remove=('headers', 'footers', 'quotes'))
    X = []
    y = []
    for i in range(min(size, len(dataset.target))):
        if dataset.target[i] == 7:  # auto 08 motorcycles
            X.append(tokens(dataset.data[i]))
            y.append(1)
        elif dataset.target[i] >= 11 and dataset.target[i] <= 14:  # science
            X.append(tokens(dataset.data[i]))
            y.append(2)
        elif dataset.target[i] == 9 or dataset.target[i] == 10:  # sport
            X.append(tokens(dataset.data[i]))
            y.append(3)
    return (X, y)


def main():
    # English word Embedding
    # TODO: Stop words, Stem
    X, y = fetch_20newsgroups_sub('train')
#     model = FastText(sentences=X, size=100, sg=1, min_count=2, window=5, negative=5)
    dim = 100
    model = Word2Vec(sentences=X, size=dim)
    
    # TODO: Chinese word Embedding
    
    # Classification
    w2v = dict(zip(model.wv.index2word, model.wv.syn0))
    svm_w2v = Pipeline([
    ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v, dim)),
    ("SVM", SVC())])
    svm_w2v.fit(X, y)
    
    # Test
    X1, y1 = fetch_20newsgroups_sub('test')
    y_hat = svm_w2v.predict(X1)
    
    hit = 0
    miss = 0
    for i in range(len(y_hat)):
        if y_hat[i] == y1[i]:
            hit += 1
        else:
            miss += 1
    print(miss, hit)


if __name__ == '__main__':
    main()
