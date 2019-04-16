import numpy as np
import thulac
import re
import io
import os
from collections import defaultdict
from sklearn.datasets import fetch_20newsgroups
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


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


def relabel20newsgroup(target):
    y = []
    for t in target:
        if t == 0 or t == 1:
            y.append(0)
        elif t == 2 or t == 3 or t == 4:
            y.append(1)
        else:
            y.append(2)
    return y


def fetchTHUnews(size=float("inf")):

    def read_files(file_name, target, X, y):
        """param: file_name, string type file directory
        param: target, int for target label
        param: X to append
        param: y to append"""
        cur = 1
        for file in os.listdir(file_name):
            file = open(file_name + "/" + file, 'r')
            X.append(file.read())
            y.append(target)
            if cur >= size: break
            else: cur += 1

    print('fetchTHUnews...')
    X = []
    y = []
    read_files("data/sport_out", 0, X, y)
    read_files("data/politics_out", 1, X, y)
    read_files("data/science_out", 2, X, y)
    return (X, y)


def THUWordSegmentation():
    """One time function to separate Chinese by space and save"""
    print('THUWordSegmentation...')
    thu = thulac.thulac(seg_only=True)  # segment only
    # sport 0-131603.txt
    # politics 339764-402849.txt
    # science 481650-644578.txt
    for i in range(50000, 60000):
        if (i / 10) == int(i / 10): print(i)
        filename = str(i)
        # TODO pick 1000 files only for speed
        thu.cut_f("data/sport/" + filename + ".txt", "data/sport_out/" + filename + ".txt")
        thu.cut_f("data/politics/3" + filename + ".txt", "data/politics_out/3" + filename + ".txt")
        thu.cut_f("data/science/5" + filename + ".txt", "data/science_out/5" + filename + ".txt")


def fetchUMCorpus():

    def read_files(file_name, target, X, y, is_token):
        """param: file_name, string type file name
        param: target, int for target label
        param: X to append
        param: y to append"""
        file = open(file_name, 'r')
        for line in file:
            if is_token == 1:
                l = line.split()
            else:
                l = line
            X.append(l)
            y.append(target)
            
    print('fetchUMCorpus...')
    X_en, y_en, X_zh, y_zh = [], [], [], []
    read_files("data/UM-Corpus/en/Bi-Laws.txt", 0, X_en, y_en, 0)
    read_files("data/UM-Corpus/en/Bi-Thesis.txt", 1, X_en, y_en, 0)
    read_files("data/UM-Corpus/zh_s/Bi-Laws.txt", 0, X_zh, y_zh, 1)
    read_files("data/UM-Corpus/zh_s/Bi-Thesis.txt", 1, X_zh, y_zh, 1)
    return (X_en, y_en, X_zh, y_zh)


def UMWordSegmentation(filenames, size=float("inf")):
    """param: filenames, filename to deal with
    param: size, max size of lines to perform
    One time function to separate English and Chinese,
    and separate Chinese by space and save"""
    print('UMWordSegmentation...')
    # separate English and Chinese 
    for filename in filenames:
        # Bi-Laws 440000 Bi-Thesis 600000
        input_file = "data/UM-Corpus/" + filename + ".txt"
        output_file_en = "data/UM-Corpus/en/" + filename + ".txt"
        output_file_zh = "data/UM-Corpus/zh/" + filename + ".txt"
        i_file = open(input_file, 'r')
        o_file_en = open(output_file_en, 'a')
        o_file_zh = open(output_file_zh, 'a')
        cur = 0
        for line in i_file:
            if (cur / 2) == int(cur / 2):  # even for English
                o_file_en.write(line)
            else:
                o_file_zh.write(line)
            if cur >= size: break
            else: cur += 1
        o_file_en.close()
        o_file_zh.close()
    # separate Chinese words
    thu = thulac.thulac(seg_only=True)  # segment only
    for file in os.listdir("data/UM-Corpus/zh"):
        thu.cut_f("data/UM-Corpus/zh/" + file, "data/UM-Corpus/zh_s/" + file)


def tokenizeEnglish(docs):
    """param: docs, a list of strings
    return: a list of lists of tokens (words)"""

    class LemmaTokenizer(object):  # for stemming and lemmatizing

        def __init__(self):
            self.wnl = WordNetLemmatizer()

        def __call__(self, doc):
            tokenizer = RegexpTokenizer(r'(?u)\b\w\w+\b')  # remove punctuations
            doc = re.sub(r'\S*@\S*\s?', '', doc.lower())  # to lower case, remove emails
            doc = ''.join([i for i in doc if not i.isdigit()])  # remove digits/numbers
            return [self.wnl.lemmatize(t) for t in tokenizer.tokenize(doc)]
    
    print('tokenizeEnglish...')
    # vectorizer = TfidfVectorizer(stop_words='english', max_df=0.3, min_df=5, tokenizer=LemmaTokenizer())
    # X = vectorizer.fit_transform(englishNews.data)
    # y = englishNews.target
    tokenizer = LemmaTokenizer()
    ret = [tokenizer(doc) for doc in docs]
    return ret


def tokenizeChinese(docs):
    """param: docs, a list of lists of tokens (strings)
    return: docs with extreme frequence, stop words, punctuation removed"""
    print('tokenizeChinese...')
    file = open("data/stopwords-zh.txt", 'r')
    stop_words_chinese = file.read().splitlines()
    tfidf = TfidfVectorizer(stop_words=stop_words_chinese, max_df=0.7, min_df=3, lowercase=False, token_pattern=r'(?u)\b\w+\b')
    weight = tfidf.fit_transform(docs).toarray()
    word = tfidf.get_feature_names()
    ret = []
    for i in range(len(weight)):
        doc = ""
        for j in range(len(word)):
            if weight[i][j] > 0:
                doc = doc + " " + word[j]
        ret.append(doc)
    return ret


def featurizeEnglish(docs, w2v):
    """param: docs, a list of lists of tokens (strings)
    w2v, a dictionary of word embeddings
    return: a matrix, a row represents a doc's feature vector"""

    class TfidfEmbeddingVectorizer(object):

        def __init__(self, word2vec, dim):
            self.word2vec = word2vec
            self.word2weight = None
            self.dim = dim

        def fit(self, X):
            """X is tokenized docs, list[list[string]]"""
            tfidf = TfidfVectorizer(analyzer=lambda x: x, stop_words='english', max_df=0.8, min_df=2)
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

    print('featurizeEnglish...')
    vectorizer = TfidfEmbeddingVectorizer(w2v, 300)
    vectorizer.fit(docs)
    X = vectorizer.transform(docs)
    return X


def featurizeChinese(docs, w2v):
    """param: docs, a list of lists of tokens (strings)
    w2v, a dictionary of word embeddings
    return: a matrix, a row represents a doc's feature vector"""
    
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
            hit = 0
            count = 0
            retl = []
            for doc in X:
                l = []
                doc = doc.split()
                for word in doc:
                    count += 1
                    if word in self.word2vec:
                        l.append(self.word2vec[word])
                        hit += 1           
                retl.append(np.mean(l or np.zeros(self.dim), axis=0))
            print("The hit rate is " + str(hit / count))
            return np.array(retl)
        
#             return np.array([
#                 np.mean([self.word2vec[word] for word in doc if word in self.word2vec]
#                         or [np.zeros(self.dim)], axis=0)
#                 for doc in X
#             ])
    
    print('featurizeChinese...')
    vectorizer = MeanEmbeddingVectorizer(w2v, 300)
    X = vectorizer.transform(docs)
    return X


def loadWordVectors(fname, size=float('inf')):
    """param: fname, file name of word embeddings
    size: for max loaded lines restriction
    return: a dictionary storing the embeddings"""
    print('loadWordVectors:' + fname + '...')
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