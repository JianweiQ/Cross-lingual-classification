import warnings
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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

# import nltk
# nltk.download('wordnet')

MAX_LINE = 100000


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    model = 1  # 1 for THU, 2 for UM-corpus
    # English corpus loading
    if model == 1:
        cat = ['rec.sport.baseball', 'talk.politics.misc', 'sci.electronics'] 
    #     # More cat lead to worse results
#         cat = ['rec.sport.baseball', 'rec.sport.hockey',
#                'talk.politics.mideast', 'talk.politics.guns', 'talk.politics.misc',
#                'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'] 
        dataset = fetch20newsgroup(cat, "all")
        X = dataset.data
        y = dataset.target
#         y = relabel20newsgroup(y)
    elif model == 2:  # 1 for THU, 2 for UM-corpus
        X, y, X_c, y_c = fetchUMCorpus()
        
    # English w2v
    X = tokenizeEnglish(X)
    w2v_en = loadWordVectors('data/wiki.en.align.vec', MAX_LINE)  
#     w2v_en = loadWordVectors('data/vectors-en.txt', MAX_LINE)
    X = featurizeEnglish(X, w2v_en)
      
    # English classifiers
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    print("English training vectors for English testing vectors:")
    classifiers(X_train, y_train, X_test, y_test)
     
    # Chinese w2v
    if model == 1:
        X_c, y_c = fetchTHUnews(1000)
    w2v_zh = loadWordVectors('data/wiki.zh.align.vec', MAX_LINE)  
#     w2v_zh = loadWordVectors('data/vectors-zh.txt', MAX_LINE)  
    X_c = featurizeChinese(X_c, w2v_zh)
    
    # Chinese classifiers
    X_c_train, X_c_test, y_c_train, y_c_test = train_test_split(X_c, y_c, test_size=0.33)
    print("Chinese training vectors for Chinese testing vectors:")
    classifiers(X_c_train, y_c_train, X_c_test, y_c_test)
    
    # English classifiers to train Chinese vectors
    print("English training vectors for Chinese testing vectors:")
    classifiers(X_train, y_train, X_c, y_c)
    
    # Chinese classifiers to train English vectors
    print("Chinese training vectors for English testing vectors:")
    classifiers(X_c_train, y_c_train, X, y)

    
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
        cur = 0
        for file in os.listdir(file_name):
            doc = []
            file = open(file_name + "/" + file, 'r')
            for line in file:
                l = line.split()
                doc = doc + l
            X.append(doc)
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


def tokenizeChinese():
    print('tokenizeChinese...')
    # TODO remove frequence, stop words, punctuation


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


def classifiers(X_train, y_train, X_test, y_test):
    """param: X_train, y_train are feature vectors for training
    param: X_test, y_test are feature vectors for testing"""

    def classifier(clf, clf_name):
        print(clf_name + ":")
        clf.fit(X_train, y_train)
        y2_pred = clf.predict(X_test)
        print("accuracy =", accuracy_score(y_test, y2_pred))
        # row for true label and column for predicted label
        print(confusion_matrix(y_test, y2_pred))
        
    classifier(LinearSVC(), "LinearSVC")
    classifier(ExtraTreesClassifier(), "ExtraTreesClassifier")
    classifier(LogisticRegression(), "LogisticRegression")
#     classifier(BernoulliNB(), "BernoulliNB")
#     classifier(MultinomialNB(), "MultinomialNB")


if __name__ == '__main__':
    main()
    # run only once to get output
#     THUWordSegmentation()
#     UMWordSegmentation(["Bi-Laws", "Bi-Thesis"], 100000)
