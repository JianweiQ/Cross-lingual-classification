from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import io
import numpy as np


def average_traditional_classifiers(X, y, embeding):
    """param: X list of list of tokens for 2 languages
    param: y list of list of target for 2 languages
    param: embedding, list of word embedding path for 2 languages
    return: Trained CNN and classifer on 2 languages""" 
    
    X_e = X[0]
    X_c = X[1]
    y_e = y[0]
    y_c = y[1]
    embed_file_en = embeding[0]
    embed_file_zh = embeding[1]
    
    embed_dim = 300
    
    # English feature (average)
    w2v = loadWordVectors(embed_file_en, 100000)  
    X_e = featurize(X_e, w2v, embed_dim)
        
    # Chinese feature (average)
    w2v = loadWordVectors(embed_file_zh)  
    X_c = featurize(X_c, w2v, embed_dim)
     
    # English classifiers
    X_train, X_test, y_train, y_test = train_test_split(X_e, y_e, test_size=0.33)
    print("English training vectors for English testing vectors:")
    classifiers(X_train, y_train, X_test, y_test)
     
    # Chinese classifiers
    X_c_train, X_c_test, y_c_train, y_c_test = train_test_split(X_c, y_c, test_size=0.33)
    print("Chinese training vectors for Chinese testing vectors:")
    classifiers(X_c_train, y_c_train, X_c_test, y_c_test)
     
    # English classifiers to train Chinese vectors
    print("English training vectors for Chinese testing vectors:")
    classifiers(X_train, y_train, X_c, y_c)
     
    # Chinese classifiers to train English vectors
    print("Chinese training vectors for English testing vectors:")
    classifiers(X_c_train, y_c_train, X_e, y_e)


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


def featurize(docs, w2v, dim=300):
    """param: docs, a list of lists of sentences (continuous)
    param: w2v, a dictionary of word embeddings
    param: dim, word embeddings dimension
    return: a matrix, a row represents a doc's feature vector"""
    print('featurizing...')  
    hit = 0
    count = 0
    retl = []
    for doc in docs:
        l = []
        doc = doc.split()
        for word in doc:
            count += 1
            if word in w2v:
                l.append(w2v[word])
                hit += 1           
        retl.append(np.mean(l or np.zeros(dim), axis=0))
    print("The hit rate for words in embedding list is " + str(hit / count))
    return np.array(retl)


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
