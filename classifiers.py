from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from plot import prepare_document_embedding_tsv
import numpy as np
import io


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
     
    X_train = np.concatenate((X_e[:1000], X_e[5000:]), axis=0)
    y_train = np.concatenate((y_e[:1000], y_e[5000:]), axis=0)
    X_test, y_test = X_e[1000:5000], y_e[1000:5000]
    X_c_train = np.concatenate((X_c[:1000], X_c[5000:]), axis=0)
    y_c_train = np.concatenate((y_c[:1000], y_c[5000:]), axis=0)
    X_c_test, y_c_test = X_c[1000:5000], y_c[1000:5000]
    
#     prepare_document_embedding_tsv(X_e, y_e, X_c, y_c, "DocEmbed_average")
    
    ret = []
    ret.append(classifiers(X_train, y_train, X_test, y_test, "EN-EN"))
    ret.append(classifiers(X_train, y_train, X_c, y_c, "EN-ZH"))
    ret.append(classifiers(X_c_train, y_c_train, X_c_test, y_c_test, "ZH-ZH"))
    ret.append(classifiers(X_c_train, y_c_train, X_e, y_e, "ZH-EN"))
    return ret


def classifiers(X_train, y_train, X_test, y_test, msg):
    """param: X_train, y_train are feature vectors for training
    param: X_test, y_test are feature vectors for testing"""

    def classifier(clf, clf_name):
        print("\n" + msg + ", " + clf_name + ":")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("accuracy =", accuracy)
        print(classification_report(y_test, y_pred, target_names=['CCAT', 'ECAT', 'GCAT', 'MCAT']))
#         compute_precision_recall_F1(y_pred, y_test, 4)
        # row for true label and column for predicted label
        matrix = confusion_matrix(y_test, y_pred)
        print(matrix)
        return (y_pred, matrix)
        
    ret = classifier(LinearSVC(), "LinearSVC")
    return ret
#     classifier(ExtraTreesClassifier(), "ExtraTreesClassifier")
#     classifier(LogisticRegression(), "LogisticRegression")
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
#             else:
#                 print(word)         
        retl.append(np.mean(l or np.zeros(dim), axis=0))
    print("The hit rate for words in embedding list is " + str(hit / count))
    return np.array(retl)


def loadWordVectors(fname, size=float('inf')):
    """param: fname, file name of word embeddings
    size: for max loaded lines restriction
    return: a dictionary storing the embeddings"""
    print('\nloadWordVectors:' + fname + '...')
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
