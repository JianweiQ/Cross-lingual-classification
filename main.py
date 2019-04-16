import warnings
from sklearn.model_selection import train_test_split
from classifiers import classifiers
from cnn import CNNCross
from preprocessing import *


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # ==========================data preprocessing==========================
    
    dataset_option = 1  # 1 for THU, 2 for UM-corpus
    # English corpus loading
    if dataset_option == 1:
        cat = ['rec.sport.baseball', 'talk.politics.misc', 'sci.electronics'] 
#         cat = ['rec.sport.baseball', 'rec.sport.hockey',
#                'talk.politics.mideast', 'talk.politics.guns', 'talk.politics.misc',
#                'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'] 
        dataset = fetch20newsgroup(cat, "all")
        X_e = dataset.data
        y_e = dataset.target
#         y_e = relabel20newsgroup(y_e)
    elif dataset_option == 1:  # 1 for THU, 2 for UM-corpus
        X_e, y_e, X_c, y_c = fetchUMCorpus()
    
    # Chinese corpus loading
    if dataset_option == 1:
        X_c, y_c = fetchTHUnews(1000)
    X_c = tokenizeChinese(X_c)  
    
    # ==========================Model training==========================
    
    model_option = 1  # 1 for traditional classifiers, 2 for CNN
    
    if model_option == 1:
        # English feature (average)
        X_e = tokenizeEnglish(X_e)
        w2v_en = loadWordVectors('data/wiki.en.align.vec', 100000)  
        X_e = featurizeEnglish(X_e, w2v_en)
           
        # Chinese feature (average)
        w2v_zh = loadWordVectors('data/wiki.zh.align.vec', 100000)  
        X_c = featurizeChinese(X_c, w2v_zh)
        
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
        
    elif model_option == 2:
        CNNCross([X_e, X_c], [y_e, y_c], ['data/wiki.en.align.vec', 'data/wiki.zh.align.vec'])
        

if __name__ == '__main__':
    main()
    # run only once to get output
#     THUWordSegmentation()
#     UMWordSegmentation(["Bi-Laws", "Bi-Thesis"], 100000)
