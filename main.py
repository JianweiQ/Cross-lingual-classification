import warnings
import sys
from collections import Counter
from preprocessing import *
from classifiers import average_traditional_classifiers
from cnn import CNNCross


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    embed_file_en = 'data/wiki.en.align.vec'
    embed_file_zh = 'data/wiki.zh.align.vec'
    
    dataset_option = 3  # 1 for 20NewsGroup&THU, 2 for UM-corpus, 3 for RCV2, else for skip
    model_option = 2  # 1 for traditional classifiers, 2 for CNN, else for skip
    
    sys.stdout = Logger('output/output' + str(dataset_option) + str(model_option) + ".log")
    
    # ==========================Data preparation==========================
     
    # load data
    if dataset_option == 1:
        cat = ['rec.sport.baseball', 'talk.politics.misc', 'sci.electronics'] 
#         cat = ['rec.sport.baseball', 'rec.sport.hockey',
#                'talk.politics.mideast', 'talk.politics.guns', 'talk.politics.misc',
#                'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'] 
        dataset = fetch20newsgroup(cat, "all")
        X_e = dataset.data
        y_e = dataset.target
#         y_e = relabel20newsgroup(y_e)
        X_c, y_c = fetchTHUnews(1000)
    elif dataset_option == 2:
        X_e, y_e, X_c, y_c = fetchUMCorpus()
    elif dataset_option == 3:
        X_e, y_e, X_c, y_c = fetchRCV2()
      
    X_e_count = countUniqueWords(X_e, y_e, 4)
    X_c_count = countUniqueWords(X_c, y_c, 4)
    y_e_count = Counter(y_e)
    y_c_count = Counter(y_c)
    print("EN Documents count is ", y_e_count)
    print("ZH Documents count is ", y_c_count)
    countbarplot(X_e_count, X_c_count, "Unique words count before pre-processing")
    countbarplot(y_e_count.values(), y_c_count.values(), "Documents count")
     
    # data pre-processing
     
    if dataset_option in set([1,2,3]):
        X_c = tokenize(X_c, 'C') 
        X_e = tokenize(X_e, 'E') 
        saveMidOutput(X_e, X_c, y_e, y_c, dataset_option)
    
    # ==========================Model training==========================
    
    X_e, X_c, y_e, y_c = loadMidOutput(3)
    
    X_e_count = countUniqueWords(X_e, y_e, 4)
    X_c_count = countUniqueWords(X_c, y_c, 4)
    countbarplot(X_e_count, X_c_count, "Unique words count after pre-processing")
    
    if model_option == 1:
        average_traditional_classifiers([X_e, X_c], [y_e, y_c], [embed_file_en, embed_file_zh])
    elif model_option == 2:
        CNNCross([X_e, X_c], [y_e, y_c], [embed_file_en, embed_file_zh])


class Logger(object):

    def __init__(self, file_name):
        self.terminal = sys.stdout
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  
    def flush(self):
        pass        


if __name__ == '__main__':
    # run once to get output saved in disk
#     THUWordSegmentation()
#     UMWordSegmentation(["Bi-Laws", "Bi-Thesis"], 100000)
#     RCV2WordSegmentationJieba()
#     RCV2WordSegmentationTHU()
    main()
