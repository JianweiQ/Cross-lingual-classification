import thulac
import jieba
import re
import os
from sklearn.datasets import fetch_20newsgroups
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


def countUniqueWords(docs, labels, k, w2v=None):
    """
    param:
        docs: a list of strings
        labels: a list of k integers (0...k-1), representing k labels/categories.
            docs and labels should have the same length
        k: int, the number of categories. k = 4 for reuters dataset
        w2v: word embedding dictionary, used to count how many words can be found
    return:
        nUniqueWords: a list of k numbers, each representing the number 
            of unique words for docs in that category
    """
    uniqueWords = [set() for _ in range(k)]
    for i in range(len(docs)):
        doc = docs[i]
        words = doc.strip().split()
        for word in words:
            if w2v:
                if word in w2v and word not in uniqueWords[labels[i]]:
                    uniqueWords[labels[i]].add(word)
            else:   
                if word not in uniqueWords[labels[i]]:
                    uniqueWords[labels[i]].add(word)
    X_count = []
    for i in range(k):
        X_count.append(len(uniqueWords[i]))
    print ("The number of unique words is:", X_count)
    return X_count
    

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


def fetchRCV2():

    def transfer_y(y_str):
        if y_str == "CCAT":
            return 0
        elif y_str == "ECAT":
            return 1
        elif y_str == "GCAT":
            return 2
        elif y_str == "MCAT":
            return 3
        print("bad label")
        
    def read_files(file_name, X, y, lang):
        """param: file_name, string type file directory
        param: X to append
        param: y to append
        param: lang, E for english and C for chinese"""
        file = open(file_name, 'r')
        for line in file:
            label, text = line.strip().split('\t') 
            if lang == 'E':
                X.append(text[2:-1])  # remove b''
            else:
                X.append(text)
            y.append(transfer_y(label))

    print('fetchRCV2...')
    X_e = []
    y_e = []
    X_c = []
    y_c = []
    read_files("data/RCV2/chinese.dev_se", X_c, y_c, 'C')
    read_files("data/RCV2/chinese.test_se", X_c, y_c, 'C')
    read_files("data/RCV2/chinese.train.1000_se", X_c, y_c, 'C')
    read_files("data/RCV2/english.dev", X_e, y_e, 'E')
    read_files("data/RCV2/english.test", X_e, y_e, 'E')
    read_files("data/RCV2/english.train.1000", X_e, y_e, 'E')
    return (X_e, y_e, X_c, y_c)


def RCV2WordSegmentationJieba():
    print('RCV2WordSegmentationJieba...')
    
    def read_files(input_file):
        """param: input_file, string type file directory"""
        output_file_zh = input_file + "_se"
        i_file = open(input_file, 'r')
        o_file_zh = open(output_file_zh, 'a')
        for line in i_file:
            label, text = line.strip().split('\t') 
            newitem = jieba.cut(text)
            newitem = " ".join(newitem)
            newitem = label + '\t' + newitem + '\n'
            o_file_zh.write(newitem)
        o_file_zh.close()
    
    read_files("data/RCV2/chinese.dev")
    read_files("data/RCV2/chinese.test")
    read_files("data/RCV2/chinese.train.1000")


def RCV2WordSegmentationTHU():
    print('RCV2WordSegmentationTHU...')
    thu = thulac.thulac(seg_only=True)
    thu.cut_f("data/RCV2/chinese.dev", "data/RCV2/chinese.dev_thu")
    thu.cut_f("data/RCV2/chinese.test", "data/RCV2/chinese.test_thu")
    thu.cut_f("data/RCV2/chinese.train.1000", "data/RCV2/chinese.train.1000_thu")
    

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


def tokenize(docs, lang):
    """param: docs, a list of lists of sentences (continuous)
    param: lang, E for english and C for chinese
    return: a list of lists of sentences (continuous)
    stemmed, docs with extreme frequence, stop words, punctuation removed"""
    
    class LemmaTokenizer(object):  # for stemming and lemmatizing

        def __init__(self):
            self.wnl = WordNetLemmatizer()

        def __call__(self, doc):
            tokenizer = RegexpTokenizer(r'(?u)\b\w\w+\b')  # remove punctuations
            doc = re.sub(r'\S*@\S*\s?', '', doc.lower())  # to lower case, remove emails
            doc = ''.join([i for i in doc if not i.isdigit()])  # remove digits/numbers
            return [self.wnl.lemmatize(t) for t in tokenizer.tokenize(doc)]
    
    if lang == 'E':
        print('tokenizeEnglish...')
        tokenizer = LemmaTokenizer()  # stemmed
        docs = [tokenizer(doc) for doc in docs]
        for i in range(len(docs)): 
            docs[i] = " ".join(docs[i])  # merge to sentences
        tfidf = TfidfVectorizer(stop_words='english', max_df=0.7, min_df=3)  # , token_pattern=u'(?u)\b\w*[a-zA-Z]\w*\b')
    else:
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


def saveMidOutput(X_e, X_c, y_e, y_c, dataset_option):

    def save(midout, name, isX=0):
        file = open("mid/" + name + str(dataset_option), 'w')
        for item in midout:
            if isX == 0: 
                item = str(item)
            file.write(item + "\n")
        file.close()

    print('saveMidOutput...')
    save(X_e, "X_e", 1)
    save(y_e, "y_e", 0)
    save(X_c, "X_c", 1)
    save(y_c, "y_c", 0)

    
def loadMidOutput(dataset_option):  

    def load(midout, name, isX=0):
        file = open("mid/" + name + str(dataset_option), 'r')
        for line in file:
            if isX == 1: 
                midout.append(line.strip())
            else:
                l = line.strip().split()
                midout.append(int(l[0]))

    print('loadMidOutput...')
    X_e, X_c, y_e, y_c = [], [], [], []
    load(X_e, "X_e", 1)
    load(y_e, "y_e", 0)
    load(X_c, "X_c", 1)
    load(y_c, "y_c", 0)
    return (X_e, X_c, y_e, y_c)

