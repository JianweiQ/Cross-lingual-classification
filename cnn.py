"""
Train convolutional network for sentiment analysis on IMDB corpus. Based on
"Convolutional Neural Networks for Sentence Classification" by Yoon Kim
http://arxiv.org/pdf/1408.5882v2.pdf
For "CNN-rand" and "CNN-non-static" gets to 88-90%, and "CNN-static" - 85% after 2-5 epochs with following settings:
embedding_dim = 50
filter_sizes = (3, 8)
num_filters = 10
dropout_prob = (0.5, 0.8)
hidden_dims = 50
Differences from original article:
- larger IMDB corpus, longer sentences; sentence length is very important, just like data size
- smaller embedding dimension, 50 instead of 300
- 2 filter sizes instead of original 3
- fewer filters; original work uses 100, experiments show that 3-10 is enough;
- random initialization is no worse than word2vec init on IMDB corpus
- sliding Max Pooling instead of original Global Pooling
"""

import numpy as np
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Embedding, Convolution1D
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from plot import plot_cnn_accuracy_history
from decimal import Decimal


class TextCNN(BaseEstimator):
    """CNN class with build model in __init__, training model, predicting model functions"""

    def __init__(self,
        model_type,  # CNN-non-static|CNN-static
        # Model Building Parameters
        embedding_dim, filter_sizes, num_filters, dropout_prob, hidden_dims, sequence_length,
        # Model Testing Parameters
        batch_size, num_epochs, verbose,
        # Embedding layer if CNN-non-static
        embed_matrix=None,
        ):
        
        self.model_type = model_type
        self.embedding_dim = embedding_dim
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.dropout_prob = dropout_prob
        self.hidden_dims = hidden_dims
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.embed_matrix = embed_matrix
        self.model = self.build()
        
    def build(self):
        # Build model
        number_of_class = 4
        
        if self.model_type == "CNN-non-static":
            vocabulary_size = len(self.embed_matrix)  # include 0 empty
            input_shape = (self.sequence_length,)
            model_input = Input(shape=input_shape)
            z = Embedding(vocabulary_size, self.embedding_dim, input_length=self.sequence_length,
                           weights=[self.embed_matrix])(model_input)
        elif self.model_type == "CNN-static":
            input_shape = (self.sequence_length, self.embedding_dim)
            model_input = Input(shape=input_shape)
            z = model_input            
        
        z = Dropout(self.dropout_prob[0])(z)
        
        # Convolutional block
        conv_blocks = []
        for sz in self.filter_sizes:
            conv = Convolution1D(filters=self.num_filters,
                                 kernel_size=sz,
                                 padding="valid",
                                 activation="relu",
                                 strides=1)(z)
            conv = MaxPooling1D(pool_size=2)(conv)
            conv = Flatten()(conv)
            conv_blocks.append(conv)
        z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
        
        z = Dropout(self.dropout_prob[1])(z)
        z = Dense(self.hidden_dims, activation="relu")(z)
        model_output = Dense(number_of_class, activation="sigmoid")(z)
        
        model = Model(model_input, model_output)
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model

    def fit(self, X_train, y_train, X_val=[], y_val=[]):
        y_train = to_categorical(y_train)
        if y_val: 
            y_val = to_categorical(y_val)
            history = self.model.fit(X_train, y_train, batch_size=self.batch_size,
                                 epochs=self.num_epochs, verbose=self.verbose,
                                validation_data=(X_val, y_val))
        else:
            history = self.model.fit(X_train, y_train, batch_size=self.batch_size,
                                 epochs=self.num_epochs, verbose=self.verbose)
        return history

    def predict(self, X_test, y_test=[]):
        y_pred = self.model.predict(X_test, batch_size=self.batch_size, verbose=self.verbose)
        return y_pred.argmax(axis=1)

    def print_score(self, y_pred, y_true):
        accuracy = accuracy_score(y_true, y_pred)
        print("Testing accuracy:", accuracy)
        print(classification_report(y_true, y_pred, target_names=['CCAT', 'ECAT', 'GCAT', 'MCAT']))
        matrix = confusion_matrix(y_true, y_pred)
        print("Confusion matrix:\n", matrix)
        return matrix
    

def CNNCross(X, y, embeding,
             # Model Parameters
            embedding_dim=300,  # 50,
            filter_sizes=(3, 8),
            num_filters=10,
            dropout_prob=(0.5, 0.8),
            hidden_dims=50,
            batch_size=64,
            num_epochs=20,
            sequence_length=400,
            verbose=False,
            ):
    """param: X list of list of docs for 2 languages
    param: y list of list of target for 2 languages
    param: embedding, list of word embedding path for 2 languages
    return: Trained CNN and classifer on 2 languages"""    

    def tokenize_sequence(tok, X):
        """
        Use tok to transfer X to one-hot key sequence
        """
        X = tok.texts_to_sequences(X)
        X = pad_sequences(X, padding='post', maxlen=sequence_length)
        return X

    # Preparing data
    X_train, X_test, X_val, y_train, y_test, y_val = ([0] * 2 for i in range(6))
    X_train_static, X_test_static, X_val_static = ([0] * 2 for i in range(3))
    tok, em = ([0] * 2 for i in range(2))
    
    for i in range(2):
        # develop 1000 test 4000 train 1000 provided by Facebook
        X_train[i], y_train[i] = X[i][:1000], y[i][:1000]
        X_test[i], y_test[i] = X[i][1000:5000], y[i][1000:5000]
        X_val[i], y_val[i] = X[i][5000:], y[i][5000:]

        tok[i] = Tokenizer(split=' ')
        tok[i].fit_on_texts(X_train[i])
        
        X_train[i] = tokenize_sequence(tok[i], X_train[i])
        X_test[i] = tokenize_sequence(tok[i], X_test[i])
        X_val[i] = tokenize_sequence(tok[i], X_val[i])
        
        em[i] = create_embedding_matrix(embeding[i], tok[i].word_index, embedding_dim)#, 100000)
        
        X_train_static[i] = np.stack([np.stack([em[i][word] for word in sentence]) for sentence in X_train[i]])
        X_test_static[i] = np.stack([np.stack([em[i][word] for word in sentence]) for sentence in X_test[i]])
        X_val_static[i] = np.stack([np.stack([em[i][word] for word in sentence]) for sentence in X_val[i]])
        print("X_train static shape: " , X_train_static[i].shape)
        print("X_test static shape: " , X_test_static[i].shape)
        print("X_val static shape: " , X_val_static[i].shape)
    
    lang = ["EN", "ZH"]
    testing_predict = []
    
    for model_type in ["CNN-static"]:#, "CNN-non-static"]:
        training_history = []
        label = []
        for i in range(1, -1, -1):#2):  # train
            for k in range(1, -1, -1):#2):  # test
                # Define parameters
                j = 1 - k if i == 1 else k
                y_train_ = y_train[i]
                y_val_ = y_val[i]
                y_test_ = y_test[j]
                training_lang = lang[i]
                testing_lang = lang[j]
                em_ = None
                if model_type == "CNN-non-static":
                    if i == j:
                        em_ = em[i]
                        X_test_ = X_test[j]
                    else:
                        em_ = np.concatenate((em[i], em[j]), axis=0)
                        X_test_ = np.copy(X_test[j])
                        for item in X_test_:
                            item += len(em[i])
                    X_train_ = X_train[i]
                    X_val_ = X_val[i]
                elif model_type == "CNN-static":
                    X_train_ = X_train_static[i]
                    X_val_ = X_val_static[i]
                    X_test_ = X_test_static[j]
                
                # Build and run
                def run_cnn():
                    """Use default parameters to train CNN model"""
                    print("\nTraining/Testing Pair: " + training_lang + "-" + testing_lang)
                    print("\nBuilding " + model_type + " model...")
                    text_model = TextCNN(model_type, embedding_dim, filter_sizes, num_filters,
                                 dropout_prob, hidden_dims, sequence_length, batch_size, num_epochs, verbose, em_)
                    history = text_model.fit(X_train_, y_train_, X_val_, y_val_)
                    print("Training accuracy: ")
                    print([float(Decimal("%.4f" % e)) for e in history.history['acc']])
                    if j == i: 
                        training_history.append(history)
                        label.append(training_lang)
                    y_pred = text_model.predict(X_test_, y_test_)
                    matrix = text_model.print_score(y_pred, y_test_)
                    testing_predict.append((y_pred, matrix))
                    
                def grid_search():
                    """Use grid search method to find best parameters"""
                    print("\nGridSearching..." + training_lang + "-" + testing_lang)
                    parameters = {
                                'embedding_dim':[300],#, 100, 300],
                                'filter_sizes':[(3, 8), (3, 4, 5)],
                                'num_filters' :[10, 50, 100],
                                'dropout_prob' : [(0.5, 0.8)],
                                'hidden_dims' : [10, 50, 100],
                                'batch_size' : [64],
                                'num_epochs' : [20],
                                'sequence_length' : [400],
                                }
                    text_model = TextCNN(model_type, embedding_dim, filter_sizes, num_filters,
                                 dropout_prob, hidden_dims, sequence_length, batch_size, num_epochs, verbose, em_)
                    gds = GridSearchCV(text_model, parameters, scoring='f1_micro', verbose=2)  # 'accuracy'
#                     gds.fit(np.concatenate((X_train_, X_val_), axis=0), np.concatenate((y_train_, y_val_), axis=0))
                    gds.fit(np.concatenate((X_train_, X_test_), axis=0), np.concatenate((y_train_, y_test_), axis=0))
                    print("Best estimator found by grid search:")
                    print(gds.best_estimator_)
                    print("Scores for each grid:")
                    print(gds.cv_results_)
                
                run_cnn()   
                grid_search()
                    
        plot_cnn_accuracy_history(training_history, label, model_type + " accuracy")
        
    return testing_predict


def create_embedding_matrix(filepath, word_index, embedding_dim, size=float('inf')):
    """
    Create embedding matrix
    filepath: file path for word embedding
    word_index: dictionary of word(key) to sequence index(value): 
    embedding_dim: load first n dimension in word vector, max=300
    size: max number of word vectors to load (not all for saving time)
    return: matrix with each row is a vector, the index of is corresponding to sequence index of a word, 
    [0,0...] if this a word is not found in embedding
    """
    print("\nCreateEmbeddingMatrix..." + filepath)
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        cur = 0
        hit = 0
        for line in f:
            if cur == 0:
                embed_sizes = line.rstrip().split(' ')
                print("load word vectors number: ", min(size, int(embed_sizes[0])))
                cur += 1
            elif cur <= size:
                tokens = line.rstrip().split(' ')
                word = tokens[0]
                if word in word_index:
                    hit += 1
                    idx = word_index[word] 
                    embedding_matrix[idx] = np.asarray(list(map(float, tokens[1:])))[:embedding_dim]
                cur += 1
            else:
                break
        print("Words found in embedding: %d/%d=%.4f" % (hit, len(word_index), hit / len(word_index)))
    return embedding_matrix

