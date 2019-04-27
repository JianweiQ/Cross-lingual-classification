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
from sklearn.metrics import confusion_matrix
from plot import plot_cnn_accuracy_history, compute_precision_recall_F1
from decimal import Decimal


class TextCNN(object):
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
        
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.verbose = verbose

        # Build model
        number_of_class = 4
        print("\nBuilding " + model_type + " model...")
        
        if model_type == "CNN-non-static":
            vocabulary_size = len(embed_matrix)  # include 0 empty
            input_shape = (sequence_length,)
            model_input = Input(shape=input_shape)
            z = Embedding(vocabulary_size, embedding_dim, input_length=sequence_length, weights=[embed_matrix])(model_input)
        elif model_type == "CNN-static":
            input_shape = (sequence_length, embedding_dim)
            model_input = Input(shape=input_shape)
            z = model_input            
        
        z = Dropout(dropout_prob[0])(z)
        
        # Convolutional block
        conv_blocks = []
        for sz in filter_sizes:
            conv = Convolution1D(filters=num_filters,
                                 kernel_size=sz,
                                 padding="valid",
                                 activation="relu",
                                 strides=1)(z)
            conv = MaxPooling1D(pool_size=2)(conv)
            conv = Flatten()(conv)
            conv_blocks.append(conv)
        z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
        
        z = Dropout(dropout_prob[1])(z)
        z = Dense(hidden_dims, activation="relu")(z)
        model_output = Dense(number_of_class, activation="sigmoid")(z)
        
        self.model = Model(model_input, model_output)
        self.model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    def fit(self, X_train, y_train, X_val, y_val):
        history = self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.num_epochs, verbose=self.verbose,
                            validation_data=(X_val, y_val))
        print("Training accuracy: ")
        print([float(Decimal("%.4f" % e)) for e in history.history['acc']])
#         print("Training loss: ")
#         print([float(Decimal("%.4f" % e)) for e in history.history['loss']])
        print("Validation accuracy: ")
        print([float(Decimal("%.4f" % e)) for e in history.history['val_acc']])
#         print("Validation loss: ")
#         print([float(Decimal("%.4f" % e)) for e in history.history['val_loss']])
        return history

    def predict(self, X_test, y_test):
        y_pred = self.model.predict(X_test, batch_size=self.batch_size, verbose=self.verbose)
        loss, accuracy = self.model.evaluate(X_test, y_test, batch_size=self.batch_size, verbose=self.verbose)
        print("Accuracy: ", accuracy)
#         print("Testing loss: ", loss)
        matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
        print("Confusion matrix:")
        print(matrix)
        compute_precision_recall_F1(y_pred.argmax(axis=1), y_test.argmax(axis=1), 4)
        return (y_pred.argmax(axis=1), matrix)


def CNNCross(X, y, embeding,
             # Model Parameters
            embedding_dim=50,
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
        
        em[i] = create_embedding_matrix(embeding[i], tok[i].word_index, embedding_dim, 100000)
        
        y_train[i] = to_categorical(y_train[i])
        y_test[i] = to_categorical(y_test[i])   
        y_val[i] = to_categorical(y_val[i])
        
        X_train_static[i] = np.stack([np.stack([em[i][word] for word in sentence]) for sentence in X_train[i]])
        X_test_static[i] = np.stack([np.stack([em[i][word] for word in sentence]) for sentence in X_test[i]])
        X_val_static[i] = np.stack([np.stack([em[i][word] for word in sentence]) for sentence in X_val[i]])
        print("X_train static shape: " , X_train_static[i].shape)
        print("X_test static shape: " , X_test_static[i].shape)
        print("X_val static shape: " , X_val_static[i].shape)
    
    lang = ["EN", "ZH"]
    
    # choose testing 
#     grid_search()
#     run_CNN()
    
    def grid_search():
        """Use grid search method to find best parameters"""
        # Parameters to change
        embedding_dim_list = [50, 100, 300]
        filter_sizes_list = [(3, 8), (3, 4, 5)]
        num_filters_list = [10, 50, 100]
        dropout_prob_list = [(0.5, 0.8)]
        hidden_dims_list = [50, 100]
        batch_size_list = [64]
        num_epochs_list = [20, 50, 100]
        sequence_length_list = [100, 500, 1000]
        
    testing_predict = []
    
    for model_type in ["CNN-static", "CNN-non-static"]:
        training_history = []
        label = []
        for i in range(2):  # train
            for k in range(2):  # test
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
                text_model = TextCNN(model_type, embedding_dim, filter_sizes, num_filters,
                             dropout_prob, hidden_dims, sequence_length, batch_size, num_epochs, verbose, em_)
                print("Training on: ", training_lang)
                history = text_model.fit(X_train_, y_train_, X_val_, y_val_)
                if j == i: 
                    training_history.append(history)
                    label.append(training_lang)
                print("Testing on: ", testing_lang)
                testing_predict.append(text_model.predict(X_test_, y_test_))
               
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

