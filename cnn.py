from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Embedding, Convolution1D
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pylab as plt
import numpy as np
from decimal import Decimal


def CNNCross(X, y, embeding):
    """param: X list of list of docs for 2 languages
    param: y list of list of target for 2 languages
    param: embedding, list of word embedding path for 2 languages
    return: Trained CNN and classifer on 2 languages"""    

    model_type = "both"  # both|CNN-non-static|CNN-static

    # Model Parameters
    embedding_dim = 50
    filter_sizes = (3, 8)
    num_filters = 10
    dropout_prob = (0.5, 0.8)
    hidden_dims = 50
    batch_size = 64
    num_epochs = 10
    sequence_length = 400
    
    def build_model(model_type, vocabulary_size=0, embed_matrix=None):
        number_of_class = 4
        print("Building " + model_type + " model...")
        
        if model_type == "CNN-non-static":
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
        
        model = Model(model_input, model_output)
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model
    
    def train_model(model, X_train, y_train, X_test, y_test, train, test, model_type):
        msg = "Calculating " + train + "-" + test + " accuracy (" + model_type + "):"
        print(msg)
        verbose = False
        history = model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=verbose,
                            validation_data=(X_test, y_test))
        print("In sample accuracy: ")
        print([float(Decimal("%.4f" % e)) for e in history.history['acc']])
        print("Out of sample accuracy: ")
        print([float(Decimal("%.4f" % e)) for e in history.history['val_acc']])
        predictions = model.predict(X_test)
        matrix = confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
        print("Confusion matrix:")
        print(matrix)
        return (history, matrix)
    
    def plot_model(model_type, ret, label):
        history = [item[0] for item in ret] 
        confusion_matrix = [item[1] for item in ret] 
        plot_accuracy_history(history, label, model_type + " out of sample accuracy")

    # Data prepare
    X_train, X_test, y_train, y_test, X_train_static, X_test_static, tok, em, v = [0] * 2, [0] * 2, [0] * 2, [0] * 2, [0] * 2, [0] * 2, [0] * 2, [0] * 2, [0] * 2
    
    for i in range(2):
        X_train[i], X_test[i], y_train[i], y_test[i] = train_test_split(X[i], y[i], test_size=0.33, random_state=1000)

        tok[i] = Tokenizer(split=' ')
        tok[i].fit_on_texts(X_train[i])
    
        X_train[i] = tok[i].texts_to_sequences(X_train[i])
        X_test[i] = tok[i].texts_to_sequences(X_test[i])
        X_train[i] = pad_sequences(X_train[i], padding='post', maxlen=sequence_length)
        X_test[i] = pad_sequences(X_test[i], padding='post', maxlen=sequence_length)
        em[i] = create_embedding_matrix(embeding[i], tok[i].word_index, embedding_dim)  # , 100000)
        v[i] = len(tok[i].word_index) + 1
        y_train[i] = to_categorical(y_train[i])
        y_test[i] = to_categorical(y_test[i])    
        if model_type == "CNN-static" or "both":
            X_train_static[i] = np.stack([np.stack([em[i][word] for word in sentence]) for sentence in X_train[i]])
            X_test_static[i] = np.stack([np.stack([em[i][word] for word in sentence]) for sentence in X_test[i]])
            print("X_train static shape: " , X_train_static[i].shape)
            print("X_test static shape: " , X_test_static[i].shape)
    
    lang = ["EN", "ZH"]
    ret = []
    label = []
    
    if model_type == "CNN-static" or "both":
        model_type_s = "CNN-static"
        model = build_model(model_type_s)
        for i in range(2):
            j = 1 - i
            ret.append(train_model(model, X_train_static[i], y_train[i], X_test_static[i], y_test[i],
                                              lang[i], lang[i], model_type_s))
            label.append(lang[i] + "-" + lang[i])
            ret.append(train_model(model, X_train_static[i], y_train[i], X_test_static[j], y_test[j],
                                              lang[i], lang[j], model_type_s))
            label.append(lang[i] + "-" + lang[j])
        
        plot_model(model_type_s, ret, label)
        
    if model_type == "CNN-non-static" or "both":
        model_type_s = "CNN-non-static"
        for i in range(2):
            j = 1 - i
            model = build_model(model_type_s, v[i], em[i])
            ret.append(train_model(model, X_train[i], y_train[i], X_test[i], y_test[i],
                                          lang[i], lang[i], model_type_s))
            
            model = build_model(model_type_s, v[i] + v[j], np.concatenate((em[i], em[j]), axis=0))
            X_test_shift = np.copy(X_test[j])
            for item in X_test_shift:
                item += v[i]
            ret.append(train_model(model, X_train[i], y_train[i], X_test_shift, y_test[j],
                                           lang[i], lang[j], model_type_s))
        
        plot_model(model_type_s, ret[4:], label)


def create_embedding_matrix(filepath, word_index, embedding_dim, size=float('inf')):
    print("CreateEmbeddingMatrix..." + filepath)
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        cur = 0
        for line in f:
            if cur == 0:
                embed_sizes = line.rstrip().split(' ')
                print("loading file size: ", min(size, int(embed_sizes[0])))
                cur += 1
            elif cur <= size:
                tokens = line.rstrip().split(' ')
                word = tokens[0]
                if word in word_index:
                    idx = word_index[word] 
                    embedding_matrix[idx] = np.asarray(list(map(float, tokens[1:])))[:embedding_dim]
                cur += 1
            else:
                break

    return embedding_matrix


def plot_accuracy_history(history, label, title):
#     plt.style.use('ggplot')
    color_set = ["darkblue", "darkblue", "orangered", "orangered"]
    linestyles = ['--', '-' , '--', '-']
#     val_acc = history.history['val_acc']
#     loss = history.history['loss']
#     val_loss = history.history['val_loss']
    x = range(1, len(history[0].history['val_acc']) + 1)
#     plt.figure(figsize=(12, 5))
    for i in range(len(label)):
        plt.plot(x, history[i].history['val_acc'], color_set[i], label=label[i], linestyle=linestyles[i])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend(loc=4)  # bottom right
    plt.savefig('output/' + title + '.png')
    plt.show()
