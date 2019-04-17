from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Embedding, Convolution1D
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt
import numpy as np


def CNNCross(X, y, embeding):
    """param: X list of list of docs for 2 languages
    param: y list of list of target for 2 languages
    param: embedding, list of word embedding path for 2 languages
    return: Trained CNN and classifer on 2 languages"""    

    # Model Parameters
    embedding_dim = 50
    filter_sizes = (3, 8)
    num_filters = 10
    dropout_prob = (0.5, 0.8)
    hidden_dims = 50
    batch_size = 64
    num_epochs = 5
    sequence_length = 400
    
    def build_model(vocabulary_size, number_of_class):
        # TODO: optimize
        input_shape = (sequence_length,)
        model_input = Input(shape=input_shape)
        z = Embedding(vocabulary_size, embedding_dim, input_length=sequence_length, name="embedding")(model_input)
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
    
    # Data prepare
    X_train, X_test, y_train, y_test, tok, em, v = [0] * 2, [0] * 2, [0] * 2, [0] * 2, [0] * 2, [0] * 2, [0] * 2
    
    for i in range(2):
        X_train[i], X_test[i], y_train[i], y_test[i] = train_test_split(X[i], y[i], test_size=0.33, random_state=1000)

        tok[i] = Tokenizer(split=' ')
        tok[i].fit_on_texts(X_train[i])
    
        X_train[i] = tok[i].texts_to_sequences(X_train[i])
        X_test[i] = tok[i].texts_to_sequences(X_test[i])
        X_train[i] = pad_sequences(X_train[i], padding='post', maxlen=sequence_length)
        X_test[i] = pad_sequences(X_test[i], padding='post', maxlen=sequence_length)
        em[i] = create_embedding_matrix(embeding[i], tok[i].word_index, embedding_dim)
        v[i] = len(tok[i].word_index) + 1
        y_train[i] = to_categorical(y_train[i])
        y_test[i] = to_categorical(y_test[i])    
    
    print("English vectors to train English vectors:")
    model = build_model(v[0], 4)
    embedding_layer = model.get_layer("embedding")
    embedding_layer.set_weights([em[0]])
    model.fit(X_train[0], y_train[0], batch_size=batch_size, epochs=num_epochs,
              validation_data=(X_test[0], y_test[0]), verbose=2)
    
    print("Chinese vectors to train Chinese vectors:")
    model = build_model(v[1], 4)
    embedding_layer = model.get_layer("embedding")
    embedding_layer.set_weights([em[1]])
    model.fit(X_train[1], y_train[1], batch_size=batch_size, epochs=num_epochs,
              validation_data=(X_test[1], y_test[1]), verbose=2)

    print("English vectors to train Chinese vectors:")
    vocabulary_size = v[1] + v[0]
    model = build_model(vocabulary_size, 4)
    emm = np.concatenate((em[0], em[1]), axis=0)
    embedding_layer = model.get_layer("embedding")
    embedding_layer.set_weights([emm])
    X_test_shift = np.copy(X_test[1])
    for item in X_test_shift:
        item += v[0]
    model.fit(X_train[0], y_train[0], batch_size=batch_size, epochs=num_epochs,
              validation_data=(X_test_shift, y_test[1]), verbose=2)
    
    print("Chinese vectors to train English vectors:")
    model = build_model(vocabulary_size, 4)
    emm = np.concatenate((em[1], em[0]), axis=0)
    embedding_layer = model.get_layer("embedding")
    embedding_layer.set_weights([emm])
    X_test_shift = np.copy(X_test[0])
    for item in X_test_shift:
        item += v[1]
    model.fit(X_train[1], y_train[1], batch_size=batch_size, epochs=num_epochs,
              validation_data=(X_test_shift, y_test[0]), verbose=2)


def create_embedding_matrix(filepath, word_index, embedding_dim):
    print("CreateEmbeddingMatrix..." + filepath)
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        cur = 0
        for line in f:
            if cur != 0:
                tokens = line.rstrip().split(' ')
                word = tokens[0]
                if word in word_index:
                    idx = word_index[word] 
                    embedding_matrix[idx] = np.asarray(list(map(float, tokens[1:])))[:embedding_dim]
            else:
                cur = 1

    return embedding_matrix


def plot_history(history):
    plt.style.use('ggplot')
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
