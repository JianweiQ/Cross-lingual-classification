from keras import layers
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from main import fetch20newsgroup, fetchTHUnews, tokenizeChinese
import matplotlib.pylab as plt
import numpy as np
from keras.utils import plot_model

plt.style.use('ggplot')
maxlen = 100
embedding_dim = 50


def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


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


def CNNCross(X, y, embeding):
    """param: X list of list of docs for 2 languages
    param: y list of list of target for 2 languages
    param: embedding, list of word embedding path for 2 languages
    return: Trained CNN and classifer on 2 languages"""    
    X_train, X_test, y_train, y_test, tok, em, v = [0] * 2, [0] * 2, [0] * 2, [0] * 2, [0] * 2, [0] * 2, [0] * 2
    
    # E-E C-C
    for i in range(len(X)):
        X_train[i], X_test[i], y_train[i], y_test[i] = train_test_split(X[i], y[i], test_size=0.33, random_state=1000)

        tok[i] = Tokenizer(num_words=5000, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')
        tok[i].fit_on_texts(X_train[i])
    
        X_train[i] = tok[i].texts_to_sequences(X_train[i])
        X_test[i] = tok[i].texts_to_sequences(X_test[i])
        X_train[i] = pad_sequences(X_train[i], padding='post', maxlen=maxlen)
        X_test[i] = pad_sequences(X_test[i], padding='post', maxlen=maxlen)
        em[i] = create_embedding_matrix(embeding[i], tok[i].word_index, embedding_dim)
        v[i] = len(tok[i].word_index) + 1
        CNN(X_train[i], y_train[i], X_test[i], y_test[i], v[i], em[i])
#     
#     # E-C
#     em = np.concatenate((em[0], em[1]), axis=0)
#     X_test_shift = np.copy(X_test[1])
#     for item in X_test_shift:
#         item += v[0]
#     CNN(X_train[0], y_train[0], X_test_shift, y_test[1], v[0] + v[1], em)
#     
#     # C-E
#     em = np.concatenate((em[1], em[0]), axis=0)
#     X_test_shift = np.copy(X_test[0])
#     for item in X_test_shift:
#         item += v[1]
#     CNN(X_train[1], y_train[1], X_test_shift, y_test[0], v[0] + v[1], em)


def CNN(X_train, y_train, X_test, y_test, vocab_size, embedding_matrix):
    """param: vocab_size, number of unique words
    param: embedding_matrix, word embeddings matrix 
    return: Trained CNN and classifer"""
    print("CreatingModel...")
    model = Sequential()
    # TODO: optimize the layers
    model.add(layers.Embedding(vocab_size, embedding_dim,
                               weights=[embedding_matrix],
                               input_length=maxlen,
                               trainable=True))
    model.add(layers.GlobalMaxPool1D())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
#     model.add(layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
#     model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#     model.add(layers.Conv2D(64, (5, 5), activation='relu'))
#     model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#     model.add(layers.Flatten())
#     model.add(layers.Dense(10, activation='relu'))
#     model.add(layers.Dense(1, activation='sigmoid'))
#     model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
    model.summary()
    print("TrainingModel...")
    history = model.fit(X_train, y_train,
                        epochs=50,
                        verbose=False,
                        validation_data=(X_test, y_test),
                        batch_size=10)
    loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))
    plot_history(history)
    plot_model(model, to_file='model.png')


def main():
    # English database
    cat = ['rec.sport.baseball', 'talk.politics.misc', 'sci.electronics'] 
    dataset = fetch20newsgroup(cat, "test")
    X_e = dataset.data
    y_e = dataset.target
    # Chinese database
    X_c, y_c = fetchTHUnews(100)
    X_c = tokenizeChinese(X_c)
    CNNCross([X_e, X_c], [y_e, y_c], ['data/wiki.en.align.vec', 'data/wiki.zh.align.vec'])


if __name__ == '__main__':
    main()
