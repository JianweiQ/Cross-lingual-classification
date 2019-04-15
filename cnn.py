from keras import layers
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from main import fetch20newsgroup, fetchTHUnews
import matplotlib.pylab as plt
import numpy as np

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
    print("create_embedding_matrix")
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


def CNN(X, y, embeding):
    
    sentences_train, sentences_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1000)
    
    tokenizer = Tokenizer(num_words=5000, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')
    tokenizer.fit_on_texts(sentences_train)
    
    X_train = tokenizer.texts_to_sequences(sentences_train)
    X_test = tokenizer.texts_to_sequences(sentences_test)
    
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
    
    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
    
    embedding_matrix = create_embedding_matrix(embeding, tokenizer.word_index, embedding_dim)
    
    model = Sequential()
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
    model.summary()
    print("Training model...")
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


def main():
    # English
    cat = ['rec.sport.baseball', 'talk.politics.misc', 'sci.electronics'] 
    dataset = fetch20newsgroup(cat, "all")
    X = dataset.data
    y = dataset.target
    CNN(X, y, 'data/wiki.en.align.vec')
    # Chinese
    X, y = fetchTHUnews(1000)
    CNN(X, y, 'data/wiki.zh.align.vec')


if __name__ == '__main__':
    main()