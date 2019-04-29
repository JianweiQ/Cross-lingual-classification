import io
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def prepare_word_embedding_tsv(embed_file_list, language_list, saved_file, size=float('inf')):
    """
    merge word embedding as middle outputs for visualization. Also Inputs in: 
    https://projector.tensorflow.org/
    param:
    embed_file_list, a list of string for path of word embedding
    language_list, a list of string for language label
    saved_file, string for saved file name
    size, maximum load size
    return: saved tsv file: mid/..._vec.tsv & mid/..._label.tsv
    """
    
    embedding_vec = open("mid/" + saved_file + "_vec.tsv", 'w')
    embedding_label = open("mid/" + saved_file + "_label.tsv", 'w')
    embedding_label.write("Word\tLanguage\n")  # 2 labels
    
    def load(file_name, label, size):
        print('\nGeneratingWordEmbeddingTSV:' + file_name + '...')
        fin = io.open(file_name, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        print(min(n, size), d)  # n is #words, d is dimension
        cur = 1
        for line in fin:
            tokens = line.rstrip().split(' ')
            embedding_label.write(tokens[0] + "\t" + label + "\n")
            for i in range(1, 300):
                embedding_vec.write(tokens[i] + "\t")
            embedding_vec.write(tokens[300] + "\n")
            if (cur >= size): break 
            else: cur += 1
    
    for i in range(len(embed_file_list)):
        load(embed_file_list[i], language_list[i], size)   
    
    embedding_vec.close()
    embedding_label.close()
    

def prepare_document_embedding_tsv(X_e, y_e, X_c, y_c, saved_file):
    """
    merge document embedding as middle outputs for visualization. Also Inputs in: 
    https://projector.tensorflow.org/
    param:
    X_e, a list of document vectors for English
    y_e, a list of document labels for English
    X_c, a list of document vectors for Chinese
    y_c, a list of document labels for Chinese
    saved_file, string for saved file name
    return: saved tsv file: mid/..._vec.tsv & mid/..._label.tsv
    """
    embedding_vec = open("mid/" + saved_file + "_vec.tsv", 'w')
    embedding_label = open("mid/" + saved_file + "_label.tsv", 'w')
    embedding_label.write("Topic\tLanguage\n")  # 2 labels
    label_list = ['CCAT', 'ECAT', 'GCAT', 'MCAT']
    
    def load(X, y, label):
        print('\nGeneratingDocumentEmbeddingTSV:' + label + '...')
        for i in range(len(X)):
            embedding_label.write(label_list[y[i]] + "\t" + label + "\n")
            for j in range(len(X[i])-1):
                embedding_vec.write("{:.4f}".format(X[i][j]) + "\t")
            embedding_vec.write("{:.4f}".format(X[i][-1]) + "\n")
    
    load(X_e, y_e, "English") 
    load(X_c, y_c, "Chinese")     
    
    embedding_vec.close()
    embedding_label.close()

def visualize_word_embedding():
    """Sample functions to visualize word embedding locally
    Run in terminal: tensorboard --logdir=logs
    open an explore and enter localhost:6006
    """
    LOG_DIR = 'logs'

    mnist = input_data.read_data_sets('MNIST_data')
    images = tf.Variable(mnist.test.images, name='images')
    
    with tf.Session() as sess:
        saver = tf.train.Saver([images])
    
        sess.run(images.initializer)
        saver.save(sess, os.path.join(LOG_DIR, 'images.ckpt'))


def plot_bar_chart_count(X_e_count, X_c_count, title):
    """
    param: X_e_count, English count list
    X_c_count, Chinese count list
    title, title of plot
    return: bar plot
    """
    n_groups = 4
     
    # create plot
    fig = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8
     
    color_set = ["darkblue", "orangered"]
     
    en = plt.bar(index, X_e_count, bar_width,
                     alpha=opacity,
                     color=color_set[0],
                     label='EN')
     
    zh = plt.bar(index + bar_width, X_c_count, bar_width,
                     alpha=opacity,
                     color=color_set[1],
                     label='ZH')
     
    plt.xlabel('Topics')
    plt.ylabel('Count')
    plt.title(title)
    plt.xticks(index + bar_width, ('CCAT', 'ECAT', 'GCAT', 'MCAT'))
    plt.legend()
     
    plt.tight_layout()
    plt.savefig('output/' + title + '.png')
#     plt.show()


def plot_confusion_matrices(matrices, titles, classes=['C', 'E', 'G', 'M']):
    """
    Plot many confusion matrices and organize them in a single figure
    matrices: a list of confusion matrices, the length must be a multiple of 4
    classes: a list of class labels, e.g., ['CCAT', 'ECAT', 'GCAT', 'MCAT']
    titles: a list of titles, each for a confusion matrix
    """

    fig, axes = plt.subplots(len(matrices) // 4, 4, sharex=True, sharey=True)
    whichMatrix = 0
    for i in range(len(matrices) // 4):
        for j in range(4):
            im = axes[i, j].imshow(matrices[whichMatrix], interpolation='nearest', cmap=plt.cm.Blues)
            axes[i, j].figure.colorbar(im, ax=axes[i, j])
            # We want to show all ticks...
            axes[i, j].set(xticks=np.arange(matrices[whichMatrix].shape[1]),
                   yticks=np.arange(matrices[whichMatrix].shape[0]),
                   # ... and label them with the respective list entries
                   xticklabels=classes, yticklabels=classes,
                   title=titles[whichMatrix],
                   ylabel='True label',
                   xlabel='Predicted label')

            # Rotate the tick labels and set their alignment.
            plt.setp(axes[i, j].get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")

            # Loop over data dimensions and create text annotations.
            fmt = 'd'  # '.2f' if normalize else 'd'
            thresh = matrices[whichMatrix].max() / 2.
            for ii in range(matrices[whichMatrix].shape[0]):
                for jj in range(matrices[whichMatrix].shape[1]):
                    axes[i, j].text(jj, ii, format(matrices[whichMatrix][ii, jj], fmt),
                            ha="center", va="center",
                            color="white" if matrices[whichMatrix][ii, jj] > thresh else "black")
            fig.tight_layout()
            whichMatrix += 1
    plt.show()
    return axes


def plot_cnn_accuracy_history(history, label, title):
    """
    Plot accuracy trend along traing epoch
    history: history list from model.fit
    label: language pair label e.g.ZH-EH
    title: title of graph
    """
    color_set = ["darkblue", "orangered"]
    x = range(1, len(history[0].history['val_acc']) + 1)
    
    def plot(y, msg):
        for i in range(len(label)):
            plt.plot(x, history[i].history[y + 'acc'], color_set[i], label=label[i] + "_Accuracy", linestyle='-')
            plt.plot(x, history[i].history[y + 'loss'], color_set[i], label=label[i] + "_Loss", linestyle='--')
        plt.xlabel('Training Steps')
        plt.ylabel('Accuracy/Loss')
        plt.title(title + msg)
        plt.legend(loc=4)  # bottom right
        plt.savefig('output/' + title + msg + '.png')
#         plt.show()
        plt.clf()
    
    plot('', " (training)")
    plot('val_', " (validation)")

    
def main():
    
    # confusion matrix
    matrices = [np.asarray([[846, 50, 27, 68],
       [ 35, 856, 57, 52],
       [ 27, 27, 972, 4],
       [ 36, 61, 8, 874]]), np.asarray([[ 645, 16, 3, 1123],
       [  15, 63, 6, 1717],
       [  19, 26, 173, 347],
       [   4, 1, 0, 1842]]), np.asarray([[ 980, 123, 17, 49],
       [ 105, 1035, 31, 44],
       [  24, 45, 293, 1],
       [  29, 73, 5, 1146]]), np.asarray([[1404, 0, 3, 82],
       [1066, 21, 52, 333],
       [ 304, 1, 1060, 183],
       [ 456, 0, 2, 1033]]), np.asarray([[854, 56, 8, 73],
       [ 44, 905, 8, 43],
       [ 46, 83, 891, 10],
       [ 44, 68, 0, 867]]), np.asarray([[ 427, 133, 0, 609],
       [  17, 373, 0, 825],
       [  12, 258, 43, 50],
       [   3, 30, 0, 1220]]), np.asarray([[ 955, 144, 53, 17],
       [  97, 1031, 57, 30],
       [  19, 30, 314, 0],
       [  38, 90, 8, 1117]]), np.asarray([[646, 126, 2, 217],
       [130, 686, 2, 182],
       [ 62, 337, 45, 586],
       [ 54, 71, 0, 854]]), np.asarray([[871, 47, 27, 46],
       [ 42, 883, 42, 33],
       [ 23, 17, 989, 1],
       [ 47, 55, 2, 875]]), np.asarray([[ 624, 7, 30, 508],
       [ 113, 103, 90, 909],
       [  33, 33, 277, 20],
       [  38, 13, 12, 1190]]), np.asarray([[1020, 105, 23, 21],
       [  88, 1100, 6, 21],
       [  35, 79, 249, 0],
       [  24, 72, 3, 1154]]), np.asarray([[523, 360, 3, 105],
       [ 60, 863, 3, 74],
       [ 38, 377, 44, 571],
       [ 64, 376, 1, 538]])]
#     [[[846, 50, 27, 68], [35, 856, 57, 52], [27, 27, 972, 4], [36, 61, 8, 874]],
#     [[645, 16, 3, 1123], [15, 63, 6, 1717], [19, 26, 173, 347], [4, 1, 0, 1842]],
#     [[980, 123, 17, 49], [105, 1035, 31, 44], [24, 45, 293, 1], [29, 73, 5, 1146]],
#     [[1404, 0, 3, 82], [1066, 21, 52, 333], [304, 1, 1060, 183], [456, 0, 2, 1033]],
#     [[288, 546, 42, 115], [9, 960, 20, 11], [16, 152, 839, 23], [45, 340, 7, 587]],
#     [[65, 46, 3, 1055], [2, 79, 1, 1133], [1, 101, 41, 220], [0, 27, 0, 1226]],
#     [[908, 179, 28, 54], [99, 973, 32, 111], [31, 100, 226, 6], [36, 105, 4, 1108]],
#     [[231, 502, 29, 229], [10, 913, 10, 67], [3, 298, 324, 405], [8, 270, 0, 701]],
#     [[773, 105, 35, 78], [28, 910, 30, 32], [69, 39, 922, 0], [35, 75, 1, 868]],
#     [[295, 42, 10, 822], [103, 25, 5, 1082], [178, 41, 38, 106], [22, 13, 0, 1218]],
#     [[1007, 106, 44, 12], [89, 1094, 11, 21], [55, 95, 207, 6], [19, 75, 14, 1145]],
#     [[5, 900, 9, 77], [0, 899, 3, 98], [0, 513, 3, 514], [0, 585, 1, 393]]]
    
    matrices = np.asarray(matrices)
    
    np.set_printoptions(precision=2)
    plot_confusion_matrices(matrices, titles=[
        'LinearSVC,EN-EN', 'LinearSVC,EN-ZH', 'LinearSVC,ZH-ZH', 'LinearSVC,ZH-EN',
        'CNN-static,EN-EN', 'CNN-static,EN-ZH', 'CNN-static,ZH-ZH', 'CNN-static,ZH-EN',
        'CNN-non-st,EN-EN', 'CNN-non-st,EN-ZH', 'CNN-non-static,ZH-ZH', 'CNN-non-static,ZH-EN',
         ])


if __name__ == '__main__':
    main()
#     prepare_word_embedding_tsv(['data/wiki.en.align.vec', 'data/wiki.zh.align.vec'
#                                   , 'data/wiki.es.align.vec', 'data/wiki.de.align.vec', 'data/wiki.fr.align.vec']
#                                   , ["English", "Chinese", "Spanish", "German", "French"], "Embed_5lang" , 5000)
#     prepare_word_embedding_tsv(['data/wiki.en.align.vec', 'data/wiki.zh.align.vec']
#                               , ["English", "Chinese"], "Embed_EN_ZH" , 5000)

