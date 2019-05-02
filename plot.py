import io
import numpy as np
import matplotlib.pyplot as plt


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
            for j in range(len(X[i]) - 1):
                embedding_vec.write("{:.4f}".format(X[i][j]) + "\t")
            embedding_vec.write("{:.4f}".format(X[i][-1]) + "\n")
    
    load(X_e, y_e, "English") 
    load(X_c, y_c, "Chinese")     
    
    embedding_vec.close()
    embedding_label.close()


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
    Plot accuracy trend along training epoch
    history: history list from model.fit
    label: training language list e.g.[EN, ZH]
    title: title of graph
    """
    color_set = ["darkblue", "orangered"]
    x = range(1, len(history[0].history['acc']) + 1)
    
    for i in range(len(label)):
        plt.plot(x, history[i].history['acc'], color_set[i], label=label[i] + "_Accuracy", linestyle='-')
        plt.plot(x, history[i].history['loss'], color_set[i], label=label[i] + "_Loss", linestyle='--')
    plt.xlabel('Training Steps')
    plt.ylabel('Accuracy/Loss')
    plt.title(title)
    plt.legend(loc=4)  # bottom right
    plt.savefig('output/' + title + '.png')
#         plt.show()
    plt.clf()


def plot_svc_cnn_outputs(svc, cnn):
    """
    print scores in formated table, plot matrix
    svc, output scores from svc model
    cnn, output scores from cnn model
    """
    titles = [
        'LinearSVC,EN-EN', 'LinearSVC,EN-ZH', 'LinearSVC,ZH-ZH', 'LinearSVC,ZH-EN',
        'CNN-static,EN-EN', 'CNN-static,EN-ZH', 'CNN-static,ZH-ZH', 'CNN-static,ZH-EN',
        'CNN-non-st,EN-EN', 'CNN-non-st,EN-ZH', 'CNN-non-static,ZH-ZH', 'CNN-non-static,ZH-EN',
         ]
    matrices = []
    accuracy = []
    precision = []
    recall = []
    f1_micro = []
    for item in svc + cnn:
        matrices.append(item[0])
        accuracy.append(item[1])
        precision.append(item[2])
        recall.append(item[3])
        f1_micro.append(item[4])
        
    np.asarray(matrices)
    print("\nConfusion matrix summary: ", matrices)
    
    print("\nAccuracy summary: ")
    for i in range(0, 12, 4):
        print(" ".join(list(map(str, accuracy[i:i + 4]))))
     
    print("\nF1s summary: ")
    for i in range(0, 12, 4):
        print(" ".join(list(map(str, precision[i:i + 4]))))
        print(" ".join(list(map(str, recall[i:i + 4]))))
        print(" ".join(list(map(str, f1_micro[i:i + 4]))))
    
#     plot_confusion_matrices(matrices, titles)
    
    
def plot_confusion_matrices_with_outputs():
    """Plot confusion matrices given by output.log"""
     
    # confusion matrix
    matrices = [np.asarray([[846, 50, 27, 68],
       [ 35, 856, 57, 52],
       [ 27, 27, 972, 4],
       [ 36, 61, 8, 874]]), np.asarray([[ 408, 10, 1, 750],
     [  12, 40, 2 , 1161],
     [  11  , 17, 107 , 228],
     [   4 , 1  , 0, 1248]]), np.asarray([[ 980, 123, 17, 49],
       [ 105, 1035, 31, 44],
       [  24, 45, 293, 1],
       [  29, 73, 5, 1146]]), np.asarray([[930, 0, 3, 58],
 [710, 16, 33, 241],
 [197, 1, 712, 120],
 [298, 0, 2, 679]]), np.asarray([[865, 45, 21, 60],
       [ 45, 882, 33, 40],
       [ 35, 31, 960, 4],
       [ 51, 61, 7, 860]]), np.asarray([[ 910, 138, 4, 117],
       [ 109, 919, 2, 185],
       [  45, 151, 163, 4],
       [  58, 94, 0, 1101]]), np.asarray([[1018, 84, 34, 33],
       [ 101, 1034, 43, 37],
       [  25, 27, 310, 1],
       [  38, 68, 0, 1147]]), np.asarray([[795, 70, 0, 126],
       [177, 726, 1, 96],
       [133, 204, 90, 603],
       [103, 100, 0, 776]]), np.asarray([[817, 57, 41, 76],
       [ 15, 901, 39, 45],
       [ 15, 31, 976, 8],
       [ 23, 49, 10, 897]]), np.asarray([[ 740, 191, 6, 232],
       [  76, 739, 8, 392],
       [  53, 152, 151, 7],
       [  28, 164, 1, 1060]]), np.asarray([[ 984, 105, 49, 31],
       [  64, 1092, 30, 29],
       [  30, 51, 281, 1],
       [  21, 76, 2, 1154]]), np.asarray([[889, 29, 0, 73],
       [329, 561, 2, 108],
       [289, 122, 143, 476],
       [311, 40, 1, 627]])]
     
    matrices = np.asarray(matrices)
     
    np.set_printoptions(precision=2)
    plot_confusion_matrices(matrices, titles=[
        'LinearSVC,EN-EN', 'LinearSVC,EN-ZH', 'LinearSVC,ZH-ZH', 'LinearSVC,ZH-EN',
        'CNN-static,EN-EN', 'CNN-static,EN-ZH', 'CNN-static,ZH-ZH', 'CNN-static,ZH-EN',
        'CNN-non-static,EN-EN', 'CNN-non-static,EN-ZH', 'CNN-non-static,ZH-ZH', 'CNN-non-static,ZH-EN',
         ])


def plot_cnn_accuracy_history_with_output():
    """Plot accuracy history given by output.log"""
    label = ["EN", "ZH"]
    
    history_static = {}
    title_static = "CNN-static accuracy and loss"
    history_non_static = {}
    title_non_static = "CNN-non-static accuracy and loss"
    
    plot_cnn_accuracy_history(history_static, label, title_static)
    plot_cnn_accuracy_history(history_non_static, label, title_non_static)


def plot_count_bar_chart_with_output():
    """Plot words count bar charts given by output.log"""
    X_count = [[33068, 29276, 55586, 35701], [8719, 8560, 12040, 8219], [8303, 8132, 11521, 7564],
               [22746, 19507, 8922, 29704], [8826, 8446, 4309, 9680], [4998, 4677, 3137, 4758]]
    labels = ["EN(Ori)", "EN(Pro)", "EN(Inc)", "ZH(Ori)", "ZH(Pro)", "ZH(Inc)"]
    n_groups = 4
    title = "Unique words count"
     
    # create plot
    fig = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.15
     
    color_set = ["darkblue", "darkblue", "darkblue", "orangered", "orangered", "orangered"]
    opacity_set = [1, 0.6, 0.3, 1, 0.6, 0.3]
    
    for i in range(len(X_count)):
        plt.bar(index + bar_width * i, X_count[i], bar_width,
                        alpha=opacity_set[i],
                         color=color_set[i],
                         label=labels[i])
     
    plt.xlabel('Topics')
    plt.ylabel('Count')
    plt.title(title)
    plt.xticks(index + bar_width, ('CCAT', 'ECAT', 'GCAT', 'MCAT'))
    plt.legend()
     
    plt.tight_layout()
    plt.savefig('output/' + title + '.png')
    plt.show()

   
if __name__ == '__main__':
    plot_confusion_matrices_with_outputs()
#     prepare_word_embedding_tsv(['data/wiki.en.align.vec', 'data/wiki.zh.align.vec'
#                                   , 'data/wiki.es.align.vec', 'data/wiki.de.align.vec', 'data/wiki.fr.align.vec']
#                                   , ["English", "Chinese", "Spanish", "German", "French"], "Embed_5lang" , 5000)
#     prepare_word_embedding_tsv(['data/wiki.en.align.vec', 'data/wiki.zh.align.vec']
#                               , ["English", "Chinese"], "Embed_EN_ZH" , 5000)

