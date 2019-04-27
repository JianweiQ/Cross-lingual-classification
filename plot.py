import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def compute_precision_recall_F1(predictions, y, numOfClass):
    """ compute precision, recall, and F1 over each class and then average them
    param: predictions and y are two lists of the same length with values from [0,numOfClass-1]
        numOfClass is the number of distinct classes
    return: a tuple of (precision, recall, F1)
    """
    precisions = []
    recalls = []
    f1Scores = []
    countPred = Counter(predictions)
    countY = Counter(y)
    for i in range(numOfClass):
        # focus on class i
        truePos = 0
        for j in range(len(y)):
            if predictions[j] == y[j] == i: truePos += 1 
        p = 1 if countPred[i] == 0 else truePos / countPred[i]
        r = 1 if countY[i] == 0 else truePos / countY[i]
        precisions.append(p)
        recalls.append(r)
        f1Scores.append(2*p*r / (p+r))
    print("Precision for each class:", precisions, "Average:", np.mean(precisions))
    print("Recall for each class:", recalls, "Average:", np.mean(recalls))
    print("F1 score for each class:", f1Scores, "Average:", np.mean(f1Scores))
    return np.mean(precisions), np.mean(recalls), np.mean(f1Scores)
    

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
    plt.show()


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
    color_set = ["darkblue", "orangered", "orangered", "orangered"]
#     linestyles = ['--', '-' , '--', '-']
    x = range(1, len(history[0].history['val_acc']) + 1)
    
    def plot(y, msg):
        for i in range(len(label)):
            plt.plot(x, history[i].history[y+'acc'], color_set[i], label=label[i]+"_Accuracy", linestyle='-')
            plt.plot(x, history[i].history[y+'loss'], color_set[i], label=label[i]+"_Loss", linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy/Loss')
        plt.title(title + msg)
        plt.legend(loc=4)  # bottom right
        plt.savefig('output/' + title + msg + '.png')
        plt.show()
        plt.clf()
    
    plot('', " (training)")
    plot('val_', " (validation)")

    
def main():
    
    pre = [0,0,0,0,2,2,3,3]
    y = [1,1,1,1,2,2,3,3]
    print(compute_precision_recall_F1(pre,y,4))

    # confusion matrix
    matrices = [[[846, 50, 27, 68], [35, 856, 57, 52], [27, 27, 972, 4], [36, 61, 8, 874]],
    [[645, 16, 3, 1123], [15, 63, 6, 1717], [19, 26, 173, 347], [4, 1, 0, 1842]],
    [[980, 123, 17, 49], [105, 1035, 31, 44], [24, 45, 293, 1], [29, 73, 5, 1146]],
    [[1404, 0, 3, 82], [1066, 21, 52, 333], [304, 1, 1060, 183], [456, 0, 2, 1033]],
    [[288, 546, 42, 115], [9, 960, 20, 11], [16, 152, 839, 23], [45, 340, 7, 587]],
    [[65, 46, 3, 1055], [2, 79, 1, 1133], [1, 101, 41, 220], [0, 27, 0, 1226]],
    [[908, 179, 28, 54], [99, 973, 32, 111], [31, 100, 226, 6], [36, 105, 4, 1108]],
    [[231, 502, 29, 229], [10, 913, 10, 67], [3, 298, 324, 405], [8, 270, 0, 701]],
    [[773, 105, 35, 78], [28, 910, 30, 32], [69, 39, 922, 0], [35, 75, 1, 868]],
    [[295, 42, 10, 822], [103, 25, 5, 1082], [178, 41, 38, 106], [22, 13, 0, 1218]],
    [[1007, 106, 44, 12], [89, 1094, 11, 21], [55, 95, 207, 6], [19, 75, 14, 1145]],
    [[5, 900, 9, 77], [0, 899, 3, 98], [0, 513, 3, 514], [0, 585, 1, 393]]]
    
    matrices = np.asarray(matrices)
    
    np.set_printoptions(precision=2)
    plot_confusion_matrices(matrices, titles=
    [
    'LinearSVC,E-E', 'LinearSVC,E-C', 'LinearSVC,C-C', 'LinearSVC,C-E',
    'CNN-static,E-E', 'CNN-static,E-C', 'CNN-static,C-C', 'CNN-static,C-E',
    'CNN-non-st,E-E', 'CNN-non-st,E-C', 'CNN-non-static,C-C', 'CNN-non-static,C-E',
     ])
    


if __name__ == '__main__':
    main()
