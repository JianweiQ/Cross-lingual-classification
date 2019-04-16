from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression


def classifiers(X_train, y_train, X_test, y_test):
    """param: X_train, y_train are feature vectors for training
    param: X_test, y_test are feature vectors for testing"""

    def classifier(clf, clf_name):
        print(clf_name + ":")
        clf.fit(X_train, y_train)
        y2_pred = clf.predict(X_test)
        print("accuracy =", accuracy_score(y_test, y2_pred))
        # row for true label and column for predicted label
        print(confusion_matrix(y_test, y2_pred))
        
    classifier(LinearSVC(), "LinearSVC")
    classifier(ExtraTreesClassifier(), "ExtraTreesClassifier")
    classifier(LogisticRegression(), "LogisticRegression")
#     classifier(BernoulliNB(), "BernoulliNB")
#     classifier(MultinomialNB(), "MultinomialNB")
