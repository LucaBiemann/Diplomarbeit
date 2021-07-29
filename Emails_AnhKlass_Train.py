import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pickle

key_word_list = ['usb stick','extortion', 'password', 'trade secret', 'intimidate', 'return','blackmail', 'reveal','inside information',
       'trust', 'major contract', 'recommend', 'insolvency', 'merge',
       'step down', 'takeover bid', 'obfuscate', 'undervaluation', 'conceal',
       'fictitious transaction', 'overvaluation', 'tax evasion', 'underrate',
       'overstate',]
names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]
dataset = pd.read_pickle('Emails_Test')
print(dataset.columns)
scaler = StandardScaler()
for c, n in zip(classifiers, names):
    for m in ['BERT', 'XLNet', 'ERNIE', 'T5']:
        scaler.fit(list(dataset['{}emb'.format(m)].values))
        for i in key_word_list:
            data_pos = dataset[dataset[i] == 1]
            data_neg = dataset[dataset[i] == 0]
            data = data_neg.sample(n=len(data_pos), random_state=42).append(data_pos)

            X_train, X_valid, y_train, y_valid = \
                train_test_split(list(data['{}emb'.format(m)].values), data[i], test_size=0.33,
                                 random_state=42,
                                 stratify=data[i])

            X_train = scaler.transform(X_train)
            X_valid = scaler.transform(X_valid)
            X_train, y_train = np.array(X_train), np.array(y_train)
            X_valid, y_valid = np.array(X_valid), np.array(y_valid)
            clf = c.fit(
                X_train, y_train)
            acc_mlp = clf.score(X_valid, y_valid)
            print('{} {} {} Accuracy:'.format(m, i,n), acc_mlp)
            filename = 'models/Anhang_Mail_Models/{}_{}_{}.sav'.format(n,m, i)
            pickle.dump(clf, open(filename, 'wb'))
