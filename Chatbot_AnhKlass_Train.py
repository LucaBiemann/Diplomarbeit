import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
dataset = pd.read_pickle('Chatbot_AusweisNr')
scaler = StandardScaler()
classifiers = [
    KNeighborsClassifier(3),
     QuadraticDiscriminantAnalysis(),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    MLPClassifier(hidden_layer_sizes=100, max_iter=1000, activation='logistic')]
names = ["Nearest Neighbors","QDA",
         "Decision Tree", "Random Forest", "AdaBoost",
         "Naive Bayes",  "MLP"]
for c, n in zip(classifiers, names):
    for m in ['BERT','DistilBERT','XLNet', 'RoBERTa','ERNIE']:
        scaler = pickle.load(open('models/chatbot/scaler_{}.pkl'.format(m), 'rb'))
        for i in [5]:
                X_train, X_valid, y_train, y_valid = \
                        train_test_split(list(dataset[m+'emb'].values), dataset['Zeichen{}'.format(i)].astype(int), test_size=0.23,
                                         random_state=42,
                                         stratify=dataset['Zeichen{}'.format(i)].astype(int))

                X_train = scaler.transform(X_train)

                X_valid = scaler.transform(X_valid)

                X_train, y_train = np.array(X_train, dtype='float64'), np.array(y_train)
                X_valid, y_valid = np.array(X_valid, dtype='float64'), np.array(y_valid)

                print("Training start:", datetime.now())
                clf = c.fit(
                    X_train, y_train)
                print("Training ende:", datetime.now())
                filename = 'models/Anhang_Chatbot_Models/{}_{}_Zeichen{}.sav'.format(n,m,i)
                pickle.dump(clf, open(filename, 'wb'))
                #accuracy_lin = clf.score(X_valid, y_valid)
                print('{} {} Zeichen {}:'.format(n,m,i))

