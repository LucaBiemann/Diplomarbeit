import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import neural_network, svm
from sklearn.preprocessing import StandardScaler
import pickle
from datetime import datetime

dataset = pd.read_pickle('Data_num')
scaler = StandardScaler()
for m in ['BERT','DistilBERT','XLNet', 'RoBERTa','ERNIE']:
    for i in range(10):
        scaler = pickle.load(open('models/scaler_{}.pkl'.format(m), 'rb'))


        X_train, X_valid, y_train, y_valid = \
                train_test_split(list(dataset[m+'emb'].values), dataset['Zeichen{}'.format(i)].astype(int), test_size=0.23,
                                 random_state=42,
                                 stratify=dataset['Zeichen{}'.format(i)].astype(int))
        print(len(X_train))
        X_train = scaler.transform(X_train)

        X_valid = scaler.transform(X_valid)

        X_train, y_train = np.array(X_train, dtype='float64'), np.array(y_train)
        X_valid, y_valid = np.array(X_valid, dtype='float64'), np.array(y_valid)
        print(y_train)
        print("Training start:", datetime.now())
        clf = neural_network.MLPClassifier(hidden_layer_sizes=100, random_state=69, max_iter=1000, activation='logistic').fit(
            X_train, y_train)
        print("Training ende:", datetime.now())
        filename = 'MLP_Zeichen{}.sav'.format(i)
        pickle.dump(clf, open(filename, 'wb'))
        accuracy_lin = clf.score(X_valid, y_valid)
        print('{} Accuracy Linear Kernel Zeichen {}:'.format(datetime.now(),i), accuracy_lin)