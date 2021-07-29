import pandas as pd
import numpy as np
import random
import pickle
from sklearn.model_selection import train_test_split
from sklearn import neural_network, svm
from sklearn.preprocessing import StandardScaler
from datetime import datetime

data_message = pd.read_pickle('Chatbot_Messages')

data_num = pd.read_pickle('Chatbot_AusweisNr')
data_num = data_num.sample(n=len(data_message), random_state=42)
data_num = data_num.rename(columns={'BertEmb':'BERTemb'})
data_num = data_num[['BERTemb','DistilBERTemb','XLNetemb', 'RoBERTaemb','ERNIEemb' ,'text']]
data_num['label'] = 1
data_message['label'] = 0
data = data_message.append(data_num)
data = data.sample(frac=0.5, random_state=42)
data.reset_index(drop=True, inplace=True)
colum_df = pd.DataFrame([])
colum_df[0] = ['BERTemb','DistilBERTemb','XLNetemb', 'RoBERTaemb', 'ERNIEemb']
data_num2 = pd.DataFrame(columns=[1])
data_num2[1] = data_num2[1].astype(object)
for index, row in data.iterrows():
    random_num = random.randrange(5)
    data.loc[index, 'label'] = random_num
    data_num2.loc[index, 1] = np.array(row[colum_df.iloc[random_num,0]])
data['vektor'] = data_num2[1]



X_train, X_valid, y_train, y_valid = \
            train_test_split(list(data['vektor'].values), data['label'], test_size=0.23,
                             random_state=42,
                             stratify=data['label'])
X_train, y_train = np.array(X_train, dtype='float64'), np.array(y_train)
X_valid, y_valid = np.array(X_valid, dtype='float64'), np.array(y_valid)
print("Training start:", datetime.now())
clf = neural_network.MLPClassifier(hidden_layer_sizes=100, random_state=69, max_iter=1000, activation='logistic').fit(
        X_train, y_train)
print("Training ende:", datetime.now())
filename = 'models/chatbot/MLP_Bot_Model.sav'
pickle.dump(clf, open(filename, 'wb'))
accuracy_lin = clf.score(X_valid, y_valid)
print('{} Accuracy:'.format(datetime.now()), accuracy_lin)
for i in ['BERT','DistilBERT','XLNet', 'RoBERTa','ERNIE']:
    scaler = pickle.load(open('models/chatbot/scaler_{}.pkl'.format(i), 'rb'))
    X_train, X_valid, y_train, y_valid = \
        train_test_split(list(data[i+'emb'].values), data['label'], test_size=0.23,
                         random_state=42,
                         stratify=data['label'])

    X_train = scaler.transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_train, y_train = np.array(X_train, dtype='float64'), np.array(y_train)
    X_valid, y_valid = np.array(X_valid, dtype='float64'), np.array(y_valid)
    print("Training start:", datetime.now())
    clf = neural_network.MLPClassifier(hidden_layer_sizes=100, random_state=69, max_iter=1000,
                                       activation='logistic').fit(
        X_train, y_train)
    print("Training ende:", datetime.now())
    filename = 'models/chatbot/{}/MLP_NumMes_{}.sav'.format(i,i)
    pickle.dump(clf, open(filename, 'wb'))
    accuracy_lin = clf.score(X_valid, y_valid)
    print('{} {} Accuracy:'.format(datetime.now(),i), accuracy_lin)
