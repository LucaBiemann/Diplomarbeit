import warnings
warnings.simplefilter(action='ignore')
import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


class BERT(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class binaryClassification(nn.Module):
    def __init__(self, num_feature, num_class):
        super(binaryClassification, self).__init__()
        # Number of input features is 12.
        self.layer_1 = nn.Linear(num_feature, 128)
        self.layer_2 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, num_class)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(64)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return torch.sigmoid(x)


key_word_list1 = ['usb stick', 'extortion', 'password', 'trade secret', 'intimidate', 'return', 'blackmail', 'reveal']
key_word_list2 = ['obfuscate', 'undervaluation', 'conceal',
                  'fictitious transaction', 'overvaluation', 'tax evasion', 'underrate',
                  'overstate']
key_word_list3 = [
    'inside information',
    'trust', 'major contract', 'recommend', 'insolvency', 'merge',
    'step down', 'takeover bid']
names1 = ['James Smith', 'Mary Johnson', 'Robert Williams']
names2 = ['Patricia Brown', 'John Jones', 'Jennifer Garcia']
names3 = ['Michael Miller', 'Linda Davis', 'William Rodriguez']

dataset = pd.read_pickle('keywordmails_test_t5')
data_man = dataset[dataset.labeled == 1]
data_man.reset_index(drop=True, inplace=True)


def AttackFunction(vektor, white_box=True):
    if white_box:
        v = 'wb'
    else:
        v = 'bb'
    vektor = np.array(vektor).reshape(1, -1)
    loaded_model = pickle.load(open('models/mail/{}/MLP_gplm.sav'.format(v), 'rb'))
    model_typ = loaded_model.predict(vektor)
    if model_typ == 0:
        model = 'BERT'
    elif model_typ == 1:
        model = 'XLNet'
    elif model_typ == 2:
        model = 'ERNIE'
    else:
        model = 'T5'
    print('Model: ', model)
    scaler = pickle.load(open('models/mail/{}/scaler_keywords_{}.pkl'.format(v, model), 'rb'))
    X_input = scaler.transform(vektor)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_dataset_binary_front = BERT(torch.from_numpy(np.array(X_input)).float(),
                                     torch.from_numpy(np.array([0])).long())

    test_loader_binary_front = DataLoader(dataset=test_dataset_binary_front, batch_size=1)
    model_binary_front = torch.load(
        'models/mail/{}/{}/Binary_Verdacht_{}'.format(v, model, model),
        map_location=torch.device('cuda:0'))
    y_pred_list = []
    with torch.no_grad():
        model_binary_front.eval()
        for X_batch, _ in test_loader_binary_front:
            X_batch = X_batch.to(device)
            y_test_pred = model_binary_front(X_batch)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.cpu().numpy())
    y_pred_list_binary_front = [a.squeeze().tolist() for a in y_pred_list]
    verd = y_pred_list_binary_front[0]
    if verd == 0:
        return print('Unverdächtig')

    filename = 'models/mail/{}/{}/MLP_{}_crime.sav'.format(v, model, model)
    loaded_model = pickle.load(open(filename, 'rb'))
    probs = loaded_model.predict(X_input)
    if probs == 0:
        return print('Keine Delikt-Art indetifiziert')

    elif probs == 1:
        print('Delikt: Industriespionage')
        keywords = key_word_list1
        names = names1
    elif probs == 2:
        print('Delikt: Bilanzfälschung')
        keywords = key_word_list2
        names = names2
    else:
        print('Delikt: Insiderhandel')
        keywords = key_word_list3
        names = names3
    keywords_list = []
    for k in keywords:
        filename = 'models/mail/{}/{}/Linear SVM_{}_{}.sav'.format(v, model, model, k)
        loaded_model = pickle.load(open(filename, 'rb'))
        probs = loaded_model.predict(X_input)[0]
        wsk = loaded_model.predict_proba(X_input)

        if probs == 1:
            keywords_list.append(k+' '+str(round(wsk[0][1]*100))+'%')

    if keywords_list:
        print('Verdächtige Wörter gefunden: ',keywords_list)
    else:
        return print('Keine vedächtigen Wörter gefunden')
    name_list = []
    for n in names:
        filename = 'models/mail/{}/{}/Linear SVM_{}_{}.sav'.format(v, model, model, n)
        loaded_model = pickle.load(open(filename, 'rb'))
        probs = loaded_model.predict(X_input)[0]
        wsk = loaded_model.predict_proba(X_input)
        if probs == 1:
            name_list.append(n+' '+str(round(wsk[0][1]*100))+'%')
    if name_list:
        print('Namen identifiziert: ',name_list)
    else:
        return print('Keine Namen indentifiziertbar')


for i in range(10):
    AttackFunction(data_man.loc[i, 'BERTemb'], white_box=False)
