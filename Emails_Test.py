import warnings
warnings.simplefilter(action='ignore')
import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from pandas import ExcelWriter

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

dataset = pd.read_pickle('Emails_SpamFilter_Test')
Data_erg_spam = pd.DataFrame([])
for i in ['BERT','XLNet','ERNIE','GPT2' ,'T5']:
    scaler = pickle.load(open('models/spam/scaler_spam_{}.pkl'.format(i), 'rb'))
    X_input = scaler.transform(list(dataset['{}emb'.format(i)].values))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_dataset_binary_front = BERT(torch.from_numpy(np.array(X_input)).float(), torch.from_numpy(np.array(dataset['label_num'])).long())
    test_loader_binary_front = DataLoader(dataset=test_dataset_binary_front, batch_size=1)
    model_binary_front = torch.load(
            'models/spam/Binary_Spam_{}'.format(i),
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
    dataset['predicted'] = y_pred_list_binary_front
    for index, row in dataset.iterrows():
        if row['predicted'] == row['label_num']:
            dataset.loc[index, 'verd'] = 1
        else:
            dataset.loc[index, 'verd'] = 0
        if row['predicted'] == 1 and row['label_num'] == 1:
            dataset.loc[index, 'man'] = 1
        elif row['predicted'] == 0 and row['label_num'] == 1:
            dataset.loc[index, 'man'] = 0
    print(i, ' Precision: ', sum(dataset['verd'])/len(dataset))
    Data_erg_spam = Data_erg_spam.append({'Klassifikator':'Spam' ,i: sum(dataset['verd'])/len(dataset)}, ignore_index=True)
Data_erg_spam = Data_erg_spam.groupby(by=['Klassifikator']).agg(max)


dataset = pd.read_pickle('Emails_Test')
data_man = dataset[dataset.labeled == 1]


for v in ['wb', 'bb']:
    Data_erg = pd.DataFrame([])
    for m in ['BERT', 'XLNet', 'ERNIE', 'T5']:
        scaler = pickle.load(open('models/mail/{}/scaler_keywords_{}.pkl'.format(v,m), 'rb'))
        X_input = scaler.transform(list(dataset['{}emb'.format(m)].values))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        test_dataset_binary_front = BERT(torch.from_numpy(np.array(X_input)).float(),
                                         torch.from_numpy(np.array(dataset['labeled'])).long())
        test_loader_binary_front = DataLoader(dataset=test_dataset_binary_front, batch_size=1)
        model_binary_front = torch.load(
            'models/mail/{}/{}/Binary_Verdacht_{}'.format(v,m, m),
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
        dataset.loc[:, 'verd_{}_pred'.format(m)] = y_pred_list_binary_front
        for index, row in dataset.iterrows():
            if row['verd_{}_pred'.format(m)] == row['labeled']:
                dataset.loc[index, 'verd_{}'.format(m)] = 1
            else:
                dataset.loc[index, 'verd_{}'.format(m)] = 0
        print(m, ' Verd Alle Mails: ', len(data_man), '| Richtig: ', sum(dataset['verd_{}_pred'.format(m)]),
              '| Precision: ',
              sum(dataset['verd_{}'.format(m)]) / len(dataset))
        Data_erg = Data_erg.append(
            {'Klassifikator': 'Verdacht', '{}'.format(m): sum(dataset['verd_{}'.format(m)]) / len(dataset)},
            ignore_index=True)

        dataset_verd = dataset[dataset['verd_{}_pred'.format(m)] == 1]
        print(sum(dataset_verd['labeled']))
        X_input = scaler.transform(list(dataset_verd['{}emb'.format(m)].values))
        filename = 'models/mail/{}/{}/MLP_{}_crime.sav'.format(v,m, m)
        loaded_model = pickle.load(open(filename, 'rb'))
        probs = loaded_model.predict(X_input)
        dataset_verd.loc[:, 'crime_{}_pred'.format(m)] = probs
        for index, row in dataset_verd.iterrows():
            if row['crime_{}_pred'.format(m)] == row['label_crime']:
                dataset_verd.loc[index, 'crime_{}'.format(m)] = 1
            else:
                dataset_verd.loc[index, 'crime_{}'.format(m)] = 0
        print(m, ' Crime Alle Mails: ', len(dataset_verd), '| Richtig: ', sum(dataset_verd['crime_{}'.format(m)]),
              '| Precision: ',
              sum(dataset_verd['crime_{}'.format(m)]) / len(dataset_verd))
        print(0, len(dataset_verd[dataset_verd['crime_{}_pred'.format(m)] == 0]))
        print(1, len(dataset_verd[dataset_verd['crime_{}_pred'.format(m)] == 1]))
        print(2, len(dataset_verd[dataset_verd['crime_{}_pred'.format(m)] == 2]))
        print(3, len(dataset_verd[dataset_verd['crime_{}_pred'.format(m)] == 3]))
        Data_erg = Data_erg.append(
            {'Klassifikator': 'Crime', '{}'.format(m): sum(dataset_verd['crime_{}'.format(m)]) / len(dataset_verd)},
            ignore_index=True)
        for c, k, n in zip([1, 2, 3], [key_word_list1, key_word_list2, key_word_list3], [names1, names2, names3]):
            dataset_crime = dataset_verd[dataset_verd['crime_{}_pred'.format(m)] == c]
            dataset_crime_all = dataset[dataset['label_crime'] == c]
            print(sum(dataset_crime.loc[dataset_crime.label_crime == c, 'labeled']))
            print(sum(dataset_crime_all['labeled']))
            X_input = scaler.transform(list(dataset_crime['{}emb'.format(m)].values))
            dataset_crime['keywords_{}'.format(m)] = 0
            dataset_crime['KeyCorr_{}'.format(m)] = 0
            for i in k:
                filename = 'models/mail/{}/{}/Linear SVM_{}_{}.sav'.format(v,m,m, i)
                loaded_model = pickle.load(open(filename, 'rb'))
                probs = loaded_model.predict(X_input)
                dataset_crime.loc[:, '{}_{}_pred'.format(i, m)] = probs

                for index, row in dataset_crime.iterrows():
                    if row['{}_{}_pred'.format(i, m)] == 1:
                        dataset_crime.loc[index, 'keywords_{}'.format(m)] += 1
                    if row['{}_{}_pred'.format(i, m)] == 1 and row[i] == 1:
                        dataset_crime.loc[index, 'KeyCorr_{}'.format(m)] = 1
                    if row['{}_{}_pred'.format(i, m)] == row[i]:
                        dataset_crime.loc[index, '{}_{}'.format(i, m)] = 1
                    else:
                        dataset_crime.loc[index, '{}_{}'.format(i, m)] = 0
                print(m, i, 'Alle Mails: ', len(dataset_crime), '| Enthält Wort: ', len(dataset_crime[dataset_crime[i] == 1]) ,'| Pred Enthält Kein Wort: ', len(dataset_crime[dataset_crime['{}_{}_pred'.format(i, m)] == 0]), '| Pred Enthält Wort: ', len(dataset_crime[dataset_crime['{}_{}_pred'.format(i, m)] == 1]),
                      '| Precision: ', sum(dataset_crime['{}_{}'.format(i, m)]) / len(dataset_crime))
                Data_erg = Data_erg.append(
                    {'Klassifikator': i, '{}'.format(m): sum(dataset_crime['{}_{}'.format(i, m)]) / len(dataset_crime)},
                    ignore_index=True)
            print(m, c, len(dataset_crime[dataset_crime['keywords_{}'.format(m)] > 0]))
            date_names = dataset_crime[dataset_crime['keywords_{}'.format(m)] > 0]
            date_names['names_{}'.format(m)] = 0
            date_names['NameCorr_{}'.format(m)] = 0
            print(m, c, len(date_names[date_names['KeyCorr_{}'.format(m)] == 1]))
            X_input = scaler.transform(list(date_names['{}emb'.format(m)].values))
            for i in n:
                filename = 'models/mail/{}/{}/Linear SVM_{}_{}.sav'.format(v,m,m, i)
                loaded_model = pickle.load(open(filename, 'rb'))
                probs = loaded_model.predict(X_input)
                date_names.loc[:, '{}_{}_pred'.format(i, m)] = probs
                for index, row in date_names.iterrows():
                    if row['{}_{}_pred'.format(i, m)] == 1:
                        date_names.loc[index, 'names_{}'.format(m)] += 1
                    if row['{}_{}_pred'.format(i, m)] == 1 and row[i] == 1:
                        date_names.loc[index, 'NameCorr_{}'.format(m)] = 1
                    if row['{}_{}_pred'.format(i, m)] == row[i]:
                        date_names.loc[index, '{}_{}'.format(i, m)] = 1
                    else:
                        date_names.loc[index, '{}_{}'.format(i, m)] = 0
                print(m, i, 'Alle Mails: ', len(date_names), '| Enthält Wort: ',
                      len(date_names[date_names[i] == 1]), '| Pred Enthält Kein Wort: ',
                      len(date_names[date_names['{}_{}_pred'.format(i, m)] == 0]), '| Pred Enthält Wort: ',
                      len(date_names[date_names['{}_{}_pred'.format(i, m)] == 1]),
                      '| Precision: ', sum(date_names['{}_{}'.format(i, m)]) / len(date_names))
                Data_erg = Data_erg.append(
                    {'Klassifikator': i, '{}'.format(m): sum(date_names['{}_{}'.format(i, m)]) / len(date_names)},
                    ignore_index=True)
            print(m, c, len(date_names[date_names['names_{}'.format(m)] > 0]))
            date_final = date_names[(date_names['names_{}'.format(m)] > 0) & (date_names['KeyCorr_{}'.format(m)] == 1)]
            print(m, c, len(date_final[date_final['NameCorr_{}'.format(m)] == 1]))
        if v == 'wb':
            Data_erg_wb = Data_erg
        else:
            Data_erg_bb = Data_erg

Data_erg_wb = Data_erg_wb.groupby(by=['Klassifikator']).agg(max)
Data_erg_bb = Data_erg_bb.groupby(by=['Klassifikator']).agg(max)
with ExcelWriter("Email_Test.xlsx") as writer:
 Data_erg_spam.to_excel(writer, sheet_name="Spam-Filter")
 Data_erg_wb.to_excel(writer, sheet_name="White-Box")
 Data_erg_bb.to_excel(writer, sheet_name="Black-Box")

