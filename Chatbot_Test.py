import warnings

warnings.simplefilter(action='ignore')
import pandas as pd
import pickle
import numpy as np
from pandas import ExcelWriter
import random

Data_erg = pd.DataFrame([])
Data_all = pd.read_pickle('Chatbot_Test')
X_input = list(Data_all['vektor'])
filename = 'models/chatbot/MLP_Bot_Model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
probs = loaded_model.predict(X_input)
Data_all['Model'] = probs
for index, row in Data_all.iterrows():
    if row['Model'] == row['label']:
        Data_all.loc[index, 'Model_pred'] = 1
    else:
        Data_all.loc[index, 'Model_pred'] = 0
print('GPLM Accuracy: ', sum(Data_all['Model_pred']) / len(Data_all))
for m in ['BERT', 'DistilBERT', 'XLNet', 'RoBERTa', 'ERNIE']:
    scaler = pickle.load(open('models/chatbot/scaler_{}.pkl'.format(m), 'rb'))
    X_input = scaler.transform(list(Data_all[m + 'emb'].values))
    filename = 'models/chatbot/{}/MLP_NumMes.sav'.format(m)
    loaded_model = pickle.load(open(filename, 'rb'))
    probs = loaded_model.predict(X_input)
    Data_all[m + 'NumMes'] = probs
    Data_Num = Data_all[Data_all[m + 'NumMes'] == 1]
    X_input = scaler.transform(list(Data_Num[m + 'emb'].values))
    Data_Num[m + 'All'] = 0
    Data_Num[m + 'Topk'] = 0
    Data_Num.reset_index(drop=True, inplace=True)
    print('Ausweisnummern erkannt: ', len(Data_Num))
    for i in range(10):
        filename = 'models/chatbot/{}/MLP_Zeichen{}.sav'.format(m, i)
        loaded_model = pickle.load(open(filename, 'rb'))
        if (i in [0, 9]) or (i in [1, 8] and m in ['BERT', 'RoBERTa', 'ERNIE']) or (
                i in [1, 2] and m in ['DistilBERT', 'XLNet']):
            probs = loaded_model.predict(X_input)
            Data_Num[m + 'erg' + str(i)] = probs
            for index, row in Data_Num.iterrows():
                if row['Zeichen' + str(i)] == row[m + 'erg' + str(i)]:
                    Data_Num.loc[index, m + 'pred' + str(i)] = 1
                    Data_Num.loc[index, m + 'All'] += 1
                else:
                    Data_Num.loc[index, m + 'pred' + str(i)] = 0
        else:
            probs = loaded_model.predict_proba(X_input)
            probs_list = np.argsort(probs, axis=1)[:, -10:]

            for index, row in Data_Num.iterrows():
                if row['Zeichen' + str(i)] in probs_list[index]:
                    Data_Num.loc[index, m + 'pred' + str(i)] = 1
                    Data_Num.loc[index, m + 'All'] += 1
                    Data_Num.loc[index, m + 'Topk'] += 1
                else:
                    Data_Num.loc[index, m + 'pred' + str(i)] = 0
        print(m, 'Zeichen ', i, 'Accuracy: ', sum(Data_Num[m + 'pred' + str(i)]) / len(Data_Num))
        Data_erg = Data_erg.append({'Zeichen': i, m: sum(Data_Num[m + 'pred' + str(i)]) / len(Data_Num)}, ignore_index=True)
    for index, row in Data_Num.iterrows():
        if row[m + 'All'] == 10:
            Data_Num.loc[index, m + 'Allpred'] = 1
        else:
            Data_Num.loc[index, m + 'Allpred'] = 0

        if row[m + 'Topk'] == 6:
            Data_Num.loc[index, m + 'Topk'] = 1
        else:
            Data_Num.loc[index, m + 'Topk'] = 0
    print(m, 'Zeichen Topk ', 'Accuracy: ', sum(Data_Num[m + 'Topk']) / len(Data_Num))
    print(m, 'All ', 'Accuracy: ', sum(Data_Num[m + 'Allpred']) / len(Data_Num))
    Data_erg = Data_erg.append({'Zeichen': 'Topk', m: sum(Data_Num[m + 'Topk']) / len(Data_Num)}, ignore_index=True)
    Data_erg = Data_erg.append({'Zeichen': 'All', m: sum(Data_Num[m + 'Allpred']) / len(Data_Num)}, ignore_index=True)
Data_erg = Data_erg.groupby(by=['Zeichen']).agg(max)
with pd.ExcelWriter("Chatbot_Test.xlsx") as writer:
    Data_erg.to_excel(writer, sheet_name="Chatbot_Test")
