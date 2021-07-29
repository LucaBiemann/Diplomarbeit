import warnings

warnings.simplefilter(action='ignore')
import pandas as pd
import pickle
import numpy as np
from pandas import ExcelWriter

Data_mes = pd.read_pickle('Messages_test')
Data_mes['NumMes'] = 0
Data_num = pd.read_pickle('Data_ausweis_test')
Data_num = Data_num.rename(columns={'XLNetEmb': 'XLNetemb', 'RoBERTaEmb': 'RoBERTaemb'})
Data_num['NumMes'] = 1
Data_all = pd.concat([Data_mes, Data_num])
Data_all = Data_all.sample(frac=1, random_state=42)
Data_all.reset_index(drop=True, inplace=True)

'''
for i in ['BERT','DistilBERT','XLNet', 'RoBERTa','ERNIE']:
    scaler = pickle.load(open('models/scaler_{}.pkl'.format(i), 'rb'))
    X_input = scaler.transform(list(Data[i+'emb'].values))
    for n in range(10):
        filename = 'models/{}/MLP_Zeichen{}.sav'.format(i, n)
        loaded_model = pickle.load(open(filename, 'rb'))
        probs = loaded_model.predict(X_input)
        Data[i+'erg'+str(n)] = probs
        for index, row in Data.iterrows():
            #print(row[i+'erg'+str(n)], row['Zeichen'+str(n)], row['text'])
            if row[i+'erg'+str(n)] == row['Zeichen'+str(n)]:
                Data.loc[index,i + 'pred' + str(n)] = 1
            else:
                Data.loc[index,i + 'pred' + str(n)] = 0
        print(i, 'Zeichen ', n, 'Accuracy: ', sum(Data[i + 'pred' + str(n)])/len(Data))

for i in ['BERT','DistilBERT','XLNet', 'RoBERTa','ERNIE']:
    scaler = pickle.load(open('models/scaler_{}.pkl'.format(i), 'rb'))
    X_input = scaler.transform(list(Data[i+'emb'].values))
    for n in [5]:
        filename = 'models/{}/MLP_Zeichen{}.sav'.format(i, n)
        loaded_model = pickle.load(open(filename, 'rb'))
        probs = loaded_model.predict_proba(X_input)
        if n == 0 or n == 9:
            u = 11
        else:
            u = 28
        for o in range(1,u):
            label_list = np.argsort(probs, axis=1)[:, -u:][0]

            for index, row in Data.iterrows():
                if row['Zeichen'+str(n)] in np.argsort(probs, axis=1)[:, -o:][index]:
                    Data.loc[index,i + 'pred' + str(n)] = 1
                else:
                    Data.loc[index,i + 'pred' + str(n)] = 0
            print(i, 'Top: ' ,o ,'Zeichen ', n, 'Accuracy: ', sum(Data[i + 'pred' + str(n)])/len(Data))
            Data_erg.loc[Data_erg.Top == o, 'Accuracy_{}'.format(i)] =sum(Data[i + 'pred' + str(n)])/len(Data)

with ExcelWriter("C:/Users/lucab/Desktop/Uni/Diplomarbeit/Daten/DatenAnhang.xlsx") as writer:
    Data_erg.to_excel(writer, sheet_name="topk")

Top = range(1, 28)
Data_erg = pd.DataFrame([])
Data_erg['Top'] = Top
for i in ['BERT', 'DistilBERT', 'XLNet', 'RoBERTa', 'ERNIE']:
    scaler = pickle.load(open('models/scaler_{}.pkl'.format(i), 'rb'))
    X_input = scaler.transform(list(Data[i + 'emb'].values))
    for n in range(2, 8):
        filename = 'models/{}/MLP_Zeichen{}.sav'.format(i, n)
        loaded_model = pickle.load(open(filename, 'rb'))
        probs = loaded_model.predict_proba(X_input)
        for o in range(1, 28):
            for index, row in Data.iterrows():
                if row['Zeichen' + str(n)] in np.argsort(probs, axis=1)[:, -o:][index]:
                    Data.loc[index, i + 'pred' + str(n) + 'top' + str(o)] = 1
                else:
                    Data.loc[index, i + 'pred' + str(n) + 'top' + str(o)] = 0
            # print(i, 'Top: ' ,o ,'Zeichen ', n, 'Accuracy: ', sum(Data[i + 'pred' + str(n)])/len(Data))
    for o in range(1, 28):
        for index, row in Data.iterrows():
            print(int(row[i + 'pred2' + 'top' + str(o)]))
            if sum([int(row[i + 'pred2' + 'top' + str(o)]), int(row[i + 'pred3' + 'top' + str(o)]),
                   int(row[i + 'pred4' + 'top' + str(o)]), int(row[i + 'pred5' + 'top' + str(o)]),
                   int(row[i + 'pred6' + 'top' + str(o)]), int(row[i + 'pred7' + 'top' + str(o)])]) == 6:
                Data.loc[index, 'True_{}_Top_{}'.format(i, o)] = 1
            else:
                Data.loc[index, 'True_{}_Top_{}'.format(i, o)] = 0
        Data_erg.loc[Data_erg.Top == o, 'Accuracy_{}'.format(i)] = sum(Data['True_{}_Top_{}'.format(i, o)]) / len(Data)
with ExcelWriter("C:/Users/lucab/Desktop/Uni/Diplomarbeit/Daten/DatenAnhang2.xlsx") as writer:
    Data_erg.to_excel(writer, sheet_name="topk")
'''
Top = range(1, 28)
Data_erg = pd.DataFrame([])
Data_erg2 = pd.DataFrame([])
Data_erg['Top'] = Top
for i in ['BERT', 'DistilBERT', 'XLNet', 'RoBERTa', 'ERNIE']:
    scaler = pickle.load(open('models/scaler_{}.pkl'.format(i), 'rb'))
    X_input = scaler.transform(list(Data_all[i + 'emb'].values))
    filename = 'models/{}/MLP_NumMes.sav'.format(i)
    loaded_model = pickle.load(open(filename, 'rb'))
    Data_all['{}NumMesPred'.format(i)] = loaded_model.predict(X_input)
    for index, row in Data_all.iterrows():
        if row['{}NumMesPred'.format(i)] == row['NumMes']:
            Data_all.loc[index, '{}NumMes'.format(i)] = 1
        else:
            Data_all.loc[index, '{}NumMes'.format(i)] = 0
    print('Model ', i, 'NumMes ', 'Accuray ', (sum(Data_all['{}NumMes'.format(i)] / len(Data_all))), 'richtig ',
          sum(Data_all['{}NumMes'.format(i)]))
    Data = Data_all[Data_all['{}NumMesPred'.format(i)] == 1]
    Data['{}PredAll'.format(i)] = 0
    Data['{}Pred2-7'.format(i)] = 0
    for k in range(1, 28):
        Data['{}Pred2-7K{}'.format(i, k)] = 0
    Data.reset_index(drop=True, inplace=True)
    X_input = scaler.transform(list(Data[i + 'emb'].values))
    for n in range(10):
        if (i not in ['DistilBERT', 'XLNet'] and n in [0, 1, 8, 9]) or (
                i in ['DistilBERT', 'XLNet'] and n in [0, 1, 2, 9]):
            filename = 'models/{}/MLP_Zeichen{}.sav'.format(i, n)
            loaded_model = pickle.load(open(filename, 'rb'))
            Data['{}Pred{}'.format(i, n)] = loaded_model.predict(X_input)
            for index, row in Data.iterrows():
                if row['Zeichen{}'.format(n)] == row['{}Pred{}'.format(i, n)]:
                    Data.loc[index, '{}Zeichen{}'.format(i, n)] = 1
                    Data.loc[index, '{}PredAll'.format(i)] += 1
                else:
                    Data.loc[index, '{}Zeichen{}'.format(i, n)] = 0
            print('Model ', i, 'Zeichen ', n, 'Accuray ', (sum(Data['{}Zeichen{}'.format(i, n)] / len(Data))))
        else:

            filename = 'models/{}/MLP_Zeichen{}.sav'.format(i, n)
            loaded_model = pickle.load(open(filename, 'rb'))
            probs = loaded_model.predict_proba(X_input)

            for k in range(1, 28):
                for index, row in Data.iterrows():
                    if row['Zeichen{}'.format(n)] in np.argsort(probs, axis=1)[:, -k:][index]:
                        Data.loc[index, '{}Zeichen{}K{}'.format(i, n, k)] = 1
                        Data.loc[index, '{}Pred2-7K{}'.format(i, k)] += 1

                    else:
                        Data.loc[index, '{}Zeichen{}K{}'.format(i, n, k)] = 0

                print('Model ', i, 'Zeichen ', n, 'K-Pool', k, 'Accuray ',
                      (sum(Data['{}Zeichen{}K{}'.format(i, n, k)] / len(Data))))
    for k in range(1, 28):
        for index, row in Data.iterrows():
            if row['{}Pred2-7K{}'.format(i, k)] == 6:
                Data.loc[index, '{}Zeichen2-7K{}'.format(i, k)] = 1
            else:
                Data.loc[index, '{}Zeichen2-7K{}'.format(i, k)] = 0
            if row['{}PredAll'.format(i)] + row['{}Pred2-7K{}'.format(i, k)] == 10:
                Data.loc[index, '{}AllK{}'.format(i, k)] = 1
            else:
                Data.loc[index, '{}AllK{}'.format(i, k)] = 0
        Data_erg.loc[Data_erg.Top == k, 'Accuracy_{}'.format(i)] = (sum(Data['{}AllK{}'.format(i, k)] / len(Data)))
        print('Model ', i, 'Zeichen 2-7', 'K-Pool', k, 'Accuray ',
              (sum(Data['{}Zeichen2-7K{}'.format(i, k)] / len(Data))))
        print('Model ', i, 'All ', 'K-Pool', k, 'Accuray ', (sum(Data['{}AllK{}'.format(i, k)] / len(Data))))
    Data_erg2 = Data_erg2.append(Data)
    print(Data_erg2)
#with ExcelWriter("C:/Users/lucab/Desktop/Uni/Diplomarbeit/Daten/DatenAnhang3.xlsx") as writer:
 #   Data_erg.to_excel(writer, sheet_name="topkGes")
pd.to_pickle(Data_erg2, 'C:/Users/lucab/PycharmProjects/dipldata/Data_num_erg2')
print(Data)
print(Data.columns)
