import warnings
warnings.simplefilter(action='ignore')
import numpy as np
import pandas as pd
import pickle


Data = pd.read_pickle('Chatbot_Test')
Top = range(1, 28)

Data_erg_topk = pd.DataFrame([])
Data_erg_pos = pd.DataFrame([])
Data_erg_zeichen = pd.DataFrame([])
Data_erg_z5 = pd.DataFrame([])
Data_erg_gestopk = pd.DataFrame([])
Data_erg = pd.DataFrame([])
Data_erg_z5['Top'] = Top
Data_erg_gestopk['Top'] = Top
Data_erg_topk['Top'] = Top


for m in ['BERT', 'DistilBERT', 'XLNet', 'RoBERTa', 'ERNIE']:
    scaler = pickle.load(open('models/chatbot/scaler_{}.pkl'.format(m), 'rb'))
    X_input = scaler.transform(list(Data[m + 'emb'].values))
    for n in range(10):
        filename = 'models/chatbot/{}/MLP_Zeichen{}.sav'.format(m, n)
        loaded_model = pickle.load(open(filename, 'rb'))
        probs = loaded_model.predict(X_input)
        Data[m + 'erg' + str(n)] = probs
        for index, row in Data.iterrows():
            if row[m + 'erg' + str(n)] == row['Zeichen' + str(n)]:
                Data.loc[index, m + 'pred' + str(n)] = 1
            else:
                Data.loc[index, m + 'pred' + str(n)] = 0
        print(m, 'Zeichen ', n, 'Accuracy: ', sum(Data[m + 'pred' + str(n)]) / len(Data))
        Data_erg_zeichen = Data_erg_zeichen.append({'Zeichen': n, m: sum(Data[m + 'pred' + str(n)]) / len(Data)},
                                                   ignore_index=True)
for i in ['BERT', 'DistilBERT', 'XLNet', 'RoBERTa', 'ERNIE']:
    scaler = pickle.load(open('models/chatbot/scaler_{}.pkl'.format(i), 'rb'))
    X_input = scaler.transform(list(Data[i + 'emb'].values))
    for n in [5]:
        filename = 'models/chatbot/{}/MLP_Zeichen{}.sav'.format(i, n)
        loaded_model = pickle.load(open(filename, 'rb'))
        probs = loaded_model.predict_proba(X_input)
        if n == 0 or n == 9:
            u = 11
        else:
            u = 28
        for o in range(1, u):
            label_list = np.argsort(probs, axis=1)[:, -u:][0]

            for index, row in Data.iterrows():
                if row['Zeichen' + str(n)] in np.argsort(probs, axis=1)[:, -o:][index]:
                    Data.loc[index, i + 'pred' + str(n)] = 1
                else:
                    Data.loc[index, i + 'pred' + str(n)] = 0
            print(i, 'Top: ', o, 'Zeichen ', n, 'Accuracy: ', sum(Data[i + 'pred' + str(n)]) / len(Data))
            Data_erg_z5.loc[Data_erg_z5.Top == o, 'Accuracy_{}'.format(i)] = sum(Data[i + 'pred' + str(n)]) / len(Data)


for i in ['BERT', 'DistilBERT', 'XLNet', 'RoBERTa', 'ERNIE']:
    scaler = pickle.load(open('models/chatbot/scaler_{}.pkl'.format(i), 'rb'))
    X_input = scaler.transform(list(Data[i + 'emb'].values))
    if i in ['BERT', 'RoBERTa', 'ERNIE']:
        for n in range(2, 8):
            filename = 'models/chatbot/{}/MLP_Zeichen{}.sav'.format(i, n)
            loaded_model = pickle.load(open(filename, 'rb'))
            probs = loaded_model.predict_proba(X_input)
            for o in range(1, 28):
                for index, row in Data.iterrows():
                    if row['Zeichen' + str(n)] in np.argsort(probs, axis=1)[:, -o:][index]:
                        Data.loc[index, i + 'pred' + str(n) + 'top' + str(o)] = 1
                    else:
                        Data.loc[index, i + 'pred' + str(n) + 'top' + str(o)] = 0

        for o in range(1, 28):
            for index, row in Data.iterrows():
                if sum([int(row[i + 'pred2' + 'top' + str(o)]), int(row[i + 'pred3' + 'top' + str(o)]),
                        int(row[i + 'pred4' + 'top' + str(o)]), int(row[i + 'pred5' + 'top' + str(o)]),
                        int(row[i + 'pred6' + 'top' + str(o)]), int(row[i + 'pred7' + 'top' + str(o)])]) == 6:
                    Data.loc[index, 'True_{}_Top_{}'.format(i, o)] = 1
                else:
                    Data.loc[index, 'True_{}_Top_{}'.format(i, o)] = 0
            print(i, 'Top: ', o, 'Zeichen ', '2-7', 'Accuracy: ', sum(Data['True_{}_Top_{}'.format(i, o)]) / len(Data))
            Data_erg_topk.loc[Data_erg_topk.Top == o, 'Accuracy_{}'.format(i)] = sum(Data['True_{}_Top_{}'.format(i, o)]) / len(Data)
    else:
        for n in range(3, 9):
            filename = 'models/chatbot/{}/MLP_Zeichen{}.sav'.format(i, n)
            loaded_model = pickle.load(open(filename, 'rb'))
            probs = loaded_model.predict_proba(X_input)
            for o in range(1, 28):
                for index, row in Data.iterrows():
                    if row['Zeichen' + str(n)] in np.argsort(probs, axis=1)[:, -o:][index]:
                        Data.loc[index, i + 'pred' + str(n) + 'top' + str(o)] = 1
                    else:
                        Data.loc[index, i + 'pred' + str(n) + 'top' + str(o)] = 0

        for o in range(1, 28):
            for index, row in Data.iterrows():
                if sum([int(row[i + 'pred3' + 'top' + str(o)]), int(row[i + 'pred4' + 'top' + str(o)]),
                        int(row[i + 'pred5' + 'top' + str(o)]), int(row[i + 'pred6' + 'top' + str(o)]),
                        int(row[i + 'pred7' + 'top' + str(o)]), int(row[i + 'pred8' + 'top' + str(o)])]) == 6:
                    Data.loc[index, 'True_{}_Top_{}'.format(i, o)] = 1
                else:
                    Data.loc[index, 'True_{}_Top_{}'.format(i, o)] = 0
            print(i, 'Top: ', o, 'Zeichen ', '3-8', 'Accuracy: ',sum(Data['True_{}_Top_{}'.format(i, o)]) / len(Data))
            Data_erg_topk.loc[Data_erg_topk.Top == o, 'Accuracy_{}'.format(i)] = sum(Data['True_{}_Top_{}'.format(i, o)]) / len(Data)




Data_mes = pd.read_pickle('Messages_test')
Data_mes['NumMes'] = 0
Data['NumMes'] = 1
Data_all = pd.concat([Data_mes, Data])
Data_all = Data_all.sample(frac=1, random_state=42)
Data_all.reset_index(drop=True, inplace=True)

for i in ['BERT', 'DistilBERT', 'XLNet', 'RoBERTa', 'ERNIE']:
    scaler = pickle.load(open('models/chatbot/scaler_{}.pkl'.format(i), 'rb'))
    X_input = scaler.transform(list(Data_all[i + 'emb'].values))
    filename = 'models/chatbot/{}/MLP_NumMes.sav'.format(i)
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
            filename = 'models/chatbot/{}/MLP_Zeichen{}.sav'.format(i, n)
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

            filename = 'models/chatbot/{}/MLP_Zeichen{}.sav'.format(i, n)
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
        Data_erg_gestopk.loc[Data_erg_gestopk.Top == k, 'Accuracy_{}'.format(i)] = (sum(Data['{}AllK{}'.format(i, k)] / len(Data)))
        print('Model ', i, 'Zeichen 2-7', 'K-Pool', k, 'Accuray ',
              (sum(Data['{}Zeichen2-7K{}'.format(i, k)] / len(Data))))
        print('Model ', i, 'All ', 'K-Pool', k, 'Accuray ', (sum(Data['{}AllK{}'.format(i, k)] / len(Data))))
    Data_erg = Data_erg.append(Data)

for m in ['BERT', 'DistilBERT', 'XLNet', 'RoBERTa', 'ERNIE']:
    Data_test = Data_erg[Data_erg['{}AllK10'.format(m)] == 1]
    Data_test = Data_test.sample(n=100, random_state=42)
    Data_test.reset_index(drop=True, inplace=True)
    for vektor, target, number in zip(Data_test[m + 'emb'], Data_test['text'], range(len(Data_test))):
        vektor = np.array(vektor).reshape(1, -1)
        loaded_model = pickle.load(open('models/chatbot/MLP_Bot_Model.sav', 'rb'))
        model_typ = loaded_model.predict(vektor)
        if model_typ == 0:
            model = 'BERT'
        elif model_typ == 1:
            model = 'DistilBERT'
        elif model_typ == 2:
            model = 'XLNet'
        elif model_typ == 3:
            model = 'RoBERTa'
        else:
            model = 'ERNIE'
        print('Model: ', model)
        scaler = pickle.load(open('models/chatbot/scaler_{}.pkl'.format(model), 'rb'))
        X_input = scaler.transform(vektor)

        loaded_model = pickle.load(open('models/chatbot/{}/MLP_NumMes.sav'.format(model), 'rb'))

        start = ['L', 'M', 'N', 'P', 'R', 'T', 'V', 'W', 'X', 'Y']
        start_map = pd.DataFrame([], columns=['start', 'label'])
        start_map['zeichen'] = start
        start_map['label'] = range(len(start))
        zeichen = ['C', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'T', 'V', 'W', 'X', 'Y', 'Z']
        zeichen_map = pd.DataFrame([], columns=['start', 'label'])
        zeichen_map['zeichen'] = zeichen
        zeichen_map['label'] = range(10, len(zeichen) + 10)
        buchstaben = pd.DataFrame(columns=['buchstabe', 'wert'])
        buchstaben['buchstabe'] = ['C', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'T', 'V', 'W', 'X', 'Y', 'Z']
        buchstaben['wert'] = [12, 15, 16, 17, 19, 20, 21, 22, 23, 25, 27, 29, 31, 32, 33, 34, 35]
        if model in ['DistilBERT', 'XLNet']:
            arrays_col = [
                ['Zeichen0', 'Zeichen0', 'Zeichen1', 'Zeichen1', 'Zeichen2', 'Zeichen2', 'Zeichen3', 'Zeichen3',
                 'Zeichen3', 'Zeichen4', 'Zeichen4', 'Zeichen4', 'Zeichen5', 'Zeichen5', 'Zeichen5', 'Zeichen6',
                 'Zeichen6',
                 'Zeichen6', 'Zeichen7', 'Zeichen7', 'Zeichen7', 'Zeichen8', 'Zeichen8', 'Zeichen8', 'Zeichen9'],
                ['Zeichen_0', 'Zeichen_0_value', 'Zeichen_1', 'Zeichen_1_value',
                 'Zeichen_2', 'Zeichen_2_value', 'Zeichen_3_wsk',
                 'Zeichen_3', 'Zeichen_3_value', 'Zeichen_4_wsk', 'Zeichen_4',
                 'Zeichen_4_value', 'Zeichen_5_wsk', 'Zeichen_5', 'Zeichen_5_value',
                 'Zeichen_6_wsk', 'Zeichen_6', 'Zeichen_6_value', 'Zeichen_7_wsk',
                 'Zeichen_7', 'Zeichen_7_value', 'Zeichen_8_wsk', 'Zeichen_8', 'Zeichen_8_value',
                 'Zeichen_9']]
        else:
            arrays_col = [
                ['Zeichen0', 'Zeichen0', 'Zeichen1', 'Zeichen1', 'Zeichen2', 'Zeichen2', 'Zeichen2', 'Zeichen3',
                 'Zeichen3',
                 'Zeichen3', 'Zeichen4', 'Zeichen4', 'Zeichen4', 'Zeichen5', 'Zeichen5', 'Zeichen5', 'Zeichen6',
                 'Zeichen6',
                 'Zeichen6', 'Zeichen7', 'Zeichen7', 'Zeichen7', 'Zeichen8', 'Zeichen8', 'Zeichen9'],
                ['Zeichen_0', 'Zeichen_0_value', 'Zeichen_1', 'Zeichen_1_value',
                 'Zeichen_2_wsk', 'Zeichen_2', 'Zeichen_2_value', 'Zeichen_3_wsk',
                 'Zeichen_3', 'Zeichen_3_value', 'Zeichen_4_wsk', 'Zeichen_4',
                 'Zeichen_4_value', 'Zeichen_5_wsk', 'Zeichen_5', 'Zeichen_5_value',
                 'Zeichen_6_wsk', 'Zeichen_6', 'Zeichen_6_value', 'Zeichen_7_wsk',
                 'Zeichen_7', 'Zeichen_7_value', 'Zeichen_8', 'Zeichen_8_value',
                 'Zeichen_9']]
        tuples_col = list(zip(*arrays_col))
        index_col = pd.MultiIndex.from_tuples(tuples_col, names=['first', 'second'])

        Data_vektor = pd.DataFrame([], index=range(10), columns=index_col)

        for i in range(10):
            filename = 'models/chatbot/{}/MLP_Zeichen{}.sav'.format(model, i)
            loaded_model = pickle.load(open(filename, 'rb'))
            probs = loaded_model.predict_proba(X_input)
            if i in [0, 9]:
                label = int(np.argsort(probs, axis=1)[:, -1:])
                value = label if i == 9 else start_map.loc[start_map.label == label, 'zeichen'].values[0]
                Data_vektor.loc[0, ('Zeichen{}'.format(i), 'Zeichen_{}'.format(i))] = value
                if i == 0:
                    produkt = buchstaben.loc[buchstaben.buchstabe == value, 'wert'].values[0] * 7
                    Data_vektor.loc[0, ('Zeichen{}'.format(i), 'Zeichen_{}_value'.format(i))] = int(str(produkt)[-1])
            elif i in [1, 8] and model not in ['DistilBERT', 'XLNet']:
                multi = 3 if i == 1 else 1
                label = int(np.argsort(probs, axis=1)[:, -1:])
                value = label if label < 10 else zeichen_map.loc[zeichen_map.label == label, 'zeichen'].values[0]
                Data_vektor.loc[0, ('Zeichen{}'.format(i), 'Zeichen_{}'.format(i))] = value
                produkt = value * multi if str(value).isnumeric() else \
                    buchstaben.loc[buchstaben.buchstabe == value, 'wert'].values[0] * multi
                Data_vektor.loc[0, ('Zeichen{}'.format(i), 'Zeichen_{}_value'.format(i))] = int(str(produkt)[-1])
            elif i in [1, 2] and model in ['DistilBERT', 'XLNet']:
                multi = 3 if i == 1 else 1
                label = int(np.argsort(probs, axis=1)[:, -1:])
                value = label if label < 10 else zeichen_map.loc[zeichen_map.label == label, 'zeichen'].values[0]
                Data_vektor.loc[0, ('Zeichen{}'.format(i), 'Zeichen_{}'.format(i))] = value
                produkt = value * multi if str(value).isnumeric() else \
                    buchstaben.loc[buchstaben.buchstabe == value, 'wert'].values[0] * multi
                Data_vektor.loc[0, ('Zeichen{}'.format(i), 'Zeichen_{}_value'.format(i))] = int(str(produkt)[-1])

            else:
                label_list = np.argsort(probs, axis=1)[:, -10:][0]
                row = 0

                Data_vektor.at[0:9, ('Zeichen{}'.format(i), 'Zeichen_{}_wsk'.format(i))] = probs[0][label_list]

                for label in label_list:
                    multi = 7 if i in [3, 6] else 3
                    value = int(label) if label < 10 else zeichen_map.loc[zeichen_map.label == label, 'zeichen'].values[
                        0]
                    Data_vektor.loc[row, ('Zeichen{}'.format(i), 'Zeichen_{}'.format(i))] = value
                    produkt = value * (multi if i in [3, 4, 6, 7] else 1) if str(value).isnumeric() else \
                        buchstaben.loc[buchstaben.buchstabe == value, 'wert'].values[0] * (
                            multi if i in [3, 4, 6, 7] else 1)
                    Data_vektor.loc[row, ('Zeichen{}'.format(i), 'Zeichen_{}_value'.format(i))] = int(str(produkt)[-1])
                    row += 1

        if model in ['DistilBERT', 'XLNet']:
            Data_vektor = Data_vektor.rename(
                columns={'Zeichen2': 'Zeichen8', 'Zeichen8': 'Zeichen2', 'Zeichen_2': 'Zeichen_8',
                         'Zeichen_2_value': 'Zeichen_8_value', 'Zeichen_8': 'Zeichen_2',
                         'Zeichen_8_value': 'Zeichen_2_value', 'Zeichen_8_wsk': 'Zeichen_2_wsk'})

        Kandidat_0, Zeichen_0 = Data_vektor.loc[0, 'Zeichen0']
        Kandidat_1, Zeichen_1 = Data_vektor.loc[0, 'Zeichen1']
        Kandidat_8, Zeichen_8 = Data_vektor.loc[0, 'Zeichen8']
        Zeichen_9 = Data_vektor.loc[0, 'Zeichen9'].values[0]
        Summe018 = sum([Zeichen_0, Zeichen_1, Zeichen_8])
        keyword_list = pickle.load(open('iter_list.pkl', 'rb'))

        Kandidaten = pd.DataFrame([])
        Kandict = {'{}{}'.format(o, i): int(Data_vektor.loc[i, ('Zeichen{}'.format(o), 'Zeichen_{}_value'.format(o))])
                   for i
                   in range(10) for o in range(2, 8)}

        for i in keyword_list:

            Summe = str(int(sum(
                [Kandict['2{}'.format(i[0])], Kandict['3{}'.format(i[1])], Kandict['4{}'.format(i[2])],
                 Kandict['5{}'.format(i[3])],
                 Kandict['6{}'.format(i[4])], Kandict['7{}'.format(i[5])], Summe018])))

            if int(Summe[-1]) == Zeichen_9:
                wsk_2, Kandidat_2, Zeichen_2 = Data_vektor.loc[int(i[0]), 'Zeichen2']
                wsk_3, Kandidat_3, Zeichen_3 = Data_vektor.loc[int(i[1]), 'Zeichen3']
                wsk_4, Kandidat_4, Zeichen_4 = Data_vektor.loc[int(i[2]), 'Zeichen4']
                wsk_5, Kandidat_5, Zeichen_5 = Data_vektor.loc[int(i[3]), 'Zeichen5']
                wsk_6, Kandidat_6, Zeichen_6 = Data_vektor.loc[int(i[4]), 'Zeichen6']
                wsk_7, Kandidat_7, Zeichen_7 = Data_vektor.loc[int(i[5]), 'Zeichen7']
                wsk = sum([wsk_2, wsk_3, wsk_4, wsk_5, wsk_6, wsk_7])
                if model in ['DistilBERT', 'XLNet']:
                    kandidat = f'{Kandidat_0}{Kandidat_1}{Kandidat_8}{Kandidat_3}{Kandidat_4}{Kandidat_5}{Kandidat_6}{Kandidat_7}{Kandidat_2}{Zeichen_9}'
                else:
                    kandidat = f'{Kandidat_0}{Kandidat_1}{Kandidat_2}{Kandidat_3}{Kandidat_4}{Kandidat_5}{Kandidat_6}{Kandidat_7}{Kandidat_8}{Zeichen_9}'
                Kandidaten = Kandidaten.append({'text': kandidat, 'wsk': wsk}, ignore_index=True)
        print('Größe des Kandidaten-Pools: ', len(Kandidaten))
        Kandidaten = Kandidaten.sort_values(by='wsk', ascending=False)
        Kandidaten.reset_index(drop=True, inplace=True)
        Position = Kandidaten[Kandidaten.text == target].index.values[0]
        print('Position:', Position + 1)
        Data_erg_pos = Data_erg_pos.append({'Number': number, m: Position}, ignore_index=True)
Data_erg_pos = Data_erg_pos.groupby(by=['Number']).agg(max)
with pd.ExcelWriter("Daten/Chatbot_Anhang.xlsx") as writer:
    Data_erg_zeichen.to_excel(writer, sheet_name="Stelle")
    Data_erg_z5.to_excel(writer, sheet_name='Zeichen5')
    Data_erg_topk.to_excel(writer, sheet_name="topk")
    Data_erg_gestopk.to_excel(writer, sheet_name='TopkGes')
    Data_erg_pos.to_excel(writer, sheet_name="Position")



