import warnings

warnings.simplefilter(action='ignore')
import numpy as np
import pandas as pd
import pickle
from gplm.BERT import BERT_embeddings
from gplm.DistilBert import DistilBERT_embeddings
from gplm.XLNet import XLNet_embeddings
from gplm.RoBERTa import RoBERTa_embeddings
from gplm.ERNIE import ERNIE_embeddings

import time
from sklearn.metrics.pairwise import euclidean_distances


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        original_return_val = func(*args, **kwargs)
        end = time.time()
        print("time elapsed in ", func.__name__, ": ", end - start, sep='')
        return original_return_val

    return wrapper


@timing_decorator
def Ausweis_Attack(vektor):
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
    print('Model: ', model, model_typ)
    scaler = pickle.load(open('models/chatbot/scaler_{}.pkl'.format(model), 'rb'))
    X_input = scaler.transform(vektor)

    loaded_model = pickle.load(open('models/chatbot/{}/MLP_NumMes.sav'.format(model), 'rb'))
    if loaded_model.predict(X_input) == 0:
        return print('Nachricht')
    else:
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
        # Data_vektor['label'] = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

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

        n = 1000
        list_df = [Kandidaten[i:i + n] for i in range(0, Kandidaten.shape[0], n)]
        count = 0
        for i in list_df:
            count += 1
            kandidat_vektor = globals()[model + '_embeddings'](i)

            print(count, '% der Vektoren berechnet')
            for index, row in kandidat_vektor.iterrows():
                kan_vec = np.array(row[model + 'emb'])
                dist = euclidean_distances(vektor, kan_vec.reshape(1, -1))
                if dist < 0.001:
                    return print('Die Ausweisnummer lautet ' + kandidat_vektor.loc[index, 'text'])
            print(count, '% der Vektoren geprüft')