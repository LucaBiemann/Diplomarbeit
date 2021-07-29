from gplm.BERT import BERT_embeddings
from gplm.DistilBert import DistilBERT_embeddings
from gplm.RoBERTa import RoBERTa_embeddings
from gplm.ERNIE import ERNIE_embeddings
from gplm.XLNet import XLNet_embeddings
from Chatbot_AusweisNr import ausweisnummern
import pandas as pd

start = ['L', 'M', 'N', 'P', 'R', 'T', 'V', 'W', 'X', 'Y']
start_map = pd.DataFrame([], columns=['start', 'label'])
start_map['start'] = start
start_map['label'] = range(len(start))
zeichen = ['C', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'T', 'V', 'W', 'X', 'Y', 'Z']
zeichen_map = pd.DataFrame([], columns=['start', 'label'])
zeichen_map['zeichen'] = zeichen
zeichen_map['label'] = range(10, len(zeichen) + 10)
Data = pd.DataFrame([])
Data['text'] = ausweisnummern(100)
for i in range(10):
    if i !=9:
        Data['Zeichen_{}'.format(i)] = Data['text'].apply(lambda x: x[i])
    else:
        Data['Zeichen{}'.format(i)] = Data['text'].apply(lambda x: x[i])
for index, row in Data.iterrows():
    Data.loc[index, 'Zeichen0'] = start_map.loc[start_map.start == row['Zeichen_0'], 'label'].values[0]
    for i in range(1,9):
        if row['Zeichen_{}'.format(i)].isnumeric():
            Data.loc[index, 'Zeichen{}'.format(i)] = int(row['Zeichen_{}'.format(i)])
        else:
            Data.loc[index, 'Zeichen{}'.format(i)] = zeichen_map.loc[zeichen_map.zeichen == row['Zeichen_{}'.format(i)], 'label'].values[0]
BERT_embeddings(Data)
DistilBERT_embeddings(Data)
RoBERTa_embeddings(Data)
ERNIE_embeddings(Data)
XLNet_embeddings(Data)
pd.to_pickle(Data, 'DataAusweisnummer')