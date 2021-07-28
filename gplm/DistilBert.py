import time
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel
import torch


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        original_return_val = func(*args, **kwargs)
        end = time.time()
        print("time elapsed in ", func.__name__, ": ", end - start, sep='')
        return original_return_val

    return wrapper


@timing_decorator
def DistilBERT_embeddings(df):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    DistilBERT_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    DistilBERT_model.to(device)
    df2 = pd.DataFrame(columns=[1])
    df2[1] = df2[1].astype(object)
    for index, row in df.iterrows():
        tokenz = tokenizer(row['text'], return_tensors="pt", max_length=1024, truncation=True)
        tokenz.to(device)
        pooled_output = DistilBERT_model(**tokenz)
        df2.loc[index, 1] = pooled_output[0][0][1].cpu().detach().numpy()
    df['DistilBERTemb'] = df2[1]
    return df


def DistilBERT_vektor(text):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    DistilBERT_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

    tokenz = tokenizer(text, return_tensors="pt")

    pooled_output = DistilBERT_model(**tokenz)
    vektor = pooled_output[0][0][1].detach().numpy()
    return vektor


