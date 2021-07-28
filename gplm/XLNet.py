import time
import torch
import pandas as pd
from transformers import XLNetTokenizer, XLNetModel


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        original_return_val = func(*args, **kwargs)
        end = time.time()
        print("time elapsed in ", func.__name__, ": ", end - start, sep='')
        return original_return_val

    return wrapper


@timing_decorator
def XLNet_embeddings(df):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    XLNet_model = XLNetModel.from_pretrained('xlnet-base-cased')
    XLNet_model.to(device)
    df2 = pd.DataFrame(columns=[1])
    df2[1] = df2[1].astype(object)
    for index, row in df.iterrows():
        tokenz = tokenizer(row['text'], return_tensors="pt", max_length=1024,
                                         truncation=True)
        tokenz.to(device)
        pooled_output = XLNet_model(**tokenz)

        df2.loc[index, 1] = pooled_output[0][0][1].cpu().detach().numpy()
    df['XLNetemb'] = df2[1]
    return df


def XLNet_vektor(text):
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    XLNet_model = XLNetModel.from_pretrained('xlnet-base-cased')

    tokenz = tokenizer(text, return_tensors="pt")

    pooled_output = XLNet_model(**tokenz)
    vektor = pooled_output[0][0][1].detach().numpy()
    return vektor





