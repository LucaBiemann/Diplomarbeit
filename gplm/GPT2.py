from transformers import GPT2Tokenizer, GPT2Model,GPT2Config
import torch
import pandas as pd
import time

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        original_return_val = func(*args, **kwargs)
        end = time.time()
        print("time elapsed in ", func.__name__, ": ", end - start, sep='')
        return original_return_val

    return wrapper

@timing_decorator
def GPT2_embeddings(df):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    GPT_model = GPT2Model.from_pretrained('gpt2')
    GPT_model.to(device)
    df2 = pd.DataFrame(columns=[1])
    df2[1] = df2[1].astype(object)
    for index, row in df.iterrows():
        tokenz = tokenizer(row['text'], return_tensors="pt", max_length=1024,
                                         truncation=True)
        tokenz.to(device)
        pooled_output = GPT_model(**tokenz)
        df2.loc[index, 1] = pooled_output[0][0][1].detach().numpy()
    df['GPT2emb'] = df2[1]
    return df

def DistilBERT_vektor(text):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    GPT_model = GPT2Model.from_pretrained('gpt2')

    tokenz = tokenizer(text, return_tensors="pt")

    pooled_output = GPT_model(**tokenz)
    vektor = pooled_output[0][0][1].detach().numpy()
    return vektor

