from transformers import T5Tokenizer, T5EncoderModel
import torch
import time
import pandas as pd



def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        original_return_val = func(*args, **kwargs)
        end = time.time()
        print("time elapsed in ", func.__name__, ": ", end - start, sep='')
        return original_return_val

    return wrapper


@timing_decorator
def T5_embeddings(df):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    T5_model = T5EncoderModel.from_pretrained('t5-base')
    T5_model.to(device)
    df2 = pd.DataFrame(columns=[1])
    df2[1] = df2[1].astype(object)
    for index, row in df.iterrows():
        tokenz = tokenizer(row['text'], return_tensors="pt", max_length=512, truncation=True)
        tokenz.to(device)
        output = T5_model(**tokenz)
        df2.loc[index, 1] = output.last_hidden_state[0][0].cpu().detach().numpy()
    df['T5emb'] = df2[1]
    return df
def T5_vektor(text):
    T5_model = T5EncoderModel.from_pretrained('t5-base')
    tokenizer =  T5Tokenizer.from_pretrained('t5-base')
    tokenz = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)

    output = T5_model(**tokenz)
    vektor = output.cpu().detach().numpy()[0]
    return vektor


