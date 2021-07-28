from transformers import RobertaTokenizer, RobertaModel, BatchEncoding
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
def RoBERTa_embeddings(df):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    RoBERTa_model = RobertaModel.from_pretrained('roberta-base')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    RoBERTa_model.to(device)
    df2 = pd.DataFrame(columns=[1])
    df2[1] = df2[1].astype(object)
    for index, row in df.iterrows():
        tokenz = tokenizer(row['text'], return_tensors="pt", max_length=1024,
                                         truncation=True)
        tokenz.to(device)
        pooled_output = RoBERTa_model(**tokenz)
        df2.loc[index, 1] = pooled_output[0][0][1].cpu().detach().numpy()
    df['RoBERTaemb'] = df2[1]
    return df


def RoBERTa_vektor(text):
    RoBERTa_model = RobertaModel.from_pretrained('roberta-base')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    RoBERTa_model.config.type_vocab_size = 2
    single_emb = RoBERTa_model.embeddings.token_type_embeddings
    RoBERTa_model.embeddings.token_type_embeddings = torch.nn.Embedding(2, single_emb.embedding_dim)
    RoBERTa_model.embeddings.token_type_embeddings.weight = torch.nn.Parameter(single_emb.weight.repeat([2, 1]))
    tokenz = tokenizer(text, return_tensors="pt")

    last_hidden_state, pooled_output = RoBERTa_model(**tokenz)
    vektor = pooled_output.detach().numpy()[0]
    return vektor
