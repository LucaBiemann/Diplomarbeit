from transformers import BertTokenizer, BertModel
import time
import torch
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
def BERT_embeddings(df):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, max_length=512)
    bert_model.to(device)
    df2 = pd.DataFrame(columns=[1])
    df2[1] = df2[1].astype(object)
    for index, row in df.iterrows():
        tokenz = tokenizer(row['text'], return_tensors="pt",truncation=True, max_length=512)
        tokenz.to(device)
        pooled_output = bert_model(**tokenz)
        df2.loc[index, 1] = pooled_output[0][0][1].cpu().detach().numpy()
    df['BERTemb'] = df2[1]
    return df
def BERT_vektor(text):
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    tokenz = tokenizer.encode(text)
    encoding = tokenizer.encode_plus(tokenz,
                                         max_length=1024,
                                         add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                                         return_token_type_ids=False,
                                         padding='max_length',
                                         return_attention_mask=True,
                                         truncation=True,
                                         return_tensors='pt',  # Return PyTorch tensors
                                         )
    last_hidden_state, pooled_output = bert_model(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask']
        )
    vektor = pooled_output.detach().numpy()[0]
    return vektor




