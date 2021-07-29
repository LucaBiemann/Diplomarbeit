import pandas as pd
import pickle

dataset = pd.read_pickle('keywordmails_test_t5')
dataset['seqlen'] = dataset['text'].apply(lambda x: len(x))
dataset = dataset.sort_values(by='seqlen')
dataset['bins'] = pd.qcut(dataset['seqlen'], q=20)
key_word_list = ['usb stick', 'extortion', 'password', 'trade secret', 'intimidate', 'return', 'blackmail', 'reveal',
                 'inside information',
                 'trust', 'major contract', 'recommend', 'insolvency', 'merge',
                 'step down', 'takeover bid', 'obfuscate', 'undervaluation', 'conceal',
                 'fictitious transaction', 'overvaluation', 'tax evasion', 'underrate',
                 'overstate']
key_word_noun = ['usb stick', 'extortion', 'password', 'trade secret', 'inside information', 'major contract',
                 'insolvency', 'takeover bid', 'undervaluation', 'fictitious transaction', 'overvaluation',
                 'tax evasion']
name_list = ['James Smith', 'Mary Johnson', 'Robert Williams', 'Patricia Brown', 'John Jones', 'Jennifer Garcia',
             'Michael Miller', 'Linda Davis', 'William Rodriguez']
names = ["Nearest Neighbors", "MLP", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "AdaBoost",
         "Naive Bayes", "QDA"]
Data_erg_key = pd.DataFrame([])
Data_erg_names = pd.DataFrame([])
Data_erg_seqlen = pd.DataFrame([])
Data_erg_art = pd.DataFrame([])
Data_erg_klass = pd.DataFrame([])
for m in ['BERT', 'XLNet', 'ERNIE', 'T5']:
    scaler = pickle.load(open('models/email/wb/scaler_keywords_{}.pkl'.format(m), 'rb'))
    for k in key_word_list:
        data_key = dataset[dataset[k] == 1]
        data_sample = data_key.append(dataset[dataset[k] == 0].sample(n=len(data_key) * 19))
        X_input = scaler.transform(list(data_sample['{}emb'.format(m)].values))
        filename = 'models/mail/wb/{}/Linear SVM_{}_{}.sav'.format(m, m, k)
        loaded_model = pickle.load(open(filename, 'rb'))
        probs = loaded_model.predict(X_input)
        data_sample.loc[:, '{}_{}_pred'.format(k, m)] = probs
        for index, row in data_sample.iterrows():

            if row['{}_{}_pred'.format(k, m)] == row[k]:
                data_sample.loc[index, '{}_{}'.format(k, m)] = 1
            else:
                data_sample.loc[index, '{}_{}'.format(k, m)] = 0
        print(m, k, 'Alle Mails: ', len(data_sample), '| Enthält Wort: ', len(data_sample[data_sample[k] == 1]),
              '| Pred Enthält Kein Wort: ', len(data_sample[data_sample['{}_{}_pred'.format(k, m)] == 0]),
              '| Pred Enthält Wort: ', len(data_sample[data_sample['{}_{}_pred'.format(k, m)] == 1]),
              '| Precision: ', sum(data_sample['{}_{}'.format(k, m)]) / len(data_sample))
        Data_erg_key = Data_erg_key.append(
            {'Klassifikator': k, '{}'.format(m): sum(data_sample['{}_{}'.format(k, m)]) / len(data_sample)},
            ignore_index=True)
        Data_erg_klass = Data_erg_klass.append(
            {'Model': "Linear SVM", '{}'.format(m): sum(data_sample['{}_{}'.format(k, m)]) / len(data_sample)},
            ignore_index=True)
        if k in key_word_noun:
            Data_erg_art = Data_erg_art.append(
                {'Klassifikator': 'Noun', '{}'.format(m): sum(data_sample['{}_{}'.format(k, m)]) / len(data_sample)},
                ignore_index=True)
        else:
            Data_erg_art = Data_erg_art.append(
                {'Klassifikator': 'Verb', '{}'.format(m): sum(data_sample['{}_{}'.format(k, m)]) / len(data_sample)},
                ignore_index=True)
        for b in dataset['bins'].unique():
            data_bin = data_sample[data_sample.bins == b]
            Data_erg_seqlen = Data_erg_seqlen.append(
                {'Bins': b, 'Klassifikator': k, '{}'.format(m): sum(data_bin['{}_{}'.format(k, m)]) / len(data_bin)},
                ignore_index=True)
    for n in name_list:
        data_name = dataset[dataset[n] == 1]

        data_sample = data_name.append(dataset[dataset[n] != 0].sample(n=len(data_name) * 19))
        X_input = scaler.transform(list(data_sample['{}emb'.format(m)].values))
        filename = 'models/mail/wb/{}/Linear SVM_{}_{}.sav'.format(m, m, n)
        loaded_model = pickle.load(open(filename, 'rb'))
        probs = loaded_model.predict(X_input)
        data_sample.loc[:, '{}_{}_pred'.format(n, m)] = probs
        for index, row in data_sample.iterrows():

            if row['{}_{}_pred'.format(n, m)] == row[n]:
                data_sample.loc[index, '{}_{}'.format(n, m)] = 1
            else:
                data_sample.loc[index, '{}_{}'.format(n, m)] = 0
        print(m, n, 'Alle Mails: ', len(data_sample), '| Enthält Wort: ', len(data_sample[data_sample[n] == 1]),
              '| Pred Enthält Kein Wort: ', len(data_sample[data_sample['{}_{}_pred'.format(n, m)] == 0]),
              '| Pred Enthält Wort: ', len(data_sample[data_sample['{}_{}_pred'.format(n, m)] == 1]),
              '| Precision: ', sum(data_sample['{}_{}'.format(n, m)]) / len(data_sample))
        Data_erg_names = Data_erg_names.append(
            {'Klassifikator': n, '{}'.format(m): sum(data_sample['{}_{}'.format(n, m)]) / len(data_sample)},
            ignore_index=True)
        Data_erg_art = Data_erg_art.append(
            {'Klassifikator': 'Name', '{}'.format(m): sum(data_sample['{}_{}'.format(n, m)]) / len(data_sample)},
            ignore_index=True)
for n in names:
    for m in ['BERT', 'XLNet', 'ERNIE', 'T5']:
        scaler = pickle.load(open('models/email/wb/scaler_keywords_{}.pkl'.format(m), 'rb'))
        for k in key_word_list:
                data_key = dataset[dataset[k] == 1]
                data_sample = data_key.append(dataset[dataset[k] == 0].sample(n=len(data_key) * 19))
                X_input = scaler.transform(list(data_sample['{}emb'.format(m)].values))
                filename = 'models/email/Anhang_Mail_Models/{}_{}_{}.sav'.format(n, m, k)
                loaded_model = pickle.load(open(filename, 'rb'))
                probs = loaded_model.predict(X_input)
                data_sample.loc[:, '{}_{}_pred'.format(k, m)] = probs
                for index, row in data_sample.iterrows():

                    if row['{}_{}_pred'.format(k, m)] == row[k]:
                        data_sample.loc[index, '{}_{}'.format(k, m)] = 1
                    else:
                        data_sample.loc[index, '{}_{}'.format(k, m)] = 0
                print( n,m, k,'Alle Mails: ', len(data_sample), '| Enthält Wort: ', len(data_sample[data_sample[k] == 1]),
                      '| Pred Enthält Kein Wort: ', len(data_sample[data_sample['{}_{}_pred'.format(k, m)] == 0]),
                      '| Pred Enthält Wort: ', len(data_sample[data_sample['{}_{}_pred'.format(k, m)] == 1]),
                      '| Precision: ', sum(data_sample['{}_{}'.format(k, m)]) / len(data_sample))
                Data_erg_klass = Data_erg_klass.append(
                    {'Model': n, '{}'.format(m): sum(data_sample['{}_{}'.format(k, m)]) / len(data_sample)},
                    ignore_index=True)
Data_erg_key = Data_erg_key.groupby(by=['Klassifikator']).agg(max)
Data_erg_names = Data_erg_names.groupby(by=['Klassifikator']).agg(max)
Data_erg_seqlen = Data_erg_seqlen.groupby(by=['Bins']).mean()
Data_erg_art = Data_erg_art.groupby(by=['Klassifikator']).mean()
Data_erg_klass = Data_erg_klass.groupby(by=['Model']).mean()


with pd.ExcelWriter("Excel/Anhang/Keywords.xlsx") as writer:
    Data_erg_key.to_excel(writer, sheet_name="Keywords")
    Data_erg_names.to_excel(writer, sheet_name="Names")
    Data_erg_seqlen.to_excel(writer, sheet_name="Length")
    Data_erg_art.to_excel(writer, sheet_name="Art")
    Data_erg_klass.to_excel(writer, sheet_name="Class")