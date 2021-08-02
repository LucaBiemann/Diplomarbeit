import pandas as pd
import pickle
key_word_list = ['usb stick','extortion', 'password', 'trade secret', 'intimidate', 'return','blackmail', 'reveal','inside information',
       'trust', 'major contract', 'recommend', 'insolvency', 'merge',
       'step down', 'takeover bid', 'obfuscate', 'undervaluation', 'conceal',
       'fictitious transaction', 'overvaluation', 'tax evasion', 'underrate',
       'overstate',]
names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "MLP", "AdaBoost",
         "Naive Bayes", "QDA"]
dataset = pd.read_pickle('Emails_Test')
Data_erg_klass = pd.DataFrame([])
for n in names:
    for m in ['BERT', 'XLNet', 'ERNIE', 'T5']:
        for k in key_word_list:
            scaler = pickle.load(open('models/email/wb/scaler_keywords_{}.pkl'.format(m), 'rb'))
            for k in key_word_list:
                data_key = dataset[dataset[k] == 1]
                data_sample = data_key.append(dataset[dataset[k] == 0].sample(n=len(data_key) * 19))
                X_input = scaler.transform(list(data_sample['{}emb'.format(m)].values))
                if n == "RBF SVM":
                     filename = 'models/Anhang_Mail_Models2/{}_{}_{}.sav'.format(n, m, k)
                else:
                     filename = 'models/Anhang_Mail_Models/{}_{}_{}.sav'.format(n, m, k)
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
                Data_erg_klass = Data_erg_klass.append(
                    {'Model': n, '{}'.format(m): sum(data_sample['{}_{}'.format(k, m)]) / len(data_sample)},
                    ignore_index=True)
Data_erg_klass = Data_erg_klass.groupby(by=['Model']).mean()
with pd.ExcelWriter("Emails_AnhKlass.xlsx") as writer:
    Data_erg_klass.to_excel(writer, sheet_name="Klassifikator")
