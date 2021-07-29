import pandas as pd
import pickle
import numpy as np
names = [ "Linear SVM","Nearest Neighbors", "RBF SVM",
         "Decision Tree", "Random Forest", "MLP", "AdaBoost",
         "Naive Bayes", "QDA"]
Data = pd.read_pickle('Chatbot_Test')
Data_erg_klass = pd.DataFrame([])
for n in names:
    for m in ['BERT', 'DistilBERT', 'XLNet', 'RoBERTa', 'ERNIE']:
        scaler = pickle.load(open('models/chatbot/scaler_{}.pkl'.format(m), 'rb'))
        X_input = scaler.transform(list(Data[m + 'emb'].values))
        filename = 'models/Anhang_Chatbot_Models/{}_{}_Zeichen5.sav'.format(n,m)
        loaded_model = pickle.load(open(filename, 'rb'))

        Data['Pred'+m+n] = loaded_model.predict(X_input)
        for index, row in Data.iterrows():
                if row['Zeichen5'] == row['Pred'+m+n]:
                    Data.loc[index, m + n] = 1
                else:
                        Data.loc[index, m  + n] = 0
        print(n, 'Top: ', m, 'Accuracy: ', sum(Data[ m+ n]) / len(Data))
        Data_erg_klass = Data_erg_klass.append({'Klassifikator': n, m:sum(Data[ m + n]) / len(Data)}, ignore_index=True)

Data_erg_klass = Data_erg_klass.groupby(by=['Klassifikator']).agg(max)
with pd.ExcelWriter("Chatbot_Klass.xlsx") as writer:
    Data_erg_klass.to_excel(writer, sheet_name="Klassifikator")
