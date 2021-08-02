import pandas as pd
import numpy as np
import random
import nltk
import spacy
from spacy.cli.download import download
download('en_core_web_sm')
random.seed(42)
nlp = spacy.load("en_core_web_sm")


dataset_wb = pd.read_pickle('Emails_Enron')
dataset_bb = pd.read_csv('Emails_yelp.csv')
dataset_bb = dataset_bb.sample(n=8000, random_state=42)
dataset_bb['content'] = dataset_bb['text']
words_is = pd.DataFrame(data={'NN': ['trade secret', 'extortion', 'usb stick', 'password'],
                              'NNS': ['trade secrets', 'extortions', 'usb sticks', 'passwords'],
                              'VB': ['reveal', 'return', 'blackmail', 'intimidate'],
                              'VBD': ['revealed', 'returned', 'blackmailed', 'intimidated'],
                              'VBG': ['revealing', 'returning', 'blackmailing', 'intimidating'],
                              'VBN': ['revealed', 'returned', 'blackmailed', 'intimidated'],
                              'VBP': ['reveal', 'return', 'blackmail', 'intimidate'],
                              'VBZ': ['reveals', 'returns', 'blackmails', 'intimidates']
                              })
words_bf = pd.DataFrame(data={'NN': ['overvaluation', 'undervaluation', 'fictitious transaction', 'tax evasion'],
                              'NNS': ['overvaluations', 'undervaluations', 'fictitious transactions', 'tax evasions'],
                              'VB': ['overstate', 'underrate', 'obfuscate', 'conceal'],
                              'VBD': ['overstated', 'underrated', 'obfuscated', 'concealed'],
                              'VBG': ['overstating', 'underrating', 'obfuscating', 'concealing'],
                              'VBN': ['overstated', 'underrated', 'obfuscated', 'concealed'],
                              'VBP': ['overstate', 'underrate', 'obfuscate', 'conceal'],
                              'VBZ': ['overstates', 'underrates', 'obfuscates', 'conceals']})
words_ih = pd.DataFrame(data={'NN': ['inside information', 'takeover bid', 'major contract', 'insolvency'],
                              'NNS': ['inside informations', 'takeover bids', 'major contracts', 'insolvencys'],
                              'VB': ['trust', 'merge', 'step down', 'recommend'],
                              'VBD': ['trusted', 'merged', 'stepped down', 'recommended'],
                              'VBG': ['trusting', 'merging', 'stepping down', 'recommending'],
                              'VBN': ['trusted', 'merged', 'stepped down', 'recommended'],
                              'VBP': ['trust', 'merge', 'step down', 'recommend'],
                              'VBZ': ['trusts', 'merges', 'steps down', 'recommends']
                              })

for v, n in zip([dataset_wb, dataset_bb], ['wb', 'bb']):
    v['tokens'] = v['content'].apply(lambda x: nltk.word_tokenize(x))
    df2 = pd.DataFrame(columns=[1, 2])
    df2[1] = df2[1].astype(object)
    df2[2] = df2[2].astype(object)
    for index, row in v.iterrows():

        count = 0
        count2 = 0
        saetze = []
        pos_tags = nltk.pos_tag(row['tokens'])
        df2.loc[index, 1] = pos_tags

        for i in pos_tags:
            count += 1
            if i[1] == '.':
                satzlaenge = count - count2 - 1
                count2 += count - count2
                saetze.append(satzlaenge)

        df2.loc[index, 2] = saetze
    v['pos'] = df2[1]
    v['saetze'] = df2[2]
    v = v.sample(frac=1, random_state=42).reset_index(drop=True)
    if n == 'wb':
        v['saetzestr'] = v['saetze'].apply(lambda x: str(x))
        v = v[v.saetzestr != '[]']
        v['maxsentence'] = v['saetze'].apply(lambda x: max(x))
        v = v[v.maxsentence >= 5]
        v['posstr'] = v['pos'].apply(lambda x: str(x))
        v = v[v['posstr'].str.contains("NN|VBP")]
        v.reset_index(drop=True, inplace=True)

    Data_is, Data_bf, Data_ih = np.array_split(v, 3)
    label = 1
    for i, j in zip([Data_is, Data_ih, Data_bf], [words_is, words_ih, words_bf]):
        i['label_crime'] = label
        i['content_new'] = i['content']
        for index, row in i.iterrows():
            satz_index = row['saetze']
            last_index = 0
            current_index = 0
            text = row['content']
            for o in range(len(satz_index)):
                current_index += satz_index[o] + 1

                if current_index - last_index >= 6:
                    number = random.randrange(2)
                    satz = row['pos'][last_index:current_index]
                    satz_typ = []
                    for m in satz:
                        satz_typ.append(str(m[1]))

                    if 'VB' or 'VBD' or 'VBG' or 'VBN' or 'VBP' or 'VBZ' in satz_typ and 'NN' or 'NNS' in satz_typ:
                        enthalten = 'beides'
                    elif 'NN' or 'NNS' in satz_typ:
                        enthalten = 'NN'
                    elif 'VB' or 'VBD' or 'VBG' or 'VBN' or 'VBP' or 'VBZ' in satz_typ:
                        enthalten = 'VB'
                    else:
                        enthalten = 'nichts'
                    word_list = []
                    if (enthalten == 'beides' and number == 0) or enthalten == 'NN':
                        for u in satz:
                            if u[1] in ['NN', 'NNS'] and u[0].lower() not in ['i', 'you', 'he', 'she', 'it', 'we',
                                                                              'they']:
                                word_list.append(u)
                        if word_list:
                            number2 = random.randrange(len(word_list))
                            word = word_list[number2][0]
                            word_index = random.randrange(len(j))
                            new_word = j.loc[word_index, word_list[number2][1]]
                            text = text.replace(word, new_word)
                            i.loc[index, j.loc[word_index, 'NN']] = 1
                    elif (enthalten == 'beides' and number == 1) or enthalten == 'VB':
                        for u in satz:
                            if u[1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
                                word_list.append(u)
                        if word_list:
                            number2 = random.randrange(len(word_list))
                            word = word_list[number2][0]
                            word_index = random.randrange(len(j))
                            new_word = j.loc[word_index, word_list[number2][1]]
                            text = text.replace(word, new_word)
                            i.loc[index, j.loc[word_index, 'VB']] = 1
                    i.loc[index, 'content_new'] = text
                last_index += satz_index[o] + 1

        i = i.fillna(0)
        label += 1
    for i in [Data_is, Data_ih, Data_bf]:
        i.iloc[:,-8:] = i.iloc[:,-8:].fillna(0)
        for index, row in i.iterrows():
            i.loc[index, 'label_sum'] = sum(row.iloc[-8:])
        for index, row in i.iterrows():
            if row['label_sum'] > 0:
                i.loc[index, 'labeled'] = 1
            else:
                i.loc[index, 'labeled'] = 0
                i.loc[index, 'label'] = 0

    Data_is = Data_is[Data_is.label_sum > 0]
    Data_ih = Data_ih[Data_ih.label_sum > 0]
    Data_bf = Data_bf[Data_bf.label_sum > 0]

    Data_final = Data_is.append(Data_ih).append(Data_bf)
    df_names = pd.DataFrame([])
    df_names[0] = ['James Smith', 'Mary Johnson', 'Robert Williams']
    df_names[1] = ['Patricia Brown', 'John Jones', 'Jennifer Garcia']
    df_names[2] = ['Michael Miller', 'Linda Davis', 'William Rodriguez']
    for index, row in Data_final.iterrows():
        doc = nlp(row['content_new'])
        Data_ner = pd.DataFrame([])
        Data_ner['word'] = [x.text for x in doc.ents]
        Data_ner['label'] = [x.label_ for x in doc.ents]
        name_list = list(set(Data_ner.loc[Data_ner['label'] == 'PERSON', 'word']))
        if len(name_list) > 0:
            crime = int(row['label_crime']) - 1

            if len(name_list) >= 3:
                text = Data_final.loc[index, 'content_new']
                index_list = list(range(len(name_list)))
                index0, index1, index2 = [random.sample(list(range(len(index_list))), k=1)[0] for i in [0, 1, 2]]
                name0, name1, name2 = name_list[index0], name_list[index1], name_list[index2]
                text = text.replace(name0,
                                    df_names.iloc[
                                        0,
                                        crime] if len(name0.split()) >= 2 else df_names.iloc[
                                        0,
                                        crime].split()[0])
                Data_final.loc[index, df_names.iloc[0, crime]] = 1
                text = text.replace(name1,
                                    df_names.iloc[
                                        1,
                                        crime] if len(
                                        name1.split()) >= 2 else
                                    df_names.iloc[
                                        1,
                                        crime].split()[
                                        0])
                Data_final.loc[index, df_names.iloc[1, crime]] = 1
                text = text.replace(name2,
                                    df_names.iloc[
                                        2,
                                        crime] if len(
                                        name2.split()) >= 2 else
                                    df_names.iloc[
                                        2,
                                        crime].split()[
                                        0])
                Data_final.loc[index, df_names.iloc[2, crime]] = 1
                Data_final.loc[index, 'text_named'] = text
            else:
                index_list = [random.sample([0, 1, 2], k=1)[0] for i in name_list]
                text = Data_final.loc[index, 'content_new']
                for i, o in zip(index_list, name_list):
                    text = text.replace(o, df_names.iloc[i, crime] if len(o.split()) >= 2 else
                    df_names.iloc[i, crime].split()[0])
                    Data_final.loc[index, df_names.iloc[i, crime]] = 1
                Data_final.loc[index, 'text_named'] = text
    for index, row in Data_final.head(10).iterrows():
        print('Vorher: ', row['content'])
        print('Nachher: ', row['content_new'])
    pd.to_pickle(Data_final, 'Emails_{}_Man'.format(n))
