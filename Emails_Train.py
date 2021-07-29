
import pandas as pd 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.optim as optim
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from tqdm.notebook import tqdm
from sklearn import neural_network
from sklearn.svm import SVC
import pickle
import numpy as np
from sklearn.model_selection import train_test_split





def get_class_distribution(obj):
    count_dict = {
        "unverd": 0,
        "verd": 0,
    }

    for i in obj:
        if i == 0:
            count_dict['unverd'] += 1
        elif i == 1:
            count_dict['verd'] += 1
        else:
            print("Check classes.")

    return count_dict


class BERT(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class binaryClassification(nn.Module):
    def __init__(self, num_feature, num_class):
        super(binaryClassification, self).__init__()
        # Number of input features is 12.
        self.layer_1 = nn.Linear(num_feature, 128)
        self.layer_2 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, num_class)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(64)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return torch.sigmoid(x)


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


dataset_wb = pd.read_pickle('Emails_WB_Train')
dataset_bb = pd.read_pickle('Emails_BB_Train')

scaler = StandardScaler()

key_word_list = ['usb stick','extortion', 'password', 'trade secret', 'intimidate', 'return','blackmail', 'reveal','inside information',
       'trust', 'major contract', 'recommend', 'insolvency', 'merge',
       'step down', 'takeover bid', 'obfuscate', 'undervaluation', 'conceal',
       'fictitious transaction', 'overvaluation', 'tax evasion', 'underrate',
       'overstate',]
key_word_list_labels = ['usb stick','extortion', 'password', 'trade secret', 'intimidate', 'return','blackmail', 'reveal','inside information',
       'trust', 'major contract', 'recommend', 'insolvency', 'merge',
       'step down', 'takeover bid', 'obfuscate', 'undervaluation', 'conceal',
       'fictitious transaction', 'overvaluation', 'tax evasion', 'underrate',
       'overstate',
       'labeled', 'label']


name_list = ['James Smith', 'Mary Johnson', 'Robert Williams','Patricia Brown', 'John Jones', 'Jennifer Garcia', 'Michael Miller', 'Linda Davis', 'William Rodriguez']

for v,n in zip([dataset_wb, dataset_bb],['wb','bb']):
    v1,v2,v3,v4 = np.array_split(v, 4)
    for d,g,l in zip([v1,v2,v3,v4],['BERT', 'XLNet', 'ERNIE', 'T5'],[0,1,2,3]):
        d['gplm'] = d['{}emb'.format(g)]
        d['gplm_label'] = l
    data_gplm = pd.concat([v1,v2,v3,v4])
    X_train, X_valid, y_train, y_valid = \
        train_test_split(list(data_gplm['gplm'].values), data_gplm['gplm_label'], test_size=0.10,
                         random_state=42,
                         stratify=data_gplm['gplm_label'])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_valid, y_valid = np.array(X_valid), np.array(y_valid)
    clf = neural_network.MLPClassifier(hidden_layer_sizes=100, random_state=42, max_iter=300,
                                       activation='logistic').fit(
        X_train, y_train)
    acc_mlp = clf.score(X_valid, y_valid)
    print('Accuracy GPLM:', acc_mlp)
    filename = 'models/mail/{}/MLP_gplm.sav'.format(n)
    pickle.dump(clf, open(filename, 'wb'))
    for m in ['BERT', 'XLNet', 'ERNIE', 'T5']:

        scaler.fit(list(v['{}emb'.format(m)].values))
        pickle.dump(scaler, open('models/mail/{}/scaler_keywords_{}.pkl'.format(n,m), 'wb'))


        X_train, X_valid, y_train, y_valid = \
            train_test_split(list(v['{}emb'.format(m)].values), v['label_crime'], test_size=0.10,
                             random_state=42,
                             stratify=v['label_crime'])

        X_train = scaler.transform(X_train)
        X_valid = scaler.transform(X_valid)
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_valid, y_valid = np.array(X_valid), np.array(y_valid)
        clf = neural_network.MLPClassifier(hidden_layer_sizes=100, random_state=42, max_iter=300,
                                           activation='logistic').fit(
            X_train, y_train)
        acc_mlp = clf.score(X_valid, y_valid)
        print('{} Accuracy Crime:'.format(m), acc_mlp)
        filename = 'models/mail/{}/{}/MLP_{}_crime.sav'.format(n,m,m)
        pickle.dump(clf, open(filename, 'wb'))

        for i in key_word_list:
            data_pos = v[v[i] == 1]
            data_neg = v[v[i] == 0]
            data = data_neg.sample(n=len(data_pos), random_state=42).append(data_pos)
            print(len(data))

            X_train, X_valid, y_train, y_valid = \
                    train_test_split(list(data['{}emb'.format(m)].values), data[i], test_size=0.33,
                                     random_state=42,
                                     stratify=data[i])

            X_train = scaler.transform(X_train)
            X_valid = scaler.transform(X_valid)
            X_train, y_train = np.array(X_train), np.array(y_train)
            X_valid, y_valid = np.array(X_valid), np.array(y_valid)
            clf = SVC(kernel="linear", C=0.025, probability=True, random_state=42).fit(
                    X_train, y_train)
            acc_mlp = clf.score(X_valid, y_valid)
            print('{} Accuracy  {}:'.format(m,i), acc_mlp)
            filename = 'models/mail/{}/{}/Linear SVM_{}_{}.sav'.format(n,m,m,i)
            pickle.dump(clf, open(filename, 'wb'))

        for i in name_list:
            data_pos = v[v[i] == 1]
            data_neg = v[v[i] == 0]
            data = data_neg.sample(n=len(data_pos), random_state=42).append(data_pos)
            print(len(data))
            X_train, X_valid, y_train, y_valid = \
                    train_test_split(list(data['{}emb'.format(m)].values), data[i], test_size=0.33,
                                     random_state=42,
                                     stratify=data[i])

            X_train = scaler.transform(X_train)
            X_valid = scaler.transform(X_valid)
            X_train, y_train = np.array(X_train), np.array(y_train)
            X_valid, y_valid = np.array(X_valid), np.array(y_valid)
            clf = SVC(kernel="linear", C=0.025, probability=True, random_state=42).fit(
                    X_train, y_train)
            acc_mlp = clf.score(X_valid, y_valid)
            print('{} Accuracy  {}:'.format(m,i), acc_mlp)
            filename = 'models/mail/{}/{}/Linear SVM_{}_{}.sav'.format(n,m,m,i)
            pickle.dump(clf, open(filename, 'wb'))

        dataset_neg = v[v.labeled == 0]
        dataset_verd = v[v.labeled == 1].sample(n=len(dataset_neg), random_state=42).append(dataset_neg)
        print(len(dataset_verd))
        EPOCHS = 200
        BATCH_SIZE = int((len(dataset_verd) * 0.9) // EPOCHS)
        NUM_FEATURES = 768
        NUM_CLASSES = 1
        X_train, X_valid, y_train, y_valid = train_test_split(list(dataset_verd['{}emb'.format(m)].values), dataset_verd['labeled'],
                                                              test_size=0.10, random_state=42,
                                                              stratify=dataset_verd['labeled'])

        X_train = scaler.transform(np.array(X_train))
        X_valid = scaler.transform(X_valid)
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_valid, y_valid = np.array(X_valid), np.array(y_valid)

        train_dataset = BERT(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
        valid_dataset = BERT(torch.from_numpy(X_valid).float(), torch.from_numpy(y_valid).long())
        target_list = []
        for _, t in train_dataset:
            target_list.append(t)

        target_list = torch.tensor(target_list)
        target_list = target_list[torch.randperm(len(target_list))]
        class_count = [i for i in get_class_distribution(y_train).values()]
        class_weights = 1. / torch.tensor(class_count, dtype=torch.float)
        class_weights_all = class_weights[target_list]
        weighted_sampler = WeightedRandomSampler(
            weights=class_weights_all,
            num_samples=len(class_weights_all),
            replacement=True
        )

        train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, sampler=weighted_sampler)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=1)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)

        model_start = binaryClassification(num_feature=NUM_FEATURES, num_class=NUM_CLASSES)
        model_start.to(device)
        weight = class_weights.to(device)
        criterion = nn.BCELoss()
        optimizer_start = optim.Adam(model_start.parameters())

        loss_stats = pd.DataFrame(data=[], columns=['train', 'valid'])
        accuracy_stats = pd.DataFrame(columns=['train', 'valid'])
        for e in tqdm(range(1, EPOCHS + 1)):
            # TRAINING
            train_epoch_loss = 0
            train_epoch_acc = 0
            model_start.train()

            for X_train_batch, y_train_batch in train_loader:
                X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
                optimizer_start.zero_grad()

                y_train_pred = model_start(X_train_batch).reshape(-1)
                train_loss = criterion(y_train_pred, y_train_batch.float())
                train_acc = binary_acc(y_train_pred, y_train_batch)

                train_loss.backward()
                optimizer_start.step()

                train_epoch_loss += train_loss.item()
                train_epoch_acc += train_acc.item()

            # VALIDATION
            with torch.no_grad():
                val_epoch_loss = 0
                val_epoch_acc = 0

                model_start.eval()
                for X_val_batch, y_val_batch in valid_loader:
                    X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

                    y_val_pred = model_start(X_val_batch).reshape(-1)

                    val_loss = criterion(y_val_pred, y_val_batch.float())
                    val_acc = binary_acc(y_val_pred, y_val_batch)
                    val_epoch_loss += val_loss.item()
                    val_epoch_acc += val_acc.item()

                loss_stats.append(
                    {'train': train_epoch_loss / len(train_loader), 'valid': val_epoch_loss / len(valid_loader)},
                    ignore_index=True)
                accuracy_stats.append(
                    {'train': train_epoch_acc / len(train_loader), 'valid': val_epoch_acc / len(valid_loader)},
                    ignore_index=True)

                print(
                    f'Epoch {e + 0:03}: | Train Loss: {train_epoch_loss / len(train_loader):.5f} | Val Loss: {val_epoch_loss / len(valid_loader):.5f} | Train Acc: {train_epoch_acc / len(train_loader):.3f}| Val Acc: {val_epoch_acc / len(valid_loader):.3f}')
        torch.save(model_start,
                   'models/mail/{}/{}/Binary_Verdacht_{}'.format(n,m,m))