import numpy as np
from itertools import *
import os
import time
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from model import myModel
import config_file
from data_process import feature_list_train, A_n, P_n, V_n, T_n

os.environ['CUDA_VISIBLE_DEVICES']='0,1'

def load_data(author_c_dict, train_ratio, valid_ratio):
    class A_C(Dataset):
        '''
        PyTorch Dataset for author classification
        '''

        def __init__(self, author_id, labels):
            self.author_id = author_id
            self.labels = labels
        
        def __getitem__(self, idx):
            return [self.author_id[idx], self.labels[idx]]
        
        def __len__(self):
            return len(self.author_id)
    
    X = list(author_c_dict.keys())
    y = list(author_c_dict.values())
    data_size = len(X)
    true_test_ratio = 1 - (train_ratio + valid_ratio)
    rest_data = data_size * (1 - true_test_ratio)
    true_valid_ratio = (rest_data - data_size * train_ratio) / rest_data

    train, test, train_labels, test_labels = train_test_split(X, y, test_size=true_test_ratio, random_state=42)
    train, valid, train_labels, valid_labels = train_test_split(train, train_labels, test_size=true_valid_ratio, random_state=42)
    
    train_set = A_C(train, train_labels)
    valid_set = A_C(valid, valid_labels)
    test_set = A_C(test, test_labels)
    train_size = len(train)
    test_size = len(test)
    valid_size = len(valid)
    
    data_size = [train_size, test_size, valid_size]
    
    return train_set, valid_set, test_set, data_size

def load_label(file, data_path):
    class_records_labels = dict()
    label_file = open(data_path + file, "r")
    for line in islice(label_file, 0, None):
        values = line.split(',')
        index = int(values[0])
        label = int(values[1])
        class_records_labels[index] = label
    label_file.close()
    return class_records_labels
    
def display(mae, macro_f1, micro_f1):
    print("MAE on test set is {:.4f}".format(mae))
    print("Macro_f1 F1-score on test set is {:.4f}".format(macro_f1))
    print("Micro_f1 F1-score on test set is {:.4f}".format(micro_f1))
    
def main():
    configs = config_file.configs
    feature_list_class = feature_list_train(configs['data_path'], 128, A_n, P_n, V_n, T_n)
    feature_list = feature_list_class.feature_list
    node_neigh_list_train = feature_list_class.node_neigh_list_train
    epochs = configs['epochs']
    batch_sz = configs['batch_sz']
    gpu = configs['gpu']
    label_dict = load_label(configs['label_file'], configs['data_path'])
    train_set, valid_set, test_set, data_size = load_data(label_dict, 0.24, 0.06)
    train_size = data_size[0]
    valid_size = data_size[1]
    model_path = os.path.join(configs['model_path'],"{}_{}.pt".format(configs['experiment'], configs['version']))
    
    print('\nEXPERIMENT: ', "{}_{}".format(configs['experiment'], configs['version']))
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model = myModel(feature_list, node_neigh_list_train, configs['embed_dimen'], configs['rank'], gpu)
    if gpu:
        model = model.cuda()
        data_t = torch.cuda.FloatTensor
    else:
        data_t = torch.FloatTensor
    print("Model initialized")
    criterion = nn.L1Loss(size_average=False)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=configs['lr'], weight_decay = 0)
    
    # setup training
    min_valid_loss = float('Inf')
    train_iterator = DataLoader(train_set, batch_size=batch_sz, num_workers=0, shuffle=True)
    valid_iterator = DataLoader(valid_set, batch_size=batch_sz, num_workers=0, shuffle=True)
    test_iterator = DataLoader(test_set, batch_size=batch_sz, num_workers=0, shuffle=True)
    
    traintimes = []
    total_time_start = time.time()
    for e in range(epochs):
        start_time = time.time()
        model.train()
        model.zero_grad()
        avg_train_loss = 0.0
        for batch in train_iterator:
            model.zero_grad()
            x = batch[0].tolist()
            y = Variable(batch[-1].view(-1, 1).float().type(data_t), requires_grad=False)
            output, _ = model(x)
            loss = criterion(output, y)
            
            avg_loss = loss.item()
            avg_train_loss += avg_loss / train_size
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print("Epoch {} complete! Average Training loss: {}".format(e+1, avg_train_loss))
        
        model.eval()
        avg_valid_loss = 0.0
        for batch in valid_iterator:
            x = batch[0].tolist()
            y = Variable(batch[-1].view(-1, 1).float().type(data_t), requires_grad=False)
            output, _ = model(x)
            valid_loss = criterion(output, y)
            avg_valid_loss += valid_loss.item()
            
        output = output.cpu().data.numpy().reshape(-1, 1)
        output = np.round(output)
        y = y.cpu().data.numpy().reshape(-1, 1)
        # time
        epo_time = time.time() - start_time
        traintimes.append(epo_time)
        # training stat
        avg_valid_loss = avg_valid_loss / valid_size
        print("Validation loss is: {:.4f}".format(avg_valid_loss))
        valid_macro_f1 = f1_score(y, output, average='macro')
        valid_micro_f1 = f1_score(y, output, average='micro')
        print("Macro_f1 F1-score on validation set is: {:.4f}".format(valid_macro_f1))
        print("Micro_f1 F1-score on validation set is: {:.4f}".format(valid_micro_f1))
        
        print("--- %s seconds ---" % (time.time() - start_time))
        print("\n")
        
        if (avg_valid_loss < min_valid_loss):
            min_valid_loss = avg_valid_loss
            torch.save(model, model_path)
            print("Found new best model, saving to disk...")
            print("\n")
    
    # testing
    best_model = torch.load(model_path)
    if gpu:
        best_model = best_model.cuda()
    best_model.eval()
    
    for batch in test_iterator:
        x = batch[0].tolist()
        y = Variable(batch[-1].view(-1, 1).float().type(data_t), requires_grad=False)
        output_test, node_embedding = best_model(x)
        
    print('Here is the final results: \n')
    output_test = output_test.cpu().data.numpy().reshape(-1, 1)
    y = y.cpu().data.numpy().reshape(-1, 1)
    output_test = output_test.reshape((len(output_test),))
    y = y.reshape((len(y),))
    mae = np.mean(np.absolute(output_test-y))
    true_label = y
    predicted_label = np.round(output_test)
    macro_f1 = f1_score(true_label, predicted_label, average='macro')
    micro_f1 = f1_score(true_label, predicted_label, average='micro')
    display(mae, macro_f1, micro_f1)
    mean_train_time = np.array(traintimes).mean()
    print('Mean training time per epoch: ', mean_train_time)
    print("--- Total time: %s seconds ---" % (time.time() - total_time_start))

if __name__ == '__main__':
    main()


