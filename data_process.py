import numpy as np
import re
from collections import Counter
import random
import os
from itertools import *
import config_file

#window = 5
#walk_L = 40
A_n = config_file.configs['A_n']
P_n = config_file.configs['P_n']
V_n = config_file.configs['V_n']
T_n = config_file.configs['T_n']

# import node content embeddings
input_dimen = 128
data_path = os.getcwd() + '/data/'

class feature_list_train(object):
    def __init__(self, data_path, input_dimen, A_n, P_n, V_n, T_n):
        self.data_path = data_path
        self.input_dimen = input_dimen
        self.P_n = P_n
        self.A_n = A_n
        self.V_n = V_n
        self.T_n = T_n
        
        a_p_list_train = [[] for k in range(self.A_n)]
        v_p_list_train = [[] for k in range(self.V_n)]
        
        relation_f = ["a_p.txt", "v_p.txt"]
        
        #store academic relational data
        for i in range(len(relation_f)):
            f_name = relation_f[i]
            neigh_f = open(self.data_path + f_name, "r")
            
            for line in neigh_f:
                line = line.strip()
                node_id = int(re.split(':', line)[0])
                neigh_list = re.split(':', line)[1]
                neigh_list_id = re.split(',', neigh_list)
                if f_name == 'a_p.txt':
                    for j in range(len(neigh_list_id)):
                        a_p_list_train[node_id].append('p'+str(neigh_list_id[j]))
                else:
                    for j in range(len(neigh_list_id)):
                        v_p_list_train[node_id].append('p'+str(neigh_list_id[j]))
            neigh_f.close()
        
        #store pre-trained network/content embedding
        a_net_embed = {x:np.zeros((1, self.input_dimen)) for x in range(self.A_n)}
        p_net_embed = {x:np.zeros((self.input_dimen)) for x in range(self.P_n)}
        v_net_embed = {x:np.zeros((1, self.input_dimen)) for x in range(self.V_n)}
        t_net_embed = {x:np.zeros((self.input_dimen)) for x in range(self.T_n)}
        
        net_e_f = open(self.data_path + "node_net_embedding.txt", "r")
        for line in islice(net_e_f, 1, None):
            line = line.strip()
            index = re.split(' ', line)[0]
            if len(re.split(' ', line)[1:]) == 0:
                continue
            else:
                if len(index) and (index[0] == 'a' or index[0] == 'v' or index[0] == 'p' or index[0] == 't'):
                    embeds = np.asarray(re.split(' ', line)[1:], dtype='float32')
                    if index[0] == 'a':
                        a_net_embed[int(index[1:])] = embeds
                    elif index[0] == 'v':
                        v_net_embed[int(index[1:])] = embeds
                    elif index[0] == 't':
                        t_net_embed[int(index[1:])] = embeds
                    else:
                        p_net_embed[int(index[1:])] = embeds
        net_e_f.close()
                
        self.a_net_embed = a_net_embed
        self.p_net_embed = p_net_embed
        self.v_net_embed = v_net_embed
        self.t_net_embed = t_net_embed
        print('size of a_net_embed: ', len(a_net_embed))
        print('size of p_net_embed: ', len(p_net_embed))
        print('size of v_net_embed: ', len(v_net_embed))
        print('size of t_net_embed: ', len(t_net_embed))
        
        p_title_embed = {x:np.zeros((1, self.input_dimen)) for x in range(self.P_n)}
        p_t_e_f = open(self.data_path + "paper_title_embeddings.txt", "r")
        index = 0
        for line in islice(p_t_e_f, 0, None):
            values = line.split()
            #index = int(values[0])
            embeds = np.asarray(values, dtype='float32')
            p_title_embed[index] = embeds
            index += 1
        p_t_e_f.close()
        self.p_title_embed = p_title_embed
        print('size of p_title_embed: ', len(p_title_embed))
        
        term_embed = {x:np.zeros((1, self.input_dimen)) for x in range(self.T_n)}
        t_e_f = open(self.data_path + "terms_embedding.txt", "r")
        index = 0
        for line in islice(t_e_f, 1, None):
            values = line.split()
            embeds = np.asarray(values[1:], dtype='float32')
            term_embed[index] = embeds
            index += 1
        t_e_f.close()
        self.term_embed = term_embed
        print('size of term_embed: ', len(term_embed))
        
        #empirically use 3 paper embedding for author content embeding generation
        a_text_embed = {x:np.zeros((1, self.input_dimen*3)) for x in range(self.A_n)}
        for i in range(self.A_n):
            if len(a_p_list_train[i]):
                feature_temp = []
                if len(a_p_list_train[i]) >= 3:
                    for j in range(3):
                        feature_temp.append(p_title_embed[int(a_p_list_train[i][j][1:])])
                else:
                    for j in range(len(a_p_list_train[i])):
                        feature_temp.append(p_title_embed[int(a_p_list_train[i][j][1:])])
                    for k in range(len(a_p_list_train[i]), 3):
                        feature_temp.append(p_title_embed[int(a_p_list_train[i][-1][1:])])
                        
                feature_temp = np.reshape(np.asarray(feature_temp), [1, -1])
                a_text_embed[i] = feature_temp
        self.a_text_embed = a_text_embed
        print('size of a_text_embed: ', len(a_text_embed))
        
        #empirically use 5 paper embedding for venue content embeding generation
        v_text_embed = {x:np.zeros((1, self.input_dimen*5)) for x in range(self.V_n)}
        for i in range(self.V_n):
            if len(v_p_list_train[i]):
                feature_temp = []
                if len(v_p_list_train[i]) >= 5:
                    for j in range(5):
                        feature_temp.append(p_title_embed[int(v_p_list_train[i][j][1:])])
                else:
                    for j in range(len(v_p_list_train[i])):
                        feature_temp.append(p_title_embed[int(v_p_list_train[i][j][1:])])
                    for k in range(len(v_p_list_train[i]), 5):
                        feature_temp.append(p_title_embed[int(v_p_list_train[i][-1][1:])])
                
                feature_temp = np.reshape(np.asarray(feature_temp), [1, -1])
                v_text_embed[i] = feature_temp
        self.v_text_embed = v_text_embed
        print('size of v_text_embed: ', len(v_text_embed))
        
        feature_list = [self.a_net_embed, self.p_net_embed, self.v_net_embed,\
                        self.t_net_embed, self.p_title_embed, self.term_embed,\
                       self.a_text_embed, self.v_text_embed]
        
        self.feature_list = feature_list
        print(len(self.feature_list))
        
        #store neighbor set from random walk sequence
        a_neigh_list_train = [[[] for i in range(self.A_n)] for j in range(4)]
        p_neigh_list_train = [[[] for i in range(self.P_n)] for j in range(4)]
        v_neigh_list_train = [[[] for i in range(self.V_n)] for j in range(4)]
        t_neigh_list_train = [[[] for i in range(self.T_n)] for j in range(4)]
        
        het_neigh_train_f = open(self.data_path + "het_neigh_train.txt", "r")
        for line in het_neigh_train_f:
            line = line.strip()
            node_id = re.split(':', line)[0]
            neigh = re.split(':', line)[1]
            neigh_list = re.split(',', neigh)
            if node_id[0] == 'a' and len(node_id) > 1:
                for j in range(len(neigh_list)):
                    if neigh_list[j][0] == 'a':
                        a_neigh_list_train[0][int(node_id[1:])].append(int(neigh_list[j][1:]))
                    elif neigh_list[j][0] == 'p':
                        a_neigh_list_train[1][int(node_id[1:])].append(int(neigh_list[j][1:]))
                    elif neigh_list[j][0] == 'v':
                        a_neigh_list_train[2][int(node_id[1:])].append(int(neigh_list[j][1:]))
                    else:
                        a_neigh_list_train[3][int(node_id[1:])].append(int(neigh_list[j][1:]))
            elif node_id[0] == 'p' and len(node_id) > 1:
                for j in range(len(neigh_list)):
                    if neigh_list[j][0] == 'a':
                        p_neigh_list_train[0][int(node_id[1:])].append(int(neigh_list[j][1:]))
                    elif neigh_list[j][0] == 'p':
                        p_neigh_list_train[1][int(node_id[1:])].append(int(neigh_list[j][1:]))
                    elif neigh_list[j][0] == 'v':
                        p_neigh_list_train[2][int(node_id[1:])].append(int(neigh_list[j][1:]))
                    else:
                        p_neigh_list_train[3][int(node_id[1:])].append(int(neigh_list[j][1:]))
            elif node_id[0] == 'v' and len(node_id) > 1:
                for j in range(len(neigh_list)):
                    if neigh_list[j][0] == 'a':
                        v_neigh_list_train[0][int(node_id[1:])].append(int(neigh_list[j][1:]))
                    elif neigh_list[j][0] == 'p':
                        v_neigh_list_train[1][int(node_id[1:])].append(int(neigh_list[j][1:]))
                    elif neigh_list[j][0] == 'v':
                        v_neigh_list_train[2][int(node_id[1:])].append(int(neigh_list[j][1:]))	
                    else:
                        v_neigh_list_train[3][int(node_id[1:])].append(int(neigh_list[j][1:]))
            else:
                if node_id[0] == 't' and len(node_id) > 1:
                    for j in range(len(neigh_list)):
                        if neigh_list[j][0] == 'a':
                            t_neigh_list_train[0][int(node_id[1:])].append(int(neigh_list[j][1:]))
                        elif neigh_list[j][0] == 'p':
                            t_neigh_list_train[1][int(node_id[1:])].append(int(neigh_list[j][1:]))
                        elif neigh_list[j][0] == 'v':
                            t_neigh_list_train[2][int(node_id[1:])].append(int(neigh_list[j][1:]))	
                        else:
                            t_neigh_list_train[3][int(node_id[1:])].append(int(neigh_list[j][1:]))
                    
        het_neigh_train_f.close()
        
        #store top neighbor set (based on frequency) from random walk sequence 
        a_neigh_list_train_top = [{i: [] for i in range(self.A_n)} for j in range(4)]
        p_neigh_list_train_top = [{i: [] for i in range(self.P_n)} for j in range(4)]
        v_neigh_list_train_top = [{i: [] for i in range(self.V_n)} for j in range(4)]
        t_neigh_list_train_top = [{i: [] for i in range(self.T_n)} for j in range(4)]
        top_k = [10, 10, 3, 3] #fix each neighor type size 
        for i in range(self.A_n):
            for j in range(4):
                a_neigh_list_train_temp = Counter(a_neigh_list_train[j][i])
                top_list = a_neigh_list_train_temp.most_common(top_k[j])
                neigh_size = 0
                if j == 0 or j == 1:
                    neigh_size = 10
                else:
                    neigh_size = 3
                for k in range(len(top_list)):
                    a_neigh_list_train_top[j][i].append(int(top_list[k][0]))
                if len(a_neigh_list_train_top[j][i]) and len(a_neigh_list_train_top[j][i]) < neigh_size:
                    for l in range(len(a_neigh_list_train_top[j][i]), neigh_size):
                        a_neigh_list_train_top[j][i].append(random.choice(a_neigh_list_train_top[j][i]))
        for i in range(self.P_n):
            for j in range(4):
                p_neigh_list_train_temp = Counter(p_neigh_list_train[j][i])
                top_list = p_neigh_list_train_temp.most_common(top_k[j])
                neigh_size = 0
                if j == 0 or j == 1:
                    neigh_size = 10
                else:
                    neigh_size = 3
                for k in range(len(top_list)):
                    p_neigh_list_train_top[j][i].append(int(top_list[k][0]))
                if len(p_neigh_list_train_top[j][i]) and len(p_neigh_list_train_top[j][i]) < neigh_size:
                    for l in range(len(p_neigh_list_train_top[j][i]), neigh_size):
                        p_neigh_list_train_top[j][i].append(random.choice(p_neigh_list_train_top[j][i]))
        
        for i in range(self.V_n):
            for j in range(4):
                v_neigh_list_train_temp = Counter(v_neigh_list_train[j][i])
                top_list = v_neigh_list_train_temp.most_common(top_k[j])
                neigh_size = 0
                if j == 0 or j == 1:
                    neigh_size = 10
                else:
                    neigh_size = 3
                for k in range(len(top_list)):
                    v_neigh_list_train_top[j][i].append(int(top_list[k][0]))
                if len(v_neigh_list_train_top[j][i]) and len(v_neigh_list_train_top[j][i]) < neigh_size:
                    for l in range(len(v_neigh_list_train_top[j][i]), neigh_size):
                        v_neigh_list_train_top[j][i].append(random.choice(v_neigh_list_train_top[j][i]))

        for i in range(self.T_n):
            for j in range(4):
                t_neigh_list_train_temp = Counter(t_neigh_list_train[j][i])
                top_list = t_neigh_list_train_temp.most_common(top_k[j])
                neigh_size = 0
                if j == 0 or j == 1:
                    neigh_size = 10
                else:
                    neigh_size = 3
                for k in range(len(top_list)):
                    t_neigh_list_train_top[j][i].append(int(top_list[k][0]))
                if len(t_neigh_list_train_top[j][i]) and len(t_neigh_list_train_top[j][i]) < neigh_size:
                    for l in range(len(t_neigh_list_train_top[j][i]), neigh_size):
                        t_neigh_list_train_top[j][i].append(random.choice(t_neigh_list_train_top[j][i]))
                    
        a_neigh_list_train[:] = []
        p_neigh_list_train[:] = []
        v_neigh_list_train[:] = []
        t_neigh_list_train[:] = []

        self.a_neigh_list_train = a_neigh_list_train_top
        self.p_neigh_list_train = p_neigh_list_train_top
        self.v_neigh_list_train = v_neigh_list_train_top
        self.t_neigh_list_train = t_neigh_list_train_top
        
        node_neigh_list_train = [self.a_neigh_list_train, self.p_neigh_list_train,\
                                self.v_neigh_list_train, self.t_neigh_list_train]
        self.node_neigh_list_train = node_neigh_list_train
