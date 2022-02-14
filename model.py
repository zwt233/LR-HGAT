import numpy as np
from itertools import *
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import dgl

class GraphAttentionLayer(nn.Module):
    def __init__(self, input_size, out_size, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.input_size = input_size
        self.out_size = out_size
        self.dropout = dropout
        self.alpha = alpha # activation param for leakyrelu
        self.concat = concat # If true, apply elu
        
        self.W = nn.Parameter(torch.zeros(size=(input_size, out_size)))  
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_size, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, input_feature, adj):
        '''
        Simple GAT layer
        
        Args:
            input_feature: node neighbour features of shape (B, N, input_size)
            adj: adjacency matrix of shape (N, N)
        Output:
            tensor of shape (batch_size, feature_input_size)
        '''
    
        h = torch.matmul(input_feature, self.W)
        N = adj.shape[0]    # N: nodes in the graph
        
        # [N, N, 2*out_size]
        a_input = torch.cat([h.repeat(1, 1, N).view(-1, N*N, self.out_size), 
                             h.repeat(1, N, 1)], dim=1).view(-1, N, N, 2*self.out_size)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3)) #[B, N, N]
        
        zero_vec = -1e12 * torch.ones_like(e) # -inf for no link
        attention = torch.where(adj>0, e, zero_vec)   # [B, N, N]
        attention = F.softmax(e, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)  # [B, N, out_size]
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime 

class GAT(nn.Module): # GAT module
    def __init__(self, input_size, hidden_size, out_size, dropout, alpha, n_heads, residual = False):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.residual = residual
        
        self.attentions = [GraphAttentionLayer(input_size, hidden_size, 
                                               dropout=dropout, alpha=alpha, concat=True) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(hidden_size * n_heads, out_size, dropout=dropout,
                                           alpha=alpha, concat=True)
    
    def forward(self, adj, input_feature):
        x = F.dropout(input_feature, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        if self.residual:
            x = x + input_feature
        return x

class myModel(nn.Module):
    def __init__(self, feature_list, node_neigh_list, embed_dimen,\
                 rank, gpu):
        super(myModel, self).__init__()
        self.feature_list = feature_list
        self.embed_dimen = embed_dimen
        self.rank = rank
        self.gpu = gpu
        
        self.a_neigh_list = node_neigh_list[0]
        self.p_neigh_list = node_neigh_list[1]
        self.v_neigh_list = node_neigh_list[2]
        self.t_neigh_list = node_neigh_list[3]
        
        self.author_input_d = self.embed_dimen[0]
        self.paper_input_d = self.embed_dimen[1]
        self.venue_input_d = self.embed_dimen[2]
        self.term_input_d = self.embed_dimen[3]
        
        self.author_aggregator = nn.Linear(4*self.author_input_d, self.author_input_d, bias=True)
        self.paper_aggregator = nn.Linear(2*self.paper_input_d, self.paper_input_d, bias=True)
        self.venue_aggregator = nn.Linear(6*self.venue_input_d, self.venue_input_d, bias=True)
        self.term_aggregator = nn.Linear(2*self.term_input_d, self.term_input_d, bias=True)
        
        self.author_gal = GAT(self.author_input_d, 128, 128, 0.5, 0.2, 8)
        self.paper_gal = GAT(self.paper_input_d, 128, 128, 0.5, 0.2, 8)
        self.venue_gal = GAT(self.venue_input_d, 128, 128, 0.5, 0.2, 8)
        self.term_gal = GAT(self.term_input_d, 128, 128, 0.5, 0.2, 8)
        self.author_gal2 = GAT(128, 128, 128, 0.5, 0.2, 1, residual=True)
        self.paper_gal2 = GAT(128, 128, 128, 0.5, 0.2, 1, residual=True)
        self.venue_gal2 = GAT(128, 128, 128, 0.5, 0.2, 1, residual=True)
        self.term_gal2 = GAT(128, 128, 128, 0.5, 0.2, 1, residual=True)
        
        self.author_weight = nn.Parameter(torch.Tensor(self.rank, self.author_input_d+1, 16))
        self.paper_weight = nn.Parameter(torch.Tensor(self.rank, self.paper_input_d+1, 16))
        self.venue_weight = nn.Parameter(torch.Tensor(self.rank, self.venue_input_d+1, 16))
        self.term_weight = nn.Parameter(torch.Tensor(self.rank, self.term_input_d+1, 16))
        self.combine_weight = nn.Parameter(torch.Tensor(1, self.rank))
        self.combine_bias = nn.Parameter(torch.Tensor(1, 16))
        self.fc = nn.Linear(16, 1, bias=True)
        
        nn.init.xavier_normal_(self.author_weight)
        nn.init.xavier_normal_(self.paper_weight)
        nn.init.xavier_normal_(self.venue_weight)
        nn.init.xavier_normal_(self.term_weight)
        nn.init.xavier_normal_(self.combine_weight)
        self.combine_bias.data.fill_(0)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)
    
    
    def batch_collector(self, id_batch, node_type):
        '''
        feature collector for each type in this batch
        
        Args:
            id_batch: node id batch of shape (1, 10*batch_size)
            node_type: type indicator
        Output:
            tensor of shape (batch_size, feature_input_size)
        '''
        if node_type == 'a':
            net_embed_batch = torch.Tensor([self.feature_list[0][x] for x in id_batch[0]])
            a_text_batch = [self.feature_list[6][x] for x in id_batch[0]]
            a_text_embed_batch_1 = torch.Tensor([x[0][:self.author_input_d] for x in a_text_batch])
            a_text_embed_batch_2 = torch.Tensor([x[0][self.author_input_d : self.author_input_d * 2] for x in a_text_batch])
            a_text_embed_batch_3 = torch.Tensor([x[0][self.author_input_d * 2 : self.author_input_d * 3] for x in a_text_batch])
            concate_embed = torch.cat((net_embed_batch, a_text_embed_batch_1,\
                                      a_text_embed_batch_2, a_text_embed_batch_3), 1)
            if self.gpu:
                concate_embed = concate_embed.cuda()
            net_embed_batch = self.author_aggregator(concate_embed)
        
        elif node_type == 'p':
            net_embed_batch = torch.Tensor([self.feature_list[1][x] for x in id_batch[0]])
            p_title_batch = torch.Tensor([self.feature_list[4][x] for x in id_batch[0]])
            concate_embed = torch.cat((net_embed_batch, p_title_batch), 1)
            if self.gpu:
                concate_embed = concate_embed.cuda()
            net_embed_batch = self.paper_aggregator(concate_embed)
        
        elif node_type == 't':
            net_embed_batch = torch.Tensor([self.feature_list[3][x] for x in id_batch[0]])
            term_embedn_batch = torch.Tensor([self.feature_list[5][x] for x in id_batch[0]])
            concate_embed = torch.cat((net_embed_batch, term_embedn_batch), 1)
            if self.gpu:
                concate_embed = concate_embed.cuda()
            net_embed_batch = self.term_aggregator(concate_embed)
        
        else:
            net_embed_batch = torch.Tensor([self.feature_list[2][x] for x in id_batch[0]])
            v_text_batch = [self.feature_list[7][x] for x in id_batch[0]]
            v_text_embed_batch_1 = torch.Tensor([x[0][:self.venue_input_d] for x in v_text_batch])
            v_text_embed_batch_2 = torch.Tensor([x[0][self.venue_input_d: 2 * self.venue_input_d] for x in v_text_batch])
            v_text_embed_batch_3 = torch.Tensor([x[0][2 * self.venue_input_d: 3 * self.venue_input_d] for x in v_text_batch])
            v_text_embed_batch_4 = torch.Tensor([x[0][3 * self.venue_input_d: 4 * self.venue_input_d] for x in v_text_batch])
            v_text_embed_batch_5 = torch.Tensor([x[0][4 * self.venue_input_d:] for x in v_text_batch])
            concate_embed = torch.cat((net_embed_batch, v_text_embed_batch_1, v_text_embed_batch_2, \
                                      v_text_embed_batch_3, v_text_embed_batch_4, v_text_embed_batch_5), 1)
            if self.gpu:
                concate_embed = concate_embed.cuda()
            net_embed_batch = self.venue_aggregator(concate_embed)
        
        return net_embed_batch
    
    def node_type_agg(self, id_batch, node_type):
        '''
        node type aggregator, need to be specific 
        
        Args:
            id_batch: node id batch of shape (batch_size, )
            node_type: specific node neighbour type
        Output:
            tensor of shape (batch_size, 128)
        '''
        
        if node_type == 0 or node_type == 1:
            batch_s = int(len(id_batch[0]) / 10)
        else:
            batch_s = int(len(id_batch[0]) / 3)
        
        if node_type == 0:
            neigh_agg = self.batch_collector(id_batch, 'a').view(batch_s, 10, self.author_input_d)
        elif node_type == 1:
            neigh_agg = self.batch_collector(id_batch, 'p').view(batch_s, 10, self.paper_input_d)
        elif node_type == 2:
            neigh_agg = self.batch_collector(id_batch, 't').view(batch_s, 3, self.term_input_d)
        else:
            neigh_agg = self.batch_collector(id_batch, 'v').view(batch_s, 3, self.venue_input_d)
        
        return neigh_agg
    
    def node_neigh_agg(self, id_batch, node_type):
        '''
        node neighbour aggregator, need to be specific 
        
        Args:
            id_batch: node id batch of shape (batch_size, 1)
            node_type: specific target node type
        Output:
            tensor tuple of shape (5, batch_size, node_type_hidden_d)
        '''
        a_neigh_batch = [[0] * 10] * len(id_batch)
        p_neigh_batch = [[0] * 10] * len(id_batch)
        t_neigh_batch = [[0] * 3] * len(id_batch)
        v_neigh_batch = [[0] * 3] * len(id_batch)
        
        for i in range(len(id_batch)):
            if node_type == 0:
                a_neigh_batch[i] = self.a_neigh_list[0][id_batch[i]]
                p_neigh_batch[i] = self.a_neigh_list[1][id_batch[i]]
                v_neigh_batch[i] = self.a_neigh_list[2][id_batch[i]]
                t_neigh_batch[i] = self.a_neigh_list[3][id_batch[i]]
            elif node_type == 1:
                a_neigh_batch[i] = self.p_neigh_list[0][id_batch[i]]
                p_neigh_batch[i] = self.p_neigh_list[1][id_batch[i]]
                v_neigh_batch[i] = self.p_neigh_list[2][id_batch[i]]
                t_neigh_batch[i] = self.p_neigh_list[3][id_batch[i]]
            elif node_type == 2:
                a_neigh_batch[i] = self.t_neigh_list[0][id_batch[i]]
                p_neigh_batch[i] = self.t_neigh_list[1][id_batch[i]]
                v_neigh_batch[i] = self.t_neigh_list[2][id_batch[i]]
                t_neigh_batch[i] = self.t_neigh_list[3][id_batch[i]]
            else:
                a_neigh_batch[i] = self.v_neigh_list[0][id_batch[i]]
                p_neigh_batch[i] = self.v_neigh_list[1][id_batch[i]]
                v_neigh_batch[i] = self.v_neigh_list[2][id_batch[i]]
                t_neigh_batch[i] = self.v_neigh_list[3][id_batch[i]]
        
        a_neigh_batch = np.reshape(a_neigh_batch, (1, -1))
        a_agg_batch = self.node_type_agg(a_neigh_batch, 0)
        p_neigh_batch = np.reshape(p_neigh_batch, (1, -1))
        p_agg_batch = self.node_type_agg(p_neigh_batch, 1)
        v_neigh_batch = np.reshape(v_neigh_batch, (1, -1))
        v_agg_batch = self.node_type_agg(v_neigh_batch, 3)
        t_neigh_batch = np.reshape(t_neigh_batch, (1, -1))
        t_agg_batch = self.node_type_agg(t_neigh_batch, 2)
        
        if node_type == 0:
            node_batch = self.batch_collector([id_batch], 'a')
        elif node_type == 1:
            node_batch = self.batch_collector([id_batch], 'p')
        elif node_type == 2:
            node_batch = self.batch_collector([id_batch], 'v')
        else:
            node_batch = self.batch_collector([id_batch], 't')
        
        node_batch = node_batch.unsqueeze(1)
        
        g_1 = dgl.DGLGraph()
        g_1.add_nodes(11)
        g_1.add_edges([n for n in range(0, 11)], 0)
        g_1.add_edges(0, [n for n in range(1, 11)])
        adj_1 = g_1.adjacency_matrix().to_dense()
        
        g_2 = dgl.DGLGraph()
        g_2.add_nodes(4)
        g_2.add_edges([n for n in range(0, 4)], 0)
        g_2.add_edges(0, [n for n in range(1, 4)])
        adj_2 = g_2.adjacency_matrix().to_dense()
        
        if self.gpu == True:
            adj_1 = adj_1.cuda()
            adj_2 = adj_2.cuda()
            
        # author type attention
        concate_embed = torch.cat((node_batch, a_agg_batch), 1).view(len(id_batch), 11, self.author_input_d)
        a_gat_batch_1 = self.author_gal(adj_1, concate_embed) #[B, N, input_size]
        a_gat_batch_2 = self.author_gal2(adj_1, a_gat_batch_1) #[B, N, input_size]
        a_gat_batch = a_gat_batch_2[:, 0, :]
        a_gat_batch = a_gat_batch.squeeze()
        
        # paper type attention
        concate_embed = torch.cat((node_batch, p_agg_batch), 1).view(len(id_batch), 11, self.paper_input_d)
        p_gat_batch_1 = self.paper_gal(adj_1, concate_embed) #[B, N, input_size]
        p_gat_batch_2 = self.paper_gal2(adj_1, p_gat_batch_1) #[B, N, input_size]
        p_gat_batch = p_gat_batch_2[:, 0, :]
        p_gat_batch = p_gat_batch.squeeze()
        
        # term type attention
        concate_embed = torch.cat((node_batch, t_agg_batch), 1).view(len(id_batch), 4, self.term_input_d)
        t_gat_batch_1 = self.term_gal(adj_2, concate_embed) #[B, N, input_size]
        t_gat_batch_2 = self.term_gal2(adj_2, t_gat_batch_1) #[B, N, input_size]
        t_gat_batch = t_gat_batch_2[:, 0, :]
        t_gat_batch = t_gat_batch.squeeze()
        
        # venue type attention
        concate_embed = torch.cat((node_batch, v_agg_batch), 1).view(len(id_batch), 4, self.venue_input_d)
        v_gat_batch_1 = self.venue_gal(adj_2, concate_embed) #[B, N, input_size]
        v_gat_batch_2 = self.venue_gal2(adj_2, v_gat_batch_1) #[B, N, input_size]
        v_gat_batch = v_gat_batch_2[:, 0, :]
        v_gat_batch = v_gat_batch.squeeze()
        
        return a_gat_batch, p_gat_batch, v_gat_batch, t_gat_batch
    
    def forward(self, id_batch):
        '''
        combination module and forward 
        
        Args:
            id_batch: node id batch of shape (batch_size, )
        '''
        
        # 0 indicates author type
        a_gat_batch, p_gat_batch, v_gat_batch, t_gat_batch = self.node_neigh_agg(id_batch, 0)
        batch_size = len(id_batch)
        
        if self.gpu == True:
            author_feature = torch.cat((Variable(torch.ones(batch_size, 1).type(torch.cuda.FloatTensor), 
                                            requires_grad=False), a_gat_batch), dim=1)
            paper_feature = torch.cat((Variable(torch.ones(batch_size, 1).type(torch.cuda.FloatTensor),
                                           requires_grad=False), p_gat_batch), dim=1)
            venue_feature = torch.cat((Variable(torch.ones(batch_size, 1).type(torch.cuda.FloatTensor), 
                                           requires_grad=False), v_gat_batch), dim=1)
            term_feature = torch.cat((Variable(torch.ones(batch_size, 1).type(torch.cuda.FloatTensor), 
                                          requires_grad=False), t_gat_batch), dim=1)
        else:
            author_feature = torch.cat((Variable(torch.ones(batch_size, 1),
                                                 requires_grad=False), a_gat_batch), dim=1)
            paper_feature = torch.cat((Variable(torch.ones(batch_size, 1),
                                                requires_grad=False), p_gat_batch), dim=1)
            venue_feature = torch.cat((Variable(torch.ones(batch_size, 1),
                                                requires_grad=False), v_gat_batch), dim=1)
            term_feature = torch.cat((Variable(torch.ones(batch_size, 1),
                                               requires_grad=False), t_gat_batch), dim=1)
        
        author_matrix = torch.matmul(author_feature, self.author_weight)
        paper_matrix = torch.matmul(paper_feature, self.paper_weight)
        venue_matrix = torch.matmul(venue_feature, self.venue_weight)
        term_matrix = torch.matmul(term_feature, self.term_weight)
        combine_matrix = author_matrix * paper_matrix * venue_matrix * term_matrix
        
        final_emb = torch.matmul(self.combine_weight, combine_matrix.permute(1, 0, 2)).squeeze() + self.combine_bias
        output = self.fc(final_emb)
        return output, final_emb

