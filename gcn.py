"""GCN using DGL nn package
References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer
from torch.autograd import Variable

class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h

class Gconv(nn.Module):
    def __init__(self, nf_input, nf_output, J, bn_bool=True):
        super(Gconv, self).__init__()
        self.J = J
        self.num_inputs = J*nf_input
        self.num_outputs = nf_output
        self.fc = nn.Linear(self.num_inputs, self.num_outputs)

        self.bn_bool = bn_bool
        if self.bn_bool:
            self.bn = nn.BatchNorm1d(self.num_outputs)

    def forward(self, input):
        W = input[0]
        x = gmul(input) # out has size (bs, N, num_inputs)
        #if self.J == 1:
        #    x = torch.abs(x)
        x_size = x.size()
        x = x.contiguous()
        x = x.view(-1, self.num_inputs)
        x = self.fc(x) # has size (bs*N, num_outputs)

        if self.bn_bool:
            x = self.bn(x)

        x = x.view(*x_size[:-1], self.num_outputs)
        return W, x

class com_adj(nn.Module):
    def __init__(self, input_features, activation='softmax', ratio=[2, 1.5, 1, 1], drop=False):
        super(com_adj, self).__init__()
        # self.num_features = nf
        nf = input_features/2
        # self.operator = operator
        self.conv2d_1 = nn.Conv2d(input_features, int(nf* ratio[0]), 1, stride=1)
        self.bn_1 = nn.BatchNorm2d(int(nf * ratio[0]))
        self.drop = drop
        if self.drop:
            self.dropout = nn.Dropout(0.3)
        self.conv2d_2 = nn.Conv2d(int(nf * ratio[0]), int(nf * ratio[1]), 1, stride=1)
        self.bn_2 = nn.BatchNorm2d(int(nf * ratio[1]))
        self.conv2d_3 = nn.Conv2d(int(nf * ratio[1]), int(nf*ratio[2]), 1, stride=1)
        self.bn_3 = nn.BatchNorm2d(int(nf*ratio[2]))
        self.conv2d_4 = nn.Conv2d(int(nf*ratio[2]), int(nf*ratio[3]), 1, stride=1)
        self.bn_4 = nn.BatchNorm2d(int(nf*ratio[3]))
        self.conv2d_last = nn.Conv2d(int(nf), 1, 1, stride=1)
        self.activation = activation

    def forward(self, x):
        x = x.unsqueeze(0)
        W_init = Variable(torch.eye(x.size(1)).unsqueeze(0).repeat(x.size(0), 1, 1).unsqueeze(3))
        W_init = W_init.cuda()
        W1 = x.unsqueeze(2)
        # print(W1.size(), 'Wcomput_W1')
        W2 = torch.transpose(W1, 1, 2) #size: bs x N x N x num_features
        # print(W2.size(), 'Wcomput_W2')
        W_new = torch.abs(W1 - W2) #size: bs x N x N x num_features
        # print(W_new.size(), 'Wcomput_W_new')
        W_new = torch.transpose(W_new, 1, 3) #size: bs x num_features x N x N
        # print(W_new.size(), 'Wcomput_W_new_transpose')
        W_new = self.conv2d_1(W_new)
        W_new = self.bn_1(W_new)
        W_new = F.leaky_relu(W_new)
        if self.drop:
            W_new = self.dropout(W_new)

        W_new = self.conv2d_2(W_new)
        W_new = self.bn_2(W_new)
        W_new = F.leaky_relu(W_new)

        W_new = self.conv2d_3(W_new)
        W_new = self.bn_3(W_new)
        W_new = F.leaky_relu(W_new)

        W_new = self.conv2d_4(W_new)
        W_new = self.bn_4(W_new)
        W_new = F.leaky_relu(W_new)

        W_new = self.conv2d_last(W_new)
        W_new = torch.transpose(W_new, 1, 3) #size: bs x N x N x 1
        
        if self.activation == 'softmax':
            W_new = W_new - W_init.expand_as(W_new) * 1e8
            W_new = torch.transpose(W_new, 2, 3)
            # Applying Softmax
            W_new = W_new.contiguous()
            W_new_size = W_new.size()
            W_new = W_new.view(-1, W_new.size(3))
            W_new = F.softmax(W_new)
            W_new = W_new.view(W_new_size)
            # Softmax applied
            W_new = torch.transpose(W_new, 2, 3)
        W_new = W_new.squeeze()
        # elif self.activation == 'sigmoid':
        #     W_new = F.sigmoid(W_new)
        #     W_new *= (1 - W_id)
        # elif self.activation == 'none':
        #     W_new *= (1 - W_id)
        # else:
        #     raise (NotImplementedError)

        # if self.operator == 'laplace':
        #     W_new = W_id - W_new
        # elif self.operator == 'J2':
        #     W_new = torch.cat([W_id, W_new], 3)
        # else:
        #     raise(NotImplementedError)
        # print(W_new.size(), 'com_adj_return')
        return W_new

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)