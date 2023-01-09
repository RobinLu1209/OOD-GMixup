import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch.nn import Tanh, Sigmoid
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv, ResGatedGraphConv, ChebConv, LEConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool, BatchNorm
from torch_geometric.nn import TopKPooling, SAGPooling, ASAPooling
from torch_geometric.utils import dropout_adj
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch.autograd import Variable
from typing import Callable, Optional, Tuple, Union
from torch import Tensor
from torch.nn import Parameter, Sigmoid
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, PairTensor
from torch_geometric.utils import accuracy, to_dense_adj, dense_to_sparse
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli, LogitRelaxedBernoulli
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import random
import numpy as np
from util_func import *


class GNNVirtualNode(nn.Module):
    def __init__(self, args):
        super(GNNVirtualNode, self).__init__()
        self.input_dim = args.input_feature
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.class_num
        self.graph_name = args.backbone
        self.layer = args.layer
        self.pooling = args.pooling
        self.dropout = args.dropout
        self.args = args
        self.build()
    
    def build(self):
        gnn_list = [
            GNNLayer(self.input_dim, self.hidden_dim, self.graph_name, residual=False, activation='relu', dropout=self.dropout, batch_norm=self.args.batch_norm)
        ]
        if self.layer > 1:
            for i in range(self.layer - 1):
                gnn_list.append(GNNLayer(self.hidden_dim, self.hidden_dim, self.graph_name, activation='relu', dropout=self.dropout, residual=self.args.residual, batch_norm=self.args.batch_norm))
        self.gnn_layer = nn.ModuleList(gnn_list)
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)

        if self.pooling == 'mean':
            self.gnn_pooling_layer = global_mean_pool
        elif self.pooling == 'max':
            self.gnn_pooling_layer = global_max_pool
        elif self.pooling == 'sum':
            self.gnn_pooling_layer = global_add_pool
        else:
            raise ValueError('Undefined pooling type called {}'.format(self.pooling))
    
    def get_virtual_node_value(self, gnn_output, data):
        batchNum = data.y.shape[0]
        scatter_index = torch.ones(data.batch.shape) * batchNum

        if torch.cuda.is_available():
            scatter_index = scatter_index.cuda()
        else:
            scatter_index = scatter_index.cpu()

        for k in range(batchNum):
            k_idx = torch.where(data.batch == k)[0][-1]
            scatter_index[k_idx] = k
        scatter_index = scatter_index.long()
        out = scatter(gnn_output, scatter_index, dim=0, reduce='max')[0:-1]
        return out

    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        gnn_input = x
        for i in range(self.layer):
            gnn_output = self.gnn_layer[i](gnn_input, edge_index, edge_attr)
            gnn_input = gnn_output
        # virtual_output = self.gnn_pooling_layer(gnn_output, batch)
        virtual_output = self.get_virtual_node_value(gnn_output, data)
        # gnn_output = F.dropout(virtual_output, p=0.2, training=self.training)
        output = self.output_layer(virtual_output)
        return output

class GNNTopKPooling(nn.Module):
    def __init__(self, args):
        super(GNNTopKPooling, self).__init__()
        print("GNNTopKPooling init.")
        self.input_dim = args.input_feature
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.class_num
        self.layer = args.layer
        self.graph_name = args.backbone
        self.pooling = args.pooling
        self.dropout = args.dropout
        self.args = args
        self.build()
    
    def build(self):
        gnn_list = [
            GNNLayer(self.input_dim, self.hidden_dim, self.graph_name, residual=False, activation='relu', dropout=self.dropout, batch_norm=self.args.batch_norm),
            TopKPooling(self.hidden_dim)
        ]
        if self.layer > 1:
            for i in range(self.layer - 1):
                gnn_list.append(GNNLayer(self.hidden_dim, self.hidden_dim, self.graph_name, activation='relu', dropout=self.dropout, residual=self.args.residual, batch_norm=self.args.batch_norm))
                gnn_list.append(TopKPooling(self.hidden_dim))
        self.gnn_layer = nn.ModuleList(gnn_list)
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)

        if self.pooling == 'mean':
            self.gnn_pooling_layer = global_mean_pool
        elif self.pooling == 'max':
            self.gnn_pooling_layer = global_max_pool
        elif self.pooling == 'sum':
            self.gnn_pooling_layer = global_add_pool
        else:
            raise ValueError('Undefined pooling type called {}'.format(self.pooling))
    
        # self.mean_pooling = global_mean_pool
        # self.max_pooling = global_max_pool

    def tensor_list_sum(self, tensor_list):
        for i in range(len(tensor_list)):
            if i == 0:
                output_tensor = tensor_list[0]
            else:
                output_tensor += tensor_list[i]
        return output_tensor
    
    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        gnn_input = x
        k = 0
        hier_gnn_list = []
        for i in range(self.layer):
            # print("i = ", i)
            gnn_output = self.gnn_layer[k](gnn_input, edge_index, edge_attr)
            # print("gnn_output shape is", gnn_output.shape)
            gnn_input, edge_index, edge_attr, batch, _, _ = self.gnn_layer[k+1](gnn_output, edge_index, edge_attr, batch)
            # print("gnn_input shape is", gnn_input.shape, " edge_index", edge_index.shape)
            # ==============
            # pool_output = torch.cat(
            #     (self.mean_pooling(gnn_input, batch), self.max_pooling(gnn_input, batch)), 1
            # )
            # hier_gnn_list.append(pool_output)
            # ==============
            hier_gnn_list.append(self.gnn_pooling_layer(gnn_input, batch))
            k = k + 2
        
        # gnn_pooling = torch.cat(hier_gnn_list, 1)
        gnn_pooling = self.tensor_list_sum(hier_gnn_list)
        # gnn_pooling = self.gnn_pooling_layer(gnn_input, batch)
        # print("gnn_pooling shape is", gnn_pooling.shape)
        # gnn_output = F.dropout(gnn_pooling, p=0.2, training=self.training)
        output = self.output_layer(gnn_pooling)
        return output


class GNNFc(nn.Module):
    def __init__(self, args):
        super(GNNFc, self).__init__()
        self.input_dim = args.input_feature
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.class_num
        self.graph_name = args.backbone
        self.layer = args.layer
        self.pooling = args.pooling
        self.dropout = args.dropout
        self.args = args
        self.build()

    def build(self):
        gnn_list = [
            GNNLayer(self.input_dim, self.hidden_dim, self.graph_name, residual=False, activation='relu', dropout=self.dropout, batch_norm=self.args.batch_norm)
        ]
        if self.layer > 1:
            for i in range(self.layer - 1):
                gnn_list.append(
                    GNNLayer(self.hidden_dim, self.hidden_dim, self.graph_name, activation='relu', dropout=self.dropout, residual=self.args.residual, batch_norm=self.args.batch_norm)
                )
        self.gnn_layer = nn.ModuleList(gnn_list)

        if self.pooling == 'mean':
            self.gnn_pooling_layer = global_mean_pool
        elif self.pooling == 'max':
            self.gnn_pooling_layer = global_max_pool
        elif self.pooling == 'sum':
            self.gnn_pooling_layer = global_add_pool
        else:
            raise ValueError('Undefined pooling type called {}'.format(self.pooling))
    
        self.predict_fc = nn.Linear(self.hidden_dim, self.output_dim)
    
    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        if self.args.dropedge:
            # edge_index, edge_attr = dropout_adj(edge_index, edge_attr, p=0.5)
            edge_index, _, _ = dropout_node(edge_index, p=0.5, num_nodes=data.num_nodes)
        gnn_input = x
        for i in range(self.layer):
            gnn_output = self.gnn_layer[i](gnn_input, edge_index, edge_attr)
            gnn_input = gnn_output
        gnn_pooling = self.gnn_pooling_layer(gnn_output, batch)
        output = self.predict_fc(gnn_pooling)
        return output


class GNNPooling(nn.Module):
    def __init__(self, args):
        super(GNNPooling, self).__init__()
        self.input_dim = args.input_feature
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.class_num
        self.graph_name = args.backbone
        self.layer = args.layer
        self.pooling = args.pooling
        self.dropout = args.dropout
        self.args = args
        self.build()

    def build(self):
        gnn_list = [
            GNNLayer(self.input_dim, self.hidden_dim, self.graph_name, residual=False, activation='relu', dropout=self.dropout, batch_norm=self.args.batch_norm)
        ]
        if self.layer > 1:
            for i in range(self.layer - 1):
                gnn_list.append(
                    GNNLayer(self.hidden_dim, self.hidden_dim, self.graph_name, activation='relu', dropout=self.dropout, residual=self.args.residual, batch_norm=self.args.batch_norm)
                )
        self.gnn_layer = nn.ModuleList(gnn_list)

        if self.pooling == 'mean':
            self.gnn_pooling_layer = global_mean_pool
        elif self.pooling == 'max':
            self.gnn_pooling_layer = global_max_pool
        elif self.pooling == 'sum':
            self.gnn_pooling_layer = global_add_pool
        else:
            raise ValueError('Undefined pooling type called {}'.format(self.pooling))
    
    def encoding(self, data):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        gnn_input = x
        for i in range(self.layer):
            gnn_output = self.gnn_layer[i](gnn_input, edge_index, edge_attr)
            gnn_input = gnn_output
        gnn_pooling = self.gnn_pooling_layer(gnn_output, batch)
        return gnn_pooling
    
    def get_prototype(self, train_dataloader):
        for idx, data in enumerate(train_dataloader):
            data = data.cuda()
            embedding = self.encoding(data)
            if idx == 0:
                embedding_all = embedding
                y_all = data.y
            else:
                embedding_all = torch.cat((embedding_all, embedding), 0)
                y_all = torch.cat((y_all, data.y), 0)
        
        class_list = torch.unique(y_all)
        # print("class_list:", class_list)
        prototype_embedding = []
        for class_id in class_list:
            class_idx = torch.where(y_all == class_id)[0]
            prototype_i = torch.mean(
                embedding_all[class_idx], 0
            )
            prototype_embedding.append(prototype_i)
        return torch.stack(prototype_embedding)
    
    def get_sample_prototype(self, spt_data):
        embedding = self.encoding(spt_data)
        y = spt_data.y
        class_list = torch.unique(y)
        prototype_embedding = []
        for class_id in class_list:
            class_idx = torch.where(y == class_id)[0]
            prototype_i = torch.mean(
                embedding[class_idx], 0
            )
            prototype_embedding.append(prototype_i)
        return torch.stack(prototype_embedding)
    

class GNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, gnn_name, batch_norm=True, activation='relu', residual=True, dropout=0.0):
        super(GNNLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gnn_name = gnn_name
        self.batch_norm = batch_norm
        self.activation = activation
        self.residual = residual
        self.drop_ratio = dropout
        self.build()
    
    def build(self):
        if self.gnn_name == 'GCN':
            self.gnn_layer = GCNConv(self.input_dim, self.output_dim)
        elif self.gnn_name == 'GAT':
            self.gnn_layer = GATConv(self.input_dim, self.output_dim, edge_dim=2)
        elif self.gnn_name == 'GraphSage':
            self.gnn_layer = SAGEConv(self.input_dim, self.output_dim)
        elif self.gnn_name == 'GIN':
            self.mlp = nn.Linear(self.input_dim, self.output_dim)
            self.gnn_layer = GINConv(self.mlp, train_eps=True)
        elif self.gnn_name == 'GatedGCN':
            self.gnn_layer = ResGatedGraphConv(self.input_dim, self.output_dim)
        elif self.gnn_name == 'TanhGCN':
            self.gnn_layer = ResGatedGraphConv(self.input_dim, self.output_dim, act=Tanh())
        elif self.gnn_name == 'ChebConv':
            self.gnn_layer = ChebConv(self.input_dim, self.output_dim, K=2)
        elif self.gnn_name == 'LEConv':
            self.gnn_layer = LEConv(self.input_dim, self.output_dim)
        else:
            raise ValueError('Undefined GNN type called {}'.format(self.gnn_name))

        if self.batch_norm:
            self.batchnorm = BatchNorm(in_channels=self.output_dim)
    
    def forward(self, x, edge_index, edge_attr):
        if self.gnn_name in ['GCN', 'GAT']:
            gnn_output = self.gnn_layer(x, edge_index, edge_attr)
        elif self.gnn_name == 'LEConv':
            gnn_output = self.gnn_layer(x, edge_index, edge_attr.view(-1))
        else:
            gnn_output = self.gnn_layer(x, edge_index)
        if self.drop_ratio != 0.0:
            gnn_output = F.dropout(gnn_output, self.drop_ratio, training = self.training)
        if self.residual:
            gnn_output = x + gnn_output
        if self.activation == 'relu':
            gnn_output = torch.relu(gnn_output)
        if self.activation == 'tanh':
            gnn_output = torch.tanh(gnn_output)
        if self.batch_norm:
            gnn_output = self.batchnorm(gnn_output)
        
        return gnn_output


class CausalManifoldMixup(nn.Module):
    def __init__(self, args):
        super(CausalManifoldMixup,self).__init__()
        self.input_dim = args.input_feature
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.class_num
        self.backbone = args.backbone
        self.layer = args.layer
        self.pooling = args.pooling
        self.dropout = args.dropout
        self.args = args
        self.build(args)
    
    def build(self, args):
        self.mask_structure = nn.Linear(self.input_dim, self.hidden_dim)
        self.gnn_backbone = GNNPooling(args=args)
    
    def get_new_adj(self, graph, adj, temperature=0.1, eps=0.5):
        x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr
        
        mask_output = self.mask_structure(x)

        # for i in range(len(self.mask_learner)):
        #     mask_output = self.mask_learner[i](x, edge_index, edge_attr)
        #     x = mask_output

        # =========================================================
        # Soft Mask
        # =========================================================
        attention = torch.mm(mask_output, mask_output.T)
        weighted_adjacency_matrix = torch.softmax(attention, dim=1)
        causal_mask = weighted_adjacency_matrix.detach().float()
        spurious_mask = 1 - causal_mask

        # =========================================================
        # Hard Mask
        # =========================================================
        # attention = torch.mm(mask_output, mask_output.T)
        # attention = torch.sigmoid(attention)
        # attention = torch.clamp(attention, 0.01, 0.99)
        # weighted_adjacency_matrix = RelaxedBernoulli(temperature=torch.Tensor([temperature]).cuda(), probs=attention).rsample()
        # causal_mask = (weighted_adjacency_matrix > eps).detach().float()
        # spurious_mask = 1 - causal_mask

        causal_adj = torch.mul(causal_mask, adj)
        spurious_adj = torch.mul(spurious_mask, adj)
        return causal_adj, spurious_adj
    
    def get_causal_graph(self, data):
        num_sample = data.num_graphs
        graphs_list = data.to_data_list()
        causal_graphs_list, spurious_graphs_list = [], []

        for graph in graphs_list:
            x = graph.x
            edge_index = graph.edge_index
            edge_attr = graph.edge_attr

            raw_adj = to_dense_adj(edge_index, max_num_nodes=graph.num_nodes)[0]
            causal_adj, spurious_adj = self.get_new_adj(graph, raw_adj)
            causal_edge_index, causal_edge_attr = dense_to_sparse(causal_adj)
            # spurious_edge_index, spurious_edge_attr = dense_to_sparse(spurious_adj)

            causal_graph = Data(x=x, edge_index=causal_edge_index, edge_attr=causal_edge_attr).cuda()
            causal_graphs_list.append(causal_graph)
            
            # spurious_graph = Data(x=x, edge_index=spurious_edge_index, edge_attr=spurious_edge_attr)
            # spurious_graphs_list.append(spurious_graph)
        
        causal_graph_loader = DataLoader(causal_graphs_list, batch_size=len(causal_graphs_list))
        # spurious_graph_loader = DataLoader(spurious_graphs_list, batch_size=len(spurious_graphs_list))
        batch_causal = next(iter(causal_graph_loader))
        # batch_spurious = next(iter(spurious_graph_loader))
        return batch_causal
    
    def encoding(self, data):
        causal_data = self.get_causal_graph(data)
        causal_embedding = self.gnn_backbone.encoding(causal_data)
        # spurious_embedding = self.gnn_backbone.encoding(spurious_data)
        return causal_embedding
    
    def get_mixup_embedding(self, data, beta=2):
        causal_embedding = self.encoding(data)
        y = data.y
        lam = get_lambda(beta)
        one_hot_y = to_one_hot(y, self.args.class_num)
        mixup_embedding, mixup_y = mixup_process(causal_embedding, one_hot_y, lam=lam)
        return mixup_embedding, mixup_y
    
    def get_sample_prototype(self, spt_data):
        embedding = self.encoding(spt_data)
        y = spt_data.y
        class_list = torch.unique(y)
        prototype_embedding = []
        for class_id in class_list:
            class_idx = torch.where(y == class_id)[0]
            prototype_i = torch.mean(
                embedding[class_idx], 0
            )
            prototype_embedding.append(prototype_i)
        return torch.stack(prototype_embedding)

    def get_prototype(self, train_dataloader):
        for idx, data in enumerate(train_dataloader):
            data = data.cuda()
            embedding = self.encoding(data)
            y = data.y
            if idx == 0:
                embedding_all = embedding
                y_all = y
            else:
                embedding_all = torch.cat((embedding_all, embedding), 0)
                y_all = torch.cat((y_all, y), 0)
        class_list = torch.unique(y_all)
        prototype_embedding = []
        for class_id in class_list:
            class_idx = torch.where(y_all == class_id)[0]
            prototype_i = torch.mean(
                embedding_all[class_idx], 0
            )
            prototype_embedding.append(prototype_i)
        return torch.stack(prototype_embedding)

class CausalFeatureMixup(nn.Module):
    def __init__(self, args):
        super(CausalFeatureMixup,self).__init__()
        self.input_dim = args.input_feature
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.class_num
        self.backbone = args.backbone
        self.layer = args.layer
        self.pooling = args.pooling
        self.dropout = args.dropout
        self.args = args
        self.build(args)
    
    def build(self, args):
        self.mask_feature = Parameter(torch.ones(self.input_dim))
        self.gnn_backbone = GNNPooling(args=args)
        
    def get_causal_graph(self, data):
        num_sample = data.num_graphs
        graphs_list = data.to_data_list()
        causal_graphs_list, spurious_graphs_list = [], []

        data.x = data.x * self.mask_feature

        # for graph in graphs_list:
        #     x = graph.x
        #     edge_index = graph.edge_index
        #     edge_attr = graph.edge_attr
        #     causal_x = x * self.mask_feature

        #     causal_graph = Data(x=causal_x, edge_index=edge_index, edge_attr=edge_attr).cuda()
        #     causal_graphs_list.append(causal_graph)
        
        # causal_graph_loader = DataLoader(causal_graphs_list, batch_size=len(causal_graphs_list))
        # batch_causal = next(iter(causal_graph_loader))
        return data
    
    def encoding(self, data):
        causal_data = self.get_causal_graph(data)
        causal_embedding = self.gnn_backbone.encoding(causal_data)
        return causal_embedding, data.y
    
    def mixup_encoding(self, causal_embedding, y, beta):
        sample_embedding, sample_y = [], []
        lambda_list = []
        class_num, class_list = len(y.unique()), y.unique()
        sample_num = int(causal_embedding.shape[0] / class_num)
        for graph_class in class_list:
            idx = torch.where(y == graph_class)[0]
            for i in range(sample_num):
                idx_1, idx_2 = int(random.choice(idx)), int(random.choice(idx))
                lambda_ratio = np.random.beta(2, beta)
                sample_embedding.append(
                    lambda_ratio * causal_embedding[idx_1] + (1-lambda_ratio) * causal_embedding[idx_2]
                )
                sample_y.append(graph_class)
        shuffle_list = [i for i in range(len(sample_embedding))]
        random.shuffle(shuffle_list)
        sample_embedding_tensor = torch.stack(sample_embedding)
        sample_y_tensor = torch.stack(sample_y)
        return sample_embedding_tensor[shuffle_list], sample_y_tensor[shuffle_list]

    def get_mixup_embedding(self, data, beta=2):
        causal_embedding, y = self.encoding(data)
        mixup_embedding, mixup_y = self.mixup_encoding(causal_embedding, y, beta)
        return mixup_embedding, mixup_y
    
    def get_sample_prototype(self, spt_data):
        embedding, y = self.encoding(spt_data)
        class_list = torch.unique(y)
        prototype_embedding = []
        for class_id in class_list:
            class_idx = torch.where(y == class_id)[0]
            prototype_i = torch.mean(
                embedding[class_idx], 0
            )
            prototype_embedding.append(prototype_i)
        return torch.stack(prototype_embedding)

    def get_prototype(self, train_dataloader):
        for idx, data in enumerate(train_dataloader):
            data = data.cuda()
            embedding, y = self.encoding(data)
            if idx == 0:
                embedding_all = embedding
                y_all = y
            else:
                embedding_all = torch.cat((embedding_all, embedding), 0)
                y_all = torch.cat((y_all, y), 0)
        class_list = torch.unique(y_all)
        prototype_embedding = []
        for class_id in class_list:
            class_idx = torch.where(y_all == class_id)[0]
            prototype_i = torch.mean(
                embedding_all[class_idx], 0
            )
            prototype_embedding.append(prototype_i)
        return torch.stack(prototype_embedding)
    


class CausalMixup(nn.Module):
    def __init__(self, args):
        super(CausalMixup,self).__init__()
        self.input_dim = args.input_feature
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.class_num
        self.backbone = args.backbone
        self.layer = args.layer
        self.pooling = args.pooling
        self.dropout = args.dropout
        self.args = args
        self.build(args)
    
    def build(self, args):
        self.mask_structure = nn.Linear(self.input_dim, self.hidden_dim)

        # mask_gnn_list = [
        #         GNNLayer(self.input_dim, self.hidden_dim, self.backbone, residual=False, batch_norm=False, dropout=0.1)
        # ]
        # self.mask_structure = nn.ModuleList(mask_gnn_list)

        self.mask_feature = Parameter(torch.ones(self.input_dim))
        self.gnn_backbone = GNNPooling(args=args)
        
    def get_new_adj(self, graph, adj, temperature=0.1, eps=0.5):
        x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr
        
        mask_output = self.mask_structure(x)
        # for i in range(len(self.mask_structure)):
        #     mask_output = self.mask_structure[i](x, edge_index, edge_attr)
        #     x = mask_output

        # =========================================================
        # Soft Mask
        # =========================================================
        attention = torch.mm(mask_output, mask_output.T)
        attention = torch.sigmoid(attention)
        weighted_adjacency_matrix = torch.softmax(attention, dim=1)
        causal_mask = weighted_adjacency_matrix.detach().float()
        spurious_mask = 1 - causal_mask

        # =========================================================
        # Hard Mask
        # =========================================================
        # attention = torch.mm(mask_output, mask_output.T)
        # attention = torch.sigmoid(attention)
        # attention = torch.clamp(attention, 0.01, 0.99)
        # weighted_adjacency_matrix = RelaxedBernoulli(temperature=torch.Tensor([temperature]).cuda(), probs=attention).rsample()
        # causal_mask = (weighted_adjacency_matrix > eps).detach().float()
        # spurious_mask = 1 - causal_mask

        causal_adj = torch.mul(causal_mask, adj)
        spurious_adj = torch.mul(spurious_mask, adj)
        return causal_adj, spurious_adj
    
    def get_causal_graph(self, data):
        num_sample = data.num_graphs
        graphs_list = data.to_data_list()
        causal_graphs_list, spurious_graphs_list = [], []

        for graph in graphs_list:
            x = graph.x
            edge_index = graph.edge_index
            edge_attr = graph.edge_attr

            raw_adj = to_dense_adj(edge_index, max_num_nodes=graph.num_nodes)[0]
            causal_adj, spurious_adj = self.get_new_adj(graph, raw_adj)
            causal_edge_index, causal_edge_attr = dense_to_sparse(causal_adj)
            # spurious_edge_index, spurious_edge_attr = dense_to_sparse(spurious_adj)

            causal_x = x * self.mask_feature

            causal_graph = Data(x=causal_x, edge_index=causal_edge_index, edge_attr=causal_edge_attr).cuda()
            causal_graphs_list.append(causal_graph)
            
            # spurious_graph = Data(x=x, edge_index=spurious_edge_index, edge_attr=spurious_edge_attr)
            # spurious_graphs_list.append(spurious_graph)
        
        causal_graph_loader = DataLoader(causal_graphs_list, batch_size=len(causal_graphs_list))
        # spurious_graph_loader = DataLoader(spurious_graphs_list, batch_size=len(spurious_graphs_list))
        batch_causal = next(iter(causal_graph_loader))
        # batch_spurious = next(iter(spurious_graph_loader))
        # return batch_causal, batch_spurious
        return batch_causal
    
    def encoding(self, data):
        # causal_data, spurious_data = self.get_causal_graph(data)
        causal_data = self.get_causal_graph(data)
        causal_embedding = self.gnn_backbone.encoding(causal_data)
        # spurious_embedding = self.gnn_backbone.encoding(spurious_data)
        # return causal_embedding, spurious_embedding, data.y
        return causal_embedding, data.y
    
    def mixup_encoding(self, causal_embedding, y, beta, times=2):
        sample_embedding, sample_y = [], []
        lambda_list = []
        class_num, class_list = len(y.unique()), y.unique()
        sample_num = int(causal_embedding.shape[0] / class_num) * times
        for graph_class in class_list:
            idx = torch.where(y == graph_class)[0]
            for i in range(sample_num):
                idx_1, idx_2 = int(random.choice(idx)), int(random.choice(idx))
                lambda_ratio = np.random.beta(2, beta)
                # spurious_mixup = lambda_ratio * spurious_embedding[idx_1] + (1-lambda_ratio) * spurious_embedding[idx_2]
                # sample_embedding.append(
                #     torch.cat((causal_embedding[idx_1], spurious_mixup), dim=0)
                # )
                # sample_y.append(graph_class)
                # sample_embedding.append(
                #     torch.cat((causal_embedding[idx_2], spurious_mixup), dim=0)
                # )
                sample_embedding.append(
                    lambda_ratio * causal_embedding[idx_1] + (1-lambda_ratio) * causal_embedding[idx_2]
                )
                sample_y.append(graph_class)
        shuffle_list = [i for i in range(len(sample_embedding))]
        random.shuffle(shuffle_list)
        sample_embedding_tensor = torch.stack(sample_embedding)
        sample_y_tensor = torch.stack(sample_y)
        return sample_embedding_tensor[shuffle_list], sample_y_tensor[shuffle_list]

    def get_mixup_embedding(self, data, beta=2):
        causal_embedding, y = self.encoding(data)
        mixup_embedding, mixup_y = self.mixup_encoding(causal_embedding, y, beta)
        return mixup_embedding, mixup_y
    
    # def get_vanilla_embedding(self, data):
    #     causal_embedding, spurious_embedding, y = self.encoding(data)
    #     vanilla_embedding = torch.cat((causal_embedding, spurious_embedding), dim=1)
    #     return vanilla_embedding, y
    
    def get_sample_prototype(self, spt_data):
        embedding, y = self.encoding(spt_data)
        class_list = torch.unique(y)
        prototype_embedding = []
        for class_id in class_list:
            class_idx = torch.where(y == class_id)[0]
            prototype_i = torch.mean(
                embedding[class_idx], 0
            )
            prototype_embedding.append(prototype_i)
        return torch.stack(prototype_embedding)

    def get_prototype(self, train_dataloader):
        for idx, data in enumerate(train_dataloader):
            data = data.cuda()
            embedding, y = self.encoding(data)
            if idx == 0:
                embedding_all = embedding
                y_all = y
            else:
                embedding_all = torch.cat((embedding_all, embedding), 0)
                y_all = torch.cat((y_all, y), 0)
        class_list = torch.unique(y_all)
        prototype_embedding = []
        for class_id in class_list:
            class_idx = torch.where(y_all == class_id)[0]
            prototype_i = torch.mean(
                embedding_all[class_idx], 0
            )
            prototype_embedding.append(prototype_i)
        return torch.stack(prototype_embedding)
    


class CausalAugment(nn.Module):
    def __init__(self, args):
        super(CausalAugment,self).__init__()
        self.input_dim = args.input_feature
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.class_num
        self.backbone = args.backbone
        self.layer = args.layer
        self.pooling = args.pooling
        self.dropout = args.dropout
        self.args = args
        self.build(args)
    
    def build(self, args):
        self.mask_structure = nn.Linear(self.input_dim, self.hidden_dim)

        # mask_gnn_list = [
        #         GNNLayer(self.input_dim, self.hidden_dim, self.backbone, residual=False, batch_norm=False, dropout=0.1)
        # ]
        # self.mask_structure = nn.ModuleList(mask_gnn_list)

        self.mask_feature = Parameter(torch.ones(self.input_dim))
        self.gnn_backbone = GNNPooling(args=args)
        
    def get_new_adj(self, graph, adj, temperature=0.1, eps=0.5):
        x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr
        
        mask_output = self.mask_structure(x)
        # for i in range(len(self.mask_structure)):
        #     mask_output = self.mask_structure[i](x, edge_index, edge_attr)
        #     x = mask_output

        # =========================================================
        # Soft Mask
        # =========================================================
        attention = torch.mm(mask_output, mask_output.T)
        attention = torch.sigmoid(attention)
        weighted_adjacency_matrix = torch.softmax(attention, dim=1)
        causal_mask = weighted_adjacency_matrix.detach().float()
        spurious_mask = 1 - causal_mask

        # =========================================================
        # Hard Mask
        # =========================================================
        # attention = torch.mm(mask_output, mask_output.T)
        # attention = torch.sigmoid(attention)
        # attention = torch.clamp(attention, 0.01, 0.99)
        # weighted_adjacency_matrix = RelaxedBernoulli(temperature=torch.Tensor([temperature]).cuda(), probs=attention).rsample()
        # causal_mask = (weighted_adjacency_matrix > eps).detach().float()
        # spurious_mask = 1 - causal_mask

        causal_adj = torch.mul(causal_mask, adj)
        spurious_adj = torch.mul(spurious_mask, adj)
        return causal_adj, spurious_adj
    
    def get_causal_graph(self, data):
        num_sample = data.num_graphs
        graphs_list = data.to_data_list()
        causal_graphs_list, spurious_graphs_list = [], []

        for graph in graphs_list:
            x = graph.x
            edge_index = graph.edge_index
            edge_attr = graph.edge_attr

            raw_adj = to_dense_adj(edge_index, max_num_nodes=graph.num_nodes)[0]
            causal_adj, spurious_adj = self.get_new_adj(graph, raw_adj)
            causal_edge_index, causal_edge_attr = dense_to_sparse(causal_adj)
            # spurious_edge_index, spurious_edge_attr = dense_to_sparse(spurious_adj)

            causal_x = x * self.mask_feature

            causal_graph = Data(x=causal_x, edge_index=causal_edge_index, edge_attr=causal_edge_attr).cuda()
            causal_graphs_list.append(causal_graph)
            
            # spurious_graph = Data(x=x, edge_index=spurious_edge_index, edge_attr=spurious_edge_attr)
            # spurious_graphs_list.append(spurious_graph)
        
        causal_graph_loader = DataLoader(causal_graphs_list, batch_size=len(causal_graphs_list))
        # spurious_graph_loader = DataLoader(spurious_graphs_list, batch_size=len(spurious_graphs_list))
        batch_causal = next(iter(causal_graph_loader))
        # batch_spurious = next(iter(spurious_graph_loader))
        # return batch_causal, batch_spurious
        return batch_causal
    
    def encoding(self, data):
        # causal_data, spurious_data = self.get_causal_graph(data)
        causal_data = self.get_causal_graph(data)
        causal_embedding = self.gnn_backbone.encoding(causal_data)
        # spurious_embedding = self.gnn_backbone.encoding(spurious_data)
        # return causal_embedding, spurious_embedding, data.y
        return causal_embedding, data.y
    
    def mixup_encoding(self, causal_embedding, y, beta):
        sample_embedding, sample_y = [], []
        lambda_list = []
        class_num, class_list = len(y.unique()), y.unique()
        sample_num = int(causal_embedding.shape[0] / class_num)
        for graph_class in class_list:
            idx = torch.where(y == graph_class)[0]
            for i in range(sample_num):
                idx_1, idx_2 = int(random.choice(idx)), int(random.choice(idx))
                lambda_ratio = np.random.beta(2, beta)
                # spurious_mixup = lambda_ratio * spurious_embedding[idx_1] + (1-lambda_ratio) * spurious_embedding[idx_2]
                # sample_embedding.append(
                #     torch.cat((causal_embedding[idx_1], spurious_mixup), dim=0)
                # )
                # sample_y.append(graph_class)
                # sample_embedding.append(
                #     torch.cat((causal_embedding[idx_2], spurious_mixup), dim=0)
                # )
                sample_embedding.append(
                    lambda_ratio * causal_embedding[idx_1] + (1-lambda_ratio) * causal_embedding[idx_2]
                )
                sample_y.append(graph_class)
        shuffle_list = [i for i in range(len(sample_embedding))]
        random.shuffle(shuffle_list)
        sample_embedding_tensor = torch.stack(sample_embedding)
        sample_y_tensor = torch.stack(sample_y)
        return sample_embedding_tensor[shuffle_list], sample_y_tensor[shuffle_list]

    def get_mixup_embedding(self, data, beta=2):
        # data.edge_index, data.edge_attr = dropout_adj(data.edge_index, data.edge_attr, p=0.5)
        data.edge_index, _, _ = dropout_node(data.edge_index, p=0.3, num_nodes=data.num_nodes)
        causal_embedding, y = self.encoding(data)
        return causal_embedding, y
    
    def get_sample_prototype(self, spt_data):
        embedding, y = self.encoding(spt_data)
        class_list = torch.unique(y)
        prototype_embedding = []
        for class_id in class_list:
            class_idx = torch.where(y == class_id)[0]
            prototype_i = torch.mean(
                embedding[class_idx], 0
            )
            prototype_embedding.append(prototype_i)
        return torch.stack(prototype_embedding)

    def get_prototype(self, train_dataloader):
        for idx, data in enumerate(train_dataloader):
            data = data.cuda()
            embedding, y = self.encoding(data)
            if idx == 0:
                embedding_all = embedding
                y_all = y
            else:
                embedding_all = torch.cat((embedding_all, embedding), 0)
                y_all = torch.cat((y_all, y), 0)
        class_list = torch.unique(y_all)
        prototype_embedding = []
        for class_id in class_list:
            class_idx = torch.where(y_all == class_id)[0]
            prototype_i = torch.mean(
                embedding_all[class_idx], 0
            )
            prototype_embedding.append(prototype_i)
        return torch.stack(prototype_embedding)
    

