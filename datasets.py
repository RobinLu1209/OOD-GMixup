import torch
import numpy as np
import torch.nn.functional as F
import torch_geometric
from torch_geometric.loader import DataLoader
from util_func import *
from models import *
import math
import argparse
from datetime import datetime  
import os.path as osp
from torch_geometric.utils import degree
import random
import torch_geometric.transforms as T

data2dir = {
    'mnist': {'root': '../../dataset', 'name': 'MNIST'},
    'cifar10': {'root': '../../dataset', 'name': 'CIFAR10'},
    'coil': {'root': '../../dataset', 'name': 'COIL-DEL'},
    'motif': {'root': '../../dataset/SPMotif-0.9', 'name':''},
    'proteins': {'root': '../../dataset', 'name': 'PROTEINS'},
    'collab': {'root': '../../dataset', 'name': 'COLLAB'},
    'reddit': {'root': '../../dataset', 'name': 'REDDIT-BINARY'},
    'molhiv': {'root': '../../dataset', 'name': 'ogbg-molhiv'},
    'imdb': {'root': '../../dataset', 'name': 'IMDB-BINARY'},
    'enzy': {'root': '../../dataset', 'name': 'ENZYMES'},
    'DD': {'root': '../../dataset', 'name': 'DD'},
    'msrc': {'root': '../../dataset', 'name': 'MSRC_21'}
}

def get_dataset_factory():
    return list(data2dir.keys())

def add_degree_feature(dataset):
    max_degree = 0
    degs = []
    for data in dataset:
        degs += [degree(data.edge_index[0], dtype=torch.long)]
        max_degree = max(max_degree, degs[-1].max().item())
    if max_degree < 1000:
        dataset.transform = T.OneHotDegree(max_degree)
    else:
        deg = torch.cat(degs, dim=0).to(torch.float)
        mean, std = deg.mean().item(), deg.std().item()
        dataset.transform = NormalizedDegree(mean, std)
    return dataset

def get_dataset(name):
    dir_root, dataset_name = data2dir[name]['root'], data2dir[name]['name']
    if name == 'mnist':
        train_dataset = torch_geometric.datasets.GNNBenchmarkDataset(root=dir_root, name='MNIST', split='train', transform=MNISTFeatureTransform())[0:6000]
        val_dataset = torch_geometric.datasets.GNNBenchmarkDataset(root=dir_root, name='MNIST', split='val', transform=MNISTFeatureTransform())[0:1000]
        test_dataset = torch_geometric.datasets.GNNBenchmarkDataset(root=dir_root, name='MNIST', split='test', transform=MNISTFeatureTransform())[0:3000]
    elif name == 'cifar10':
        train_dataset = torch_geometric.datasets.GNNBenchmarkDataset(root=dir_root, name='CIFAR10', split='train', transform=CIFARCombineFeature())[0:6000]
        val_dataset = torch_geometric.datasets.GNNBenchmarkDataset(root=dir_root, name='CIFAR10', split='val', transform=CIFARCombineFeature())[0:1000]
        test_dataset = torch_geometric.datasets.GNNBenchmarkDataset(root=dir_root, name='CIFAR10', split='test', transform=CIFARTestTransform())[0:3000]
    elif name == 'coil':
        dataset = torch_geometric.datasets.TUDataset(dir_root, "COIL-DEL")
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)
        
        print(dataset[0])

        train_size, val_size = 2000, 800
        test_size = len(dataset) - train_size - val_size
        train_dataset = dataset[0: train_size]
        val_dataset = dataset[train_size: train_size+val_size]
        test_dataset = dataset[train_size+val_size:]
        # feature shift 
        for i in range(len(test_dataset)):
            edge_num = test_dataset[i].edge_attr.shape[0]
            disturb_time = int(edge_num / 50)
            for k in range(disturb_time):
                _idx = random.randint(0, edge_num-1)
                test_dataset[i].edge_attr[_idx] = 1 - test_dataset[i].edge_attr[_idx]
                
    elif name == 'motif':
        train_dataset = SPMotif(osp.join(dir_root), mode='train')
        val_dataset = SPMotif(osp.join(dir_root), mode='val')
        test_dataset = SPMotif(osp.join(dir_root), mode='test')
    elif name in ['proteins', 'collab', 'reddit', 'imdb', 'enzy', 'DD', 'msrc']:
        dataset = torch_geometric.datasets.TUDataset(dir_root, dataset_name)
        if dataset.data.x is None:
            dataset = add_degree_feature(dataset)
        if name == 'proteins':
            train_idx, test_idx = [], []
            num_threshold = 25
            train_size, val_size = 400, 100

            # idx = [i for i in range(len(dataset))]
            # random.shuffle(idx)
            # train_idx = idx[0: train_size + val_size]
            # test_idx = idx[train_size + val_size:]

            for idx in range(len(dataset)):
                if dataset[idx].x.shape[0] <= num_threshold and len(train_idx) <= train_size + val_size:
                    train_idx.append(idx)
                else:
                    test_idx.append(idx)
                    
        elif name == 'collab':
            train_size, val_size = 1500, 500
            edge_threshold = 1000
            train_idx, test_idx = [], []
            for idx in range(len(dataset)):
                if dataset[idx].edge_index.shape[1] <= edge_threshold and len(train_idx) < train_size + val_size:
                    train_idx.append(idx)
                else:
                    test_idx.append(idx)
        
        elif name == 'enzy':
            train_size, val_size = 280, 100
            test_size = len(dataset) - train_size - val_size
            ratio_threshold = 3.86
            train_idx, test_idx = [], []
            for idx in range(len(dataset)):
                num_nodes = dataset[idx].num_nodes
                num_edges = dataset[idx].edge_index.shape[1]
                ratio = num_edges / num_nodes
                if ratio < ratio_threshold and len(test_idx) < test_size:
                    test_idx.append(idx)
                else:
                    train_idx.append(idx)

        elif name == 'DD':
            train_size, val_size = 400, 100
            edge_threshold = 1431
            train_idx, test_idx = [], []
            for idx in range(len(dataset)):
                if dataset[idx].edge_index.shape[1] <= edge_threshold and len(train_idx) < train_size + val_size:
                    train_idx.append(idx)
                else:
                    test_idx.append(idx)
        
        elif name == 'imdb':
            train_idx, test_idx = [], []
            num_threshold = 20
            train_size, val_size = 400, 100
            for idx in range(len(dataset)):
                if dataset[idx].num_nodes <= num_threshold and len(train_idx) <= train_size + val_size:
                    train_idx.append(idx)
                else:
                    test_idx.append(idx)

        elif name == 'reddit':
            train_size, val_size = 1200, 200
            ratio_threshold = 2.4
            train_idx, test_idx = [], []
            for idx in range(len(dataset)):
                num_nodes = dataset[idx].num_nodes
                num_edges = dataset[idx].edge_index.shape[1]
                ratio = num_edges / num_nodes
                if ratio < ratio_threshold and len(train_idx) < train_size + val_size:
                    train_idx.append(idx)
                else:
                    test_idx.append(idx)
        
        elif name == 'msrc':
            train_size, val_size = 300, 63
            edge_threshold = 450
            train_idx, test_idx = [], []
            for idx in range(len(dataset)):
                if dataset[idx].edge_index.shape[1] <= edge_threshold and len(train_idx) < train_size + val_size:
                    train_idx.append(idx)
                else:
                    test_idx.append(idx)

        else:
            raise ValueError('Undefined dataset called {}'.format(name))
        random.seed(1024)
        random.shuffle(train_idx)
        train_dataset = dataset[train_idx][0: train_size]
        val_dataset = dataset[train_idx][train_size: train_size + val_size]
        test_dataset = dataset[test_idx]

    elif name == 'molhiv':
        dataset = PygGraphPropPredDataset(name = "ogbg-molhiv", root = 'dataset/')
        split_idx = dataset.get_idx_split() 
        train_dataset, val_dataset, test_dataset = dataset[split_idx["train"]], dataset[split_idx["valid"]], dataset[split_idx["test"]]

    # elif name == 'msrc':
    #     dataset = torch_geometric.datasets.TUDataset(dir_root, dataset_name)
    #     train_num, val_num = 300, 63
    #     all_idx = [i for i in range(len(dataset))]
    #     random.seed(1024)
    #     random.shuffle(all_idx)
    #     train_idx, val_idx, test_idx = all_idx[0:train_num], all_idx[train_num: train_num+val_num], all_idx[train_num+val_num:]
    #     train_dataset = dataset[train_idx]
    #     val_dataset = dataset[val_idx]
    #     test_dataset = dataset[test_idx]
        
    #     # feature shift: random select one bit of node feature 0->1 or 1->0
    #     for i in range(len(test_dataset)):
    #         data_x = test_dataset[i].x
    #         for j in range(data_x.shape[0]):
    #             random_idx = random.randint(0, data_x.shape[1]-1)
    #             data_x[j][random_idx] = 1 - data_x[j][random_idx]
        

    else:
        raise ValueError('Undefined dataset called {}'.format(name))
        return -1
    return train_dataset, val_dataset, test_dataset
    
        

