import torch
import numpy as np
import torch
from torch import Tensor
import math
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree, subgraph
import random
from torch_geometric.data import Batch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn

def mixup_process(out, target_reweighted, lam):
    indices = np.random.permutation(out.size(0))
    out = out*lam + out[indices]*(1-lam)
    target_shuffled_onehot = target_reweighted[indices]
    target_reweighted = target_reweighted * lam + target_shuffled_onehot * (1 - lam)
    return out, target_reweighted

def get_lambda(alpha=1.0):
    '''Return lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    return lam

def to_one_hot(inp, num_classes):
    y_onehot = torch.FloatTensor(inp.size(0), num_classes)
    y_onehot.zero_()

    y_onehot.scatter_(1, inp.unsqueeze(1).data.cpu(), 1)
    
    return Variable(y_onehot.cuda(),requires_grad=False)

def set_random_seed(seed_i):
    print("INFO: set random seed", seed_i)
    torch.manual_seed(seed_i)
    torch.cuda.manual_seed_all(seed_i)
    np.random.seed(seed_i)
    random.seed(seed_i)
    torch.backends.cudnn.deterministic = True

def mixup_criterion(pred, y, mixup_y, lamb):
    criterion = torch.nn.CrossEntropyLoss()
    return lamb * criterion(pred, y) + (1-lamb) * criterion(pred, mixup_y)

def extraploation_mixup(embedding, y, beta=2):
    sample_embedding, sample_y = [], []
    class_num, class_list = len(y.unique()), y.unique()
    sample_num = int(embedding.shape[0] / class_num)
    # print(f"Graph class: {class_num}, sample number: {sample_num}.")
    for graph_class in class_list:
        idx = torch.where(y == graph_class)[0]
        # sample_num = len
        for i in range(sample_num):
            idx_1, idx_2 = int(random.choice(idx)), int(random.choice(idx))
            lambda_ratio = np.random.beta(2, 2)
            ood_lambda_ratio = np.random.beta(2, 2)
            sample_embedding.append(
                ood_lambda_ratio * embedding[idx_1] + (1 - lambda_ratio) * embedding[idx_2]
            )
            sample_embedding.append(
                -1 * lambda_ratio * embedding[idx_1] + (1 + lambda_ratio) * embedding[idx_2]
            )
            sample_embedding.append(
                (1 + lambda_ratio) * embedding[idx_1] + -1 * lambda_ratio * embedding[idx_2]
            )
            sample_y.append(graph_class)
            sample_y.append(graph_class)
            sample_y.append(graph_class)
    shuffle_list = [i for i in range(len(sample_embedding))]
    random.shuffle(shuffle_list)
    sample_embedding_tensor = torch.stack(sample_embedding)
    sample_y_tensor = torch.stack(sample_y)
    return sample_embedding_tensor[shuffle_list], sample_y_tensor[shuffle_list]

# def patchup(embedding, y):
    

def mixup(embedding, y, more=1, beta=2):
    '''
        embedding shape is [batch_size, hidden_dim]
        y shape is [batch_size]
    '''
    sample_embedding, sample_y = [], []
    lambda_list = []
    class_num, class_list = len(y.unique()), y.unique()
    sample_num = int(embedding.shape[0] / class_num)
    # print(f"Graph class: {class_num}, sample number: {sample_num}.")
    for graph_class in class_list:
        idx = torch.where(y == graph_class)[0]
        for i in range(sample_num * more):
            idx_1, idx_2 = int(random.choice(idx)), int(random.choice(idx))
            lambda_ratio = np.random.beta(2, beta)
            lambda_list.append(lambda_ratio)
            sample_embedding.append(
                lambda_ratio * embedding[idx_1] + (1 - lambda_ratio) * embedding[idx_2]
            )
            sample_y.append(graph_class)
    shuffle_list = [i for i in range(len(sample_embedding))]
    random.shuffle(shuffle_list)
    sample_embedding_tensor = torch.stack(sample_embedding)
    sample_y_tensor = torch.stack(sample_y)
    return sample_embedding_tensor[shuffle_list], sample_y_tensor[shuffle_list]

def get_idx_by_class(dataset):
    idx_by_class = {}
    for i in range(len(dataset)):
        class_y = dataset[i].y
        if class_y.shape is not 1:
            class_y = int(class_y[0])
        else:
            class_y = int(class_y)
        if class_y not in idx_by_class.keys():
            idx_by_class[class_y] = [i]
        else:
            idx_by_class[class_y].append(i)
    return idx_by_class

def cosine_distance(tensor1, tensor2):
  dot_product = np.dot(tensor1, tensor2.T)
  norm_tensor1 = np.linalg.norm(tensor1)
  norm_tensor2 = np.linalg.norm(tensor2)
  return dot_product / (norm_tensor1 * norm_tensor2)

def euclidean_dist(x, y):
    # x: N x D query
    # y: M x D prototype
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)  # N x M
    # return torch.pow(x - y, 1).sum(2)

def get_task_data(dataset, idx_by_class, spt_num):
    class_list = list(idx_by_class.keys())
    # spt_list, qry_list = [], []
    spt_list = []
    for class_id in class_list:
        spt_qry_idx = random.sample(idx_by_class[class_id], spt_num)
        spt_list += spt_qry_idx[0: spt_num]
        # qry_list += spt_qry_idx[spt_num: spt_num + qry_num]
    random.shuffle(spt_list)
    # random.shuffle(qry_list)
    # print(f"spt_list: {spt_list}")
    # print(f"qry_list: {qry_list}")
    spt_data = Batch.from_data_list(dataset[spt_list]).cuda()
    # qry_data = Batch.from_data_list(dataset[qry_list]).cuda()
    return spt_data

from typing import Optional, Tuple
def dropout_node(edge_index: Tensor, p: float = 0.5,
                 num_nodes: Optional[int] = None,
                 training: bool = True) -> Tuple[Tensor, Tensor, Tensor]:
    r"""Randomly drops nodes from the adjacency matrix
    :obj:`edge_index` with probability :obj:`p` using samples from
    a Bernoulli distribution.

    The method returns (1) the retained :obj:`edge_index`, (2) the edge mask
    indicating which edges were retained. (3) the node mask indicating
    which nodes were retained.

    Args:
        edge_index (LongTensor): The edge indices.
        p (float, optional): Dropout probability. (default: :obj:`0.5`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        training (bool, optional): If set to :obj:`False`, this operation is a
            no-op. (default: :obj:`True`)

    :rtype: (:class:`LongTensor`, :class:`BoolTensor`, :class:`BoolTensor`)

    Examples:

        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> edge_index, edge_mask, node_mask = dropout_node(edge_index)
        >>> edge_index
        tensor([[0, 1],
                [1, 0]])
        >>> edge_mask
        tensor([ True,  True, False, False, False, False])
        >>> node_mask
        tensor([ True,  True, False, False])
    """
    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if not training or p == 0.0:
        node_mask = edge_index.new_ones(num_nodes, dtype=torch.bool)
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask, node_mask

    prob = torch.rand(num_nodes, device=edge_index.device)
    node_mask = prob > p
    edge_index, _, edge_mask = subgraph(node_mask, edge_index,
                                        num_nodes=num_nodes,
                                        return_edge_mask=True)
    return edge_index, edge_mask, node_mask


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=True, delta=0, model_name='checkpoint.pt', save_model=True):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            上次验证集损失值改善后等待几个epoch
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            如果是True，为每个验证集损失值改善打印一条信息
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            监测数量的最小变化，以符合改进的要求
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.model_name = model_name
        self.save_model = save_model

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''
        Saves model when validation loss decrease.
        验证损失减少时保存模型。
        '''
        # if self.verbose:
        #     print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Val Acc {acc:.4f}. Saving model ...')
        
        if self.save_model:
            torch.save(model.state_dict(), self.model_name)     # 这里会存储迄今最优模型的参数
            # torch.save(model, 'finish_model.pkl')                 # 这里会存储迄今最优的模型
        self.val_loss_min = val_loss

class VirtualNode(BaseTransform):
    r"""Appends a virtual node to the given homogeneous graph that is connected
    to all other nodes, as described in the `"Neural Message Passing for
    Quantum Chemistry" <https://arxiv.org/abs/1704.01212>`_ paper.
    The virtual node serves as a global scratch space that each node both reads
    from and writes to in every step of message passing.
    This allows information to travel long distances during the propagation
    phase.

    Node and edge features of the virtual node are added as zero-filled input
    features.
    Furthermore, special edge types will be added both for in-coming and
    out-going information to and from the virtual node.
    """
    def __call__(self, data: Data) -> Data:
        num_nodes, (row, col) = data.num_nodes, data.edge_index
        edge_type = data.get('edge_type', torch.zeros_like(row))

        arange = torch.arange(num_nodes, device=row.device)
        full = row.new_full((num_nodes, ), num_nodes)
        row = torch.cat([row, arange, full], dim=0)
        col = torch.cat([col, full, arange], dim=0)
        edge_index = torch.stack([row, col], dim=0)

        new_type = edge_type.new_full((num_nodes, ), int(edge_type.max()) + 1)
        edge_type = torch.cat([edge_type, new_type, new_type + 1], dim=0)

        for key, value in data.items():
            if key == 'edge_index' or key == 'edge_type':
                continue

            if isinstance(value, Tensor):
                dim = data.__cat_dim__(key, value)
                size = list(value.size())

                fill_value = None
                if key == 'edge_weight':
                    size[dim] = 2 * num_nodes
                    fill_value = 1.
                elif data.is_edge_attr(key):
                    size[dim] = 2 * num_nodes
                    fill_value = 0.
                elif data.is_node_attr(key):
                    size[dim] = 1
                    fill_value = 0.
                
                if key == 'pos':
                    size[dim] = 1
                    fill_value = 0.

                if fill_value is not None:
                    new_value = value.new_full(size, fill_value)
                    data[key] = torch.cat([value, new_value], dim=dim)

        data.edge_index = edge_index
        data.edge_type = edge_type

        if 'num_nodes' in data:
            data.num_nodes = data.num_nodes + 1

        return data


def rotate_test(theta, test_dataset_shift, model, args, prototype_embedding=None):
    test_pos = test_dataset_shift.data.pos
    pos_x, pos_y = test_pos[:, 0], test_pos[:, 1]
    test_pos[:, 0] = pos_x * math.cos(theta) + pos_y * math.sin(theta)
    test_pos[:, 1] = pos_y * math.cos(theta) - pos_x * math.sin(theta)
    test_dataset_shift.data.pos = test_pos
    test_dataloader_shift = DataLoader(test_dataset_shift, batch_size=args.batch_size)
    correct = 0
    if prototype_embedding == None:
        for batch_test, data in enumerate(test_dataloader_shift):
            data = data.to(args.device)
            y = data.y.to(args.device)
            if args.dataset == 'mnist':
                _x = torch.cat((data.x.repeat(1,3), data.pos), dim=1)
                data.x = _x
            pred_output = model(data)
            pred_label = pred_output.argmax(dim=1)
            correct += int((pred_label == data.y).sum())
    else:
        for batch_test, data in enumerate(test_dataloader_shift):
            data = data.to(args.device)
            y = data.y.to(args.device)
            if args.dataset == 'mnist':
                _x = torch.cat((data.x.repeat(1,3), data.pos), dim=1)
                data.x = _x
            query_embedding = model.encoding(data)
            dists = euclidean_dist(query_embedding, prototype_embedding)
            dist_output = F.log_softmax(-dists, dim=1)
            pred_label = dist_output.argmax(dim=1)
            correct += int((pred_label == data.y).sum())
    correct = correct / len(test_dataloader_shift.dataset)
    return correct

def colored_test(test_dataset_shift, model, args, prototype_embedding=None):
    # print("OOD test: color changing...")
    test_dataloader_shift = DataLoader(test_dataset_shift, batch_size=args.batch_size)
    correct = 0
    for batch_test, data in enumerate(test_dataloader_shift):
        data = data.to(args.device)
        y = data.y.to(args.device)
        if args.dataset == 'mnist':
            # color_x = torch.rand([data.x.shape[0], 2]).to('cuda')
            # _x = torch.cat((data.x, color_x), dim=1)
            # _x = torch.cat((_x, data.pos), dim=1)
            _x = torch.cat((data.x.repeat(1,3), data.pos), dim=1)
            # _x[:, 0], _x[:, 1] = _x[:, 0] + torch.rand(_x.shape[0]).cuda() * 0.2, _x[:, 1] + torch.rand(_x.shape[0]).cuda() * 0.2
            _x[:, 0] = torch.max(
                _x[:, 0] + 0.4 * torch.rand(_x.shape[0]).cuda() - 0.2, torch.zeros(_x.shape[0]).cuda()
            )
            _x[:, 1] = torch.max(
                _x[:, 1] + 0.4 * torch.rand(_x.shape[0]).cuda() - 0.2, torch.zeros(_x.shape[0]).cuda()
            )
            data.x = _x
        if prototype_embedding == None:
            pred_output = model(data)
            pred_label = pred_output.argmax(dim=1)
        else:
            query_embedding = model.encoding(data)
            dists = euclidean_dist(query_embedding, prototype_embedding)
            dist_output = F.log_softmax(-dists, dim=1)
            pred_label = dist_output.argmax(dim=1)
        correct += int((pred_label == data.y).sum())
    correct = correct / len(test_dataloader_shift.dataset)
    return correct



class CIFARCombineFeature(BaseTransform):
    def __call__(self, data: Data) -> Data:
        x_feature, pos = data.x, data.pos
        _x = torch.cat((x_feature, pos), 1)
        data.x = _x
        return data

class CIFARTestTransform(BaseTransform):
    def __call__(self, data: Data) -> Data:
        theta = math.pi / 18
        test_pos = data.pos
        pos_x, pos_y = test_pos[:, 0], test_pos[:, 1]
        test_pos[:, 0] = pos_x * math.cos(theta) + pos_y * math.sin(theta)
        test_pos[:, 1] = pos_y * math.cos(theta) - pos_x * math.sin(theta)
        data.pos = test_pos
        _x = torch.cat((data.x, data.pos), 1)
        data.x = _x
        return data

class MNISTFeatureTransform(BaseTransform):
    def __call__(self, data: Data) -> Data:
        x_feature, pos = data.x, data.pos
        _x = torch.cat((x_feature.repeat(1,3), pos), dim=1)
        data.x = _x
        return data

class MNISTTestTransform(BaseTransform):
    def __call__(self, data: Data) -> Data:
        x_feature, pos = data.x[:, 0:1], data.x[:, 1:]
        _x = torch.cat((x_feature.repeat(1,3), pos), dim=1)
        data.x = _x
        return data

class yAddDimension(BaseTransform):
    def __call__(self, data: Data) -> Data:
        data.y = data.y.unsqueeze(0)
        return data

class rotate_transform(BaseTransform):
    def __call__(self, data: Data) -> Data:
        theta = math.pi / 18
        test_pos = data.pos
        pos_x, pos_y = test_pos[:, 0], test_pos[:, 1]
        test_pos[:, 0] = pos_x * math.cos(theta) + pos_y * math.sin(theta)
        test_pos[:, 1] = pos_y * math.cos(theta) - pos_x * math.sin(theta)
        data.pos = test_pos
        # data.y = data.y.unsqueeze(0)
        _x = torch.cat((data.x.repeat(1,3), data.pos), dim=1)
        data.x = _x
        return data

class color_transform(BaseTransform):
    def __call__(self, data: Data) -> Data:
        _x = torch.cat((data.x.repeat(1,3), data.pos), dim=1)
        _x[:, 0], _x[:, 1] = _x[:, 0] + torch.rand(_x.shape[0]) * 0.2, _x[:, 1] + torch.rand(_x.shape[0]) * 0.2
        data.x = _x
        return data

class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data

class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        # batch_loss = -alpha*log_p
        #print('-----bacth_loss------')
        #print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss