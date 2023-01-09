import torch
import numpy as np
import torch.nn.functional as F
import torch_geometric
from torch_geometric.loader import DataLoader
from util_func import *
from models import *
from datasets import *
import math
import argparse
from datetime import datetime
from torch_geometric.utils import degree
import random
import torch_geometric.transforms as T
from tqdm import tqdm
import os
import time

from utils.openmax import create_weibull_model, compute_weibull_score, create_weibull_model_prototype
from utils.openmax import build_weibull


def weibull_loss(dist_output, mixup_y, weibull_score, bias=1, upper_bound=0.6):
    loss_value = F.nll_loss(dist_output, mixup_y, reduce=False)
    select_idx = torch.where(loss_value > 0.2)
    # print("weibull_score shape is", weibull_score.shape)
    # print("weibull_score:", weibull_score)
    weibull_score = torch.softmax(weibull_score[select_idx], dim=0)
    # print("After softmax:", weibull_score)
    # print("Before reweighting:", F.nll_loss(dist_output, mixup_y).item())
    loss = torch.sum(loss_value[select_idx] * weibull_score)
    # print("After reweighting:", loss.item())
    # print("================================================")
    return loss

# def weibull_loss_old(dist_output, mixup_y, weibull_score, bias=1, upper_bound=0.6):
#     loss_value = F.nll_loss(dist_output, mixup_y, reduce=False)
#     # proportion_print(weibull_score)
#     weibull_score[torch.where(weibull_score > upper_bound)] = 0
#     weibull_score = weibull_score + bias
#     # print("weibull_score shape:", weibull_score.shape)
#     # weibull_score = torch.softmax(weibull_score, dim=0)
#     # print(f"prototype loss: {torch.mean(loss_value)}, classifier loss: {torch.mean(loss_classifier)}.")
#     loss = torch.mean(loss_value * weibull_score)
#     # loss = torch.sum(loss_value * weibull_score)
#     return loss

def main():
    
    datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    print(datetime_now)

    if args.tips == 'default':
        args.tips = datetime_now

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        print("INFO: GPU")
    else:
        args.device = torch.device('cpu')
        print("INFO: CPU")

    # dataset
    train_dataset, val_dataset, test_dataset = get_dataset(name=args.dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    args.class_num, args.input_feature = train_dataset.num_classes, train_dataset.num_node_features
    print(f"INFO: {args.dataset} dataset with class_num:  {args.class_num}, input_feature: {args.input_feature}")
    print(f"train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")

    train_idx_by_class, val_idx_by_class, test_idx_by_class = get_idx_by_class(train_dataset), get_idx_by_class(val_dataset), get_idx_by_class(test_dataset)

    if args.dataset == 'motif':
        args.backbone = 'LEConv'
        args.layer = 3
        args.hidden_dim = 32
    elif args.dataset == 'proteins':
        args.backbone = 'GCN'
        args.hidden_dim = 32
        args.layer = 3
        args.pooling = 'max'
    elif args.dataset == 'reddit':
        args.backbone = 'GIN'
        args.layer = 2
        args.hidden_dim = 16
        args.dropout = 0.1
    elif args.dataset == 'collab':
        args.backbone = 'GCN'
        args.hidden_dim = 32
        args.layer = 3
        args.pooling = 'max' 
        args.distance_type = 'euclidean'
    elif args.dataset == 'enzy':
        args.backbone = 'GCN'
        args.layer = 2
        args.hidden_dim = 32
        args.pooling = 'mean'
        args.spt_num = 30
    elif args.dataset == 'imdb':
        args.backbone = 'GCN'
        args.layer = 3
        args.hidden_dim = 32
        args.pooling = 'mean'
    elif args.dataset == 'DD':
        args.backbone = 'GCN'
        args.layer = 3
        args.hidden_dim = 32
        args.pooling = 'mean'
    elif args.dataset == 'coil':
        args.backbone = 'GAT'
        args.layer = 3
        args.hidden_dim = 32
        args.pooling = 'max'
        args.spt_num = 10
    else:
        raise ValueError('Undefined dataset called {}'.format(args.dataset))
    

    if not os.path.exists('./model_pkl/{}'.format(args.dataset)):
        os.makedirs('./model_pkl/{}'.format(args.dataset))

    experiment_name = f'{args.dataset}_mixup_{args.pooling}pool_hidden{args.hidden_dim}_layer{args.layer}_{args.distance_type}_{args.upper_bound}_sample_{args.tips}_tail_{args.tail}'

    print(args)
    args.seed = [i for i in range(args.seed_num)]
    train_acc_box, val_acc_box, test_acc_box = [0] * args.seed_num, [0] * args.seed_num, [0] * args.seed_num

    test_acc_list = []

    for seed_i in tqdm(args.seed):

        set_random_seed(seed_i)
        model_pkl_name = "model_pkl/" + args.dataset + "/" + experiment_name + "_seed_" + str(seed_i) + ".pkl"

        train_losses, val_losses = [], []
        avg_train_losses, avg_val_losses = [], []

        model = CausalMixup(args).to(args.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = torch.nn.CrossEntropyLoss()
        early_stopping = EarlyStopping(patience=args.patience, verbose=True, model_name=model_pkl_name, save_model=args.save_model)

        # ==========================================
        # Training
        # ==========================================

        train_time_all = 0
        train_epoch = 0

        for epoch in tqdm(range(args.epoch)):
            
            mean, distance = create_weibull_model_prototype(model, train_dataloader, train_idx_by_class, args, args.device)
            if mean == None and distance == None:
                weibull_score_flag = 0
            else:
                weibull_score_flag = 1
                weibull_model = build_weibull(mean, distance, tail=args.tail, distance_type=args.distance_type)
            
            # weibull_score_flag = 0

            model.train()
            start_time = time.time()
            for train_idx, data in enumerate(train_dataloader):
                optimizer.zero_grad()
                data = data.to(args.device)
                y = data.y.to(args.device)
                spt_data = get_task_data(train_dataset, train_idx_by_class, spt_num=args.spt_num)
                prototype_embedding = model.get_sample_prototype(spt_data)

                mixup_embedding, mixup_y = model.get_mixup_embedding(data, beta=args.beta)

                dists = euclidean_dist(mixup_embedding, prototype_embedding)
                
                dist_output = F.log_softmax(-dists, dim=1)
                if weibull_score_flag == 1:
                    weibull_score = compute_weibull_score(weibull_model, mixup_embedding, mixup_y, args.class_num, args.distance_type, args=args)
                    loss = weibull_loss(dist_output, mixup_y, weibull_score, upper_bound=args.upper_bound, bias=args.bias)
                    # print("weibull loss:", loss)
                else:
                    # loss = F.nll_loss(dist_output, mixup_y)
                    focal_loss = FocalLoss(class_num=args.class_num)
                    loss = focal_loss(dist_output, mixup_y)
                    # print("vanilla loss:", loss)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            end_time = time.time()
            train_time_all += end_time - start_time

            model.eval()
            val_correct = 0
            prototype_embedding = model.get_prototype(train_dataloader)
            for val_idx, data in enumerate(val_dataloader):
                data = data.to(args.device)
                query_embedding, y = model.encoding(data)
                # =====================================================
                # mixup_embedding, mixup_y = mixup(query_embedding, data.y)
                # val_embedding = torch.cat((mixup_embedding, query_embedding), 0)
                # val_y = torch.cat((mixup_y, y), 0)
                # dists = euclidean_dist(val_embedding, prototype_embedding)
                # dist_output = F.log_softmax(-dists, dim=1)
                # pred_label = dist_output.argmax(dim=1)
                # val_correct += int((pred_label == val_y).sum())
                # loss = F.nll_loss(dist_output, val_y)
                # val_losses.append(loss.item())
                # =====================================================
                dists = euclidean_dist(query_embedding, prototype_embedding)
                dist_output = F.log_softmax(-dists, dim=1)
                pred_label = dist_output.argmax(dim=1)
                val_correct += int((pred_label == y).sum())
                loss = F.nll_loss(dist_output, y)
                val_losses.append(loss.item())
                # =====================================================
                

            val_correct = val_correct / len(val_dataloader.dataset)
            train_loss = np.average(train_losses)
            val_loss = np.average(val_losses)
            avg_train_losses.append(train_loss)
            avg_val_losses.append(val_loss)
            c_lr = optimizer.state_dict()['param_groups'][0]['lr']

            val_acc_box[seed_i] = val_correct
            
            train_losses, val_losses = [], []

            # scheduler.step(val_loss)
            train_epoch += 1
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                break
            
        print("avg training time is", train_time_all / train_epoch)
        # ==========================================
        # Testing
        # ==========================================
        model.load_state_dict(torch.load(model_pkl_name))
        model.eval()
        correct = 0

        prototype_embedding = model.get_prototype(train_dataloader)
        for batch_test, data in enumerate(test_dataloader):
            data = data.to(args.device)
            y = data.y.to(args.device)
            query_embedding, y = model.encoding(data)
            dists = euclidean_dist(query_embedding, prototype_embedding)
            dist_output = F.log_softmax(-dists, dim=1)
            pred_label = dist_output.argmax(dim=1)
            correct += int((pred_label == y).sum())
            # pred_output = model(data)
            # pred_label = pred_output.argmax(dim=1)
            # correct += int((pred_label == data.y).sum())
        correct = correct / len(test_dataloader.dataset)
        print(f"seed {seed_i}/{len(args.seed)} test acc: {correct:.4f}")
        test_acc_list.append(correct)
        test_acc_box[seed_i] = correct


    test_acc_arr = np.array(test_acc_list)
    test_acc_mean, test_acc_std = np.mean(test_acc_arr), np.std(test_acc_arr)
    print("-------------------------------------")
    # print(f"Test list: {test_acc_list}")
    print(args.tips)
    print(f"Test Acc: {test_acc_mean * 100:.2f}±{test_acc_std * 100:.2f}% ")
    print("-------------------------------------")



if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, default=200)
    argparser.add_argument('--pretrain_epoch', type=int, default=200)
    argparser.add_argument('--task_num', type=int, default=10)
    argparser.add_argument('--batch_size', type=int, default=128)
    argparser.add_argument('--backbone', type=str, default='GCN')
    argparser.add_argument('--pooling', type=str, default='mean')
    argparser.add_argument('--batch_norm', type=bool, default=True)
    argparser.add_argument('--residual', type=bool, default=True)
    argparser.add_argument('--hidden_dim', type=int, default=32)
    argparser.add_argument('--spt_num', type=int, default=50)
    argparser.add_argument('--bias', type=float, default=1)
    argparser.add_argument('--upper_bound', type=float, default=0.8)
    argparser.add_argument('--dropout', type=float, default=0.1)
    argparser.add_argument('--patience', type=int, default=10)
    argparser.add_argument('--tail', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=1e-3)
    argparser.add_argument('--layer', type=int, default=3)
    argparser.add_argument('--dataset', type=str, default='proteins')
    argparser.add_argument('--seed_num',  type=int, default=10)
    argparser.add_argument('--save_model', type=bool, default=True)
    argparser.add_argument('--distance_type', type=str, default='my_dis')
    argparser.add_argument('--euc', type=float, default=0.05)
    argparser.add_argument('--cos', type=float, default=1.0)
    argparser.add_argument('--beta', type=int, default=2)
    argparser.add_argument('--tips', type=str, default='default', help='message for different model settings')
    args = argparser.parse_args()

    main()