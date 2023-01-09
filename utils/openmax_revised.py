# import keras
# from keras.datasets import mnist
# from keras import backend as K
from torch_geometric.data import Batch

from utils.evt_fitting import weibull_tailfitting, query_weibull
from utils.compute_openmax import recalibrate_scores
from utils.openmax_utils import compute_vector_distance

import scipy.spatial.distance as spd

import numpy as np
import matplotlib.pyplot as plt

# from PIL import Image
import torch
# from nepali_characters import *
from util_func import euclidean_dist

# train_x,train_y,test_x,text_y,valid_x,valid_y = split(0.9,0.05,0.05)

label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def seperate_data(x, y):
    ind = y.argsort()
    # print(f"[BB Debug] ind shape is {ind.shape}")
    sort_x = x[ind[::-1]]
    sort_y = y[ind[::-1]]
    # print(f"[BB Debug] sort_y shape is {sort_y.shape}")
    dataset_x = []
    dataset_y = []
    mark = 0

    for a in range(len(sort_y)-1):
        if sort_y[a] != sort_y[a+1]:
            # print(f"[BB Deubg] a = {a}: sort_y[a]={sort_y[a]}, sort_y[a+1]={sort_y[a+1]}")
            dataset_x.append(np.array(sort_x[mark:a]))
            dataset_y.append(np.array(sort_y[mark:a]))
            mark = a + 1    # here mark should be updated to the next index.
        if a == len(sort_y)-2:
            dataset_x.append(np.array(sort_x[mark:len(sort_y)]))
            dataset_y.append(np.array(sort_y[mark:len(sort_y)]))
    return dataset_x, dataset_y


def compute_feature(x, model):
    score = get_activations(model, 8, x)
    fc8 = get_activations(model, 7, x)
    return score, fc8


def compute_mean_vector(feature):
    # return np.mean(feature, axis=0)
    return torch.mean(feature, axis=0).unsqueeze(0)


def compute_distances(mean_feature, feature):
    eucos_dist, eu_dist, cos_dist = [], [], []
    eu_dist, cos_dist, eucos_dist = [], [], []
    for feat in feature:
        # eu_dist += [spd.euclidean(mean_feature, feat)]
        # cos_dist += [spd.cosine(mean_feature, feat)]
        # eucos_dist += [spd.euclidean(mean_feature, feat)/20. + spd.cosine(mean_feature, feat)]
        feat = feat[np.newaxis, :]
        # print(f"mean_feature: {mean_feature.shape}, feat: {feat.shape}")
        eucos_dist += [euclidean_dist(mean_feature, feat)]
        # eucos_dist += [spd.euclidean(mean_feature, feat)/20.]
    # distances = {'eucos': eucos_dist, 'cosine': cos_dist, 'euclidean': eu_dist}
    distances = {'eucos': eucos_dist}
    return distances

def get_activations(model, layer, X_batch):
    # print (model.layers[6].output)
    get_activations = K.function(
        [model.layers[0].input, K.learning_phase()],
        [model.layers[layer].output])
    activations = get_activations([X_batch, 0])[0]
    # print (activations.shape)
    return activations


def get_correct_classified(pred, y):
    pred = (pred > 0.5) * 1
    res = np.all(pred == y, axis=1)
    return res


def create_weibull_model_prototype(model, dataloader, idx_by_class, args, device='cuda'):
    model.eval()
    for idx, data in enumerate(dataloader):
        data = data.to(device)
        y = data.y
        embedding = model(data)
        if idx == 0:
            y_all = y
            embedding_all = embedding
        else:
            y_all = torch.cat((y_all, y), 0)
            embedding_all = torch.cat((embedding_all, embedding), 0)
    
    num_classes = idx_by_class.keys()
    
    weibull_model = {}
    feature_mean = []
    feature_distance = []
    follow_batch_list = []

    for j in num_classes:
        weibull_model[j] = {}
        idx_i = idx_by_class[j]
        embedding_i = embedding_all[idx_i]
        mean = compute_mean_vector(embedding_i)
        mean, embedding_i = mean.detach().cpu().numpy(), embedding_i.detach().cpu().numpy()
        # print(f"BB Debug @113 mean {mean.shape}, embedding_i {embedding_i.shape}")
        distance = compute_distances(mean, embedding_i)
        feature_mean.append(mean)
        feature_distance.append(distance)
    return feature_mean, feature_distance


def create_weibull_model(model, dataloader, dataset, args, device='cuda', tips='null'):
    model.eval()
    for idx, data in enumerate(dataloader):
        data = data.to(device)
        y = data.y
        pred = model(data)
        if idx == 0:
            y_all = y
            pred_all = pred
        else:
            y_all = torch.cat((y_all, y), 0)
            pred_all = torch.cat((pred_all, pred), 0)
    
    # print(f"BB Debug: y_all shape is {y_all.shape}.")
    pred_label = pred_all.argmax(dim=1) 
    # print(f"BB Debug: pred_label shape is {pred_label.shape}.")    
    pred_label, y_all = pred_label.detach().cpu().numpy(), y_all.detach().cpu().numpy()

    num_classes = dataloader.dataset.num_classes
    index = {}
    for m in range(num_classes):
        index[int(m)] = []
    true_sample_num = 0
    # print("BB Debug: y distinct is", np.unique(y_all))
    for n in range(y_all.shape[0]):
        # print(f"n={n}, pred_label: {pred_label[n]}, y label: {y_all[n][0]}")
        if args.dataset == 'molhiv':
            if pred_label[n] == y_all[n][0]:
                true_sample_num += 1
                index[int(y_all[n])].append(n)
        else:
            if pred_label[n] == y_all[n]:
                true_sample_num += 1
                index[int(y_all[n])].append(n)
    # print(f"After pretrain, validation acc is {true_sample_num / y_all.shape[0] * 100:.4f}")

    weibull_model = {}
    feature_mean = []
    feature_distance = []
    follow_batch_list = []
    # print("BB Debug: dataset", dataset)
    for j in range(num_classes):
        weibull_model[j] = {}
        idx_i = index[int(j)]
        print(f"j = {j}, idx_i: {len(idx_i)} samples")
        if idx_i == []:
            return None, None
        data_i = Batch.from_data_list(dataset[idx_i]).cuda()
        embedding = model.encoding(data_i)
        # print("embedding shape is", embedding.shape)
        mean = compute_mean_vector(embedding)
        # mean, embedding = mean.detach().cpu().numpy(), embedding.detach().cpu().numpy()
        distance = compute_distances(mean, embedding)
        feature_mean.append(mean)
        feature_distance.append(distance)
    # print(f"np saving mean_{dataloader.dataset.name} & distance_{dataloader.dataset.name}")
    return feature_mean, feature_distance
    # if tips == 'null':
    #     np.save(f'data/mean_{args.dataset}.npy', feature_mean)
    #     np.save(f'data/distance_{args.dataset}.npy', feature_distance)
    # else:
    #     np.save(f'data/mean_{args.dataset}_{tips}.npy', feature_mean)
    #     np.save(f'data/distance_{args.dataset}_{tips}.npy', feature_distance)


def create_model(model, data):
    # output = model.layers[-1]

    # Combining the train and test set
    # print (x_train.shape,x_test.shape)
    # exit()
    # x_train, x_test, y_train, y_test = get_train_test()
    x_train, x_test, y_train, y_test = data
    x_all = np.concatenate((x_train, x_test), axis=0)
    y_all = np.concatenate((y_train, y_test), axis=0)
    pred = model.predict(x_all)
    # print(f"[BB Debug] pred shape is {pred.shape}.")
    # print(f"[BB Debug] pred[0] is {pred[0]}")
    index = get_correct_classified(pred, y_all)

    # print(f"[BB Debug] index shape is {index.shape}.") 
    # print(f"[BB Debug] index[0] is {index[0]}")
    
    x1_test = x_all[index]
    y1_test = y_all[index]

    # print(f"[BB Debug] y1_test shape is {y1_test.shape}.")

    y1_test1 = y1_test.argmax(1)

    sep_x, sep_y = seperate_data(x1_test, y1_test1)
    
    feature = {}
    feature["score"] = []
    feature["fc8"] = []
    weibull_model = {}
    feature_mean = []
    feature_distance = []

    for i in range(len(sep_y)):
        print(i, sep_x[i].shape)
        weibull_model[label[i]] = {}
        score, fc8 = compute_feature(sep_x[i], model)
        # print(f"score: {score.shape}, fc8: {fc8.shape}.")
        mean = compute_mean_vector(fc8)
        distance = compute_distances(mean, fc8)
        feature_mean.append(mean)
        feature_distance.append(distance)
    np.save('data/mean', feature_mean)
    np.save('data/distance', feature_distance)


def build_weibull(mean, distance, tail):
    weibull_model = {}
    for i in range(len(mean)):
        weibull_model[label[i]] = {}
        weibull = weibull_tailfitting(mean[i], distance[i], tailsize=tail)
        weibull_model[label[i]] = weibull
    return weibull_model

def compute_weibull_score(weibull_model, mixup_embedding, mixup_y, class_num, distance_type='eucos'):
    weibull_score = []
    for i in range(mixup_y.shape[0]):
        category_weibull = query_weibull(
            int(mixup_y[i]), weibull_model, distance_type=distance_type
        )
        # query_channel = mixup_embedding[i].detach().cpu().numpy()
        query_channel = mixup_embedding[i]
        mean_vec = category_weibull[0]
        print(f"mean_vec: {mean_vec.shape}, query_channel: {query_channel.shape}")
        if distance_type == 'eucos':
            channel_distance = euclidean_dist(mean_vec, query_channel) 
            # print(f"euclidean: {spd.euclidean(mean_vec, query_channel):.4f}, cos: {spd.cosine(mean_vec, query_channel):.4f}")
            # channel_distance = spd.euclidean(
            # mean_vec, query_channel)/20 + spd.cosine(mean_vec, query_channel)
            # channel_distance = spd.euclidean(mean_vec, query_channel)/20
        elif distance_type == 'euclidean':
            channel_distance = spd.euclidean(mean_vec, query_channel)/20
        elif distance_type == 'cosine':
            channel_distance = spd.cosine(mean_vec, query_channel)
        else:
            print(
                "distance type not known: enter either of eucos," +
                " euclidean or cosine")
        wscore = category_weibull[2][0].w_score(channel_distance)
        weibull_score.append(wscore)
    # print("weibull_score: ", weibull_score)
    return torch.tensor(weibull_score).cuda()


def compute_openmax(model, imagearr):
    mean = np.load('data/mean.npy', allow_pickle=True)
    distance = np.load('data/distance.npy', allow_pickle=True)
    # print(f"mean: {mean.shape}. distance: {distance.shape}.")
    # Use loop to find the good parameters
    # alpharank_list = [1,2,3,4,5,5,6,7,8,9,10]
    # tail_list = list(range(0,21))

    alpharank_list = [10]
    # tail_list = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    tail_list = [5]
    # total = 0
    for alpha in alpharank_list:
        weibull_model = {}
        openmax = None
        softmax = None
        for tail in tail_list:
            # print ('Alpha ',alpha,' Tail ',tail)
            # print ('++++++++++++++++++++++++++++')
            weibull_model = build_weibull(mean, distance, tail)
            openmax, softmax = recalibrate_scores(
                weibull_model, label, imagearr, alpharank=alpha)

            # print ('Openmax: ',np.argmax(openmax))
            # print ('Softmax: ',np.argmax(softmax))
            # print ('opemax lenght',openmax.shape)
            # print ('openmax',np.argmax(openmax))
            # print ('openmax',openmax)
            # print ('softmax',softmax.shape)
            # print ('softmax',np.argmax(softmax))
            # if np.argmax(openmax) == np.argmax(softmax):
            # if np.argmax(openmax) == 0 and np.argmax(softmax) == 0:
            # print ('########## Parameters found ############')
            # print ('Alpha ',alpha,' Tail ',tail)
            # print ('########## Parameters found ############')
            #    total += 1
            # print ('----------------------------')
    return np.argmax(softmax), np.argmax(openmax)


def process_input(model, ind, data):
    x_train, x_test, y_train, y_test = data
    imagearr = {}
    plt.imshow(np.squeeze(x_train[ind]))
    plt.show()
    image = np.reshape(x_train[ind], (1, 28, 28, 1))
    score5, fc85 = compute_feature(image, model)
    imagearr['scores'] = score5
    imagearr['fc8'] = fc85
    # print (score5)
    return imagearr


def compute_activation(model, img):
    imagearr = {}
    # img = np.squeeze(img)
    img = np.array(
        Image.fromarray(
            (np.squeeze(img)).astype(np.uint8)).resize((28, 28)))
    # img = scipy.misc.imresize(np.squeeze(img),(28,28))
    # img = img[:,0:28*28]
    img = np.reshape(img, (1, 28, 28, 1))
    score5, fc85 = compute_feature(img, model)
    imagearr['scores'] = score5
    imagearr['fc8'] = fc85
    return imagearr


def image_show(img, labels):
    # print(img.shape)
    # img = scipy.misc.imresize(np.squeeze(img), (28, 28))
    # img = np.array(
    #     Image.fromarray(
    #         (np.squeeze(img)).astype(np.uint8)).resize((28, 28)))
    # print(img.shape)
    # img = img[:, 0:28*28]
    plt.imshow(np.squeeze(img), cmap='gray')
    # print ('Character Label: ',np.argmax(label))
    title = "Original: " + str(
        labels[0]) + " Softmax: " + str(
            labels[1]) + " Openmax: " + str(labels[0])
    plt.title(title, fontsize=8)
    plt.show()


# def openmax_unknown_class(model):
#     f = h5py.File('HWDB1.1subset.hdf5','r')
#     # total = 0
#     i = np.random.randint(0, len(f['tst/y']))
#     print('label', np.argmax(f['tst/y'][i]))
#     print(f['tst/x'][i].shape)
#     # exit()
#     imagearr = process_other_input(model, f['tst/x'][i])
#     compute_openmax(model, imagearr)
    #     if compute_openmax(model, imagearr)    >= 4:
    #        total += 1
    # print ('correctly classified',total,'total set',len(y2))


def openmax_known_class(model, y, data):
    x_train, x_test, y_train, y_test = data
    # total = 0
    for i in range(15):
        # print ('label', y[i])
        j = np.random.randint(0, len(y_train[i]))
        imagearr = process_input(model, j)
        print(compute_openmax(model, imagearr))
        #    total += 1
    # print ('correct classified',total,'total set',len(y))


"""
def main():
    #model = load_model("MNIST_CNN_tanh.h5")
    model = load_model("MNIST_CNN.h5")
    #create_model(model)
    #openmax_known_class(model,y_test)
    openmax_unknown_class(model)

if __name__ == '__main__':
    main()
"""