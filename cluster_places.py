import os
import argparse
import random
import numpy as np

import torch
import datasets

from cluster import *

places_root = "/data/vision/torralba/ade20k-places/data"
places_split00_ann_file = "/data/vision/torralba/ade20k-places/data/annotation/places_challenge/train_files/iteration0/split00/pred.json"
places_car_ann_file = "/data/vision/torralba/ade20k-places/data/annotation/places_challenge/train_files/iteration0/predictions/categories/car.json"

def cluster_places():
    ckpt = "output/all/model_best.pth.tar"
    checkpoint = torch.load(ckpt, map_location='cpu')
    lemniscate = checkpoint['lemniscate']
    train_dataset = datasets.coco.COCODataset(places_root, places_split00_ann_file)

    # Load X, y, idxs
    X = lemniscate.memory.numpy()
    y = np.array(train_dataset.get_targets())
    idxs = np.arange(len(train_dataset))
    X, y, idxs = balance_categories(X, y, idxs, max_freq=1000)
    X, y, idxs = filter_categories(X, y, idxs, [i for i in range(11)])

    # Vis nearest neighbor
    nearest_neighbors(X, idxs, train_dataset, name="places")

    # Cluster
    print(X.shape, y.shape)
    label = np.array([train_dataset.get_label(i) for i in y])
    pca_clustering(X, label, name="places")
    tsne_clustering(X, label, name="places")
    return X, label

def cluster_places_car():
    ckpt = "output/car_new/checkpoint.pth.tar"
    checkpoint = torch.load(ckpt, map_location='cpu')
    lemniscate = checkpoint['lemniscate']
    train_dataset = datasets.coco.COCODataset(places_root, places_car_ann_file)

    # Load X, y, idxs
    X = lemniscate.memory.numpy()
    y = np.array(train_dataset.get_targets())
    idxs = np.arange(len(train_dataset))
    X, y, idxs = balance_categories(X, y, idxs, max_freq=10000)

    # Vis nearest neighbor
    nearest_neighbors(X, idxs, train_dataset, name="places_car")

    # Cluster
    print(X.shape, y.shape)
    label = np.array([train_dataset.get_label(i) for i in y])
    pca_clustering(X, label, name="places_car")
    tsne_clustering(X, label, name="places_car")
    return X, label


# ade20k
ade_root = "./data/ade20k/images"
ade_train_ann_file = "./data/ade20k/annotations/predictions_train.json"
ade_val_ann_file = "./data/ade20k/annotations/predictions_val.json"
def cluster_ade_car():
    ckpt = "output/car_new/checkpoint.pth.tar"
    checkpoint = torch.load(ckpt, map_location='cpu')
    lemniscate = checkpoint['lemniscate_ade']
    train_dataset = datasets.coco_iou.COCOIOUDataset(ade_root, ade_train_ann_file, cat_name="car")

    # Load X, y, idxs
    X = lemniscate.memory.numpy()
    y = np.array(train_dataset.get_targets())
    idxs = np.arange(len(train_dataset))
    X, y, idxs = balance_categories(X, y, idxs, max_freq=10000)

    # Cluster
    print(X.shape, y.shape)
    label = np.array([train_dataset.get_label(i) for i in y])
    pca_clustering(X, label, name="ade_car")
    tsne_clustering(X, label, name="ade_car")
    return X, label

def cluster_ade_car_and_places_car():
    X, label = cluster_places_car()
    X_ade, label_ade = cluster_ade_car()

    X = np.concantenate([X, X_ade], axis=0)
    label = np.concantenate([label, label_ade], axis=0)

    # Cluster
    print(X.shape, y.shape)
    label = np.array([train_dataset.get_label(i) for i in y])
    pca_clustering(X, label, name="ade_car_and_places_car")
    tsne_clustering(X, label, name="ade_car_and_places_car")
    return X, label


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--dataset', default='places', type=str,
                    help='dataset to cluster')
    args = parser.parse_args()
    print(args)

    if args.dataset == "places":
        cluster_places()
    elif args.dataset == "places_car":
        cluster_places_car()
    elif args.dataset == "ade_car":
        cluster_places_car()
    elif args.dataset == "ade_car_and_places_car":
        cluster_ade_car_and_places_car()
