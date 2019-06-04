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

# ade20k
ade_root = "./data/ade20k/images"
ade_train_ann_file = "./data/ade20k/annotations/predictions_train.json"
ade_val_ann_file = "./data/ade20k/annotations/predictions_val.json"

def cluster_places():
    places_ann_file = places_split00_ann_file
    train_dataset = datasets.coco.COCODataset(places_root, places_ann_file)

    ckpt = "output/all/model_best.pth.tar"
    checkpoint = torch.load(ckpt, map_location='cpu')
    lemniscate = checkpoint['lemniscate']

    X = lemniscate.memory.numpy()
    y = np.array(train_dataset.targets)
    idxs = np.arange(len(train_dataset))
    X, y, idxs = balance_categories(X, y, idxs, max_freq=10000)
    X, y, idxs = filter_categories(X, y, idxs, [4])

    print(X.shape, y.shape)
    label = np.array([train_dataset.get_label(i) for i in y])
    score = np.array([int(train_dataset.get_ann_info(i)["score"]*10) for i in idxs])
    pca_clustering(X, score, name="places")
    tsne_clustering(X, score, name="places")

    nearest_neighbors(X, idxs, train_dataset, name="places")

def cluster_places_car():
    places_ann_file = places_split00_ann_file
    train_dataset = datasets.coco.COCODataset(
        places_root, places_ann_file)

    train_dataset_ade = datasets.coco_iou.COCOIOUDataset(
        ade_root, ade_train_ann_file, cat_name="car")

    ckpt = "output/car_new/model_best.pth.tar"
    checkpoint = torch.load(ckpt, map_location='cpu')
    lemniscate = checkpoint['lemniscate']
    lemniscate_ade = checkpoint['lemniscate_ade']

    X = lemniscate.memory.numpy()
    y = np.array(train_dataset.targets)
    X, y = balance_categories(X, y)
    X, y = filter_categories(X, y, [9])
    label = np.array([train_dataset.get_label(i) for i in y])

    X_ade = lemniscate_ade.memory.numpy()
    y_ade = np.array(train_dataset_ade.targets)
    label_ade = np.array([train_dataset_ade.get_label(i) for i in y_ade])

    X = np.concantenate([X, X_ade], axis=0)
    label = np.concantenate([label, label_ade], axis=0)

    print(X.shape, label.shape)
    pca_clustering(X, label, name="places_car")
    tsne_clustering(X, label, name="places_car")
    # nearest_neighbors(train_dataset, X, name="places_car")



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
