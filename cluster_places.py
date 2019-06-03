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
    places_ann_file = places_split00_ann_file
    train_dataset = datasets.coco.COCODataset(
        places_ann_file, places_root, cat_name=None)

    ckpt = "output/all_objects/model_best.pth.tar"
    checkpoint = torch.load(ckpt, map_location='cpu')
    lemniscate = checkpoint['lemniscate']

    X = lemniscate.memory.numpy()
    y = np.array(train_dataset.targets)
    X, y = balance_data(X, y)
    print(X.shape, y.shape)
    label = np.array([train_dataset.get_cat_info(i)["name"] for i in y])

    pca_clustering(X, label, name="places")
    tsne_clustering(X, label, name="places")

def cluster_places_car():
    train_dataset = datasets.coco.COCODataset(
        places_ann_file, places_root, cat_name="car")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--dataset', default='places', type=str,
                    help='dataset to cluster')
    args = parser.parse_args()
    print(args)

    cluster_places()
