import torch
import time
import datasets
from lib.utils import AverageMeter
import torchvision.transforms as transforms
import numpy as np

import os
import argparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    args = parser.parse_args()
    print(args)

    places_root = "/data/vision/torralba/ade20k-places/data"
    places_split00_ann_file = "/data/vision/torralba/ade20k-places/data/annotation/places_challenge/train_files/iteration0/split00/pred.json"
    places_car_ann_file = "/data/vision/torralba/ade20k-places/data/annotation/places_challenge/train_files/iteration0/predictions/categories/car.json"

    cat_name = None
    places_ann_file = places_split00_ann_file
    # cat_name = "car"
    # places_ann_file = places_car_ann_file

    train_dataset = datasets.coco.COCODataset(
        places_ann_file, places_root,
        cat_name=cat_name,
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5,1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=False,
        num_workers=16, pin_memory=True, sampler=None)

    checkpoint = torch.load(args.resume, map_location='cpu')
    lemniscate = checkpoint['lemniscate']

    trainFeatures = lemniscate.memory.numpy()
    if hasattr(train_loader.dataset, 'imgs'):
        trainLabels = [y for (p, y) in train_loader.dataset.imgs]
    else:
        trainLabels = train_loader.dataset.train_labels


    unique_elements, counts_elements = np.unique(trainLabels, return_counts=True)
    print(unique_elements, counts_elements)

    features = []
    labels = []
    for x,y in zip(trainFeatures, trainLabels):
        if y in [1,2]:
            features.append(x)
            labels.append(y)
            
    tsne_clustering(features, labels)

if __name__ == '__main__':
    main()

