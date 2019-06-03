import os
import argparse
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

import torch
import datasets

def cluster_mnist():
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

    print(X.shape, y.shape)
    pca_clustering(X, y)
    tsne_clustering(X, y)

def cluster_cifar():
    cifar_checkpoint = "checkpoint/ckpt.t7"
    checkpoint = torch.load(cifar_checkpoint, map_location='cpu')
    lemniscate = checkpoint['lemniscate']
    train_dataset = datasets.CIFAR10Instance(root='./data/cifar', train=True, download=True, transform=None)

    X = lemniscate.memory.numpy()
    y = np.array(train_dataset.targets)
    X, y = balance_categories(X, y)
    label = np.array([train_dataset.classes[i] for i in y])

    print(X.shape, label.shape)
    pca_clustering(X, label, name="cifar")
    tsne_clustering(X, label, name="cifar")
    nearest_neighbors(train_dataset, X, n=5, name="cifar")

def cluster_imagenet():
    imagenet_checkpoint = "checkpoint/lemniscate_resnet18.pth.tar"
    checkpoint = torch.load(imagenet_checkpoint, map_location='cpu')
    lemniscate = checkpoint['lemniscate']
    train_dataset = datasets.ImageFolderInstance("./data/imagenet/train")

    X = lemniscate.memory.numpy()
    y = np.array(train_dataset.targets)
    X, y = balance_categories(X, y)
    X, y = filter_categories(X, y, range(10))
    label = np.array([train_dataset.classes[i] for i in y])

    print(X.shape, label.shape)
    pca_clustering(X, label, name="imagenet")
    tsne_clustering(X, label, name="imagenet")
    nearest_neighbors(train_dataset, X, n=5, name="imagenet")

def filter_categories(X, y, cats):
    X_out = []
    y_out = []
    for x, l in zip(X, y):
        if l in cats:
            X_out.append(x)
            y_out.append(l)
    X_out = np.array(X_out)
    y_out = np.array(y_out)
    print("Filtered categories {}: {}, {} -> {}, {}".format(cats, X.shape, y.shape, X_out.shape, y_out.shape))
    return X_out, y_out

def balance_categories(X, y, max_freq=1000):
    counts = {}
    X_out = []
    y_out = []
    for x, l in zip(X, y):
        if l not in counts:
            counts[l] = 0
        if counts[l] < max_freq:
            counts[l] += 1
            X_out.append(x)
            y_out.append(l)
    X_out = np.array(X_out)
    y_out = np.array(y_out)
    print("Balanced categories {}: {}, {} -> {}, {}".format(max_freq, X.shape, y.shape, X_out.shape, y_out.shape))
    return X_out, y_out

def nearest_neighbors(dataset, X, n=5, name="name"):
    print("Getting nearest neighbors...")
    start = time.time()

    nbrs = NearestNeighbors(n_neighbors=n+1, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    grid_imgs = []
    grid_targets = []
    for row in indices[:30]:
        imgs = []
        targets = []
        for idx in row:
            img, target, index = dataset[idx]
            imgs.append(img)
            targets.append(target)

        # Concatenate images
        imgs = [np.array(img, dtype="uint8") for img in imgs]
        imgs = np.concatenate(imgs, axis=1)
        grid_imgs.append(imgs)
        grid_targets.append(targets)

    grid_imgs = np.concatenate(grid_imgs, axis=0)
    grid_imgs = Image.fromarray(grid_imgs)
    grid_imgs.save('plots/{}_NN_grid.png'.format(name))
    grid_targets = np.array(grid_targets)
    print(grid_targets)
    print("Done. {:0.2f} secs".format(time.time() - start))

def tsne_clustering(X, label, name="name"):
    print("TSNE clustering...")
    start = time.time()

    df = pd.DataFrame(X)
    df['label'] = label

    N = 10000
    np.random.seed(42)
    rndperm = np.random.permutation(df.shape[0])
    df = df.loc[rndperm[:N],:].copy()

    label = df['label']
    df = df.drop('label', axis=1)
    X = df.values
    df['label'] = label

    assert X.shape[1] == 128
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(X)
    df['tsne-one'] = tsne_results[:,0]
    df['tsne-two'] = tsne_results[:,1]

    plot_scatter(df, x="tsne-one", y="tsne-two", fn="{}_tsne_scatter.png".format(name))
    print("Done. {:0.2f} secs".format(time.time() - start))

def pca_clustering(X, label, name="name"):
    print("PCA clustering...")
    start = time.time()

    df = pd.DataFrame(X)

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(X)
    df['pca-one'] = pca_result[:,0]
    df['pca-two'] = pca_result[:,1] 
    df['pca-three'] = pca_result[:,2]
    df['label'] = label

    plot_scatter(df, x="pca-one", y="pca-two", fn="{}_pca_scatter.png".format(name))
    print("Done. {:0.2f} secs".format(time.time() - start))

def plot_scatter(df, x, y, fn="scatter.png"):
    C = len(np.unique(df['label']))

    plt.figure()
    scatter_plot = sns.scatterplot(
        x=x, y=y,
        hue="label",
        palette=sns.color_palette(n_colors=C),
        data=df,
        legend="full",
        alpha=0.3
    )

    plots_dir = "./plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    plt.savefig(os.path.join(plots_dir, fn))
    print("Plotted scatter", os.path.join(plots_dir, fn))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--dataset', default='cifar', type=str,
                    help='dataset to cluster')
    args = parser.parse_args()
    print(args)

    if args.dataset == "mnist":
        cluster_mnist()
    elif args.dataset == "cifar":
        cluster_cifar()
    elif args.dataset == "imagenet":
        cluster_imagenet()

