import os
import argparse
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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
    trainset = datasets.CIFAR10Instance(root='./data/cifar', train=True, download=True, transform=None)

    X = lemniscate.memory.numpy()
    y = np.array(trainset.targets)
    X, y = balance_data(X, y)

    print(X.shape, y.shape)
    pca_clustering(X, y, name="cifar")
    tsne_clustering(X, y, name="cifar")

def cluster_imagenet():
    imagenet_checkpoint = "checkpoint/lemniscate_resnet18.pth.tar"
    checkpoint = torch.load(imagenet_checkpoint, map_location='cpu')
    lemniscate = checkpoint['lemniscate']
    train_dataset = datasets.ImageFolderInstance("./data/imagenet/train")

    X = lemniscate.memory.numpy()
    y = np.array(train_dataset.targets)
    X, y = balance_data(X, y)

    print(X.shape, y.shape)
    pca_clustering(X, y, name="imagenet")
    tsne_clustering(X, y, name="imagenet")

def balance_data(X, y, max_freq=1000):
    counts = {}
    X_bal = []
    y_bal = []
    for x, l in zip(X, y):
        if l not in counts:
            counts[l] = 0
        if counts[l] < max_freq:
            counts[l] += 1
            if l <= 10:
                X_bal.append(x)
                y_bal.append(l)
    X = np.array(X_bal)
    y = np.array(y_bal)
    return X, y

def tsne_clustering(X, y, name="name"):
    print("TSNE clustering...")
    df = pd.DataFrame(X)
    df['y'] = y

    N = 10000
    np.random.seed(42)
    rndperm = np.random.permutation(df.shape[0])
    df = df.loc[rndperm[:N],:].copy()

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(df.to_numpy())
    df['tsne-one'] = tsne_results[:,0]
    df['tsne-two'] = tsne_results[:,1]

    plot_scatter(df, x="tsne-one", y="tsne-two", fn="{}_tsne_scatter.png".format(name))

def pca_clustering(X, y, name="name"):
    print("PCA clustering...")
    df = pd.DataFrame(X)
    df['y'] = y

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(X)
    df['pca-one'] = pca_result[:,0]
    df['pca-two'] = pca_result[:,1] 
    df['pca-three'] = pca_result[:,2]

    plot_scatter(df, x="pca-one", y="pca-two", fn="{}_pca_scatter.png".format(name))

def umap_clustering(X, y):
    #from umap import UMAP
    print("UMAP clustering...")
    df = pd.DataFrame(X)
    df['y'] = y

    N = 100
    np.random.seed(42)
    rndperm = np.random.permutation(df.shape[0])
    df = df.loc[rndperm[:N],:].copy()

    umap = UMAP(random_state=42)
    umap_result = umap.fit_transform(X)
    df['umap-one'] = umap_result[:,0]
    df['umap-two'] = umap_result[:,1]

    plot_scatter(df, x="umap-one", y="umap-two")

def plot_scatter(df, x, y, fn="scatter.png"):
    print("Plotting scatter...")
    C = len(np.unique(df['y']))

    plt.figure()
    scatter_plot = sns.scatterplot(
        x=x, y=y,
        hue="y",
        palette=sns.color_palette(n_colors=C),
        data=df,
        legend="full",
        alpha=0.3
    )

    plots_dir = "./plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    plt.savefig(os.path.join(plots_dir, fn))


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

