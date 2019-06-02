import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as pltp
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP


def cluster_mnist():
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    print(X.shape, y.shape)

    # pca_clustering(X, y)
    # tsne_clustering(X, y)
    umap_clustering(X, y)

def cluster_lemniscate():
    checkpoint = torch.load(args.resume, map_location='cpu')
    lemniscate = checkpoint['lemniscate']
    
def umap_clustering(X, y):
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

def tsne_clustering(X, y):
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

    plot_scatter(df, x="tsne-one", y="tsne-two")

def pca_clustering(X, y):
    print("PCA clustering...")
    df = pd.DataFrame(X)
    df['y'] = y

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(X)
    df['pca-one'] = pca_result[:,0]
    df['pca-two'] = pca_result[:,1] 
    df['pca-three'] = pca_result[:,2]

    plot_scatter(df, x="pca-one", y="pca-two")

def plot_scatter(df, x, y, fn="scatter.png"):
    print("Plotting scatter...")
    C = len(np.unique(df['y']))
    scatter_plot = sns.scatterplot(
        x=x, y=y,
        hue="y",
        palette=sns.color_palette("hls", C),
        data=df,
        legend="full",
        alpha=0.3
    )

    fig = scatter_plot.get_figure()
    fig.savefig(fn)


if __name__ == '__main__':
    cluster_mnist()

