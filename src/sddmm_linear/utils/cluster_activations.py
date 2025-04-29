
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



def cluster_activations(activations, num_clusters):
    cluster_results = []
    for layer in activations:
        layer_results = {}
        for op, single_activations in layer.items():
            layer_results[op] = cluster_single_activations(single_activations, num_clusters)
        cluster_results.append(layer_results)
    return cluster_results


def cluster_single_activations(single_activations, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
    kmeans.fit(single_activations)

    return {
        "activations": single_activations,
        "cluster_centers": kmeans.cluster_centers_,
        "labels": kmeans.labels_,
        "n_clusters": num_clusters,
    }

def visualize_cluster_results(cluster_results, layer_idx, op):
    data = cluster_results[layer_idx][op]["activations"]
    labels = cluster_results[layer_idx][op]["labels"]
    cluster_centers = cluster_results[layer_idx][op]["cluster_centers"]
    n_clusters = cluster_results[layer_idx][op]["n_clusters"]

    n_samples, n_features = data.shape

    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    centers_2d = pca.transform(cluster_centers)

    fig, ax = plt.subplots(figsize=(8, 6))
    unique_labels = np.unique(labels)

    # Use a color map suitable for categorical data
    if n_clusters <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    elif n_clusters <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
    else: # Fallback to continuous map for many clusters
        colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))

    # Plot data points cluster by cluster
    for i, label in enumerate(unique_labels):
        cluster_points = data_2d[labels == label]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i],
                   label=f'Cluster {label}', alpha=0.7, s=50) # s controls point size

    # Plot cluster centers
    ax.scatter(centers_2d[:, 0], centers_2d[:, 1], marker='X', s=250, # Larger size for centers
               c='red', edgecolors='black', linewidths=1, # Make centers stand out
               label='Cluster Centers (2D proj.)', zorder=5) # zorder puts centers on top

    ax.set_xlabel(f'PCA Component 1')
    ax.set_ylabel(f'PCA Component 2')
    title = (f'KMeans Clustering (k={n_clusters}) on {n_samples} samples ({n_features} features)\n'
             f"Layer {layer_idx}, Operation: {op}")
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout() # Adjust layout to prevent labels overlapping
    plt.show()
