import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn import manifold
try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None

if sns is not None:
    sns.set(rc={'figure.figsize': (11.7, 8.27)})
    palette = sns.color_palette("bright", 2)
else:
    plt.rcParams['figure.figsize'] = (11.7, 8.27)
    palette = None


def plot_embedding(X_org, y, title=None, new=True):
    X_org = np.asarray(X_org)
    Y = np.asarray(y)

    if X_org.shape[0] < 2:
        print(f"Skipping TSNE for {title}: need at least 2 samples, got {X_org.shape[0]}")
        return

    cache_path = str(title) + '-tsne-features.json'
    if not new and os.path.exists(cache_path):
        with open(cache_path, 'r') as file:
            _x, _y = json.load(file)
        X = np.array(_x)
        Y = np.array(_y)
    else:
        perplexity = min(30, X_org.shape[0] - 1)
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity=perplexity)
        print('Fitting TSNE!')
        X = tsne.fit_transform(X_org)
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        denom = np.where((x_max - x_min) == 0, 1, (x_max - x_min))
        X = (X - x_min) / denom

        with open(cache_path, 'w') as file_:
            json.dump([X.tolist(), Y.tolist()], file_)

    if sns is not None:
        sns.set(style='white')
    plt.figure(figsize=(10, 10), edgecolor='black')
    plt.scatter(X[Y == 0][:, 0], X[Y == 0][:, 1],
                marker='.', c="tab:blue", s=12, linewidth=3.5, label='Non-Vuln')
    plt.scatter(X[Y == 1][:, 0], X[Y == 1][:, 1],
                marker='^', c="tab:orange", s=12, linewidth=3.5, label='Vuln')
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title("")
    plt.tight_layout()
    plt.savefig(str(title) + '.jpeg', dpi=1000)
    plt.close()


if __name__ == '__main__':
    x_a = np.random.uniform(0, 1, size=(32, 256))
    targets = np.random.randint(0, 2, size=(32))
    print(targets)
    plot_embedding(x_a, targets, title='tsne_demo')
    print("Computing t-SNE embedding")
