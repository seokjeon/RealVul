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


# 고차원 벡터를 2차원으로 축소하는 로직
def _fit_embedding(X_org):
    if X_org.shape[0] < 2:
        return None

    perplexity = min(30, X_org.shape[0] - 1) # 한 점이 주변 몇 개 이웃을 중요하게 볼지 결정하는 하이퍼파라미터. 일반적으로 5~50 사이의 값을 사용하며, 데이터 샘플 수보다 작은 값이어야 합니다.
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity=perplexity) # 순서대로 - n_components: 축소된 차원 수 (여기서는 2차원), init: 초기화 방법 (점의 시작위치를 PCA로 초기화), random_state: 난수 시드
    print('Fitting TSNE!')
    X = tsne.fit_transform(X_org) # 고차원 벡터 X_org을 2차원으로 축소하여 X에 저장: (160, 768) -> (160, 2)
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    denom = np.where((x_max - x_min) == 0, 1, (x_max - x_min)) # (x_max - x_min)이 0인 경우에는 나눗셈 오류를 방지하기 위해 1로 대체, 그렇지 않은 경우에는 원래의 (x_max - x_min) 값을 사용
    return (X - x_min) / denom # 정규화


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
        X = _fit_embedding(X_org)
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


def plot_paired_embedding(fine_X_org, fine_y, raw_X_org, raw_y, title=None, new=True):
    fine_X_org = np.asarray(fine_X_org) # fine-tuned 모델의 feature matrix
    fine_y = np.asarray(fine_y)         # fine-tuned 모델의 label(취약/비취약)
    raw_X_org = np.asarray(raw_X_org)   # raw 모델의 feature matrix
    raw_y = np.asarray(raw_y)           # raw 모델의 label(취약/비취약)


    combined_features = np.concatenate([fine_X_org, raw_X_org], axis=0)
    combined_labels = np.concatenate([fine_y, raw_y], axis=0)
    model_source = np.concatenate(
        [
            np.zeros(fine_X_org.shape[0], dtype=np.int64),
            np.ones(raw_X_org.shape[0], dtype=np.int64),
        ],
        axis=0,
    )

    cache_path = str(title) + '-tsne-features.json'

    # new=False로 설정하면 TSNE 계산을 건너뛰고, 기존에 저장된 캐시를 활용하여 jpeg 이미지만 새로 생성합니다. 
    if not new and os.path.exists(cache_path):
        with open(cache_path, 'r') as file:
            payload = json.load(file)
        X = np.array(payload['embedding'])
        combined_labels = np.array(payload['labels'])
        model_source = np.array(payload['model_source'])
    else:
        X = _fit_embedding(combined_features) # 고차원 벡터를 2차원으로 축소하는 로직
        with open(cache_path, 'w') as file_:
            json.dump(
                {
                    'embedding': X.tolist(),
                    'labels': combined_labels.tolist(),
                    'model_source': model_source.tolist(),
                },
                file_,
            )

    if sns is not None:
        sns.set(style='white')
    plt.figure(figsize=(10, 10), edgecolor='black')

    group_specs = [
        {
            'label': 'Before Fine-tuned / Vuln',
            'model_value': 1,
            'target_value': 1,
            'marker': 'o',
            'facecolor': 'tab:red',
            'edgecolor': 'tab:red',
            'linewidth': 0.8,
        },
        {
            'label': 'Before Fine-tuned / Non-Vuln',
            'model_value': 1,
            'target_value': 0,
            'marker': 'o',
            'facecolor': 'tab:blue',
            'edgecolor': 'tab:blue',
            'linewidth': 0.8,
        },
        {
            'label': 'After Fine-tuned / Vuln',
            'model_value': 0,
            'target_value': 1,
            'marker': 'o',
            'facecolor': 'none',
            'edgecolor': 'tab:red',
            'linewidth': 1.2,
        },
        {
            'label': 'After Fine-tuned / Non-Vuln',
            'model_value': 0,
            'target_value': 0,
            'marker': 'o',
            'facecolor': 'none',
            'edgecolor': 'tab:blue',
            'linewidth': 1.2,
        },
    ]
    for spec in group_specs:
        mask = (model_source == spec['model_value']) & (combined_labels == spec['target_value'])
        points = X[mask]
        if points.size == 0:
            continue
        plt.scatter(
            points[:, 0],
            points[:, 1],
            marker=spec['marker'],
            facecolors=spec['facecolor'],
            edgecolors=spec['edgecolor'],
            c=None,
            s=72,
            linewidths=spec['linewidth'],
            label=spec['label'],
        )

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title("")
    plt.legend(
        frameon=False,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.03),
        ncol=4,
        borderaxespad=0.0,
        fontsize=13,
        markerscale=1.8,
        handletextpad=0.6,
        columnspacing=1.4,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.9))
    plt.savefig(str(title) + '.jpeg', dpi=1000)
    plt.close()


if __name__ == '__main__':
    x_a = np.random.uniform(0, 1, size=(32, 256))
    targets = np.random.randint(0, 2, size=(32))
    print(targets)
    plot_embedding(x_a, targets, title='tsne_demo')
    print("Computing t-SNE embedding")
