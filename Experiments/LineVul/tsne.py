import json
from pathlib import Path

import matplotlib
import numpy as np
from sklearn import manifold

matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None

if sns is not None:
    sns.set(rc={'figure.figsize': (11.7, 8.27)})
else:
    plt.rcParams['figure.figsize'] = (11.7, 8.27)


RAW_MODEL_EVAL_DIRNAME = 'raw_model_eval'
DEFAULT_TSNE_FIGSIZE = (10, 10)
DEFAULT_TSNE_MARKER_SIZE = 3
DEFAULT_TSNE_MARKER_LINEWIDTH = 0.2
DEFAULT_TSNE_MARKER_ALPHA = 0.35
DEFAULT_TSNE_MARKER_ZORDER = 2
DEFAULT_TSNE_SHUFFLE_SEED = 0
BEFORE_NON_VULN_TSNE_COLOR = '#31a354'
AFTER_NON_VULN_TSNE_COLOR = '#3182bd'
BEFORE_VULN_TSNE_COLOR = '#cb181d'
AFTER_VULN_TSNE_COLOR = '#f16913'
SINGLE_TSNE_TITLE = 'LineVul t-SNE on SARD-Juliet SA TracePair'
COMBINED_TSNE_TITLE = 'LineVul t-SNE on SARD-Juliet SA TracePair (Combined)'
DEFAULT_LEGEND_FONT_SIZE = 12
DEFAULT_LEGEND_Y = 1.03


def _resolve_group_style(stage_label: str, target_value: int) -> dict:
    if stage_label == 'Before Fine-tuned' and target_value == 0:
        return {'color': BEFORE_NON_VULN_TSNE_COLOR}
    if stage_label == 'After Fine-tuned' and target_value == 0:
        return {'color': AFTER_NON_VULN_TSNE_COLOR}
    if stage_label == 'Before Fine-tuned' and target_value == 1:
        return {'color': BEFORE_VULN_TSNE_COLOR}
    return {'color': AFTER_VULN_TSNE_COLOR}


def _build_group_spec(
    stage_label: str,
    target_value: int,
    *,
    model_value=None,
    zorder=None,
) -> dict:
    style = _resolve_group_style(stage_label, target_value)
    spec = {
        'label': f"{stage_label} / {'Vuln' if target_value == 1 else 'Non-Vuln'}",
        'target_value': target_value,
        'marker': 'o',
        'facecolor': style['color'],
        'edgecolor': style['color'],
        'alpha': DEFAULT_TSNE_MARKER_ALPHA,
        'linewidth': DEFAULT_TSNE_MARKER_LINEWIDTH,
    }
    if model_value is not None:
        spec['model_value'] = model_value
    spec['zorder'] = DEFAULT_TSNE_MARKER_ZORDER if zorder is None else zorder
    return spec


def _resolve_single_group_specs(title=None):
    title_str = str(title) if title is not None else ''
    is_raw_model = RAW_MODEL_EVAL_DIRNAME in title_str
    stage_label = 'Before Fine-tuned' if is_raw_model else 'After Fine-tuned'
    return [
        _build_group_spec(stage_label, 0),
        _build_group_spec(stage_label, 1),
    ]


def _resolve_paired_group_specs():
    return [
        _build_group_spec(
            'Before Fine-tuned',
            1,
            model_value=1,
        ),
        _build_group_spec(
            'After Fine-tuned',
            1,
            model_value=0,
        ),
        _build_group_spec(
            'Before Fine-tuned',
            0,
            model_value=1,
        ),
        _build_group_spec(
            'After Fine-tuned',
            0,
            model_value=0,
        ),
    ]


def _scatter_groups(X, labels, group_specs, *, model_source=None):
    draw_queue = []
    for spec in group_specs:
        mask = labels == spec['target_value']
        if model_source is not None:
            mask &= model_source == spec['model_value']

        point_indices = np.flatnonzero(mask)
        if point_indices.size == 0:
            continue

        plt.scatter(
            [],
            [],
            marker=spec['marker'],
            facecolors=spec['facecolor'],
            edgecolors=spec['edgecolor'],
            c=None,
            s=DEFAULT_TSNE_MARKER_SIZE,
            alpha=spec['alpha'],
            linewidths=spec['linewidth'],
            label=spec['label'],
            zorder=spec.get('zorder', DEFAULT_TSNE_MARKER_ZORDER),
        )
        for point_index in point_indices:
            draw_queue.append((point_index, spec))

    if not draw_queue:
        return

    rng = np.random.default_rng(DEFAULT_TSNE_SHUFFLE_SEED)
    for queue_index in rng.permutation(len(draw_queue)):
        point_index, spec = draw_queue[queue_index]
        point = X[point_index]
        plt.scatter(
            point[0],
            point[1],
            marker=spec['marker'],
            facecolors=spec['facecolor'],
            edgecolors=spec['edgecolor'],
            c=None,
            s=DEFAULT_TSNE_MARKER_SIZE,
            alpha=spec['alpha'],
            linewidths=spec['linewidth'],
            label='_nolegend_',
            zorder=spec.get('zorder', DEFAULT_TSNE_MARKER_ZORDER),
        )


def _apply_tsne_layout(
    plot_title,
    *,
    legend_ncol,
    top_margin=0.9,
    legend_fontsize=DEFAULT_LEGEND_FONT_SIZE,
):
    axes = plt.gca()
    axes.set_box_aspect(1)
    plt.xticks([]), plt.yticks([])
    if plot_title is not None:
        plt.title(plot_title)
    plt.legend(
        frameon=False,
        loc='lower center',
        bbox_to_anchor=(0.5, DEFAULT_LEGEND_Y),
        ncol=legend_ncol,
        borderaxespad=0.0,
        fontsize=legend_fontsize,
        markerscale=5.0,
        handletextpad=0.6,
        columnspacing=1.4,
    )
    plt.tight_layout(rect=(0, 0, 1, top_margin))


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
        print(f'Skipping TSNE for {title}: need at least 2 samples, got {X_org.shape[0]}')
        return False

    cache_path = str(title) + '-tsne-features.json'
    if not new and Path(cache_path).exists():
        with open(cache_path, 'r', encoding='utf-8') as file:
            _x, _y = json.load(file)
        X = np.array(_x)
        Y = np.array(_y)
    else:
        X = _fit_embedding(X_org)
        if X is None:
            print(f'Skipping TSNE for {title}: need at least 2 samples, got {X_org.shape[0]}')
            return False
        with open(cache_path, 'w', encoding='utf-8') as file_:
            json.dump([X.tolist(), Y.tolist()], file_)

    if sns is not None:
        sns.set(style='white')
    plt.figure(figsize=DEFAULT_TSNE_FIGSIZE, edgecolor='black')
    _scatter_groups(X, Y, _resolve_single_group_specs(title))
    _apply_tsne_layout(
        SINGLE_TSNE_TITLE if title is not None else None,
        legend_ncol=2,
        top_margin=0.9,
    )
    plt.savefig(str(title) + '.jpeg', dpi=1000)
    plt.close()
    return True


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
    if not new and Path(cache_path).exists():
        with open(cache_path, 'r', encoding='utf-8') as file:
            payload = json.load(file)
        X = np.array(payload['embedding'])
        combined_labels = np.array(payload['labels'])
        model_source = np.array(payload['model_source'])
    else:
        X = _fit_embedding(combined_features)
        if X is None:
            print(
                f'Skipping paired TSNE for {title}: '
                f'need at least 2 samples, got {combined_features.shape[0]}'
            )
            return False
        with open(cache_path, 'w', encoding='utf-8') as file_:
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
    plt.figure(figsize=DEFAULT_TSNE_FIGSIZE, edgecolor='black')
    _scatter_groups(
        X,
        combined_labels,
        _resolve_paired_group_specs(),
        model_source=model_source,
    )
    _apply_tsne_layout(
        COMBINED_TSNE_TITLE if title is not None else None,
        legend_ncol=2,
        top_margin=0.86,
    )
    plt.savefig(str(title) + '.jpeg', dpi=1000)
    plt.close()
    return True


if __name__ == '__main__':
    x_a = np.random.uniform(0, 1, size=(32, 256))
    targets = np.random.randint(0, 2, size=(32))
    print(targets)
    plot_embedding(x_a, targets, title='tsne_demo')
    print('Computing t-SNE embedding')
