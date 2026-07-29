from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import json

def plot_one_step_rollout_pca(
    rollout_data,
    save_path=None,
    title="One-step dynamics PCA",
    true_key="true_z",
    pred_key="pred_z",
    current_key="current_z",
    draw_connections=True,
):
    """
    Encoder(o_{t+1}) と Predictor(Encoder(o_t), a_t) を
    Encoder側でfitした同一PCA空間に可視化する。

    Args:
        rollout_data:
            true_z:    (N,D)
            pred_z:    (N,D)
            current_z: (N,D), optional

        draw_connections:
            各時刻の true と pred を線で結ぶか
    """
    true_z = np.asarray(rollout_data[true_key])
    pred_z = np.asarray(rollout_data[pred_key])

    if true_z.ndim != 2:
        raise ValueError(
            f"true_z must be (N,D), got {true_z.shape}"
        )

    if pred_z.ndim != 2:
        raise ValueError(
            f"pred_z must be (N,D), got {pred_z.shape}"
        )

    length = min(
        true_z.shape[0],
        pred_z.shape[0],
    )

    true_z = true_z[:length]
    pred_z = pred_z[:length]

    if length < 3:
        raise ValueError(
            f"At least 3 samples are required for 3D PCA, "
            f"got {length}"
        )

    # Encoder表現を基準としてPCAをfit
    pca = PCA(n_components=3)
    true_pca = pca.fit_transform(true_z)
    pred_pca = pca.transform(pred_z)

    explained_variance = pca.explained_variance_ratio_

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 真のencoder表現の軌跡
    ax.plot(
        true_pca[:, 0],
        true_pca[:, 1],
        true_pca[:, 2],
        color="black",
        linewidth=2.0,
        marker="o",
        markersize=3,
        label=r"Encoder $z_{t+1}$",
    )

    # predictorの1-step予測軌跡
    ax.plot(
        pred_pca[:, 0],
        pred_pca[:, 1],
        pred_pca[:, 2],
        color="firebrick",
        linewidth=2.0,
        marker="x",
        markersize=4,
        label=r"Predictor $\hat{z}_{t+1}$",
    )

    # 各時刻の真値と予測値を線で結ぶ
    if draw_connections:
        for idx in range(length):
            ax.plot(
                [
                    true_pca[idx, 0],
                    pred_pca[idx, 0],
                ],
                [
                    true_pca[idx, 1],
                    pred_pca[idx, 1],
                ],
                [
                    true_pca[idx, 2],
                    pred_pca[idx, 2],
                ],
                color="gray",
                linewidth=0.6,
                alpha=0.35,
            )

    # 開始点
    ax.scatter(
        true_pca[0, 0],
        true_pca[0, 1],
        true_pca[0, 2],
        color="black",
        marker="o",
        s=80,
        label="Encoder start",
    )

    ax.scatter(
        pred_pca[0, 0],
        pred_pca[0, 1],
        pred_pca[0, 2],
        color="firebrick",
        marker="x",
        s=100,
        label="Predictor start",
    )

    # 終了点
    ax.scatter(
        true_pca[-1, 0],
        true_pca[-1, 1],
        true_pca[-1, 2],
        color="black",
        marker="s",
        s=70,
    )

    ax.scatter(
        pred_pca[-1, 0],
        pred_pca[-1, 1],
        pred_pca[-1, 2],
        color="firebrick",
        marker="s",
        s=70,
    )

    ax.set_title(
        f"{title}\n"
        f"explained variance: "
        f"{explained_variance[0]:.3f}, "
        f"{explained_variance[1]:.3f}, "
        f"{explained_variance[2]:.3f}"
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend()

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        plt.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight",
        )

    plt.close(fig)

    return {
        "pca": pca,
        "true_pca": true_pca,
        "pred_pca": pred_pca,
        "explained_variance_ratio": explained_variance,
    }