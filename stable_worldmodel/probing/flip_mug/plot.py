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
    draw_connections=False,
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
        # marker="o",
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
        # marker="x",
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






def plot_closed_loop_rollout_pca(
    rollout_data,
    save_path,
    title="Flip Mug Closed-loop Dynamics",
    draw_connections=False,
):
    """
    true/predを同じPCA空間へ射影して3次元表示する。

    Args:
        rollout_data:
            true_z: (H+1, D)
            pred_z: (H+1, D)

        save_path:
            保存先

    Returns:
        dict:
            true_pca: (H+1, 3)
            pred_pca: (H+1, 3)
            explained_variance_ratio: (3,)
    """
    true_z = rollout_data["true_z"]
    pred_z = rollout_data["pred_z"]

    if true_z.shape != pred_z.shape:
        raise ValueError(
            f"Shape mismatch: true_z={true_z.shape}, "
            f"pred_z={pred_z.shape}"
        )

    # 同一のPCA基底に射影するため、結合してfitする
    combined_z = np.concatenate(
        [true_z, pred_z],
        axis=0,
    )

    pca = PCA(n_components=3)
    combined_pca = pca.fit_transform(combined_z)

    num_steps = true_z.shape[0]

    true_pca = combined_pca[:num_steps]
    pred_pca = combined_pca[num_steps:]

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        true_pca[:, 0],
        true_pca[:, 1],
        true_pca[:, 2],
        color="black",
        linewidth=2.5,
        label="Encoder",
    )

    ax.plot(
        pred_pca[:, 0],
        pred_pca[:, 1],
        pred_pca[:, 2],
        color="firebrick",
        linewidth=2.5,
        label="Closed-loop Predictor",
    )

    font_size = 30
    
    # 初期状態
    ax.scatter(
        true_pca[0, 0],
        true_pca[0, 1],
        true_pca[0, 2],
        color="black",
        s=100,
        marker="o",
    )

    ax.text(
        true_pca[0, 0],
        true_pca[0, 1],
        true_pca[0, 2],
        " S",   
        color="black",
        fontsize=font_size,
        fontweight="bold",
    )


    # 最終状態
    ax.scatter(
        true_pca[-1, 0],
        true_pca[-1, 1],
        true_pca[-1, 2],
        s=100,
        marker="s",
        color="black",
        label="Encoder end",
    )

    ax.text(
        true_pca[-1, 0],
        true_pca[-1, 1],
        true_pca[-1, 2],
        " E",
        color="black",
        fontsize=font_size,
        fontweight="bold",
    )


    ax.scatter(
        pred_pca[-1, 0],
        pred_pca[-1, 1],
        pred_pca[-1, 2],
        s=100,
        marker="x",
        color="firebrick",
        label="Predictor end",
    )
    ax.text(
        pred_pca[-1, 0],
        pred_pca[-1, 1],
        pred_pca[-1, 2],
        " E",
        color="firebrick",
        fontsize=font_size,
        fontweight="bold",
    )

    # 同じ時刻同士を線で結ぶ
    if draw_connections:
        for h in range(num_steps):
            ax.plot(
                [true_pca[h, 0], pred_pca[h, 0]],
                [true_pca[h, 1], pred_pca[h, 1]],
                [true_pca[h, 2], pred_pca[h, 2]],
                linewidth=0.6,
                alpha=0.35,
            )

    explained = pca.explained_variance_ratio_

    ax.set_title(
        f"{title}\n"
        f"prediction horizon: {num_steps - 1} steps\n"
        f"explained variance: "
        f"{explained[0]:.3f}, "
        f"{explained[1]:.3f}, "
        f"{explained[2]:.3f}"
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend()

    save_path = Path(save_path)
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.tight_layout()
    fig.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    return {
        "true_pca": true_pca,
        "pred_pca": pred_pca,
        "explained_variance_ratio": explained,
    }
    
    




def plot_episode_closed_rollout_whiskers_pca(
    rollout_data,
    save_path,
    title="Flip Mug Closed-loop Dynamics",
    draw_true_segments=False,
    draw_endpoint_connections=True,
):
    """
    episode全体のEncoder軌跡を幹として、
    各時刻からの短期closed-loop予測を髭状に描く。

    Args:
        rollout_data:
            episode_true_z:
                (L, D)

            pred_rollouts:
                list[(H_i+1, D)]

            true_rollouts:
                list[(H_i+1, D)]

            rollout_start_positions:
                (N,)

        draw_true_segments:
            各髭に対応する真の短期軌跡も描くか

        draw_endpoint_connections:
            各rolloutの最終予測と最終真値を結ぶか
    """
    episode_true_z = np.asarray(
        rollout_data["episode_true_z"]
    )

    pred_rollouts = rollout_data["pred_rollouts"]
    true_rollouts = rollout_data["true_rollouts"]

    start_positions = np.asarray(
        rollout_data["rollout_start_positions"]
    )

    if episode_true_z.ndim != 2:
        raise ValueError(
            f"episode_true_z must be (L,D), "
            f"got {episode_true_z.shape}"
        )

    if len(pred_rollouts) != len(true_rollouts):
        raise ValueError(
            "pred_rollouts and true_rollouts must have "
            "the same number of elements."
        )

    if len(pred_rollouts) != len(start_positions):
        raise ValueError(
            "rollout_start_positions length mismatch."
        )

    if episode_true_z.shape[0] < 3:
        raise ValueError(
            "At least 3 episode samples are required for 3D PCA."
        )

    # -------------------------------------------------
    # PCAはEncoderのepisode軌跡だけでfit
    # -------------------------------------------------
    pca = PCA(n_components=3)

    episode_true_pca = pca.fit_transform(
        episode_true_z
    )

    pred_rollouts_pca = [
        pca.transform(np.asarray(seq))
        for seq in pred_rollouts
    ]

    true_rollouts_pca = [
        pca.transform(np.asarray(seq))
        for seq in true_rollouts
    ]

    explained = pca.explained_variance_ratio_

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    # -------------------------------------------------
    # episode全体の真の軌跡 = 幹
    # -------------------------------------------------
    ax.plot(
        episode_true_pca[:, 0],
        episode_true_pca[:, 1],
        episode_true_pca[:, 2],
        color="black",
        linewidth=2.5,
        alpha=0.9,
        label="Encoder episode trajectory",
        zorder=3,
    )

    # episode開始点
    ax.scatter(
        episode_true_pca[0, 0],
        episode_true_pca[0, 1],
        episode_true_pca[0, 2],
        color="black",
        marker="o",
        s=90,
        label="Episode start",
        zorder=5,
    )

    # episode終了点
    ax.scatter(
        episode_true_pca[-1, 0],
        episode_true_pca[-1, 1],
        episode_true_pca[-1, 2],
        color="black",
        marker="s",
        s=80,
        label="Episode end",
        zorder=5,
    )

    # -------------------------------------------------
    # 各時刻からのclosed-loop rollout = 髭
    # -------------------------------------------------
    for i, (
        pred_seq,
        true_seq,
        start_pos,
    ) in enumerate(
        zip(
            pred_rollouts_pca,
            true_rollouts_pca,
            start_positions,
        )
    ):
        pred_label = (
            "Closed-loop prediction"
            if i == 0
            else None
        )

        # 予測髭
        ax.plot(
            pred_seq[:, 0],
            pred_seq[:, 1],
            pred_seq[:, 2],
            color="firebrick",
            linewidth=1.0,
            alpha=1.0,
            label=pred_label,
            zorder=2,
        )

        # 髭の開始点
        ax.scatter(
            pred_seq[0, 0],
            pred_seq[0, 1],
            pred_seq[0, 2],
            color="firebrick",
            marker=".",
            s=12,
            alpha=1.0,
            zorder=4,
        )

        # 予測終端
        ax.scatter(
            pred_seq[-1, 0],
            pred_seq[-1, 1],
            pred_seq[-1, 2],
            color="firebrick",
            marker="x",
            s=22,
            alpha=1.0,
            zorder=4,
        )

        # 対応する真の短期軌跡
        if draw_true_segments:
            true_label = (
                "Corresponding true segment"
                if i == 0
                else None
            )

            ax.plot(
                true_seq[:, 0],
                true_seq[:, 1],
                true_seq[:, 2],
                color="gray",
                linewidth=0.8,
                linestyle="--",
                alpha=0.25,
                label=true_label,
                zorder=1,
            )

        # 最終horizonでの真値と予測値を結ぶ
        if draw_endpoint_connections:
            ax.plot(
                [
                    true_seq[-1, 0],
                    pred_seq[-1, 0],
                ],
                [
                    true_seq[-1, 1],
                    pred_seq[-1, 1],
                ],
                [
                    true_seq[-1, 2],
                    pred_seq[-1, 2],
                ],
                color="gray",
                linewidth=0.6,
                alpha=0.25,
                zorder=1,
            )

    pred_step = rollout_data.get(
        "pred_step",
        None,
    )

    plot_interval = rollout_data.get(
        "plot_interval",
        None,
    )

    subtitle = []

    if pred_step is not None:
        subtitle.append(
            f"prediction horizon: {pred_step} steps"
        )

    if plot_interval is not None:
        subtitle.append(
            f"start interval: {plot_interval} frames"
        )

    subtitle.append(
        "explained variance: "
        f"{explained[0]:.3f}, "
        f"{explained[1]:.3f}, "
        f"{explained[2]:.3f}"
    )

    ax.set_title(
        title + "\n" + "\n".join(subtitle)
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend()

    save_path = Path(save_path)
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.tight_layout()
    fig.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    return {
        "pca": pca,
        "episode_true_pca": episode_true_pca,
        "pred_rollouts_pca": pred_rollouts_pca,
        "true_rollouts_pca": true_rollouts_pca,
        "explained_variance_ratio": explained,
    }