from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import json


def plot_rollout_pca_all_fit(
    rollout_data,
    save_path=None,
    title="Rollout PCA (3D)",
    true_key="true_z",
    pred_key="pred_z",
):
    true_z = rollout_data[true_key]  # (N, D)
    pred_z = rollout_data[pred_key]  # (N+1, D) or (N, D)

    # pred_z[0] は初期 true_z_0 なので, 比較しやすいように長さを揃える
    if pred_z.shape[0] == true_z.shape[0] + 1:
        pred_z_plot = pred_z[:-1]
    else:
        pred_z_plot = pred_z[: true_z.shape[0]]

    true_z_plot = true_z[: pred_z_plot.shape[0]]

    all_z = np.concatenate([true_z_plot, pred_z_plot], axis=0)
    # all_z = true_z_plot

    pca = PCA(n_components=3)
    all_z_pca = pca.fit_transform(all_z)

    n = true_z_plot.shape[0]
    true_pca = all_z_pca[:n]
    pred_pca = all_z_pca[n:]

    var = pca.explained_variance_ratio_

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        true_pca[:, 0],
        true_pca[:, 1],
        true_pca[:, 2],
        color="black",
        linewidth=1.5,
        label="Encoder",
    )

    ax.plot(
        pred_pca[:, 0],
        pred_pca[:, 1],
        pred_pca[:, 2],
        color="firebrick",
        linewidth=1.5,
        label="Closed",
    )

    ax.scatter(
        true_pca[0, 0],
        true_pca[0, 1],
        true_pca[0, 2],
        color="black",
        marker="o",
        s=40,
    )
    ax.scatter(
        pred_pca[0, 0],
        pred_pca[0, 1],
        pred_pca[0, 2],
        color="firebrick",
        marker="x",
        s=50,
    )

    ax.set_title(
        f"{title} | var exp: {var[0]:.2f}, {var[1]:.2f}, {var[2]:.2f}"
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend()

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.close(fig)

    return {
        "pca": pca,
        "true_pca": true_pca,
        "pred_pca": pred_pca,
        "explained_variance_ratio": var,
    }




def plot_rollout_pca_enc_fit(
    rollout_data,
    save_path=None,
    title="Rollout PCA (3D)",
    true_key="true_z",
    pred_key="pred_z",
    plot_line = False, #closed forward を一本線で見るか否か
):
    true_z = rollout_data[true_key]
    pred_z = rollout_data[pred_key]

    pca = PCA(n_components=3)

    # ============================================================
    # Case 1: closed dataset rollout
    # true_z: (N, T, D)
    # pred_z: (N, T, D)
    # N本の軌跡を描く
    # ============================================================
    if true_z.ndim == 3 and not plot_line:
        N, T, D = true_z.shape

        if pred_z.ndim != 3:
            raise ValueError(f"true_z is 3D but pred_z is not 3D: pred_z.shape={pred_z.shape}")

        # 長さを揃える
        T_plot = min(true_z.shape[1], pred_z.shape[1])
        true_z_plot = true_z[:, :T_plot, :]
        pred_z_plot = pred_z[:, :T_plot, :]

        # PCA fit は encoder 出力のみで行う
        true_flat = true_z_plot.reshape(-1, D)          # (N*T, D)
        pred_flat = pred_z_plot.reshape(-1, D)          # (N*T, D)

        true_pca_flat = pca.fit_transform(true_flat)    # (N*T, 3)
        pred_pca_flat = pca.transform(pred_flat)        # (N*T, 3)

        
        true_pca = true_pca_flat.reshape(N, T_plot, 3)
        pred_pca = pred_pca_flat.reshape(N, T_plot, 3)

        var = pca.explained_variance_ratio_

        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection="3d")

        for i in range(N):
            label_true = "Encoder" if i == 0 else None
            label_pred = "Pred" if i == 0 else None

            ax.plot(
                true_pca[i, :, 0],
                true_pca[i, :, 1],
                true_pca[i, :, 2],
                color="black",
                linewidth=1.0,
                alpha=0.25,
                label=label_true,
            )


            # encoder の方向を矢印で描く
            arrow_step = max(1, T_plot // 10)  # 10本くらいに間引く

            arrow_scale = 0.4

            for t in range(0, T_plot - 1, arrow_step):
                x, y, z = true_pca[i, t]
                dx, dy, dz = (true_pca[i, t + 1] - true_pca[i, t]) * arrow_scale

                ax.quiver(
                    x, y, z,
                    dx, dy, dz,
                    color="b",
                    length=1.0,
                    normalize=False,
                    alpha=0.9,
                    linewidth=1.2,
                )



            ax.plot(
                pred_pca[i, :, 0],
                pred_pca[i, :, 1],
                pred_pca[i, :, 2],
                color="firebrick",
                linewidth=1.0,
                alpha=0.35,
                label=label_pred,
            )

            ax.scatter(
                true_pca[i, 0, 0],
                true_pca[i, 0, 1],
                true_pca[i, 0, 2],
                color="black",
                marker="o",
                s=10,
                alpha=0.3,
            )

            ax.scatter(
                pred_pca[i, 0, 0],
                pred_pca[i, 0, 1],
                pred_pca[i, 0, 2],
                color="firebrick",
                marker="x",
                s=12,
                alpha=0.35,
            )

    # ============================================================
    # Case 2: existing behavior
    # true_z: (N, D)
    # pred_z: (N+1, D) or (N, D)
    # ============================================================
    else:
        if true_z.ndim == 3:
            N, T, D = true_z.shape
            true_z = true_z.reshape(-1, D)
            pred_z = pred_z.reshape(-1, D)
            
        
        print("pred_z.shape:", pred_z.shape)
        print("true_z.shape:", true_z.shape)
        
        # pred_z[0] は初期 true_z_0 なので, 比較しやすいように長さを揃える
        L = min(pred_z.shape[0], true_z.shape[0])
        pred_z_plot = pred_z[:L]
        true_z_plot = true_z[:L]
        
        # if pred_z.shape[0] == true_z.shape[0] + 1:
        #     pred_z_plot = pred_z[:-1]
        # else:
        #     pred_z_plot = pred_z[: true_z.shape[0]]

        # true_z_plot = true_z[: pred_z_plot.shape[0]]

        true_pca = pca.fit_transform(true_z_plot)
        pred_pca = pca.transform(pred_z_plot)

        var = pca.explained_variance_ratio_

        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot(
            true_pca[:, 0],
            true_pca[:, 1],
            true_pca[:, 2],
            color="black",
            linewidth=1.5,
            label="Encoder",
        )

        ax.plot(
            pred_pca[:, 0],
            pred_pca[:, 1],
            pred_pca[:, 2],
            color="firebrick",
            linewidth=1.5,
            label="Closed",
        )

        ax.scatter(
            true_pca[0, 0],
            true_pca[0, 1],
            true_pca[0, 2],
            color="black",
            marker="o",
            s=40,
        )
        ax.scatter(
            pred_pca[0, 0],
            pred_pca[0, 1],
            pred_pca[0, 2],
            color="firebrick",
            marker="x",
            s=50,
        )

    ax.set_title(
        f"{title} | var exp: {var[0]:.2f}, {var[1]:.2f}, {var[2]:.2f}"
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend()

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.close(fig)

    return {
        "pca": pca,
        "true_pca": true_pca,
        "pred_pca": pred_pca,
        "explained_variance_ratio": var,
    }





#潜在次元毎の表現の分散を可視化
def plot_latent_spread_over_latent_dim(
    rollout_data,
    save_path=None,
    title="Latent variance per dimension",
    true_key="true_z",
):
    """
    rollout_data[true_key]: (N, D)
        N: samples
        D: latent dimension
    """

    true_z = rollout_data[true_key]  # (N, D)

    true_var = true_z.var(axis=0)  # (D,)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(true_var, linewidth=1.5)

    ax.set_title(
        f"{title}\n"
        f"mean={true_var.mean():.4f}, "
        f"min={true_var.min():.4f}, "
        f"max={true_var.max():.4f}"
    )

    ax.set_xlabel("Latent dimension")
    ax.set_ylabel("Variance")
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.close(fig)

    return {
        "latent_variance": true_var,
        "mean": true_var.mean(),
        "min": true_var.min(),
        "max": true_var.max(),
    }


#潜在次元毎の表現の分散を可視化
def plot_latent_spread_over_time(
    rollout_data,
    save_path=None,
    title="Latent variance per time",
    true_key="true_z",
):
    """
    rollout_data[true_key]: (N, D)
        N: samples
        D: latent dimension
    """

    true_z = rollout_data[true_key]  # (N, D)

    true_var = true_z.var(axis=1)  # (N,)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(true_var, linewidth=1.5)

    ax.set_title(
        f"{title}\n"
        f"mean={true_var.mean():.4f}, "
        f"min={true_var.min():.4f}, "
        f"max={true_var.max():.4f}"
    )

    ax.set_xlabel("time")
    ax.set_ylabel("Variance")
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.close(fig)

    return {
        "latent_variance": true_var,
        "mean": true_var.mean(),
        "min": true_var.min(),
        "max": true_var.max(),
    }




def plot_direction_rollout_pca(
    direction_data,
    save_path=None,
    title="Direction Action Rollout PCA",
):
    pred_z = direction_data["pred_z"]  # (4,T,D)
    labels = direction_data["labels"]

    K, T, D = pred_z.shape

    flat = pred_z.reshape(-1, D)

    pca = PCA(n_components=3)
    pred_pca_flat = pca.fit_transform(flat)
    pred_pca = pred_pca_flat.reshape(K, T, 3)

    var = pca.explained_variance_ratio_

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    for k in range(K):
        line, = ax.plot(
            pred_pca[k, :, 0],
            pred_pca[k, :, 1],
            pred_pca[k, :, 2],
            linewidth=2.0,
            marker="o",
            markersize=4,
            label=labels[k],
        )
        
        color = line.get_color()

        ax.scatter(
            pred_pca[k, 0, 0],
            pred_pca[k, 0, 1],
            pred_pca[k, 0, 2],
            marker="o",
            s=50,
            color=color,
        )

        ax.scatter(
            pred_pca[k, -1, 0],
            pred_pca[k, -1, 1],
            pred_pca[k, -1, 2],
            marker="x",
            s=70,
            color=color,
        )

    ax.set_title(
        f"{title} | var exp: {var[0]:.2f}, {var[1]:.2f}, {var[2]:.2f}"
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend()

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.close(fig)

    return {
        "pca": pca,
        "pred_pca": pred_pca,
        "explained_variance_ratio": var,
    }


def plot_direction_xy_trajectory(
    direction_data,
    save_path=None,
    title="Direction Action XY Trajectory",
):
    xyz_seq = direction_data["xyz_seq"]   # (4,T,3)
    labels = direction_data["labels"]

    K, T, _ = xyz_seq.shape

    fig, ax = plt.subplots(figsize=(7, 7))

    for k in range(K):
        x = xyz_seq[k, :, 0]
        y = xyz_seq[k, :, 1]

        line, = ax.plot(
            x,
            y,
            linewidth=2.0,
            marker="o",
            markersize=4,
            label=labels[k],
        )

        color = line.get_color()

        # start
        ax.scatter(
            x[0],
            y[0],
            marker="o",
            s=80,
            color=color,
        )

        # end
        ax.scatter(
            x[-1],
            y[-1],
            marker="x",
            s=120,
            color=color,
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

    ax.set_aspect("equal")

    ax.grid(True)
    ax.legend()

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.close(fig)






def analyze_latent_isotropy(
    latents,
    save_dir=None,
    prefix="encoder_latent",
    eps=1e-8,
):
    """
    latents: (N, D)
    Encoder 出力 z が等方的かを確認する.

    等方的なら:
      mean ≈ 0
      各次元の var ≈ 1
      covariance ≈ I
      eigvals ≈ 全部同じ
      PCA explained variance ratio が極端に偏らない
    """
    latents = np.asarray(latents)
    assert latents.ndim == 2, f"latents must be (N,D), got {latents.shape}"

    N, D = latents.shape

    mean = latents.mean(axis=0)          # (D,)
    std = latents.std(axis=0)            # (D,)
    var = latents.var(axis=0)            # (D,)

    centered = latents - mean[None, :]
    cov = centered.T @ centered / max(N - 1, 1)   # (D,D)

    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]

    diag = np.diag(cov)
    offdiag = cov - np.diag(diag)

    metrics = {
        "N": int(N),
        "D": int(D),

        # mean が 0 に近いか
        "mean_abs_mean": float(np.mean(np.abs(mean))),
        "mean_l2_norm": float(np.linalg.norm(mean)),

        # 各次元の分散が揃っているか
        "var_mean": float(var.mean()),
        "var_std": float(var.std()),
        "var_min": float(var.min()),
        "var_max": float(var.max()),
        "var_cv": float(var.std() / (var.mean() + eps)),

        # 共分散の非対角成分が小さいか
        "offdiag_abs_mean": float(np.mean(np.abs(offdiag))),
        "offdiag_abs_max": float(np.max(np.abs(offdiag))),

        # 固有値が揃っているか
        "eig_mean": float(eigvals.mean()),
        "eig_std": float(eigvals.std()),
        "eig_min": float(eigvals.min()),
        "eig_max": float(eigvals.max()),
        "eig_cv": float(eigvals.std() / (eigvals.mean() + eps)),
        "eig_condition": float(eigvals.max() / (eigvals.min() + eps)),

        # 上位主成分に分散が偏っていないか
        "top1_explained_ratio": float(eigvals[0] / (eigvals.sum() + eps)),
        "top3_explained_ratio": float(eigvals[:3].sum() / (eigvals.sum() + eps)),
        "top10_explained_ratio": float(eigvals[:10].sum() / (eigvals.sum() + eps)),
    }

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / f"{prefix}_isotropy_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # 1. dimension-wise mean
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(mean)
        ax.axhline(0.0, linestyle="--", linewidth=1.0)
        ax.set_title("Latent mean per dimension")
        ax.set_xlabel("latent dim")
        ax.set_ylabel("mean")
        ax.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_dir / f"{prefix}_mean_per_dim.png", dpi=200)
        plt.close(fig)

        # 2. dimension-wise variance
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(var)
        ax.axhline(1.0, linestyle="--", linewidth=1.0)
        ax.set_title("Latent variance per dimension")
        ax.set_xlabel("latent dim")
        ax.set_ylabel("variance")
        ax.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_dir / f"{prefix}_var_per_dim.png", dpi=200)
        plt.close(fig)

        # 3. covariance matrix
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cov, vmin=-1.5, vmax=1.5)
        ax.set_title("Latent covariance matrix")
        ax.set_xlabel("latent dim")
        ax.set_ylabel("latent dim")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(save_dir / f"{prefix}_covariance.png", dpi=200)
        plt.close(fig)

        # 4. eigenvalue spectrum
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(eigvals)
        ax.axhline(eigvals.mean(), linestyle="--", linewidth=1.0)
        ax.set_title("Covariance eigenvalue spectrum")
        ax.set_xlabel("rank")
        ax.set_ylabel("eigenvalue")
        ax.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_dir / f"{prefix}_eigenvalues.png", dpi=200)
        plt.close(fig)

        # 5. PCA explained variance ratio
        explained = eigvals / (eigvals.sum() + eps)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(explained)
        ax.set_title("PCA explained variance ratio")
        ax.set_xlabel("principal component")
        ax.set_ylabel("explained variance ratio")
        ax.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_dir / f"{prefix}_explained_ratio.png", dpi=200)
        plt.close(fig)




        # 6. PCA 3D scatter: (B, D) -> (B, 3)
        pca3 = PCA(n_components=3)
        latents_pca3 = pca3.fit_transform(latents)  # (B, 3)
        pca3_var = pca3.explained_variance_ratio_

        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(
            latents_pca3[:, 0],
            latents_pca3[:, 1],
            latents_pca3[:, 2],
            s=8,
            alpha=0.45,
        )

        ax.set_title(
            "Latent PCA 3D scatter\n"
            f"var exp: {pca3_var[0]:.3f}, {pca3_var[1]:.3f}, {pca3_var[2]:.3f}"
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")

        plt.tight_layout()
        plt.savefig(save_dir / f"{prefix}_pca3d_scatter.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        metrics["pca3_top1_explained_ratio"] = float(pca3_var[0])
        metrics["pca3_top3_explained_ratio"] = float(pca3_var.sum())





    return metrics


def plot_shaded_dataset_pca(
    shaded_data,
    save_path=None,
    title="Shaded / Clear / No-box Encoder PCA",
    latent_key="latents",
    label_key="label",
    pos_key="bluebox_pos",
    n_components=3,
):
    """
    shaded_data:
        {
            "latents": (N, D),
            "targets": {
                "label": (N,),
                "bluebox_pos": (N, 3), optional
            }
        }
    """

    latents = np.asarray(shaded_data[latent_key])

    targets = shaded_data.get("targets", {})
    labels = np.asarray(targets[label_key]).reshape(-1)

    assert latents.ndim == 2, f"latents must be (N,D), got {latents.shape}"
    assert labels.shape[0] == latents.shape[0], (
        f"labels length {labels.shape[0]} != latents length {latents.shape[0]}"
    )

    pca = PCA(n_components=n_components)
    z_pca = pca.fit_transform(latents)
    var = pca.explained_variance_ratio_

    label_names = {
        0: "shadow",
        1: "clear",
        2: "no_box",
    }

    markers = {
        0: "o",
        1: "^",
        2: "s",
    }

    fig = plt.figure(figsize=(9, 8))

    if n_components == 3:
        ax = fig.add_subplot(111, projection="3d")

        for lab in sorted(np.unique(labels)):
            mask = labels == lab
            ax.scatter(
                z_pca[mask, 0],
                z_pca[mask, 1],
                z_pca[mask, 2],
                s=30,
                alpha=0.75,
                marker=markers.get(int(lab), "o"),
                label=label_names.get(int(lab), f"label={lab}"),
            )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title(
            f"{title}\n"
            f"var exp: {var[0]:.3f}, {var[1]:.3f}, {var[2]:.3f}"
        )

    else:
        ax = fig.add_subplot(111)

        for lab in sorted(np.unique(labels)):
            mask = labels == lab
            ax.scatter(
                z_pca[mask, 0],
                z_pca[mask, 1],
                s=30,
                alpha=0.75,
                marker=markers.get(int(lab), "o"),
                label=label_names.get(int(lab), f"label={lab}"),
            )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(
            f"{title}\n"
            f"var exp: {var[0]:.3f}, {var[1]:.3f}"
        )
        ax.grid(True, alpha=0.3)

    ax.legend()
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.close(fig)

    return {
        "pca": pca,
        "z_pca": z_pca,
        "labels": labels,
        "explained_variance_ratio": var,
    }



def plot_ee_goal_cost_map(
    cost_data,
    save_path=None,
    title="EE position vs goal latent cost",
    show_min=True,
    show_goal=True,
    show_box=True,
):
    """
    EE位置を変えたときの goal latent との cost map を可視化する.

    Args:
        cost_data: collect_ee_goal_cost_map() の返り値
        save_path: 保存先
        title: 図タイトル
        show_min: 最小cost位置を表示するか
        show_goal: goal EE位置を表示するか
        show_box: box位置を表示するか
    """

    goal_img = cost_data.get("goal_img", None)

    if goal_img is not None and save_path is not None:
        goal_path = Path(save_path).with_name(
            Path(save_path).stem + "_goal.png"
        )

        plt.imsave(goal_path, goal_img)


    cost_map = np.asarray(cost_data["cost_map"])  # (num_x, num_y)
    xs = np.asarray(cost_data["xs"])
    ys = np.asarray(cost_data["ys"])

    goal_ee_pos = np.asarray(cost_data.get("goal_ee_pos", None))
    goal_bluebox_pos = np.asarray(cost_data.get("goal_bluebox_pos", None))
    fixed_bluebox_pos = np.asarray(cost_data.get("fixed_bluebox_pos", None))
    min_pos = np.asarray(cost_data.get("min_pos", None))

    min_cost = cost_data.get("min_cost", None)
    cost_type = cost_data.get("cost_type", "cost")

    fig, ax = plt.subplots(figsize=(7, 6))

    # cost_map[ix, iy] = x方向, y方向なので transpose して表示
    im = ax.imshow(
        cost_map,
        origin="lower",
        extent=[ys[0], ys[-1], xs[0], xs[-1]],
        aspect="equal",
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"Latent {cost_type} cost")

    if show_goal and goal_ee_pos is not None and goal_ee_pos.size >= 2:
        ax.scatter(
            goal_ee_pos[1],
            goal_ee_pos[0],
            marker="*",
            s=180,
            label="goal EE",
        )

    if show_box and fixed_bluebox_pos is not None and fixed_bluebox_pos.size >= 2:
        ax.scatter(
            fixed_bluebox_pos[1],
            fixed_bluebox_pos[0],
            marker="s",
            s=100,
            label="fixed box",
        )

    if show_box and goal_bluebox_pos is not None and goal_bluebox_pos.size >= 2:
        ax.scatter(
            goal_bluebox_pos[1],
            goal_bluebox_pos[0],
            marker="o",
            s=100,
            label="goal box",
        )

    if show_min and min_pos is not None and min_pos.size >= 2:
        label = "min cost"
        if min_cost is not None:
            label += f" ({min_cost:.4g})"

        ax.scatter(
            min_pos[1],
            min_pos[0],
            marker="x",
            s=140,
            label=label,
        )

    ax.set_xlabel("EE Y")
    ax.set_ylabel("EE X")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(False)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()

    return fig


def plot_box_goal_cost_map(
    cost_data,
    save_path=None,
    title="Box position vs goal latent cost",
    show_min=True,
    show_goal=True,
    show_ee=True,
):

    goal_img = cost_data.get("goal_img", None)

    if goal_img is not None and save_path is not None:
        goal_path = Path(save_path).with_name(
            Path(save_path).stem + "_goal.png"
        )

        plt.imsave(goal_path, goal_img)



    cost_map = np.asarray(cost_data["cost_map"])
    xs = np.asarray(cost_data["xs"])
    ys = np.asarray(cost_data["ys"])

    goal_ee_pos = np.asarray(cost_data.get("goal_ee_pos", None))
    goal_bluebox_pos = np.asarray(cost_data.get("goal_bluebox_pos", None))
    fixed_ee_pos = np.asarray(cost_data.get("fixed_ee_pos", None))
    min_pos = np.asarray(cost_data.get("min_pos", None))

    min_cost = cost_data.get("min_cost", None)
    cost_type = cost_data.get("cost_type", "cost")

    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(
        cost_map,
        origin="lower",
        extent=[ys[0], ys[-1], xs[0], xs[-1]],
        aspect="equal",
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"Latent {cost_type} cost")

    if show_goal and goal_bluebox_pos is not None and goal_bluebox_pos.size >= 2:
        ax.scatter(
            goal_bluebox_pos[1],
            goal_bluebox_pos[0],
            marker="*",
            s=180,
            label="goal box",
        )

    if show_ee and fixed_ee_pos is not None and fixed_ee_pos.size >= 2:
        ax.scatter(
            fixed_ee_pos[1],
            fixed_ee_pos[0],
            marker="s",
            s=100,
            label="fixed EE",
        )

    if show_ee and goal_ee_pos is not None and goal_ee_pos.size >= 2:
        ax.scatter(
            goal_ee_pos[1],
            goal_ee_pos[0],
            marker="o",
            s=100,
            label="goal EE",
        )

    if show_min and min_pos is not None and min_pos.size >= 2:
        label = "min cost"
        if min_cost is not None:
            label += f" ({min_cost:.4g})"

        ax.scatter(
            min_pos[1],
            min_pos[0],
            marker="x",
            s=140,
            label=label,
        )

    ax.set_xlabel("Y")
    ax.set_ylabel("X")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(False)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()

    return fig


def plot_relative_pair_goal_cost_map_2d(
    cost_data,
    save_path=None,
    title="Relative EE-Box pair position vs goal latent cost",
    show_min=True,
    show_goal=True,
    show_task_initial=True,
    task_initial_box_pos=None,
):
    goal_img = cost_data.get("goal_img", None)

    if goal_img is not None and save_path is not None:
        goal_path = Path(save_path).with_name(
            Path(save_path).stem + "_goal.png"
        )
        plt.imsave(goal_path, goal_img)

    cost_map = np.asarray(cost_data["cost_map"])
    xs = np.asarray(cost_data["xs"])
    ys = np.asarray(cost_data["ys"])

    goal_ee_pos = np.asarray(cost_data.get("goal_ee_pos", None))
    goal_bluebox_pos = np.asarray(cost_data.get("goal_bluebox_pos", None))
    delta = np.asarray(cost_data.get("delta", goal_bluebox_pos - goal_ee_pos))
    
    min_ee_pos = np.asarray(cost_data.get("min_ee_pos", None))
    min_box_pos = np.asarray(cost_data.get("min_box_pos", None))

    min_cost = cost_data.get("min_cost", None)
    cost_type = cost_data.get("cost_type", "cost")

    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(
        cost_map,
        origin="lower",
        extent=[ys[0], ys[-1], xs[0], xs[-1]],
        aspect="equal",
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"Latent {cost_type} cost")

    if show_goal:
        ax.scatter(
            goal_ee_pos[1],
            goal_ee_pos[0],
            marker="o",
            s=100,
            label="goal EE",
        )
        ax.scatter(
            goal_bluebox_pos[1],
            goal_bluebox_pos[0],
            marker="*",
            s=180,
            label="goal box",
        )
        ax.plot(
            [goal_ee_pos[1], goal_bluebox_pos[1]],
            [goal_ee_pos[0], goal_bluebox_pos[0]],
            linestyle="--",
            linewidth=1.5,
            # label="goal relation",
        )

    if show_min and min_ee_pos is not None and min_ee_pos.size >= 2:
        label = "min EE"
        if min_cost is not None:
            label += f" ({min_cost:.4g})"

        ax.scatter(
            min_ee_pos[1],
            min_ee_pos[0],
            marker="x",
            s=140,
            label=label,
        )

        if min_box_pos is not None and min_box_pos.size >= 2:
            ax.scatter(
                min_box_pos[1],
                min_box_pos[0],
                marker="s",
                s=100,
                label="min box",
            )
            ax.plot(
                [min_ee_pos[1], min_box_pos[1]],
                [min_ee_pos[0], min_box_pos[0]],
                linestyle=":",
                linewidth=1.5,
                # label="min relation",
            )
            
    if show_task_initial and task_initial_box_pos is not None:
        task_initial_box_pos = np.asarray(task_initial_box_pos, dtype=np.float32)

        task_initial_ee_pos = task_initial_box_pos - delta
        task_initial_ee_pos[2] = goal_ee_pos[2]

        ax.scatter(
            task_initial_ee_pos[1],
            task_initial_ee_pos[0],
            marker="^",
            s=120,
            label="task initial EE",
        )

        ax.scatter(
            task_initial_box_pos[1],
            task_initial_box_pos[0],
            marker="D",
            s=100,
            label="task initial box",
        )

        ax.plot(
            [task_initial_ee_pos[1], task_initial_box_pos[1]],
            [task_initial_ee_pos[0], task_initial_box_pos[0]],
            linestyle="--",
            linewidth=1.5,
            # label="task initial relation",
        )

    ax.set_xlabel("Y")
    ax.set_ylabel("X")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(False)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()

    return fig


def plot_relative_pair_goal_cost_map_3d(
    cost_data,
    save_path=None,
    title="Relative EE-Box pair position vs goal latent cost 3D",
    show_min=True,
    show_goal=True,
    show_task_initial=True,
    task_initial_box_pos=None,
    elev=35,
    azim=-60,
    zlim=(0, 7),
):
    cost_map = np.asarray(cost_data["cost_map"])
    xs = np.asarray(cost_data["xs"])  # X positions
    ys = np.asarray(cost_data["ys"])  # Y positions

    goal_ee_pos = np.asarray(cost_data.get("goal_ee_pos", None))
    goal_bluebox_pos = np.asarray(cost_data.get("goal_bluebox_pos", None))
    delta = np.asarray(cost_data.get("delta", goal_bluebox_pos - goal_ee_pos))

    min_ee_pos = np.asarray(cost_data.get("min_ee_pos", None))
    min_box_pos = np.asarray(cost_data.get("min_box_pos", None))

    min_cost = cost_data.get("min_cost", None)
    cost_type = cost_data.get("cost_type", "cost")

    # meshgrid
    Y, X = np.meshgrid(ys, xs)  # cost_map shape: (len(xs), len(ys))
    Z = cost_map

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        Y,
        X,
        Z,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        alpha=0.85,
    )

    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label=f"Latent {cost_type} cost")

    def interp_cost_at_pos(pos):
        """
        pos: [x, y, z]
        xs, ys の最近傍の cost を返す
        """
        ix = np.argmin(np.abs(xs - pos[0]))
        iy = np.argmin(np.abs(ys - pos[1]))
        return cost_map[ix, iy]
    
    

    z_min = np.nanmin(cost_map)
    z_max = np.nanmax(cost_map)
    eps = 0.5 * (z_max - z_min)
    if show_goal:
        goal_ee_cost = interp_cost_at_pos(goal_ee_pos)
        goal_box_cost = interp_cost_at_pos(goal_bluebox_pos)
        goal_pair_cost = interp_cost_at_pos(goal_ee_pos)

        # ax.scatter(
        #     goal_ee_pos[1],
        #     goal_ee_pos[0],
        #     goal_pair_cost + eps,
        #     marker="o",
        #     s=180,
        #     color="blue",
        #     edgecolors="black",
        #     linewidths=1.5,
        #     depthshade=False,
        #     label="goal EE",
        # )

        # ax.scatter(
        #     goal_bluebox_pos[1],
        #     goal_bluebox_pos[0],
        #     goal_pair_cost + eps,
        #     marker="*",
        #     s=260,
        #     color="orange",
        #     edgecolors="black",
        #     linewidths=1.2,
        #     depthshade=False,
        #     label="goal box",
        # )

        # ax.plot(
        #     [goal_ee_pos[1], goal_bluebox_pos[1]],
        #     [goal_ee_pos[0], goal_bluebox_pos[0]],
        #     [goal_pair_cost + eps, goal_pair_cost + eps],
        #     linestyle="--",
        #     linewidth=2.0,
        #     color="black",
        # )

    if show_min and min_ee_pos is not None and min_ee_pos.size >= 2:
        if min_cost is None:
            min_cost = interp_cost_at_pos(min_ee_pos)

        label = f"min EE ({min_cost:.4g})"

        # ax.scatter(
        #     min_ee_pos[1],
        #     min_ee_pos[0],
        #     min_cost + eps,
        #     marker="x",
        #     s=220,
        #     color="green",
        #     linewidths=3,
        #     depthshade=False,
        #     label=f"min EE ({min_cost:.4g})",
        # )

        if min_box_pos is not None and min_box_pos.size >= 2:
            min_box_cost = interp_cost_at_pos(min_box_pos)


            # ax.scatter(
            #     min_box_pos[1],
            #     min_box_pos[0],
            #     min_cost + eps,
            #     marker="s",
            #     s=160,
            #     color="red",
            #     edgecolors="black",
            #     linewidths=1.5,
            #     depthshade=False,
            #     label="min box",
            # )

            # ax.plot(
            #     [min_ee_pos[1], min_box_pos[1]],
            #     [min_ee_pos[0], min_box_pos[0]],
            #     [min_cost + eps, min_cost + eps],
            #     linestyle=":",
            #     linewidth=2.0,
            #     color="black",
            # )

    if show_task_initial and task_initial_box_pos is not None:
        task_initial_box_pos = np.asarray(task_initial_box_pos, dtype=np.float32)

        task_initial_ee_pos = task_initial_box_pos - delta
        task_initial_ee_pos[2] = goal_ee_pos[2]

        task_initial_ee_cost = interp_cost_at_pos(task_initial_ee_pos)
        task_initial_box_cost = interp_cost_at_pos(task_initial_box_pos)
        

        task_initial_pair_cost = interp_cost_at_pos(task_initial_ee_pos)

        # ax.scatter(
        #     task_initial_ee_pos[1],
        #     task_initial_ee_pos[0],
        #     task_initial_pair_cost + eps,
        #     marker="^",
        #     s=180,
        #     color="purple",
        #     edgecolors="black",
        #     linewidths=1.5,
        #     depthshade=False,
        #     label="task initial EE",
        # )

        # ax.scatter(
        #     task_initial_box_pos[1],
        #     task_initial_box_pos[0],
        #     task_initial_pair_cost + eps,
        #     marker="D",
        #     s=150,
        #     color="brown",
        #     edgecolors="black",
        #     linewidths=1.5,
        #     depthshade=False,
        #     label="task initial box",
        # )

        # ax.plot(
        #     [task_initial_ee_pos[1], task_initial_box_pos[1]],
        #     [task_initial_ee_pos[0], task_initial_box_pos[0]],
        #     [task_initial_pair_cost + eps, task_initial_pair_cost + eps],
        #     linestyle="--",
        #     linewidth=2.0,
        #     color="black",
        # )


    ax.set_xlabel("Y")
    ax.set_ylabel("X")
    ax.set_zlabel(f"Latent {cost_type} cost")
    ax.set_title(title)
    
    if zlim is not None:
        ax.set_zlim(*zlim)

    ax.view_init(elev=elev, azim=azim)
    # ax.legend(loc="best")

    fig.tight_layout()
 
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()

    return fig



def plot_action_predictor_goal_cost_map(
    cost_data,
    save_path=None,
    title="Action position vs predictor goal latent cost",
    show_min=True,
    show_init=True,
    show_goal=True,
):
    cost_map = np.asarray(cost_data["cost_map"])
    xs = np.asarray(cost_data["xs"])
    ys = np.asarray(cost_data["ys"])

    init_ee_pos = np.asarray(cost_data.get("init_ee_pos", None))
    init_bluebox_pos = np.asarray(cost_data.get("init_bluebox_pos", None))
    goal_ee_pos = np.asarray(cost_data.get("goal_ee_pos", None))
    goal_bluebox_pos = np.asarray(cost_data.get("goal_bluebox_pos", None))
    min_action_pos = np.asarray(cost_data.get("min_action_pos", None))

    min_cost = cost_data.get("min_cost", None)
    cost_type = cost_data.get("cost_type", "cost")

    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(
        cost_map,
        origin="lower",
        extent=[ys[0], ys[-1], xs[0], xs[-1]],
        aspect="equal",
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"Predictor latent {cost_type} cost")

    if show_init and init_ee_pos is not None and init_ee_pos.size >= 2:
        ax.scatter(
            init_ee_pos[1],
            init_ee_pos[0],
            marker="*",
            s=180,
            label="init EE",
        )

    if show_init and init_bluebox_pos is not None and init_bluebox_pos.size >= 2:
        ax.scatter(
            init_bluebox_pos[1],
            init_bluebox_pos[0],
            marker="s",
            s=100,
            label="init box",
        )

    if show_goal and goal_ee_pos is not None and goal_ee_pos.size >= 2:
        ax.scatter(
            goal_ee_pos[1],
            goal_ee_pos[0],
            marker="*",
            s=140,
            label="goal EE",
            color="cyan",
        )

    if show_goal and goal_bluebox_pos is not None and goal_bluebox_pos.size >= 2:
        ax.scatter(
            goal_bluebox_pos[1],
            goal_bluebox_pos[0],
            marker="o",
            s=100,
            label="goal box",
            color="tab:green",
        )

    if show_min and min_action_pos is not None and min_action_pos.size >= 2:
        label = "min action"
        if min_cost is not None:
            label += f" ({min_cost:.4g})"

        ax.scatter(
            min_action_pos[1],
            min_action_pos[0],
            marker="x",
            s=140,
            label=label,
            color="tab:red",
        )

    ax.set_xlabel("Action Y")
    ax.set_ylabel("Action X")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(False)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()

    return fig




def plot_cross_position_xy_trajectory(
    cross_data,
    save_path=None,
    title="Cross EE position trajectory",
    color_map=None,
):
    ee_pos = np.asarray(cross_data["ee_pos"])      # (N, 3)
    labels = np.asarray(cross_data["labels"])      # (N,)

    assert ee_pos.ndim == 2 and ee_pos.shape[1] >= 2, \
        f"ee_pos must be (N,>=2), got {ee_pos.shape}"

    if color_map is None:
        color_map = {
            "x_axis": "C0",
            "y_axis": "C1",
        }

    fig, ax = plt.subplots(figsize=(7, 7))

    for lab in sorted(np.unique(labels)):
        mask = labels == lab
        pos = ee_pos[mask]

        color = color_map.get(str(lab), None)

        ax.plot(
            pos[:, 0],
            pos[:, 1],
            marker="o",
            linewidth=2.0,
            markersize=4,
            label=str(lab),
            color=color,
        )

        # start
        ax.scatter(
            pos[0, 0],
            pos[0, 1],
            marker="o",
            s=80,
            color=color,
        )

        # end
        ax.scatter(
            pos[-1, 0],
            pos[-1, 1],
            marker="x",
            s=120,
            color=color,
        )

    # if "bluebox_pos" in cross_data:
    #     box = np.asarray(cross_data["bluebox_pos"])
    #     if box.size >= 2:
    #         ax.scatter(
    #             box[0],
    #             box[1],
    #             marker="s",
    #             s=120,
    #             label="bluebox",
    #             color="C2",
    #         )

    if "center" in cross_data:
        center = np.asarray(cross_data["center"])
        if center.size >= 2:
            ax.scatter(
                center[0],
                center[1],
                marker="*",
                s=180,
                label="center",
                color="C3",
            )

    ax.set_xlabel("EE X")
    ax.set_ylabel("EE Y")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.close(fig)

    return {
        "ee_pos": ee_pos,
        "labels": labels,
        "color_map": color_map,
    }
    


def plot_episode_closed_rollout_whiskers_pca(
    rollout_data,
    save_path=None,
    title="Franka Push Closed-loop Dynamics",
    draw_true_segments=False,
    draw_endpoint_connections=True,
):
    """
    episode全体のEncoder軌跡を幹として、
    各開始時刻からの短期closed-loop予測を髭状に描く。
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
            "pred_rollouts and true_rollouts length mismatch."
        )

    if len(pred_rollouts) != len(start_positions):
        raise ValueError(
            "rollout_start_positions length mismatch."
        )

    if episode_true_z.shape[0] < 3:
        raise ValueError(
            "At least 3 episode samples are required "
            "for 3D PCA."
        )

    # =========================================================
    # PCAはepisode全体のEncoder表現だけでfit
    # =========================================================
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

    # =========================================================
    # episode全体のEncoder軌跡 = 幹
    # =========================================================
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

    # =========================================================
    # 各短期closed-loop rollout = 髭
    # =========================================================
    for i, (pred_seq, true_seq) in enumerate(
        zip(
            pred_rollouts_pca,
            true_rollouts_pca,
        )
    ):
        ax.plot(
            pred_seq[:, 0],
            pred_seq[:, 1],
            pred_seq[:, 2],
            color="firebrick",
            linewidth=1.0,
            alpha=1,
            label=(
                "Closed-loop prediction"
                if i == 0
                else None
            ),
            zorder=2,
        )

        # 予測終端
        ax.scatter(
            pred_seq[-1, 0],
            pred_seq[-1, 1],
            pred_seq[-1, 2],
            color="firebrick",
            marker="x",
            s=22,
            alpha=1,
            zorder=4,
        )

        # 対応する真の短期区間
        if draw_true_segments:
            ax.plot(
                true_seq[:, 0],
                true_seq[:, 1],
                true_seq[:, 2],
                color="gray",
                linewidth=0.8,
                linestyle="--",
                alpha=0.25,
                label=(
                    "Corresponding true segment"
                    if i == 0
                    else None
                ),
                zorder=1,
            )

        # 予測終端と真の終端を結ぶ
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

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

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


def plot_closed_loop_rollout_pca(
    rollout_data,
    save_path=None,
    title="Closed-loop Dynamics",
    true_key="true_z",
    pred_key="pred_z",
    draw_connections=False,
    connection_interval=1,
    font_size=12,
):
    """
    1本の closed-loop rollout を3次元PCA空間に可視化する。

    想定入力:
        rollout_data[true_key]: (H + 1, D)
        rollout_data[pred_key]: (H + 1, D)

    PCAはEncoder軌跡 true_z のみにfitし、
    Predictor軌跡 pred_z は同じPCAでtransformする。

    Args:
        rollout_data:
            true_z, pred_zを含むdict。

        save_path:
            画像の保存先。Noneの場合は保存しない。

        title:
            図のタイトル。

        true_key:
            Encoder潜在表現のキー。

        pred_key:
            Predictor潜在表現のキー。

        draw_connections:
            同じhorizonのEncoder点とPredictor点を線で結ぶか。

        connection_interval:
            connectionを何stepごとに描くか。

        font_size:
            始点・終点ラベルのフォントサイズ。

    Returns:
        {
            "pca": PCA object,
            "true_pca": (H + 1, 3),
            "pred_pca": (H + 1, 3),
            "explained_variance_ratio": (3,),
        }
    """
    true_z = np.asarray(rollout_data[true_key])
    pred_z = np.asarray(rollout_data[pred_key])

    if true_z.ndim != 2:
        raise ValueError(
            f"{true_key} must have shape (T, D), "
            f"but got {true_z.shape}"
        )

    if pred_z.ndim != 2:
        raise ValueError(
            f"{pred_key} must have shape (T, D), "
            f"but got {pred_z.shape}"
        )

    if true_z.shape[1] != pred_z.shape[1]:
        raise ValueError(
            "Latent dimensions do not match: "
            f"true_z.shape={true_z.shape}, "
            f"pred_z.shape={pred_z.shape}"
        )

    if connection_interval <= 0:
        raise ValueError(
            "connection_interval must be positive, "
            f"got {connection_interval}"
        )

    # 長さが異なる場合にも安全に描画できるよう、短い方へ揃える
    plot_length = min(
        true_z.shape[0],
        pred_z.shape[0],
    )

    if plot_length < 3:
        raise ValueError(
            "At least 3 latent states are required for 3D PCA, "
            f"but got plot_length={plot_length}"
        )

    true_z_plot = true_z[:plot_length]
    pred_z_plot = pred_z[:plot_length]

    if true_z_plot.shape[1] < 3:
        raise ValueError(
            "Latent dimension must be at least 3 for 3D PCA, "
            f"but got D={true_z_plot.shape[1]}"
        )

    # ------------------------------------------------------------
    # PCAはEncoder軌跡だけでfit
    # ------------------------------------------------------------
    pca = PCA(n_components=3)

    true_pca = pca.fit_transform(true_z_plot)
    pred_pca = pca.transform(pred_z_plot)

    explained_variance_ratio = pca.explained_variance_ratio_

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        true_pca[:, 0],
        true_pca[:, 1],
        true_pca[:, 2],
        color="black",
        linewidth=2.0,
        label="Encoder",
    )

    ax.plot(
        pred_pca[:, 0],
        pred_pca[:, 1],
        pred_pca[:, 2],
        color="firebrick",
        linewidth=2.0,
        label="Predictor",
    )

    # ------------------------------------------------------------
    # 同じhorizonのEncoderとPredictorを接続
    # ------------------------------------------------------------
    if draw_connections:
        for h in range(
            0,
            plot_length,
            connection_interval,
        ):
            ax.plot(
                [
                    true_pca[h, 0],
                    pred_pca[h, 0],
                ],
                [
                    true_pca[h, 1],
                    pred_pca[h, 1],
                ],
                [
                    true_pca[h, 2],
                    pred_pca[h, 2],
                ],
                color="gray",
                linewidth=0.8,
                alpha=0.35,
            )

    # ------------------------------------------------------------
    # Start point
    # true_z[0] == pred_z[0] のため、共通の始点として描く
    # ------------------------------------------------------------
    ax.scatter(
        true_pca[0, 0],
        true_pca[0, 1],
        true_pca[0, 2],
        color="black",
        marker="o",
        s=55,
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

    # ------------------------------------------------------------
    # Encoder end point
    # ------------------------------------------------------------
    ax.scatter(
        true_pca[-1, 0],
        true_pca[-1, 1],
        true_pca[-1, 2],
        color="black",
        marker="o",
        s=55,
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

    # ------------------------------------------------------------
    # Predictor end point
    # ------------------------------------------------------------
    ax.scatter(
        pred_pca[-1, 0],
        pred_pca[-1, 1],
        pred_pca[-1, 2],
        color="firebrick",
        marker="x",
        s=70,
        linewidths=2.0,
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

    ax.set_title(
        f"{title}\n"
        f"var exp: "
        f"{explained_variance_ratio[0]:.3f}, "
        f"{explained_variance_ratio[1]:.3f}, "
        f"{explained_variance_ratio[2]:.3f}"
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend()

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight",
        )

    plt.close(fig)

    return {
        "pca": pca,
        "true_pca": true_pca,
        "pred_pca": pred_pca,
        "explained_variance_ratio": explained_variance_ratio,
    }