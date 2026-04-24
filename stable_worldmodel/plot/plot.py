from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.cm as cm



def plot_task_result_xy(
    bluebox_traj,
    ee_traj,
    goal_pos,
    save_path=None,
    workspace_y=(-0.2, 0.2),
    workspace_x=(0.315, 0.715),
    box_half_size=(0.05, 0.05),
    title="Task Visualization",
):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    # workspace
    ws = Rectangle(
        (workspace_y[0], workspace_x[0]),
        workspace_y[1] - workspace_y[0],
        workspace_x[1] - workspace_x[0],
        fill=False,
        linestyle="--",
        linewidth=1.5,
        edgecolor="gray",
        label="workspace",
    )
    ax.add_patch(ws)

    # bluebox trajectory
    ax.plot(
        bluebox_traj[:, 1],   # Y -> x軸
        bluebox_traj[:, 0],   # X -> y軸
        linewidth=2,
        label="bluebox_center_traj",
    )
    ax.scatter(
        bluebox_traj[0, 1],
        bluebox_traj[0, 0],
        marker="x",
        s=50,
        linewidths=2,
        label="bluebox_start",
    )


    # ---- end-effector (color gradient) ----
    points = np.stack([ee_traj[:, 1], ee_traj[:, 0]], axis=1)  # (T, 2)
    segments = np.concatenate([points[:-1, None, :], points[1:, None, :]], axis=1)

    # 時間に応じた色 (0 → 1)
    t = np.linspace(0, 1, len(segments))

    # 赤系カラーマップ（濃くなる）
    cmap = cm.get_cmap("Reds")

    lc = LineCollection(
        segments,
        cmap=cmap,
        norm=plt.Normalize(0, 1),
    )

    lc.set_array(t)
    lc.set_linewidth(2.5)

    ax.add_collection(lc)

    # 最終位置だけマーカー（見やすく）
    ax.scatter(
        ee_traj[-1, 1],
        ee_traj[-1, 0],
        color="darkred",
        s=30,
        label="end-effector",
        zorder=5,
    )
    
    ax.scatter(
    ee_traj[0, 1],
    ee_traj[0, 0],
    color="pink",
    s=30,
    label="_nolegend_",
    )
    
    

    # goal
    ax.scatter(
        goal_pos[1],
        goal_pos[0],
        s=40,
        label="goal",
        zorder=5,
    )

    # start / goal box
    hx, hy = box_half_size
    start_rect = Rectangle(
        (bluebox_traj[0, 1] - hy, bluebox_traj[0, 0] - hx),
        2 * hy,
        2 * hx,
        fill=False,
        linewidth=1.5,
        alpha=0.8,
    )
    # goal_rect = Rectangle(
    #     (goal_pos[1] - hy, goal_pos[0] - hx),
    #     2 * hy,
    #     2 * hx,
    #     fill=False,
    #     linewidth=1.5,
    #     alpha=0.8,
    # )
    # ---- bluebox final position box ----
    final_rect = Rectangle(
        (bluebox_traj[-1, 1] - hy, bluebox_traj[-1, 0] - hx),
        2 * hy,
        2 * hx,
        fill=False,
        linewidth=1.8,
        linestyle="--",
        alpha=0.9,
        label="bluebox_final_region",
    )
    ax.add_patch(final_rect)
    
    
    # ラベル（F = Final）
    ax.text(
        bluebox_traj[-1, 1],
        bluebox_traj[-1, 0] + 0.01,
        "E",
        fontsize=18,
        weight="bold",
        ha="center",
        va="center",
    )


    
    ax.add_patch(start_rect)
    # ax.add_patch(goal_rect)

    ax.text(bluebox_traj[0, 1], bluebox_traj[0, 0] + 0.01, "S",
            fontsize=20, weight="bold", ha="center")
    ax.text(goal_pos[1] - 0.02, goal_pos[0] - 0.02, "G",
            fontsize=20, weight="bold", ha="center")

    ax.set_xlabel("Y [m]")
    ax.set_ylabel("X [m]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    
    ax.relim()
    ax.autoscale_view()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    plt.show()
    



def plot_joint_angle_comparison(
    target_actions,
    actual_qpos,
    save_path=None,
    title="Joint angle comparison",
    episode_idx=None,
):
    """
    target_actions: (T, 7)
    actual_qpos:    (T, 7) or (T+1, 7)

    画像のイメージに合わせて、
    action_t と q_{t+1} を比較したい場合は
    actual_qpos が T+1 長なら actual_qpos[1:] を使う。
    """
    target_actions = np.asarray(target_actions)
    actual_qpos = np.asarray(actual_qpos)

    if target_actions.ndim != 2 or target_actions.shape[1] != 7:
        raise ValueError(f"target_actions must be (T, 7), got {target_actions.shape}")

    if actual_qpos.ndim != 2 or actual_qpos.shape[1] != 7:
        raise ValueError(f"actual_qpos must be (T or T+1, 7), got {actual_qpos.shape}")

    # action_t vs q_{t+1} を比べる
    if actual_qpos.shape[0] == target_actions.shape[0] + 1:
        qplot = actual_qpos[1:]
        xlabel = "timestep (t, comparing a_t vs q_{t+1})"
    elif actual_qpos.shape[0] == target_actions.shape[0]:
        qplot = actual_qpos
        xlabel = "timestep"
    else:
        raise ValueError(
            f"Incompatible lengths: target_actions={target_actions.shape[0]}, "
            f"actual_qpos={actual_qpos.shape[0]}"
        )

    T = target_actions.shape[0]

    fig, axes = plt.subplots(7, 1, figsize=(10, 12), sharex=True, dpi=150)

    if episode_idx is None:
        fig.suptitle(title, fontsize=13)
    else:
        fig.suptitle(f"{title} (episode idx={episode_idx})", fontsize=13)

    for j in range(7):
        ax = axes[j]
        ax.plot(
            np.arange(T),
            target_actions[:, j],
            label="target (action)" if j == 0 else None,
            linewidth=1.4,
        )
        ax.plot(
            np.arange(T),
            qplot[:, j],
            "--",
            label="actual (qpos)" if j == 0 else None,
            linewidth=1.4,
        )
        ax.set_ylabel(f"Joint {j+1}")
        ax.grid(True, alpha=0.3)

        if j == 0:
            ax.legend(loc="best")

    axes[-1].set_xlabel(xlabel)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    plt.show()