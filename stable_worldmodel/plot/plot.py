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