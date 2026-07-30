from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
import xml.etree.ElementTree as ET
import torch

from matplotlib.colors import LinearSegmentedColormap

def quat_wxyz_to_yaw(quat):
    """
    MuJoCo quaternion [w, x, y, z] -> yaw [rad]
    """
    w, x, y, z = quat
    yaw = np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    return yaw


def plot_task_result_xy(
    bluebox_traj,
    ee_traj,
    goal_pos,
    bluebox_quat_traj=None,
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
    # hx, hy = box_half_size
    # start_rect = Rectangle(
    #     (bluebox_traj[0, 1] - hy, bluebox_traj[0, 0] - hx),
    #     2 * hy,
    #     2 * hx,
    #     fill=False,
    #     linewidth=1.5,
    #     alpha=0.8,
    # )

    # # ---- bluebox final position box ----
    # final_rect = Rectangle(
    #     (bluebox_traj[-1, 1] - hy, bluebox_traj[-1, 0] - hx),
    #     2 * hy,
    #     2 * hx,
    #     fill=False,
    #     linewidth=1.8,
    #     linestyle="--",
    #     alpha=0.9,
    #     label="bluebox_final_region",
    # )

    hx, hy = box_half_size

    if bluebox_quat_traj is not None:
        start_yaw = quat_wxyz_to_yaw(bluebox_quat_traj[0])
        final_yaw = quat_wxyz_to_yaw(bluebox_quat_traj[-1])
    else:
        start_yaw = 0.0
        final_yaw = 0.0

    # matplotlib の angle は degree
    start_angle = np.rad2deg(-start_yaw)
    final_angle = np.rad2deg(-final_yaw)

    start_rect = Rectangle(
        (bluebox_traj[0, 1] - hy, bluebox_traj[0, 0] - hx),
        2 * hy,
        2 * hx,
        angle=start_angle,
        rotation_point="center",
        fill=False,
        linewidth=1.5,
        alpha=0.8,
        label="bluebox_start_region",
    )

    final_rect = Rectangle(
        (bluebox_traj[-1, 1] - hy, bluebox_traj[-1, 0] - hx),
        2 * hy,
        2 * hx,
        angle=final_angle,
        rotation_point="center",
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
    
    



def load_panda_joint_limits(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # panda default joint range
    default_joint = root.find(".//default[@class='panda']/joint")
    if default_joint is None or "range" not in default_joint.attrib:
        raise ValueError("Default panda joint range was not found.")

    default_range = np.array(
        list(map(float, default_joint.attrib["range"].split())),
        dtype=np.float32,
    )

    joint_min = []
    joint_max = []

    for i in range(1, 8):
        joint_name = f"joint{i}"
        joint = root.find(f".//joint[@name='{joint_name}']")

        if joint is not None and "range" in joint.attrib:
            r = np.array(
                list(map(float, joint.attrib["range"].split())),
                dtype=np.float32,
            )
        else:
            r = default_range

        joint_min.append(r[0])
        joint_max.append(r[1])

    return np.asarray(joint_min), np.asarray(joint_max)


def normalize_joint_to_minus1_plus1(q, joint_min, joint_max):
    q = np.asarray(q, dtype=np.float32)
    return 2.0 * (q - joint_min) / (joint_max - joint_min) - 1.0


def plot_joint_angle_comparison(
    target_actions,
    actual_qpos,
    save_path=None,
    title="Joint angle comparison",
    episode_idx=None,
    xml_path="/home/shonosukehida/work/LeWorldModel/mujoco_menagerie/franka_emika_panda/panda.xml",
    normalize=True,
):
    """
    target_actions: (T, 7)
    actual_qpos:    (T, 7) or (T+1, 7)

    normalize=True の場合,
    panda.xml の joint range を使って各jointを [-1, 1] に正規化する。
    """
    target_actions = np.asarray(target_actions)
    actual_qpos = np.asarray(actual_qpos)

    if target_actions.ndim != 2 or target_actions.shape[1] != 7:
        raise ValueError(f"target_actions must be (T, 7), got {target_actions.shape}")

    if actual_qpos.ndim != 2 or actual_qpos.shape[1] != 7:
        raise ValueError(f"actual_qpos must be (T or T+1, 7), got {actual_qpos.shape}")

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

    if normalize:
        joint_min, joint_max = load_panda_joint_limits(xml_path)
        target_actions = normalize_joint_to_minus1_plus1(
            target_actions, joint_min, joint_max
        )
        qplot = normalize_joint_to_minus1_plus1(qplot, joint_min, joint_max)
        ylabel_suffix = "normalized"
        ylim = (-1.1, 1.1)
    else:
        ylabel_suffix = "rad"
        ylim = None

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
            label="target (IK qpos)" if j == 0 else None,
            linewidth=1.4,
        )
        ax.plot(
            np.arange(T),
            qplot[:, j],
            "--",
            label="actual (qpos)" if j == 0 else None,
            linewidth=1.4,
        )

        ax.set_ylabel(f"J{j+1}\n({ylabel_suffix})")
        ax.grid(True, alpha=0.3)

        if ylim is not None:
            ax.set_ylim(*ylim)

        if j == 0:
            ax.legend(loc="best")

    axes[-1].set_xlabel(xlabel)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    plt.show()


def plot_cartesian_comparison(
    target_xyz,
    actual_ee_pos,
    save_path=None,
    title="Cartesian comparison",
    episode_idx=None,
    workspace=None,  # 👈追加
):
    target_xyz = np.asarray(target_xyz)
    actual_ee_pos = np.asarray(actual_ee_pos)

    if target_xyz.ndim != 2 or target_xyz.shape[1] != 3:
        raise ValueError(f"target_xyz must be (T, 3), got {target_xyz.shape}")

    if actual_ee_pos.ndim != 2 or actual_ee_pos.shape[1] != 3:
        raise ValueError(f"actual_ee_pos must be (T, 3), got {actual_ee_pos.shape}")

    T = target_xyz.shape[0]
    labels = ["x", "y", "z"]

    # 👇 デフォルト workspace
    if workspace is None:
        workspace = {
            "x": (0.315, 0.715),
            "y": (-0.2, 0.2),
            "z": (0.1, 0.1),
        }

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True, dpi=150)

    if episode_idx is None:
        fig.suptitle(title, fontsize=13)
    else:
        fig.suptitle(f"{title} (episode idx={episode_idx})", fontsize=13)

    for j, axis_name in enumerate(["x", "y", "z"]):
        ax = axes[j]

        # --- plot trajectory ---
        ax.plot(
            np.arange(T),
            target_xyz[:, j],
            label="target xyz" if j == 0 else None,
        )
        ax.plot(
            np.arange(T),
            actual_ee_pos[:, j],
            "--",
            label="actual ee" if j == 0 else None,
        )

        # --- workspace表示 ---
        low, high = workspace[axis_name]

        if low == high:
            # zのように固定値
            ax.axhline(low, color="green", linestyle=":", alpha=0.7, label="workspace" if j == 0 else None)
        else:
            # 範囲
            ax.axhspan(low, high, color="green", alpha=0.1)

            # 境界線
            ax.axhline(low, color="green", linestyle=":", alpha=0.7)
            ax.axhline(high, color="green", linestyle=":", alpha=0.7)

        ax.set_ylabel(axis_name)
        ax.grid(True, alpha=0.3)

        if j == 0:
            ax.legend(loc="best")

    axes[-1].set_xlabel("timestep")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    plt.show()




def plot_cem_cost_convergence(outputs, env_idx=0, save_dir=None, timestep=None):
    elite_mean = torch.stack(outputs["elite_cost_mean_n_steps"])[:, env_idx]
    elite_min = torch.stack(outputs["elite_cost_min_n_steps"])[:, env_idx]

    plt.figure()
    plt.plot(elite_mean.numpy(), label="elite mean cost")
    plt.plot(elite_min.numpy(), label="elite min cost")
    plt.xlabel("CEM iteration")
    plt.ylabel("cost")
    plt.legend()
    plt.title(f"CEM cost convergence (t={timestep})")
    
    save_dir = save_dir / "cost"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"cem_step_{timestep}"
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()




def plot_cem_sequence_transition_colormap(
    outputs,
    env_idx=0,
    save_dir=None,
    timestep=None,
    action_processor=None
):
    """
    横軸: horizon
    線: CEM iteration
    色: 白 -> firebrick で CEM iteration progression
    """

    action_mean = torch.stack(outputs["mean_n_steps"])[:, env_idx]
    action_std = torch.stack(outputs["std_n_steps"])[:, env_idx]

    # shape:
    # (n_steps, horizon, action_dim)

    n_steps, horizon, action_dim = action_mean.shape

    # =====================================
    # colormap
    # =====================================

    cmap = LinearSegmentedColormap.from_list(
        "firebrick_fade",
        ["white", "firebrick"]
    )

    # =====================================
    # mean plot
    # =====================================

    fig, axes = plt.subplots(
        action_dim,
        1,
        figsize=(8, 3 * action_dim),
        sharex=True,
    )

    if action_dim == 1:
        axes = [axes]

    for d in range(action_dim):

        for step in range(n_steps):

            color = cmap(step / (n_steps - 1))

            axes[d].plot(
                action_mean[step, :, d].numpy(),
                color=color,
                linewidth=2,
            )
            
        if (
            action_processor is not None
            and hasattr(action_processor, "normed_min_")
            and hasattr(action_processor, "normed_max_")
        ):
            ymin = float(action_processor.normed_min_.reshape(-1)[d])
            ymax = float(action_processor.normed_max_.reshape(-1)[d])

            margin = 0.05 * (ymax - ymin + 1e-8)
            axes[d].set_ylim(ymin - margin, ymax + margin)

            axes[d].axhline(ymin, color="gray", linestyle="--", linewidth=1)
            axes[d].axhline(ymax, color="gray", linestyle="--", linewidth=1)

        axes[d].set_ylabel(f"dim {d}")
        axes[d].grid(True, linestyle=":", alpha=0.5)

    axes[-1].set_xlabel("Horizon")

    fig.suptitle(f"CEM action mean sequence transition (t={timestep})")

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_dir is not None:

        mean_save_dir = save_dir / "mean_sequence_colormap"
        mean_save_dir.mkdir(parents=True, exist_ok=True)

        save_path = mean_save_dir / f"cem_mean_sequence_{timestep}.png"

        plt.savefig(save_path, bbox_inches="tight")

    else:
        plt.show()

    plt.close()

    # =====================================
    # std plot
    # =====================================

    fig, axes = plt.subplots(
        action_dim,
        1,
        figsize=(8, 3 * action_dim),
        sharex=True,
    )

    if action_dim == 1:
        axes = [axes]

    for d in range(action_dim):

        for step in range(n_steps):

            color = cmap(step / (n_steps - 1))

            axes[d].plot(
                action_std[step, :, d].numpy(),
                color=color,
                linewidth=2,
            )

        axes[d].set_ylabel(f"dim {d}")
        axes[d].grid(True, linestyle=":", alpha=0.5)

    axes[-1].set_xlabel("Horizon")

    fig.suptitle(f"CEM action std sequence transition (t={timestep})")

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_dir is not None:

        std_save_dir = save_dir / "std_sequence_colormap"
        std_save_dir.mkdir(parents=True, exist_ok=True)

        save_path = std_save_dir / f"cem_std_sequence_{timestep}.png"

        plt.savefig(save_path, bbox_inches="tight")

    else:
        plt.show()

    plt.close()



def plot_cem_rollout_cost(cem_cost_over_time, save_dir, env_idx=0):
    if len(cem_cost_over_time) == 0:
        return

    save_dir = Path(save_dir) / "cem" / "cost_over_rollout"
    save_dir.mkdir(parents=True, exist_ok=True)

    timesteps = [x["timestep"] for x in cem_cost_over_time]

    initial_mean = [x["initial_mean"] for x in cem_cost_over_time]
    final_mean   = [x["final_mean"] for x in cem_cost_over_time]
    initial_min  = [x["initial_min"] for x in cem_cost_over_time]
    final_min    = [x["final_min"] for x in cem_cost_over_time]

    # =========================
    # cost transition
    # =========================
    plt.figure(figsize=(8, 5))
    plt.plot(timesteps, initial_mean, marker="o", label="initial elite mean")
    plt.plot(timesteps, final_mean, marker="o", label="final elite mean")
    # plt.plot(timesteps, initial_min, marker="o", label="initial elite min")
    # plt.plot(timesteps, final_min, marker="o", label="final elite min")

    plt.xlabel("Environment timestep")
    plt.ylabel("CEM cost")
    plt.title(f"CEM cost over rollout (env={env_idx})")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend()
    plt.savefig(save_dir / f"cem_cost_over_rollout_env{env_idx}.png", bbox_inches="tight")
    plt.close()

    # =========================
    # improvement amount
    # =========================
    improvement_mean = [
        ini - fin for ini, fin in zip(initial_mean, final_mean)
    ]
    improvement_min = [
        ini - fin for ini, fin in zip(initial_min, final_min)
    ]

    plt.figure(figsize=(8, 5))
    plt.plot(timesteps, improvement_mean, marker="o", label="mean cost improvement")
    # plt.plot(timesteps, improvement_min, marker="o", label="min cost improvement")

    plt.xlabel("Environment timestep")
    plt.ylabel("Cost decrease")
    plt.title(f"CEM cost improvement over rollout (env={env_idx})")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend()
    plt.savefig(save_dir / f"cem_cost_improvement_env{env_idx}.png", bbox_inches="tight")
    plt.close()

    # =========================
    # improvement ratio
    # =========================
    eps = 1e-8
    improvement_ratio_mean = [
        (ini - fin) / (abs(ini) + eps)
        for ini, fin in zip(initial_mean, final_mean)
    ]
    improvement_ratio_min = [
        (ini - fin) / (abs(ini) + eps)
        for ini, fin in zip(initial_min, final_min)
    ]

    plt.figure(figsize=(8, 5))
    plt.plot(timesteps, improvement_ratio_mean, marker="o", label="mean improvement ratio")
    # plt.plot(timesteps, improvement_ratio_min, marker="o", label="min improvement ratio")

    plt.xlabel("Environment timestep")
    plt.ylabel("Improvement ratio")
    plt.title(f"CEM cost improvement ratio over rollout (env={env_idx})")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend()
    plt.savefig(save_dir / f"cem_cost_improvement_ratio_env{env_idx}.png", bbox_inches="tight")
    plt.close()
