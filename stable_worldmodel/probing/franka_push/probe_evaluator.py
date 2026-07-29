import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Callable, Sequence
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import hdf5plugin
import numpy as np
import torch
from gymnasium.vector import VectorEnv
from loguru import logger as logging
from rich import print
from tqdm import tqdm
import json

from stable_worldmodel.data.utils import get_cache_dir
from stable_worldmodel.policy import Policy

# from .wrapper import MegaWrapper, SyncWorld, VariationWrapper

from stable_worldmodel.plot.plot import plot_task_result_xy, plot_joint_angle_comparison, plot_cartesian_comparison
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from stable_worldmodel.utils import _make_env
from stable_worldmodel.probing.franka_push.plot import *



class ProbingEvaluator:
    def __init__(
        self,
        dataset,
        model,
        config = None,
        device = "cuda", 
        transform = None,
        process = None,
        plot_max_horizon = 10,
        results_path: Path | None = None,
        val_dataset = None,
        plot_all_train_data = True, 
        plot_all_val_data = False,
        plot_closed_data = True, 
        plot_open_data = True,
        closed_pred_step = 5,
        max_samples = 1000,
        plot_line = False,
        env = None,
        check_isotropy = False,
        check_shaded_images = False,
        shaded_dataset = None,
    ):
        self.dataset = dataset
        self.model = model.to(device).eval() 
        self.config = config
        self.device = device 
        self.transform = transform 
        self.process = process

        self.action_key = ""
        for col_name in self.dataset.column_names:
            if "action" in col_name:
                self.action_key = col_name
                break
        assert "action" in self.action_key, "self.action_key doesn't have 'action'"
        
        self.plot_max_horizon = plot_max_horizon
        self.results_path = results_path
        
        self.val_dataset = val_dataset
        # print("self.val_dataset:", val_dataset)
        # print(self.val_dataset.column_names)
        # print(self.val_dataset.get_col_data("pixels").shape)
        
        self.plot_all_train_data = self.config.plot_all_train_data
        
        
        self.plot_all_val_data = self.config.plot_all_val_data
        
        self.plot_closed_data = self.config.plot_closed_data 
        self.plot_open_data = self.config.plot_open_data 
        
        self.closed_pred_step = self.config.closed_pred_step
        self.max_samples = self.config.max_samples 
        # print("self.max_samples: ", self.max_samples)
        
        self.plot_line = self.config.plot_line
        # print("self.plot_line:", plot_line)
        
        self.env = env 
        # print("self.env:", self.env)
        
        self.check_isotropy = self.config.check_isotropy
        
        self.shaded_dataset = shaded_dataset
        
        
        with h5py.File(self.dataset.h5_path, "r") as f:
            self.x_range = tuple(f.attrs["x_range"])
            self.y_range = tuple(f.attrs["y_range"])
            self.z_range = tuple(f.attrs["z_range"])
        
        print("stable_worldmodel/probing/probe_evaluator.py")
        print("self.x_range:", self.x_range)
        print("self.y_range:", self.y_range)
        print("self.z_range:", self.z_range)
        
        
        
        
        

    @torch.no_grad()
    def collect_frame_latents(
        self,
        max_samples=1000,
        pixel_key="pixels",
        target_keys=("bluebox_pos", "ee_pos", "qpos", "qvel"),
        is_val = False,
    ):
        latents = []
        targets = {k: [] for k in target_keys}
        
        dataset = self._get_dataset(is_val)

        n = min(max_samples, len(dataset))

        for idx in tqdm(range(n), desc="Collecting latents"):
            sample = dataset[idx]

            if pixel_key not in sample:
                raise KeyError(f"{pixel_key} not found in sample keys: {sample.keys()}")

            pixels = sample[pixel_key]  # (T, C, H, W)

            # まずは最後の観測だけ使う
            if pixels.ndim == 4:
                pixels = pixels[-1]  # (C, H, W)

            # transform が必要なら適用
            if self.transform is not None:
                pixels = self.transform[pixel_key](pixels)

            pixels = pixels.unsqueeze(0).to(self.device)  # (1, C, H, W)

            z = self._encode_pixels(pixels)  # (1, D)
            latents.append(z.squeeze(0).cpu().numpy())

            for k in target_keys:
                if k not in sample:
                    continue

                v = sample[k]
                if torch.is_tensor(v):
                    if v.ndim >= 2:
                        v = v[-1]
                    v = v.cpu().numpy()
                else:
                    v = np.asarray(v)
                    if v.ndim >= 2:
                        v = v[-1]

                targets[k].append(v)

        latents = np.stack(latents, axis=0)

        targets = {
            k: np.stack(v, axis=0)
            for k, v in targets.items()
            if len(v) > 0
        }

        return {
            "latents": latents,
            "targets": targets,
        }


    @torch.no_grad()
    def collect_open_rollout_latents(
        self,
        max_horizon=100,
        pixel_key="pixels",
        action_key="action",
        target_keys=("bluebox_pos", "ee_pos", "qpos", "qvel"),
        is_val = False,

    ):
        true_list = []
        pred_list = []
        action_list = []
        targets = {k: [] for k in target_keys}
        
        dataset = self._get_dataset(is_val)

        n = min(max_horizon, len(dataset))
        z_cur = None

        for idx in tqdm(range(n), desc="Collecting rollout latents"):
            sample = dataset[idx]

            pixels = sample[pixel_key]      # (1, C, H, W)
            actions = sample[action_key]    # (1, A)

            # action 正規化
            if self.process is not None and action_key in self.process:
                actions_np = actions.cpu().numpy() if torch.is_tensor(actions) else np.asarray(actions)
                actions_np = self.process[action_key].transform(actions_np)
                actions = torch.from_numpy(actions_np).float()
            else:
                actions = actions.float() if torch.is_tensor(actions) else torch.from_numpy(np.asarray(actions)).float()

            if self.transform is not None:
                pixels = torch.stack([self.transform[pixel_key](p) for p in pixels], dim=0)

            pixels = pixels.to(self.device)
            actions = actions.to(self.device)

            true_z = self._encode_pixels(pixels)  # (1, D)
            true_list.append(true_z.squeeze(0).cpu().numpy())
            action_list.append(actions.squeeze(0).cpu().numpy())

            # 初期値だけ真の encoder latent
            if z_cur is None:
                z_cur = true_z                  # (1, D)
                pred_list.append(z_cur.squeeze(0).cpu().numpy())

            # action_t で pred_z_{t+1} を作る
            a_t = actions.unsqueeze(0)          # (1, 1, A)

            if hasattr(self.model, "action_encoder"):
                a_emb = self.model.action_encoder(a_t)  # (1, 1, 192)
            else:
                a_emb = a_t

            z_next = self.model.predictor(
                z_cur.unsqueeze(0),  # (1, 1, D)
                a_emb,               # (1, 1, D)
            )

            while z_next.ndim > 2:
                z_next = z_next.squeeze(0)      # (1, D)

            if hasattr(self.model, "pred_proj"):
                z_next = self.model.pred_proj(z_next)

            pred_list.append(z_next.squeeze(0).cpu().numpy())

            # 次stepは予測latentから進める
            z_cur = z_next

            for k in target_keys:
                if k in sample:
                    v = sample[k]
                    if torch.is_tensor(v):
                        v = v.squeeze(0).cpu().numpy()
                    else:
                        v = np.asarray(v).squeeze(0)
                    targets[k].append(v)

        true_z = np.stack(true_list, axis=0)       # (N, D)
        pred_z = np.stack(pred_list, axis=0)       # (N+1, D)
        actions = np.stack(action_list, axis=0)    # (N, A)

        targets = {
            k: np.stack(v, axis=0)
            for k, v in targets.items()
            if len(v) > 0
        }

        return {
            "true_z": true_z,
            "pred_z": pred_z,
            "actions": actions,
            "targets": targets,
        }
        

    @torch.no_grad()
    def collect_closed_rollout_latents(
        self,
        start_idx=0,
        max_horizon=100,
        pixel_key="pixels",
        action_key="action",
        target_keys=("bluebox_pos", "ee_pos", "qpos", "qvel"),
        is_val = False,
    ):
        true_list = []
        pred_list = []
        action_list = []
        targets = {k: [] for k in target_keys}

        z_cur = None
        
        dataset = self._get_dataset(is_val)

        for idx in tqdm(range(start_idx, start_idx + max_horizon), desc="Collecting AR rollout latents"):
            sample = dataset[idx]

            pixels = sample[pixel_key]      # (1, C, H, W)
            actions = sample[action_key]    # (1, A)

            # image transform
            if self.transform is not None:
                pixels = torch.stack([self.transform[pixel_key](p) for p in pixels], dim=0)

            pixels = pixels.to(self.device)


            true_z = self._encode_pixels(pixels)  # (1, D)
            true_list.append(true_z.squeeze(0).cpu().numpy())


            if z_cur is None:
                z_cur = true_z  # (1, D)
                pred_list.append(z_cur.squeeze(0).cpu().numpy())

            # action normalization
            if self.process is not None and action_key in self.process:
                actions_np = actions.cpu().numpy() if torch.is_tensor(actions) else np.asarray(actions)
                actions_np = self.process[action_key].transform(actions_np)
                actions = torch.from_numpy(actions_np).float()
            else:
                actions = actions.float() if torch.is_tensor(actions) else torch.from_numpy(np.asarray(actions)).float()

            actions = actions.to(self.device)
            action_list.append(actions.squeeze(0).cpu().numpy())

            # action encoding
            a_t = actions.unsqueeze(0)  # (1, 1, A)

            if hasattr(self.model, "action_encoder"):
                a_emb = self.model.action_encoder(a_t)  # (1, 1, 192)
            else:
                a_emb = a_t

            # autoregressive prediction
            z_next = self.model.predictor(
                z_cur.unsqueeze(0),  # (1, 1, D)
                a_emb,               # (1, 1, D)
            )

            while z_next.ndim > 2:
                z_next = z_next.squeeze(0)  # (1, D)

            if hasattr(self.model, "pred_proj"):
                z_next = self.model.pred_proj(z_next)

            pred_list.append(z_next.squeeze(0).cpu().numpy())

            #autoregressive
            z_cur = z_next

            for k in target_keys:
                if k in sample:
                    v = sample[k]
                    if torch.is_tensor(v):
                        v = v.squeeze(0).cpu().numpy()
                    else:
                        v = np.asarray(v).squeeze(0)
                    targets[k].append(v)

        true_z = np.stack(true_list, axis=0)     # (H, D)
        pred_z = np.stack(pred_list, axis=0)     # (H+1, D)
        actions = np.stack(action_list, axis=0)  # (H, A)

        targets = {
            k: np.stack(v, axis=0)
            for k, v in targets.items()
            if len(v) > 0
        }

        return {
            "true_z": true_z,
            "pred_z": pred_z,
            "actions": actions,
            "targets": targets,
        }
        
        
    @torch.no_grad()
    def collect_dataset_one_step_latents(
        self,
        max_samples=None,
        pixel_key="pixels",
        action_key="action",
        is_val = False,
    ):
        true_cur_list = []
        true_next_list = []
        pred_next_list = []
        action_list = []

        dataset = self._get_dataset(is_val)
        
        n_total = len(dataset) - 1
        n = n_total if max_samples is None else min(max_samples, n_total)
        

        for idx in tqdm(range(n), desc="Collecting dataset one-step dynamics"):
            sample_t = dataset[idx]
            sample_tp1 = dataset[idx + 1]

            pixels_t = sample_t[pixel_key]       # (1, C, H, W) or (C, H, W)
            pixels_tp1 = sample_tp1[pixel_key]   # (1, C, H, W) or (C, H, W)
            actions = sample_t[action_key]       # (1, A) or (A,)

            # -------------------------
            # pixel shape adjustment
            # -------------------------
            if pixels_t.ndim == 4:
                pixels_t = pixels_t[-1]
            if pixels_tp1.ndim == 4:
                pixels_tp1 = pixels_tp1[-1]

            if self.transform is not None:
                pixels_t = self.transform[pixel_key](pixels_t)
                pixels_tp1 = self.transform[pixel_key](pixels_tp1)

            pixels_t = pixels_t.unsqueeze(0).to(self.device)       # (1, C, H, W)
            pixels_tp1 = pixels_tp1.unsqueeze(0).to(self.device)   # (1, C, H, W)

            true_cur_z = self._encode_pixels(pixels_t)       # (1, D)
            true_next_z = self._encode_pixels(pixels_tp1)    # (1, D)

            # -------------------------
            # action
            # -------------------------
            if torch.is_tensor(actions):
                actions_np = actions.cpu().numpy()
            else:
                actions_np = np.asarray(actions)

            if actions_np.ndim == 1:
                actions_np = actions_np[None, :]   # (1, A)

            if self.process is not None and action_key in self.process:
                actions_np = self.process[action_key].transform(actions_np)

            actions = torch.from_numpy(actions_np).float().to(self.device)  # (1, A)

            a_in = actions.unsqueeze(0)          # (1, 1, A)
            z_in = true_cur_z.unsqueeze(0)       # (1, 1, D)

            if hasattr(self.model, "action_encoder"):
                a_in = self.model.action_encoder(a_in)

            pred_next_z = self.model.predictor(z_in, a_in)

            while pred_next_z.ndim > 2:
                pred_next_z = pred_next_z.squeeze(0)

            if hasattr(self.model, "pred_proj"):
                pred_next_z = self.model.pred_proj(pred_next_z)

            true_cur_list.append(true_cur_z.squeeze(0).cpu().numpy())
            true_next_list.append(true_next_z.squeeze(0).cpu().numpy())
            pred_next_list.append(pred_next_z.squeeze(0).cpu().numpy())
            action_list.append(actions.squeeze(0).cpu().numpy())

        if len(true_cur_list) == 0:
            raise RuntimeError("No valid one-step samples were collected. Check pixel/action shapes.")

        return {
            "true_cur_z": np.stack(true_cur_list, axis=0),
            "true_z": np.stack(true_next_list, axis=0),
            "pred_z": np.stack(pred_next_list, axis=0),
            "actions": np.stack(action_list, axis=0),
        }


    def _encode_pixels(self, pixels):
        """
        pixels: (B, C, H, W)
        return: (B, D)
        """
        z = self.model.encoder(pixels, interpolate_pos_encoding=True,)

        # HuggingFace ViTModel の場合
        if hasattr(z, "last_hidden_state"):
            z = z.last_hidden_state[:, 0]

        # LeWM は projector を通した embedding を使うのが自然
        if hasattr(self.model, "projector"):
            z = self.model.projector(z)

        return z

    @torch.no_grad()
    def collect_dataset_closed_rollout_latents(
        self,
        pred_step=5,
        max_samples=None,
        pixel_key="pixels",
        action_key="action",
        is_val=False,
    ):
        """
        Dataset 全体から closed-loop latent rollout を収集する.

        Returns:
            true_z:        (N, pred_step + 1, D)
            pred_z:        (N, pred_step + 1, D)
            actions:       (N, pred_step, A)
            start_indices: (N,)
        """
        true_rollouts = []
        pred_rollouts = []
        action_rollouts = []
        start_indices = []

        dataset = self._get_dataset(is_val)

        # idx + pred_step まで見るので, 最後 pred_step 個は start に使えない
        n_total = len(dataset) - pred_step
        if n_total <= 0:
            raise RuntimeError(
                f"Dataset too short for pred_step={pred_step}. "
                f"len(dataset)={len(dataset)}"
            )

        n = n_total if max_samples is None else min(max_samples, n_total)

        for start_idx in tqdm(range(n), desc="Collecting dataset closed-loop dynamics"):
            true_z_seq = []
            pred_z_seq = []
            action_seq = []

            # -------------------------
            # initial true latent z_t
            # -------------------------
            sample0 = dataset[start_idx]
            pixels0 = sample0[pixel_key]

            if pixels0.ndim == 4:
                pixels0 = pixels0[-1]  # (C, H, W)

            if self.transform is not None:
                pixels0 = self.transform[pixel_key](pixels0)

            pixels0 = pixels0.unsqueeze(0).to(self.device)  # (1, C, H, W)

            z_cur = self._encode_pixels(pixels0)  # (1, D)

            # pred_z の初期値は true z_t に揃える
            pred_z_seq.append(z_cur.squeeze(0).cpu().numpy())

            # true_z も z_t から入れる
            true_z_seq.append(z_cur.squeeze(0).cpu().numpy())

            # -------------------------
            # closed-loop rollout
            # -------------------------
            for h in range(pred_step):
                idx = start_idx + h
                sample_t = dataset[idx]
                sample_tp1 = dataset[idx + 1]

                # ---- true next latent z_{t+h+1} ----
                pixels_tp1 = sample_tp1[pixel_key]

                if pixels_tp1.ndim == 4:
                    pixels_tp1 = pixels_tp1[-1]  # (C, H, W)

                if self.transform is not None:
                    pixels_tp1 = self.transform[pixel_key](pixels_tp1)

                pixels_tp1 = pixels_tp1.unsqueeze(0).to(self.device)
                true_next_z = self._encode_pixels(pixels_tp1)  # (1, D)
                true_z_seq.append(true_next_z.squeeze(0).cpu().numpy())

                # ---- action a_{t+h} ----
                actions = sample_t[action_key]

                if torch.is_tensor(actions):
                    actions_np = actions.cpu().numpy()
                else:
                    actions_np = np.asarray(actions)

                if actions_np.ndim == 1:
                    actions_np = actions_np[None, :]  # (1, A)

                if self.process is not None and action_key in self.process:
                    actions_np = self.process[action_key].transform(actions_np)

                actions = torch.from_numpy(actions_np).float().to(self.device)  # (1, A)
                action_seq.append(actions.squeeze(0).cpu().numpy())

                a_in = actions.unsqueeze(0)       # (1, 1, A)
                z_in = z_cur.unsqueeze(0)         # (1, 1, D)

                if hasattr(self.model, "action_encoder"):
                    a_in = self.model.action_encoder(a_in)

                pred_next_z = self.model.predictor(z_in, a_in)

                while pred_next_z.ndim > 2:
                    pred_next_z = pred_next_z.squeeze(0)

                if hasattr(self.model, "pred_proj"):
                    pred_next_z = self.model.pred_proj(pred_next_z)

                pred_z_seq.append(pred_next_z.squeeze(0).cpu().numpy())

                # ここが closed-loop の本質:
                # 次 step の入力に encoder true_z ではなく predictor 出力を使う
                z_cur = pred_next_z

            true_rollouts.append(np.stack(true_z_seq, axis=0))      # (H+1, D)
            pred_rollouts.append(np.stack(pred_z_seq, axis=0))      # (H+1, D)
            action_rollouts.append(np.stack(action_seq, axis=0))    # (H, A)
            start_indices.append(start_idx)

        if len(true_rollouts) == 0:
            raise RuntimeError("No valid closed-loop samples were collected.")

        return {
            "true_z": np.stack(true_rollouts, axis=0),       # (N, H+1, D)
            "pred_z": np.stack(pred_rollouts, axis=0),       # (N, H+1, D)
            "actions": np.stack(action_rollouts, axis=0),    # (N, H, A)
            "start_indices": np.array(start_indices),        # (N,)
        }


    def _get_dataset(self, is_val=False):
        if is_val:
            if self.val_dataset is None:
                raise ValueError("is_val=True but val_dataset is None")
            return self.val_dataset
        return self.dataset

    @torch.no_grad()
    def render_state_image(
        self,
        ee_pos,
        bluebox_pos,
        image_size=(64, 64),
        reset=True,
    ):
        """
        指定した EE位置 / bluebox位置 の状態を env 上に構築し,
        観測画像を返す.

        Args:
            ee_pos:       (3,)
            bluebox_pos:  (3,)
            image_size:   render size
            reset:        毎回 physics.reset() するか

        Returns:
            image: (H, W, C) uint8
        """

        if self.env is None:
            raise ValueError("self.env is None")

        ee_pos = np.asarray(ee_pos, dtype=np.float32)
        bluebox_pos = np.asarray(bluebox_pos, dtype=np.float32)

        if reset:
            self.env.physics.reset()

        # ----------------------------------------
        # bluebox 配置
        # ----------------------------------------
        self.env.reset_and_place_all(
            box_pos=bluebox_pos,
            init_ee_pos=ee_pos,
        )

        self.env.physics.forward()

        # ----------------------------------------
        # render
        # ----------------------------------------
        img = self.env.render_image(size=image_size)

        return img


    @torch.no_grad()
    def encode_rendered_image(
        self,
        img,
        pixel_key="pixels",
    ):
        """
        envでrenderした画像 (H,W,C) を encoder latent に変換する.
        """
        pixels = torch.from_numpy(img.copy())

        # HWC -> CHW
        if pixels.ndim == 3 and pixels.shape[-1] == 3:
            pixels = pixels.permute(2, 0, 1)

        if self.transform is not None:
            pixels = self.transform[pixel_key](pixels)

        pixels = pixels.unsqueeze(0).to(self.device)  # (1,C,H,W)
        z = self._encode_pixels(pixels)               # (1,D)
        return z



    @torch.no_grad()
    def collect_center_direction_rollouts(
        self,
        pred_step=5,
        step_size=0.01,
        action_key=None,
        pixel_key="pixels",
        image_size=(64, 64),
        x_range=(0.315, 0.715),
        y_range=(-0.2, 0.2),
        z_value=0.1,
        bluebox_pos=None,
    ):
        """
        作業領域中心を初期観測として,
        +x, -x, +y, -y の4方向に直進する action 列を作り,
        predictor を closed-loop で rollout する.

        Returns:
            pred_z:   (4, pred_step+1, D)
            actions:  (4, pred_step, A)  # 正規化後 action
            xyz_seq:  (4, pred_step, 3)  # 正規化前 target xyz
            labels:   ["+x", "-x", "+y", "-y"]
            init_img: (H,W,C)
        """
        if self.env is None:
            raise ValueError("self.env is None. Please pass env to ProbingEvaluator.")

        if action_key is None:
            action_key = self.action_key

        center = np.array(
            [
                (x_range[0] + x_range[1]) / 2.0,
                (y_range[0] + y_range[1]) / 2.0,
                z_value,
            ],
            dtype=np.float32,
        )

        # bluebox位置を指定しない場合は, とりあえず中心付近に置く
        if bluebox_pos is None:
            bluebox_pos = center.copy()
        bluebox_pos = np.asarray(bluebox_pos, dtype=np.float32)

        # ------------------------------------------------------------
        # 初期観測: EEを作業領域中心, blueboxを指定位置に置いて render
        # ------------------------------------------------------------
        init_img = self.render_state_image(
            ee_pos=center,
            bluebox_pos=bluebox_pos,
            image_size=image_size,
            reset=True,
        )

        z0 = self.encode_rendered_image(
            init_img,
            pixel_key=pixel_key,
        )  # (1,D)

        directions = {
            "+x": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "-x": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            "+y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "-y": np.array([0.0, -1.0, 0.0], dtype=np.float32),
        }

        pred_rollouts = []
        action_rollouts = []
        xyz_rollouts = []
        labels = []

        for label, direction in directions.items():
            z_cur = z0

            z_seq = [z_cur.squeeze(0).cpu().numpy()]
            action_seq = []
            xyz_seq = []

            for t in range(pred_step):
                # 絶対座標 action: center から direction 方向へ直進
                target_xyz = center + direction * step_size * (t + 1)

                target_xyz[0] = np.clip(target_xyz[0], x_range[0], x_range[1])
                target_xyz[1] = np.clip(target_xyz[1], y_range[0], y_range[1])
                target_xyz[2] = z_value

                if action_key == "action_cartesian":
                    action_np = target_xyz[None, :].astype(np.float32)  # (1,3)
                else:
                    ik_result = self.env.calc_inverse_kinematic(target_xyz)

                    if not ik_result.success:
                        raise RuntimeError(f"IK failed at label={label}, t={t}, target_xyz={target_xyz}")

                    joint_action = ik_result.qpos[:7].astype(np.float32)

                    action_np = joint_action[None, :]   # (1,7)

                # 学習時と同じ action 正規化
                if self.process is not None and action_key in self.process:
                    action_np_norm = self.process[action_key].transform(action_np)
                else:
                    action_np_norm = action_np

                action = torch.from_numpy(action_np_norm).float().to(self.device)  # (1,A)

                a_in = action.unsqueeze(0)  # (1,1,A)

                if hasattr(self.model, "action_encoder"):
                    a_in = self.model.action_encoder(a_in)

                z_in = z_cur.unsqueeze(0)  # (1,1,D)

                z_next = self.model.predictor(z_in, a_in)

                while z_next.ndim > 2:
                    z_next = z_next.squeeze(0)

                if hasattr(self.model, "pred_proj"):
                    z_next = self.model.pred_proj(z_next)

                z_seq.append(z_next.squeeze(0).cpu().numpy())
                action_seq.append(action.squeeze(0).cpu().numpy())
                xyz_seq.append(target_xyz.copy())

                # closed-loop: 次の入力は predictor 出力
                z_cur = z_next

            pred_rollouts.append(np.stack(z_seq, axis=0))
            action_rollouts.append(np.stack(action_seq, axis=0))
            xyz_rollouts.append(np.stack(xyz_seq, axis=0))
            labels.append(label)

        return {
            "pred_z": np.stack(pred_rollouts, axis=0),       # (4,T+1,D)
            "actions": np.stack(action_rollouts, axis=0),    # (4,T,A)
            "xyz_seq": np.stack(xyz_rollouts, axis=0),       # (4,T,3)
            "labels": labels,
            "center": center,
            "bluebox_pos": bluebox_pos,
            "init_img": init_img,
        }

    @torch.no_grad()
    def evaluate_encoder_isotropy(
        self,
        max_samples=1000,
        pixel_key="pixels",
        is_val=False,
        save_name="encoder_latent",
    ):
        data = self.collect_frame_latents(
            max_samples=max_samples,
            pixel_key=pixel_key,
            target_keys=(),
            is_val=is_val,
        )

        latents = data["latents"]  # (B, D)

        save_dir = None
        if self.results_path is not None:
            save_dir = self.results_path / "probing" / "isotropy"

        metrics = analyze_latent_isotropy(
            latents,
            save_dir=save_dir,
            prefix=save_name,
        )

        print("[Latent isotropy metrics]")
        for k, v in metrics.items():
            print(f"{k}: {v}")

        return metrics


    @torch.no_grad()
    def collect_shaded_dataset_latents(
        self,
        max_samples=None,
        pixel_key="pixels",
        target_keys=("bluebox_pos", "ee_pos", "qpos", "qvel", "label"),
    ):
        if self.shaded_dataset is None:
            raise ValueError("self.shade_check_dataset is None")

        dataset = self.shaded_dataset

        latents = []
        targets = {k: [] for k in target_keys}
        print(targets.keys())

        n = len(dataset) if max_samples is None else min(max_samples, len(dataset))

        for idx in tqdm(range(n), desc="Collecting shaded dataset latents"):
            sample = dataset[idx]

            if pixel_key not in sample:
                raise KeyError(f"{pixel_key} not found in sample keys: {sample.keys()}")

            pixels = sample[pixel_key]

            # (T,C,H,W) の場合は最後のフレームを使う
            if pixels.ndim == 4:
                pixels = pixels[-1]

            if self.transform is not None:
                pixels = self.transform[pixel_key](pixels)

            pixels = pixels.unsqueeze(0).to(self.device)

            z = self._encode_pixels(pixels)  # (1,D)
            latents.append(z.squeeze(0).cpu().numpy())

            for k in target_keys:
                if k not in sample:
                    continue

                v = sample[k]

                if torch.is_tensor(v):
                    v = v.cpu().numpy()
                else:
                    v = np.asarray(v)

                if v.ndim >= 2:
                    v = v[-1]

                targets[k].append(v)

        latents = np.stack(latents, axis=0)

        targets = {
            k: np.stack(v, axis=0)
            for k, v in targets.items()
            if len(v) > 0
        }

        return {
            "latents": latents,
            "targets": targets,
        }



    @torch.no_grad()
    def collect_ee_goal_cost_map(
        self,
        goal_ee_pos,
        goal_bluebox_pos,
        fixed_bluebox_pos=None,
        x_range=(0.45, 0.85),
        y_range=(-0.20, 0.20),
        z_value=(0.10, 0.10),
        num_x=50,
        num_y=50,
        image_size=(64, 64),
        pixel_key="pixels",
        cost_type="mse",
    ):
        """
        EE位置をグリッド状に変えたときの goal latent との距離を計算する.

        Args:
            goal_ee_pos:       goal画像におけるEE位置 (3,)
            goal_bluebox_pos:  goal画像におけるbox位置 (3,)
            fixed_bluebox_pos: cost map計算中に固定するbox位置 (3,)
                            Noneなら goal_bluebox_pos を使う
            x_range:           EEのX探索範囲
            y_range:           EEのY探索範囲
            z_value:           EEのZ固定値
            num_x:             X方向のグリッド数
            num_y:             Y方向のグリッド数
            image_size:        render画像サイズ
            pixel_key:         transform用の画像key
            cost_type:         "mse" or "l2"

        Returns:
            dict:
                cost_map: (num_x, num_y)
                xs:       (num_x,)
                ys:       (num_y,)
        """

        if self.env is None:
            raise ValueError("self.env is None. Please pass env to ProbingEvaluator.")

        goal_ee_pos = np.asarray(goal_ee_pos, dtype=np.float32)
        goal_bluebox_pos = np.asarray(goal_bluebox_pos, dtype=np.float32)

        if fixed_bluebox_pos is None:
            fixed_bluebox_pos = goal_bluebox_pos
        fixed_bluebox_pos = np.asarray(fixed_bluebox_pos, dtype=np.float32)

        xs = np.linspace(x_range[0], x_range[1], num_x, dtype=np.float32)
        ys = np.linspace(y_range[0], y_range[1], num_y, dtype=np.float32)

        # -------------------------
        # goal latent
        # -------------------------
        goal_img = self.render_state_image(
            ee_pos=goal_ee_pos,
            bluebox_pos=goal_bluebox_pos,
            image_size=image_size,
            reset=True,
        )

        z_goal = self.encode_rendered_image(
            goal_img,
            pixel_key=pixel_key,
        )  # (1, D)

        cost_map = np.zeros((num_x, num_y), dtype=np.float32)

        # 必要なら確認用に最小位置も記録
        min_cost = float("inf")
        min_pos = None

        # -------------------------
        # EE grid scan
        # -------------------------
        for ix, x in enumerate(tqdm(xs, desc="Collecting EE goal cost map")):
            for iy, y in enumerate(ys):
                ee_pos = np.array([x, y, z_value[0]], dtype=np.float32)

                img = self.render_state_image(
                    ee_pos=ee_pos,
                    bluebox_pos=fixed_bluebox_pos,
                    image_size=image_size,
                    reset=True,
                )

                z = self.encode_rendered_image(
                    img,
                    pixel_key=pixel_key,
                )  # (1, D)

                diff = z - z_goal

                if cost_type == "mse":
                    cost = diff.pow(2).mean().item()
                elif cost_type == "l2":
                    cost = diff.pow(2).sum().sqrt().item()
                else:
                    raise ValueError(f"Unknown cost_type: {cost_type}")

                cost_map[ix, iy] = cost

                if cost < min_cost:
                    min_cost = cost
                    min_pos = ee_pos.copy()

        return {
            "cost_map": cost_map,
            "xs": xs,
            "ys": ys,
            "goal_ee_pos": goal_ee_pos,
            "goal_bluebox_pos": goal_bluebox_pos,
            "fixed_bluebox_pos": fixed_bluebox_pos,
            "min_cost": min_cost,
            "min_pos": min_pos,
            "goal_img": goal_img,
            "cost_type": cost_type,
        }


    @torch.no_grad()
    def collect_box_goal_cost_map(
        self,
        goal_ee_pos,
        goal_bluebox_pos,
        fixed_ee_pos=None,
        x_range=(0.45, 0.85),
        y_range=(-0.20, 0.20),
        z_value=(0.05, 0.05),
        num_x=50,
        num_y=50,
        image_size=(64, 64),
        pixel_key="pixels",
        cost_type="mse",
    ):
        """
        box位置をグリッド状に変えたときの goal latent との距離を計算する.
        EE位置は fixed_ee_pos に固定する.
        """

        if self.env is None:
            raise ValueError("self.env is None. Please pass env to ProbingEvaluator.")

        goal_ee_pos = np.asarray(goal_ee_pos, dtype=np.float32)
        goal_bluebox_pos = np.asarray(goal_bluebox_pos, dtype=np.float32)

        if fixed_ee_pos is None:
            fixed_ee_pos = goal_ee_pos
        fixed_ee_pos = np.asarray(fixed_ee_pos, dtype=np.float32)

        xs = np.linspace(x_range[0], x_range[1], num_x, dtype=np.float32)
        ys = np.linspace(y_range[0], y_range[1], num_y, dtype=np.float32)

        goal_img = self.render_state_image(
            ee_pos=goal_ee_pos,
            bluebox_pos=goal_bluebox_pos,
            image_size=image_size,
            reset=True,
        )

        z_goal = self.encode_rendered_image(
            goal_img,
            pixel_key=pixel_key,
        )

        cost_map = np.zeros((num_x, num_y), dtype=np.float32)

        min_cost = float("inf")
        min_pos = None

        for ix, x in enumerate(tqdm(xs, desc="Collecting box goal cost map")):
            for iy, y in enumerate(ys):
                box_pos = np.array([x, y, z_value[0]], dtype=np.float32)

                img = self.render_state_image(
                    ee_pos=fixed_ee_pos,
                    bluebox_pos=box_pos,
                    image_size=image_size,
                    reset=True,
                )

                z = self.encode_rendered_image(
                    img,
                    pixel_key=pixel_key,
                )

                diff = z - z_goal

                if cost_type == "mse":
                    cost = diff.pow(2).mean().item()
                elif cost_type == "l2":
                    cost = diff.pow(2).sum().sqrt().item()
                else:
                    raise ValueError(f"Unknown cost_type: {cost_type}")

                cost_map[ix, iy] = cost

                if cost < min_cost:
                    min_cost = cost
                    min_pos = box_pos.copy()

        return {
            "cost_map": cost_map,
            "xs": xs,
            "ys": ys,
            "goal_ee_pos": goal_ee_pos,
            "goal_bluebox_pos": goal_bluebox_pos,
            "fixed_ee_pos": fixed_ee_pos,
            "min_cost": min_cost,
            "min_pos": min_pos,
            "goal_img": goal_img,
            "cost_type": cost_type,
        }


    @torch.no_grad()
    def collect_relative_pair_goal_cost_map(
        self,
        goal_ee_pos,
        goal_bluebox_pos,
        x_range=(0.45, 0.85),
        y_range=(-0.20, 0.20),
        z_value=0.05,
        num_x=50,
        num_y=50,
        image_size=(64, 64),
        pixel_key="pixels",
        cost_type="mse",
    ):
        """
        goal時の EE -> box の相対位置を保ったまま、
        EE位置をグリッド状に動かし、goal latent との距離を計算する。

        この場合:
            delta = [0.0, 0.15, 0.0]
            box_pos = ee_pos + delta
        """

        if self.env is None:
            raise ValueError("self.env is None. Please pass env to ProbingEvaluator.")

        goal_ee_pos = np.asarray(goal_ee_pos, dtype=np.float32)
        goal_bluebox_pos = np.asarray(goal_bluebox_pos, dtype=np.float32)

        delta = goal_bluebox_pos - goal_ee_pos

        xs = np.linspace(x_range[0], x_range[1], num_x, dtype=np.float32)
        ys = np.linspace(y_range[0], y_range[1], num_y, dtype=np.float32)

        goal_img = self.render_state_image(
            ee_pos=goal_ee_pos,
            bluebox_pos=goal_bluebox_pos,
            image_size=image_size,
            reset=True,
        )

        z_goal = self.encode_rendered_image(
            goal_img,
            pixel_key=pixel_key,
        )

        cost_map = np.full((num_x, num_y), np.nan, dtype=np.float32)

        min_cost = float("inf")
        min_ee_pos = None
        min_box_pos = None

        for ix, x in enumerate(tqdm(xs, desc="Collecting relative pair goal cost map")):
            for iy, y in enumerate(ys):
                ee_pos = np.array([x, y, z_value], dtype=np.float32)
                box_pos = ee_pos + delta
                box_pos[2] = goal_bluebox_pos[2]


                if not (x_range[0] <= box_pos[0] <= x_range[1]):
                    continue
                if not (y_range[0] <= box_pos[1] <= y_range[1]):
                    continue

                img = self.render_state_image(
                    ee_pos=ee_pos,
                    bluebox_pos=box_pos,
                    image_size=image_size,
                    reset=True,
                )

                z = self.encode_rendered_image(
                    img,
                    pixel_key=pixel_key,
                )

                diff = z - z_goal

                if cost_type == "mse":
                    cost = diff.pow(2).mean().item()
                elif cost_type == "l2":
                    cost = diff.pow(2).sum().sqrt().item()
                else:
                    raise ValueError(f"Unknown cost_type: {cost_type}")

                cost_map[ix, iy] = cost

                if cost < min_cost:
                    min_cost = cost
                    min_ee_pos = ee_pos.copy()
                    min_box_pos = box_pos.copy()

        return {
            "cost_map": cost_map,
            "xs": xs,
            "ys": ys,
            "goal_ee_pos": goal_ee_pos,
            "goal_bluebox_pos": goal_bluebox_pos,
            "delta": delta,
            "min_cost": min_cost,
            "min_ee_pos": min_ee_pos,
            "min_box_pos": min_box_pos,
            "goal_img": goal_img,
            "cost_type": cost_type,
        }



    @torch.no_grad()
    def _encode_pixels_patch_map(
        self,
        pixels,
        pixel_key="pixels",
        layer_idx=None,
    ):
        """
        pixels: (B,C,H,W)
        return:
            feat_map: (B,C,h,w)
        """
        out = self.model.encoder(
            pixels,
            interpolate_pos_encoding=True,
            output_hidden_states=True,
            return_dict=True,
        )

        if layer_idx is None:
            tokens = out.last_hidden_state          # (B, 1+N, D)
        else:
            tokens = out.hidden_states[layer_idx]   # (B, 1+N, D)

        # CLS token を除く
        patch_tokens = tokens[:, 1:, :]             # (B,N,D)

        B, N, D = patch_tokens.shape
        h = w = int(np.sqrt(N))

        if h * w != N:
            raise ValueError(f"patch token number is not square: N={N}")

        feat_map = patch_tokens.reshape(B, h, w, D).permute(0, 3, 1, 2)
        return feat_map.contiguous()



    @torch.no_grad()
    def collect_action_predictor_goal_cost_map(
        self,
        init_ee_pos,
        init_bluebox_pos,
        goal_ee_pos,
        goal_bluebox_pos,
        x_range=(0.45, 0.85),
        y_range=(-0.20, 0.20),
        z_value=0.05,
        num_x=50,
        num_y=50,
        image_size=(64, 64),
        pixel_key="pixels",
        action_key=None,
        cost_type="l2",
    ):
        """
        初期状態 z_t から、各絶対位置 action=[x,y,z] を predictor に入力し、
        predicted next latent と goal latent の距離を action 座標上に可視化するための cost_map を集める。
        """

        if self.env is None:
            raise ValueError("self.env is None. Please pass env to ProbingEvaluator.")

        if action_key is None:
            action_key = self.action_key

        init_ee_pos = np.asarray(init_ee_pos, dtype=np.float32)
        init_bluebox_pos = np.asarray(init_bluebox_pos, dtype=np.float32)
        goal_ee_pos = np.asarray(goal_ee_pos, dtype=np.float32)
        goal_bluebox_pos = np.asarray(goal_bluebox_pos, dtype=np.float32)

        xs = np.linspace(x_range[0], x_range[1], num_x, dtype=np.float32)
        ys = np.linspace(y_range[0], y_range[1], num_y, dtype=np.float32)

        # -------------------------
        # initial latent z_t
        # -------------------------
        init_img = self.render_state_image(
            ee_pos=init_ee_pos,
            bluebox_pos=init_bluebox_pos,
            image_size=image_size,
            reset=True,
        )
        z_init = self.encode_rendered_image(init_img, pixel_key=pixel_key)  # (1, D)

        # -------------------------
        # goal latent z_g
        # -------------------------
        goal_img = self.render_state_image(
            ee_pos=goal_ee_pos,
            bluebox_pos=goal_bluebox_pos,
            image_size=image_size,
            reset=True,
        )
        z_goal = self.encode_rendered_image(goal_img, pixel_key=pixel_key)  # (1, D)

        cost_map = np.zeros((num_x, num_y), dtype=np.float32)

        min_cost = float("inf")
        min_action_pos = None

        for ix, x in enumerate(tqdm(xs, desc="Collecting action predictor goal cost map")):
            for iy, y in enumerate(ys):
                target_xyz = np.array([x, y, z_value], dtype=np.float32)

                # 今回は action が絶対 cartesian 座標である前提
                action_np = target_xyz[None, :].astype(np.float32)  # (1, 3)

                # 学習時と同じ action normalization
                if self.process is not None and action_key in self.process:
                    action_np = self.process[action_key].transform(action_np)

                action = torch.from_numpy(action_np).float().to(self.device)  # (1, A)
                a_in = action.unsqueeze(0)  # (1, 1, A)

                if hasattr(self.model, "action_encoder"):
                    a_in = self.model.action_encoder(a_in)

                z_in = z_init.unsqueeze(0)  # (1, 1, D)

                z_pred = self.model.predictor(z_in, a_in)

                while z_pred.ndim > 2:
                    z_pred = z_pred.squeeze(0)

                if hasattr(self.model, "pred_proj"):
                    z_pred = self.model.pred_proj(z_pred)

                diff = z_pred - z_goal

                if cost_type == "mse":
                    cost = diff.pow(2).mean().item()
                elif cost_type == "l2":
                    cost = diff.pow(2).sum().sqrt().item()
                else:
                    raise ValueError(f"Unknown cost_type: {cost_type}")

                cost_map[ix, iy] = cost

                if cost < min_cost:
                    min_cost = cost
                    min_action_pos = target_xyz.copy()

        return {
            "cost_map": cost_map,
            "xs": xs,
            "ys": ys,
            "init_ee_pos": init_ee_pos,
            "init_bluebox_pos": init_bluebox_pos,
            "goal_ee_pos": goal_ee_pos,
            "goal_bluebox_pos": goal_bluebox_pos,
            "min_cost": min_cost,
            "min_action_pos": min_action_pos,
            "init_img": init_img,
            "goal_img": goal_img,
            "cost_type": cost_type,
        }




    @torch.no_grad()
    def plot_dataset_pca_rgb(
        self,
        max_samples=8,
        pixel_key="pixels",
        is_val=False,
        save_dir=None,
        title_prefix="encoder_pca_rgb",
        upsample=(224, 224),
        layer_idx=None,
    ):
        import gc
        import itertools
        import numpy as np
        import torch.nn.functional as F
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        dataset = self._get_dataset(is_val)
        n = min(max_samples, len(dataset))

        if save_dir is None:
            save_dir = self.results_path / "probing" / "pca_rgb"
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        def _minmax01(x, eps=1e-8):
            x = x - x.min()
            return x / (x.max() + eps)

        def _resize_rgb(rgb_hw3, out_hw):
            if out_hw is None:
                return rgb_hw3

            rgb_t = torch.from_numpy(rgb_hw3).permute(2, 0, 1).unsqueeze(0).float()
            rgb_t = F.interpolate(
                rgb_t,
                size=out_hw,
                mode="bilinear",
                align_corners=False,
            )
            rgb = rgb_t[0].permute(1, 2, 0).cpu().numpy()
            return np.clip(rgb, 0.0, 1.0)

        def _best_perm(rgb_hw3):
            best_img = rgb_hw3
            best_score = -1e18

            for p in itertools.permutations([0, 1, 2]):
                cand = rgb_hw3[:, :, p]
                score = cand.std(axis=(0, 1)).sum()
                if score > best_score:
                    best_score = score
                    best_img = cand

            return best_img

        figs = []

        for idx in range(n):
            sample = dataset[idx]
            pixels = sample[pixel_key]

            # (T,C,H,W) の場合は最後のフレーム
            if pixels.ndim == 4:
                pixels = pixels[-1]

            raw_img = pixels.detach().cpu()

            if self.transform is not None:
                pixels = self.transform[pixel_key](pixels)

            pixels = pixels.unsqueeze(0).to(self.device)  # (1,C,H,W)

            feat_map = self._encode_pixels_patch_map(
                pixels,
                pixel_key=pixel_key,
                layer_idx=layer_idx,
            )  # (1,C,h,w)

            feat = feat_map[0]  # (C,h,w)
            print("feat.shape:", feat.shape)
            C, h, w = feat.shape

            Z = (
                feat.permute(1, 2, 0)
                .reshape(h * w, C)
                .float()
                .cpu()
                .numpy()
            )

            pca = PCA(n_components=min(3, C))
            Y = pca.fit_transform(Z)
            expl = pca.explained_variance_ratio_

            Y_norm = np.zeros((h * w, 3), dtype=np.float32)
            for c in range(Y.shape[1]):
                Y_norm[:, c] = _minmax01(Y[:, c])

            rgb = Y_norm.reshape(h, w, 3)
            rgb = _best_perm(rgb)
            rgb = _resize_rgb(rgb, upsample)

            # 元画像可視化
            img = raw_img.permute(1, 2, 0).float().numpy()
            img_vis = img.copy()
            for ch in range(img_vis.shape[2]):
                img_vis[:, :, ch] = _minmax01(img_vis[:, :, ch])
            img_vis = _resize_rgb(np.clip(img_vis, 0.0, 1.0), upsample)

            fig, axes = plt.subplots(1, 2, figsize=(6, 3), dpi=160)

            axes[0].imshow(img_vis)
            axes[0].set_title("Image")
            axes[0].axis("off")

            axes[1].imshow(rgb)
            axes[1].set_title(
                f"PCA-RGB\n{expl[0]:.3f}, {expl[1]:.3f}, {expl[2]:.3f}"
            )
            axes[1].axis("off")

            fig.suptitle(f"{title_prefix} | idx={idx}", fontsize=10)
            fig.tight_layout()

            save_path = save_dir / f"{title_prefix}_idx{idx}.png"
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            plt.close(fig)

            figs.append(save_path)

        gc.collect()
        torch.cuda.empty_cache()

        return figs



    @torch.no_grad()
    def make_dataset_sequence_pca_rgb_video(
        self,
        start_idx=0,
        horizon=20,
        pixel_key="pixels",
        is_val=False,
        save_path=None,
        layer_idx=None,
        upsample=(224, 224),
        fps=4,
    ):
        import gc
        import itertools
        import numpy as np
        import torch
        import torch.nn.functional as F
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        import imageio.v2 as imageio
        from pathlib import Path

        dataset = self._get_dataset(is_val)

        if save_path is None:
            save_dir = self.results_path / "probing" / "pca_rgb_video"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"seq_{start_idx}_{horizon}_pca_rgb.gif"
        else:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

        def _minmax01(x, eps=1e-8):
            x = x - x.min()
            return x / (x.max() + eps)

        def _resize_rgb(rgb_hw3, out_hw):
            if out_hw is None:
                return rgb_hw3
            rgb_t = torch.from_numpy(rgb_hw3).permute(2, 0, 1).unsqueeze(0).float()
            rgb_t = F.interpolate(rgb_t, size=out_hw, mode="nearest")
            return rgb_t[0].permute(1, 2, 0).cpu().numpy().clip(0, 1)

        feat_list = []
        img_list = []

        for idx in range(start_idx, min(start_idx + horizon, len(dataset))):
            sample = dataset[idx]
            pixels = sample[pixel_key]

            # (1,C,H,W) なら最後のフレームを使う
            if pixels.ndim == 4:
                pixels = pixels[-1]
            # print("(raw) pixels:", pixels.shape)
            raw_img = pixels.detach().cpu()

            if self.transform is not None:
                pixels = self.transform[pixel_key](pixels)
                # print("(after transform) pixels:", pixels.shape)

            pixels = pixels.unsqueeze(0).to(self.device)
            # print("(encoder input) pixels:", pixels.shape)

            feat_map = self._encode_pixels_patch_map(
                pixels,
                pixel_key=pixel_key,
                layer_idx=layer_idx,
            )

            feat_list.append(feat_map[0].detach().cpu())
            img_list.append(raw_img)

        feats = torch.stack(feat_list, dim=0)  # (T,C,h,w)
        T, C, h, w = feats.shape

        print("PCA video feature shape:", feats.shape)

        Z = (
            feats.permute(0, 2, 3, 1)
            .reshape(T * h * w, C)
            .float()
            .numpy()
        )

        pca = PCA(n_components=min(3, C))
        Y = pca.fit_transform(Z)
        expl = pca.explained_variance_ratio_

        Y_norm = np.zeros((T * h * w, 3), dtype=np.float32)
        for c in range(Y.shape[1]):
            Y_norm[:, c] = _minmax01(Y[:, c])

        rgb_seq = Y_norm.reshape(T, h, w, 3)

        frames = []

        for t in range(T):
            img = img_list[t].permute(1, 2, 0).float().numpy()
            img_vis = img.copy()

            for ch in range(img_vis.shape[2]):
                img_vis[:, :, ch] = _minmax01(img_vis[:, :, ch])

            img_vis = _resize_rgb(np.clip(img_vis, 0, 1), upsample)
            rgb = _resize_rgb(rgb_seq[t], upsample)

            fig, axes = plt.subplots(1, 2, figsize=(6, 3), dpi=120)

            axes[0].imshow(img_vis, interpolation="nearest")
            axes[0].set_title(f"Image t={t}")
            axes[0].axis("off")

            axes[1].imshow(rgb, interpolation="nearest")
            axes[1].set_title(
                f"PCA-RGB\n{expl[0]:.3f}, {expl[1]:.3f}, {expl[2]:.3f}"
            )
            axes[1].axis("off")

            fig.suptitle(f"start={start_idx} | layer={layer_idx}", fontsize=10)
            fig.tight_layout()

            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            frames.append(frame)
            plt.close(fig)

        imageio.mimsave(save_path, frames, fps=fps)

        gc.collect()
        torch.cuda.empty_cache()

        return {
            "save_path": save_path,
            "num_frames": len(frames),
            "feature_shape": (T, C, h, w),
            "explained_variance_ratio": expl,
        }



    @torch.no_grad()
    def collect_cross_position_latents(
        self,
        center=None,
        radius=0.05,
        num_per_axis=11,
        bluebox_pos=None,
        z_value=0.05,
        image_size=(64, 64),
        pixel_key="pixels",
    ):
        """
        EE を十字状(+x, -x, +y, -y)に配置し、
        各状態画像を render して encoder latent を collect する。

        Returns:
            {
                "latents": (N, D),
                "ee_pos": (N, 3),
                "bluebox_pos": (3,),
                "labels": (N,),
                "images": list[H,W,C],
            }
        """
        if self.env is None:
            raise ValueError("self.env is None. Please pass env to ProbingEvaluator.")

        if center is None:
            center = np.array(
                [
                    (self.x_range[0] + self.x_range[1]) / 2.0,
                    (self.y_range[0] + self.y_range[1]) / 2.0,
                    z_value,
                ],
                dtype=np.float32,
            )
        else:
            center = np.asarray(center, dtype=np.float32)

        if bluebox_pos is None:
            bluebox_pos = np.array([center[0], center[1], self.z_range[0]], dtype=np.float32)
        else:
            bluebox_pos = np.asarray(bluebox_pos, dtype=np.float32)

        offsets = np.linspace(-radius, radius, num_per_axis, dtype=np.float32)

        ee_positions = []
        labels = []

        # x軸方向
        for dx in offsets:
            pos = center.copy()
            pos[0] = np.clip(center[0] + dx, self.x_range[0], self.x_range[1])
            pos[1] = center[1]
            pos[2] = z_value
            ee_positions.append(pos)
            labels.append("x_axis")

        # y軸方向
        for dy in offsets:
            # center は x_axis 側ですでに入っているので重複を避ける
            if abs(float(dy)) < 1e-8:
                continue

            pos = center.copy()
            pos[0] = center[0]
            pos[1] = np.clip(center[1] + dy, self.y_range[0], self.y_range[1])
            pos[2] = z_value
            ee_positions.append(pos)
            labels.append("y_axis")

        latents = []
        images = []

        for ee_pos in tqdm(ee_positions, desc="Collecting cross position latents"):
            img = self.render_state_image(
                ee_pos=ee_pos,
                bluebox_pos=bluebox_pos,
                image_size=image_size,
                reset=True,
            )

            z = self.encode_rendered_image(
                img,
                pixel_key=pixel_key,
            )

            latents.append(z.squeeze(0).cpu().numpy())
            images.append(img)

        return {
            "latents": np.stack(latents, axis=0),
            "ee_pos": np.stack(ee_positions, axis=0),
            "bluebox_pos": bluebox_pos,
            "center": center,
            "labels": np.asarray(labels),
            "images": images,
        }




    
    def run(self):
        
        if self.check_isotropy:
            self.evaluate_encoder_isotropy(
                max_samples=self.max_samples,
                pixel_key="pixels",
                is_val=False,
                save_name="train_encoder_latent",
            )
        
        
        open_rollout_data = self.collect_open_rollout_latents(
            max_horizon=self.plot_max_horizon,
            action_key=self.action_key
        )
        
        pca_result = plot_rollout_pca_enc_fit(
            open_rollout_data,
            save_path=self.results_path / "probing" / "open_rollout_pca_3d.png",
            title="Franka rollout PCA (3D)",
        )


        closed_rollout_data = self.collect_closed_rollout_latents(
            max_horizon=self.plot_max_horizon,
            action_key=self.action_key
        )
        
        open_pca_result = plot_rollout_pca_enc_fit(
            open_rollout_data,
            save_path=self.results_path / "probing" / "open_rollout_pca_3d.png",
            title="Franka rollout PCA (3D)",
        )
        

        
        
        if self.plot_all_train_data:
 

            if self.plot_open_data:
                dataset_dyn_data = self.collect_dataset_one_step_latents(
                    max_samples=self.max_samples,
                    pixel_key="pixels",
                    action_key=self.action_key,
                )
                print("dataset_dyn_data(true_z).shape: ", dataset_dyn_data["true_z"].shape)
                
                plot_latent_spread_over_latent_dim(
                    dataset_dyn_data,
                    save_path=self.results_path / "probing" / "dataset_true_z_latent_variance_over_latent_dim.png",
                    title="Dataset true_z latent variance",
                    true_key="true_z",
                )
                plot_latent_spread_over_time (
                    dataset_dyn_data,
                    save_path=self.results_path / "probing" / "dataset_true_z_latent_variance_over_time.png",
                    title="Dataset true_z latent variance",
                    true_key="true_z",
                )
                
                # plot_rollout_pca_all_fit(
                #     dataset_dyn_data,
                #     save_path=self.results_path / "probing" / "dataset_one_step_dynamics_pca_3d_all_fit.png",
                #     title="Dataset one-step dynamics PCA",
                # )
                
                plot_rollout_pca_enc_fit(
                    dataset_dyn_data,
                    save_path=self.results_path / "probing" / "dataset_one_step_dynamics_pca_3d_enc_fit.png",
                    title="Dataset one-step dynamics PCA",
                )
            
            if self.plot_closed_data:
                dataset_closed_dyn_data = self.collect_dataset_closed_rollout_latents(
                        pred_step=self.closed_pred_step,
                        max_samples = self.max_samples,
                        action_key = self.action_key,
                    )
                

                
                plot_rollout_pca_enc_fit(
                    dataset_closed_dyn_data,
                    save_path=self.results_path / "probing" / "dataset_closed_dynamics_pca_3d_enc_fit.png",
                    title="Dataset closed dynamics PCA",
                    plot_line = self.plot_line
                )
                



        if self.plot_all_val_data:
            if self.plot_open_data:
                val_dataset_dyn_data = self.collect_dataset_one_step_latents(
                    max_samples=self.max_samples,
                    pixel_key="pixels",
                    action_key=self.action_key,
                    is_val = True,
                )
                
                plot_rollout_pca_enc_fit(
                    val_dataset_dyn_data,
                    save_path=self.results_path / "probing" / "val_dataset_one_step_dynamics_pca_3d_enc_fit.png",
                    title="Dataset one-step dynamics PCA",
                )
            
            if self.plot_closed_data:
                val_dataset_closed_dyn_data = self.collect_dataset_closed_rollout_latents(
                        pred_step=self.closed_pred_step,
                        max_samples = self.max_samples,
                        action_key = self.action_key,
                        is_val=True,
                    )
                plot_rollout_pca_enc_fit(
                    val_dataset_closed_dyn_data,
                    save_path=self.results_path / "probing" / "val_dataset_closed_dynamics_pca_3d_enc_fit.png",
                    title="Dataset closed dynamics PCA",
                    plot_line = self.plot_line,
                )
                
                
        direction_data = self.collect_center_direction_rollouts(
            pred_step=5,
            step_size=0.005,
            action_key=self.action_key,
            bluebox_pos=[0.55, 0.0, 0.02],
        )

        plot_direction_rollout_pca(
            direction_data,
            save_path=self.results_path / "probing" / "center_direction_rollout_pca_3d.png",
            title="Center Direction Rollout PCA",
        )
        
        plot_direction_xy_trajectory(
            direction_data,
            save_path=self.results_path / "probing" / "center_direction_xy_trajectory.png",
            title="Center Direction XY Trajectory",
        )


        if self.shaded_dataset is not None:
            shaded_data = self.collect_shaded_dataset_latents(
                max_samples=self.max_samples,
                pixel_key="pixels",
            )

            plot_shaded_dataset_pca(
                shaded_data,
                save_path=self.results_path / "probing" / "shaded_clear_nobox_pca_3d.png",
                title="Shaded / Clear / No-box Encoder PCA",
                n_components=3,
            )
            plot_shaded_dataset_pca(
                shaded_data,
                save_path=self.results_path / "probing" / "shaded_clear_nobox_pca_2d.png",
                title="Shaded / Clear / No-box Encoder PCA",
                n_components=2,
            )
            
            
        if self.config.check_ee_cost_map.check: 
            cost_data = self.collect_ee_goal_cost_map(
                goal_ee_pos=self.config.check_ee_cost_map.goal_ee_pos,
                goal_bluebox_pos=self.config.check_ee_cost_map.goal_bluebox_pos,
                fixed_bluebox_pos=self.config.check_ee_cost_map.fixed_bluebox_pos,
                x_range=self.x_range,
                y_range=self.y_range,
                z_value=self.z_range,
                num_x=self.config.check_ee_cost_map.num_x,
                num_y=self.config.check_ee_cost_map.num_y,
                cost_type=self.config.check_ee_cost_map.cost_type,
            )
            
            plot_ee_goal_cost_map(
                cost_data,
                save_path=self.results_path / "probing" / "ee_goal_cost_map.png",
                title="EE position vs goal latent cost",
            )
        
        
        if self.config.check_box_cost_map.check: 
            box_cost_data = self.collect_box_goal_cost_map(
                goal_ee_pos = self.config.check_box_cost_map.goal_ee_pos, 
                goal_bluebox_pos = self.config.check_box_cost_map.goal_bluebox_pos,
                fixed_ee_pos = self.config.check_box_cost_map.fixed_ee_pos, 
                x_range=self.x_range,
                y_range=self.y_range,
                z_value=self.z_range,            
                num_x=self.config.check_box_cost_map.num_x,
                num_y=self.config.check_box_cost_map.num_y,
                image_size=self.config.check_box_cost_map.image_size,
                cost_type=self.config.check_box_cost_map.cost_type,
            )
            plot_box_goal_cost_map(
                box_cost_data,
                save_path=self.results_path / "probing" / "box_goal_cost_map.png",
            )
        
        
        if self.config.check_ee_box_cost_map.check: 
            cost_data = self.collect_relative_pair_goal_cost_map(
                goal_ee_pos=self.config.check_ee_box_cost_map.goal_ee_pos,
                goal_bluebox_pos=self.config.check_ee_box_cost_map.goal_bluebox_pos,
                x_range=self.x_range,
                y_range=self.y_range,
                z_value=self.z_range[0],
                num_x=self.config.check_ee_box_cost_map.num_x,
                num_y=self.config.check_ee_box_cost_map.num_y,
                cost_type="mse",
            )

            plot_relative_pair_goal_cost_map_2d(
                cost_data,
                save_path=self.results_path / "probing" / "relative_pair_goal_cost_map_2d.png",
                title="Relative EE-Box pair vs goal latent cost(2D)",
                show_task_initial=self.config.check_ee_box_cost_map.show_task_initial, 
                task_initial_box_pos=self.config.check_ee_box_cost_map.task_initial_box_pos,
            )
            
            plot_relative_pair_goal_cost_map_3d(
                cost_data,
                save_path=self.results_path / "probing" / "relative_pair_goal_cost_map_3d.png",
                title="Relative EE-Box pair vs goal latent cost(3D)",
                show_task_initial=self.config.check_ee_box_cost_map.show_task_initial, 
                task_initial_box_pos=self.config.check_ee_box_cost_map.task_initial_box_pos,
            )
        if self.config.check_ac_pred_cost_map.check: 
            
            action_pred_cost_data = self.collect_action_predictor_goal_cost_map(
                init_ee_pos=self.config.check_ac_pred_cost_map.init_ee_pos,        # 作業領域中心など
                init_bluebox_pos=self.config.check_ac_pred_cost_map.init_bluebox_pos,  # タスク初期box
                goal_ee_pos=self.config.check_ac_pred_cost_map.goal_ee_pos,
                goal_bluebox_pos=self.config.check_ac_pred_cost_map.goal_bluebox_pos,
                x_range=self.x_range,
                y_range=self.y_range,
                z_value=self.z_range[0],
                num_x=self.config.check_ac_pred_cost_map.num_x,
                num_y=self.config.check_ac_pred_cost_map.num_y,
                image_size=tuple(self.config.check_ac_pred_cost_map.image_size),
                action_key=self.action_key,
                cost_type="l2",
            )

            plot_action_predictor_goal_cost_map(
                action_pred_cost_data,
                save_path=self.results_path / "probing" / "action_predictor_goal_cost_map.png",
            )
            
            
            
            
        if self.config.encoder_rgb_pca.check:
            # self.plot_dataset_pca_rgb(
            #     max_samples=8,
            #     is_val=True,
            #     save_dir=self.results_path / "probing" / "pca_rgb_val",
            #     title_prefix="val_pca_rgb",
            #     layer_idx=None,      # 最終層
            #     upsample=None,
            # )
            
            out = self.make_dataset_sequence_pca_rgb_video(
                start_idx=0,
                horizon=50,
                is_val=False,
                save_path=self.results_path / "probing" / "pca_rgb_video" / "seq0_pca_rgb.mp4",
                layer_idx=None,
                upsample=(224, 224),
                fps=4,
            )

            print(out)


        if self.config.plot_cross_position_latent_pca.check:
            cross_data = self.collect_cross_position_latents(
                radius=0.08,
                num_per_axis=100,
                bluebox_pos=[0.55, 0.0, 0.02],
                z_value=self.z_range[0],
            )

            color_map = {
                "x_axis": "C0",
                "y_axis": "C1",
            }

            plot_cross_position_latent_pca_2d(
                cross_data,
                save_path=self.results_path / "probing" / "cross_position_latent_pca_2d.png",
                title="Cross EE position encoder latent PCA 2D",
            )

            plot_cross_position_xy_trajectory(
                cross_data,
                save_path=self.results_path / "probing" / "cross_position_xy_trajectory.png",
                title="Cross EE position trajectory",
                color_map=color_map,
            )
            
            
