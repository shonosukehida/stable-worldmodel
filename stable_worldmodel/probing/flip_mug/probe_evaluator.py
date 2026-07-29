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

from stable_worldmodel.probing.flip_mug.plot import plot_one_step_rollout_pca


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
        
        
        sample = self.dataset[0]

        print("sample shapes:")
        for key, value in sample.items():
            if torch.is_tensor(value):
                print(f"{key}: shape={tuple(value.shape)}, dtype={value.dtype}")
            else:
                value_np = np.asarray(value)
                print(f"{key}: shape={value_np.shape}, dtype={value_np.dtype}")


    @torch.no_grad()
    def collect_one_step_rollout_latents(
        self,
        start_idx=0,
        max_horizon=100,
        pixel_key="pixels",
        action_key=None,
        target_keys=(),
        is_val=False,
    ):
        """
        連続するデータについて、各時刻の真のencoder表現から
        predictorの1-step予測を計算する。

        評価する対応:
            true_z[t] = Encoder(o_{t+1})
            pred_z[t] = Predictor(Encoder(o_t), a_t)

        Returns:
            true_z:      (N, D)
            pred_z:      (N, D)
            current_z:   (N, D)
            actions:     (N, A)
            targets:     dict[str, np.ndarray]
            indices:     (N,)
        """
        if action_key is None:
            action_key = self.action_key

        dataset = self._get_dataset(is_val)

        if start_idx < 0:
            raise ValueError(f"start_idx must be non-negative, got {start_idx}")

        # idx + 1まで参照するため、len(dataset) - 1が上限
        end_idx = min(
            start_idx + max_horizon,
            len(dataset) - 1,
        )

        if end_idx <= start_idx:
            raise RuntimeError(
                f"No transitions are available: "
                f"start_idx={start_idx}, len(dataset)={len(dataset)}"
            )

        current_list = []
        true_next_list = []
        pred_next_list = []
        action_list = []
        index_list = []

        targets = {key: [] for key in target_keys}

        for idx in tqdm(
            range(start_idx, end_idx),
            desc="Collecting one-step rollout latents",
        ):
            sample_t = dataset[idx]
            sample_tp1 = dataset[idx + 1]

            # -------------------------------------------------
            # エピソード境界をまたいでいないか確認
            # -------------------------------------------------
            if not self._is_consecutive_transition(
                sample_t,
                sample_tp1,
            ):
                break

            # -------------------------------------------------
            # o_t, o_{t+1}
            # -------------------------------------------------
            pixels_t = self._prepare_pixels(
                sample_t[pixel_key],
                pixel_key=pixel_key,
            )

            pixels_tp1 = self._prepare_pixels(
                sample_tp1[pixel_key],
                pixel_key=pixel_key,
            )

            # -------------------------------------------------
            # z_t = Encoder(o_t)
            # z_{t+1} = Encoder(o_{t+1})
            # -------------------------------------------------
            current_z = self._encode_pixels(pixels_t)
            true_next_z = self._encode_pixels(pixels_tp1)

            # -------------------------------------------------
            # a_t
            # -------------------------------------------------
            action = self._prepare_action(
                sample_t[action_key],
                action_key=action_key,
            )

            # -------------------------------------------------
            # z_hat_{t+1} = Predictor(z_t, a_t)
            #
            # 毎ステップ current_z はencoder出力を使う。
            # predictor出力を次の入力には戻さない。
            # -------------------------------------------------
            pred_next_z = self._predict_next_latent(
                current_z,
                action,
            )

            current_list.append(
                current_z.squeeze(0).cpu().numpy()
            )
            true_next_list.append(
                true_next_z.squeeze(0).cpu().numpy()
            )
            pred_next_list.append(
                pred_next_z.squeeze(0).cpu().numpy()
            )
            action_list.append(
                action.squeeze(0).cpu().numpy()
            )
            index_list.append(idx)

            for key in target_keys:
                if key not in sample_tp1:
                    continue

                value = sample_tp1[key]

                if torch.is_tensor(value):
                    value = value.detach().cpu().numpy()
                else:
                    value = np.asarray(value)

                if value.ndim >= 2:
                    value = value[-1]

                targets[key].append(value)

        if len(true_next_list) == 0:
            raise RuntimeError(
                "No valid one-step transitions were collected. "
                "The selected start index may be at an episode boundary."
            )

        targets = {
            key: np.stack(values, axis=0)
            for key, values in targets.items()
            if values
        }

        return {
            "current_z": np.stack(current_list, axis=0),
            "true_z": np.stack(true_next_list, axis=0),
            "pred_z": np.stack(pred_next_list, axis=0),
            "actions": np.stack(action_list, axis=0),
            "targets": targets,
            "indices": np.asarray(index_list),
        }



    def _get_dataset(self, is_val=False):
        if is_val:
            if self.val_dataset is None:
                raise ValueError("is_val=True but val_dataset is None")
            return self.val_dataset
        return self.dataset


    def _prepare_pixels(
        self,
        pixels,
        pixel_key="pixels",
    ):
        """
        Datasetの画像をencoder入力の (1,C,H,W) に変換する。
        """
        if not torch.is_tensor(pixels):
            pixels = torch.as_tensor(pixels)

        # historyを含む場合は最新画像を使用
        if pixels.ndim == 4:
            pixels = pixels[-1]

        if pixels.ndim != 3:
            raise ValueError(
                f"Expected pixels shape (C,H,W) or (T,C,H,W), "
                f"got {tuple(pixels.shape)}"
            )

        if self.transform is not None:
            pixels = self.transform[pixel_key](pixels)

        return pixels.unsqueeze(0).to(self.device)

    def _prepare_action(
        self,
        action,
        action_key,
    ):
        """
        Datasetのactionをpredictor入力の (1,A) に変換する。
        """
        if torch.is_tensor(action):
            action_np = action.detach().cpu().numpy()
        else:
            action_np = np.asarray(action)

        # historyを含む場合は最新actionを使用
        if action_np.ndim == 2:
            action_np = action_np[-1]

        if action_np.ndim != 1:
            raise ValueError(
                f"Expected action shape (A,) or (T,A), "
                f"got {action_np.shape}"
            )

        action_np = action_np[None, :]

        if self.process is not None and action_key in self.process:
            action_np = self.process[action_key].transform(
                action_np
            )

        return torch.from_numpy(
            np.asarray(action_np, dtype=np.float32)
        ).to(self.device)



    @torch.no_grad()
    def _predict_next_latent(
        self,
        current_z,
        action,
    ):
        """
        Args:
            current_z: (B,D)
            action:    (B,A)

        Returns:
            next_z:    (B,D)
        """
        z_in = current_z.unsqueeze(1)  # (B,1,D)
        action_in = action.unsqueeze(1)  # (B,1,A)

        if hasattr(self.model, "action_encoder"):
            action_in = self.model.action_encoder(action_in)

        next_z = self.model.predictor(
            z_in,
            action_in,
        )

        # 典型的には (B,1,D) -> (B,D)
        if next_z.ndim == 3 and next_z.shape[1] == 1:
            next_z = next_z[:, 0]

        # 実装によって余分な先頭次元が付く場合への対応
        while next_z.ndim > 2 and next_z.shape[0] == 1:
            next_z = next_z.squeeze(0)

        if next_z.ndim == 1:
            next_z = next_z.unsqueeze(0)

        if next_z.ndim != 2:
            raise RuntimeError(
                f"Unexpected predictor output shape: "
                f"{tuple(next_z.shape)}"
            )

        if hasattr(self.model, "pred_proj"):
            next_z = self.model.pred_proj(next_z)

        return next_z




    def _is_consecutive_transition(
        self,
        sample_t,
        sample_tp1,
    ):
        """
        2サンプルが同一エピソード内で連続しているか確認する。

        episode情報がDatasetにない場合はTrueを返す。
        """
        episode_keys = (
            "episode_index",
            "episode_idx",
            "episode_id",
        )

        timestep_keys = (
            "frame_index",
            "timestep",
            "time_index",
        )

        for key in episode_keys:
            if key in sample_t and key in sample_tp1:
                ep_t = self._scalar_value(sample_t[key])
                ep_tp1 = self._scalar_value(sample_tp1[key])

                if ep_t != ep_tp1:
                    return False

                break

        for key in timestep_keys:
            if key in sample_t and key in sample_tp1:
                t = self._scalar_value(sample_t[key])
                tp1 = self._scalar_value(sample_tp1[key])

                if tp1 != t + 1:
                    return False

                break

        return True


    @staticmethod
    def _scalar_value(value):
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()

        value = np.asarray(value)
        return int(value.reshape(-1)[-1])

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



    def run(self):


        if self.results_path is None:
            raise ValueError(
                "results_path must be specified"
            )

        save_dir = (
            Path(self.results_path)
            / "probing"
            / "one_step_dynamics"
        )
        save_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        rollout_data = self.collect_one_step_rollout_latents(
            start_idx=0,
            max_horizon=self.plot_max_horizon,
            pixel_key="pixels",
            action_key=self.action_key,
            is_val=False,
        )

        plot_result = plot_one_step_rollout_pca(
            rollout_data,
            save_path=save_dir / "one_step_pca.png",
            title="Flip Mug One-step Dynamics",
            draw_connections=True,
        )

        np.savez_compressed(
            save_dir / "one_step_rollout_data.npz",
            current_z=rollout_data["current_z"],
            true_z=rollout_data["true_z"],
            pred_z=rollout_data["pred_z"],
            actions=rollout_data["actions"],
            indices=rollout_data["indices"],
            true_pca=plot_result["true_pca"],
            pred_pca=plot_result["pred_pca"],
            explained_variance_ratio=(
                plot_result["explained_variance_ratio"]
            ),
        )

        return {
            "rollout_data": rollout_data,
            "plot_result": plot_result,
        }