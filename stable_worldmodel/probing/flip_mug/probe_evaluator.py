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

from stable_worldmodel.probing.flip_mug.plot import *

import gc

import imageio.v2 as imageio
import torch.nn.functional as F


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


        self.plot_max_horizon = config.max_samples
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
                
        self._validate_dataset_sample(
            self.dataset,
            "training",
        )

        if self.val_dataset is not None:
            self._validate_dataset_sample(
                self.val_dataset,
                "validation",
            )


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

        保存する対応:
            true_z[0] = Encoder(o_0)
            pred_z[0] = Encoder(o_0)

            true_z[t+1] = Encoder(o_{t+1})
            pred_z[t+1] = Predictor(Encoder(o_t), a_t)

        Returns:
            true_z:      (N+1, D)
            pred_z:      (N+1, D)
            current_z:   (N, D)
            actions:     (N, A)
            targets:     dict[str, np.ndarray]
            indices:     (N,)
        """
        if action_key is None:
            action_key = self.action_key

        dataset = self._get_dataset(is_val)

        if start_idx < 0:
            raise ValueError(
                f"start_idx must be non-negative, got {start_idx}"
            )

        end_idx = min(
            start_idx + max_horizon,
            len(dataset) - 1,
        )
        print("len(dataset): ", len(dataset))

        if end_idx <= start_idx:
            raise RuntimeError(
                f"No transitions are available: "
                f"start_idx={start_idx}, len(dataset)={len(dataset)}"
            )

        current_list = []
        true_list = []
        pred_list = []
        action_list = []
        index_list = []

        targets = {key: [] for key in target_keys}

        # -------------------------------------------------
        # 初期状態 z_0 を true/pred の両方に保存
        # -------------------------------------------------
        initial_sample = dataset[start_idx]
        
        initial_z = self._encode_state(
            initial_sample,
            pixel_key=pixel_key,
            proprio_key="proprio",
        )

        initial_z_np = initial_z.squeeze(0).cpu().numpy()

        true_list.append(initial_z_np)
        pred_list.append(initial_z_np.copy())

        # -------------------------------------------------
        # 1-step prediction
        # -------------------------------------------------
        for idx in tqdm(
            range(start_idx, end_idx),
            desc="Collecting one-step rollout latents",
        ):
            sample_t = dataset[idx]
            sample_tp1 = dataset[idx + 1]

            if not self._is_consecutive_transition(
                sample_t,
                sample_tp1,
            ):
                break

            current_z = self._encode_state(
                sample_t,
                pixel_key=pixel_key,
                proprio_key="proprio",
            )

            true_next_z = self._encode_state(
                sample_tp1,
                pixel_key=pixel_key,
                proprio_key="proprio",
            )

            action = self._prepare_action(
                sample_t[action_key],
                action_key=action_key,
            )

            pred_next_z = self._predict_next_latent(
                current_z,
                action,
            )

            current_list.append(
                current_z.squeeze(0).cpu().numpy()
            )

            true_list.append(
                true_next_z.squeeze(0).cpu().numpy()
            )

            pred_list.append(
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

        if len(action_list) == 0:
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
            "true_z": np.stack(true_list, axis=0),
            "pred_z": np.stack(pred_list, axis=0),
            "actions": np.stack(action_list, axis=0),
            "targets": targets,
            "indices": np.asarray(index_list),
        }



    @torch.no_grad()
    def collect_closed_loop_rollout_latents(
        self,
        start_idx=0,
        pred_step=50,
        pixel_key="pixels",
        action_key=None,
        is_val=False,
    ):
        """
        1つのエピソード内でclosed-loop latent rolloutを行う。

        保存される系列:
            true_z[0] = Encoder(o_t)
            pred_z[0] = Encoder(o_t)

            true_z[h+1] = Encoder(o_{t+h+1})
            pred_z[h+1] = Predictor(pred_z[h], a_{t+h})

        Returns:
            true_z:     (H+1, D)
            pred_z:     (H+1, D)
            actions:    (H, A)
            indices:    (H,)
        """
        if action_key is None:
            action_key = self.action_key

        dataset = self._get_dataset(is_val)

        if start_idx < 0 or start_idx >= len(dataset) - 1:
            raise ValueError(
                f"Invalid start_idx={start_idx}, len(dataset)={len(dataset)}"
            )

        true_list = []
        pred_list = []
        action_list = []
        index_list = []

        # -------------------------------------------------
        # 初期潜在 z_t
        # -------------------------------------------------
        sample0 = dataset[start_idx]
        initial_z = self._encode_state(
            sample0,
            pixel_key=pixel_key,
            proprio_key="proprio",
        )

        true_list.append(
            initial_z.squeeze(0).cpu().numpy()
        )
        pred_list.append(
            initial_z.squeeze(0).cpu().numpy().copy()
        )

        # Predictorへの入力。
        # 2 step目以降はpredictor出力で更新される。
        pred_current_z = initial_z

        max_end_idx = min(
            start_idx + pred_step,
            len(dataset) - 1,
        )

        for idx in tqdm(
            range(start_idx, max_end_idx),
            desc="Collecting closed-loop rollout latents",
        ):
            sample_t = dataset[idx]
            sample_tp1 = dataset[idx + 1]

            # エピソード境界を跨がない
            if not self._is_consecutive_transition(
                sample_t,
                sample_tp1,
            ):
                print(
                    f"Reached episode boundary at idx={idx}. "
                    f"Collected {len(action_list)} transitions."
                )
                break

            # -------------------------------------------------
            # 真の次状態 Encoder(o_{t+h+1})
            # -------------------------------------------------
            true_next_z = self._encode_state(
                sample_tp1,
                pixel_key=pixel_key,
                proprio_key="proprio",
            )

            # -------------------------------------------------
            # 実際の行動 a_{t+h}
            # -------------------------------------------------
            action = self._prepare_action(
                sample_t[action_key],
                action_key=action_key,
            )  # (1, A)

            # -------------------------------------------------
            # closed-loop prediction
            #
            # h=0:
            #   Predictor(Encoder(o_t), a_t)
            #
            # h>=1:
            #   Predictor(previous prediction, a_{t+h})
            # -------------------------------------------------
            pred_next_z = self._predict_next_latent(
                pred_current_z,
                action,
            )  # (1, D)

            true_list.append(
                true_next_z.squeeze(0).cpu().numpy()
            )
            pred_list.append(
                pred_next_z.squeeze(0).cpu().numpy()
            )
            action_list.append(
                action.squeeze(0).cpu().numpy()
            )
            index_list.append(idx)

            # closed-loopの本質
            pred_current_z = pred_next_z

        if len(action_list) == 0:
            raise RuntimeError(
                "No valid closed-loop transitions were collected."
            )

        return {
            "true_z": np.stack(true_list, axis=0),
            "pred_z": np.stack(pred_list, axis=0),
            "actions": np.stack(action_list, axis=0),
            "indices": np.asarray(index_list),
        }



    @torch.no_grad()
    def collect_episode_closed_rollouts(
        self,
        start_idx=0,
        pred_step=5,
        plot_interval=1,
        max_episode_steps=None,
        pixel_key="pixels",
        action_key=None,
        is_val=False,
    ):
        """
        1 episode 内の複数時刻を始点として、
        pred_step-step closed-loop rolloutを収集する。

        Returns:
            episode_true_z:
                (L, D)
                episode全体のEncoder軌跡

            pred_rollouts:
                list[np.ndarray]
                各要素は (H_i + 1, D)
                episode終端付近では H_i < pred_step の場合がある

            true_rollouts:
                list[np.ndarray]
                各予測rolloutと対応する真の短期軌跡

            rollout_start_positions:
                (N,)
                episode_true_z上の開始位置

            rollout_start_indices:
                (N,)
                dataset上の開始index
        """
        if action_key is None:
            action_key = self.action_key

        dataset = self._get_dataset(is_val)

        if start_idx < 0 or start_idx >= len(dataset):
            raise ValueError(
                f"Invalid start_idx={start_idx}, len(dataset)={len(dataset)}"
            )

        if pred_step <= 0:
            raise ValueError(
                f"pred_step must be positive, got {pred_step}"
            )

        if plot_interval <= 0:
            raise ValueError(
                f"plot_interval must be positive, got {plot_interval}"
            )

        # -------------------------------------------------
        # まずstart_idxが属するepisode全体を特定する
        # -------------------------------------------------
        episode_indices = [start_idx]

        idx = start_idx

        while idx + 1 < len(dataset):
            if (
                max_episode_steps is not None
                and len(episode_indices) >= max_episode_steps
            ):
                break

            sample_t = dataset[idx]
            sample_tp1 = dataset[idx + 1]

            if not self._is_consecutive_transition(
                sample_t,
                sample_tp1,
            ):
                break

            idx += 1
            episode_indices.append(idx)

        if len(episode_indices) < 2:
            raise RuntimeError(
                f"Episode starting at {start_idx} has fewer than 2 frames."
            )

        # -------------------------------------------------
        # episode全体の真のEncoder軌跡
        # -------------------------------------------------
        episode_true_list = []

        for idx in tqdm(
            episode_indices,
            desc="Encoding episode trajectory",
        ):
            sample = dataset[idx]

            z = self._encode_state(
                sample,
                pixel_key=pixel_key,
                wrist_pixel_key="wrist_pixels",
                proprio_key="proprio",
            )

            # z = self._encode_pixels(pixels)

            episode_true_list.append(
                z.squeeze(0).cpu().numpy()
            )

        episode_true_z = np.stack(
            episode_true_list,
            axis=0,
        )

        # -------------------------------------------------
        # 各時刻から短期closed-loop rollout
        # -------------------------------------------------
        true_rollouts = []
        pred_rollouts = []
        action_rollouts = []
        rollout_start_positions = []
        rollout_start_indices = []

        # 最終フレームからはactionがないので除く
        for start_pos in tqdm(
            range(
                0,
                len(episode_indices) - 1,
                plot_interval,
            ),
            desc="Collecting episode closed-loop whiskers",
        ):
            rollout_start_idx = episode_indices[start_pos]

            sample0 = dataset[rollout_start_idx]

            initial_z = self._encode_state(
                sample0,
                pixel_key=pixel_key,
                proprio_key="proprio",
            )

            true_seq = [
                initial_z.squeeze(0).cpu().numpy()
            ]
            pred_seq = [
                initial_z.squeeze(0).cpu().numpy().copy()
            ]
            action_seq = []

            pred_current_z = initial_z

            for h in range(pred_step):
                current_pos = start_pos + h
                next_pos = current_pos + 1

                if next_pos >= len(episode_indices):
                    break

                idx_t = episode_indices[current_pos]
                idx_tp1 = episode_indices[next_pos]

                sample_t = dataset[idx_t]
                sample_tp1 = dataset[idx_tp1]

                true_next_z = self._encode_state(
                    sample_tp1,
                    pixel_key=pixel_key,
                    proprio_key="proprio",
                )

                action = self._prepare_action(
                    sample_t[action_key],
                    action_key=action_key,
                )

                pred_next_z = self._predict_next_latent(
                    pred_current_z,
                    action,
                )

                true_seq.append(
                    true_next_z.squeeze(0).cpu().numpy()
                )
                pred_seq.append(
                    pred_next_z.squeeze(0).cpu().numpy()
                )
                action_seq.append(
                    action.squeeze(0).cpu().numpy()
                )

                # closed-loop:
                # 次の入力にはpredictor出力を使う
                pred_current_z = pred_next_z

            if len(action_seq) == 0:
                continue

            true_rollouts.append(
                np.stack(true_seq, axis=0)
            )
            pred_rollouts.append(
                np.stack(pred_seq, axis=0)
            )
            action_rollouts.append(
                np.stack(action_seq, axis=0)
            )

            rollout_start_positions.append(start_pos)
            rollout_start_indices.append(rollout_start_idx)

        if len(pred_rollouts) == 0:
            raise RuntimeError(
                "No closed-loop whiskers were collected."
            )

        return {
            "episode_true_z": episode_true_z,
            "true_rollouts": true_rollouts,
            "pred_rollouts": pred_rollouts,
            "actions": action_rollouts,
            "episode_indices": np.asarray(episode_indices),
            "rollout_start_positions": np.asarray(
                rollout_start_positions
            ),
            "rollout_start_indices": np.asarray(
                rollout_start_indices
            ),
            "pred_step": pred_step,
            "plot_interval": plot_interval,
        }



    def _get_dataset(self, is_val=False):
        if is_val:
            if self.val_dataset is None:
                raise ValueError("is_val=True but val_dataset is None")
            return self.val_dataset
        return self.dataset

    #画像latentを取得 (propioを捨てる)
    @torch.no_grad()
    def collect_frame_latents(
        self,
        max_samples=1000,
        pixel_key="pixels",
        target_keys=(),
        is_val=False,
        sample_interval=1,
    ):
        """
        Datasetの各フレームをEncoderへ入力し、
        画像全体の潜在表現を収集する。

        Returns:
            latents:
                (N, D)

            targets:
                dict[str, np.ndarray]

            indices:
                (N,)
        """
        dataset = self._get_dataset(is_val)

        if max_samples is not None and max_samples <= 0:
            raise ValueError(
                f"max_samples must be positive or None, "
                f"got {max_samples}"
            )

        if sample_interval <= 0:
            raise ValueError(
                f"sample_interval must be positive, "
                f"got {sample_interval}"
            )

        candidate_indices = range(
            0,
            len(dataset),
            sample_interval,
        )

        if max_samples is not None:
            candidate_indices = list(candidate_indices)[
                :max_samples
            ]

        latent_list = []
        index_list = []
        targets = {
            key: []
            for key in target_keys
        }

        for idx in tqdm(
            candidate_indices,
            desc=(
                "Collecting validation frame latents"
                if is_val
                else "Collecting training frame latents"
            ),
        ):
            sample = dataset[idx]

            if pixel_key not in sample:
                raise KeyError(
                    f"{pixel_key!r} was not found at idx={idx}. "
                    f"Available keys: {list(sample.keys())}"
                )

            pixels = self._prepare_pixels(
                sample[pixel_key],
                pixel_key=pixel_key,
            )

            # Isotropy is evaluated only on the image representation.
            # _encode_state() concatenates the proprioception embedding,
            # which would add non-image dimensions to the covariance matrix.
            z = self._encode_pixels(pixels, pixel_key)

            if z.ndim != 2 or z.shape[0] != 1:
                raise RuntimeError(
                    "Expected encoder output shape (1,D), "
                    f"got {tuple(z.shape)}"
                )

            latent_list.append(
                z[0].detach().cpu().float().numpy()
            )
            index_list.append(idx)

            for key in target_keys:
                if key not in sample:
                    continue

                value = sample[key]

                if torch.is_tensor(value):
                    value = value.detach().cpu().numpy()
                else:
                    value = np.asarray(value)

                # history付きなら最新時刻
                if value.ndim >= 2:
                    value = value[-1]

                targets[key].append(value)

        if len(latent_list) == 0:
            raise RuntimeError(
                "No latent samples were collected."
            )

        targets = {
            key: np.stack(values, axis=0)
            for key, values in targets.items()
            if len(values) == len(latent_list)
        }

        return {
            "latents": np.stack(
                latent_list,
                axis=0,
            ),
            "targets": targets,
            "indices": np.asarray(
                index_list,
                dtype=np.int64,
            ),
        }


    @torch.no_grad()
    def evaluate_encoder_isotropy(
        self,
        max_samples=1000,
        pixel_key="pixels",
        is_val=False,
        save_name="encoder_latent",
        sample_interval=1,
    ):
        """
        Encoderの画像全体表現について等方性を評価する。
        """
        data = self.collect_frame_latents(
            max_samples=max_samples,
            pixel_key=pixel_key,
            target_keys=(),
            is_val=is_val,
            sample_interval=sample_interval,
        )

        latents = data["latents"]

        if latents.ndim != 2:
            raise RuntimeError(
                f"Expected latents shape (N,D), "
                f"got {latents.shape}"
            )

        save_dir = None

        if self.results_path is not None:
            split_name = "val" if is_val else "train"

            save_dir = (
                Path(self.results_path)
                / "probing"
                / "isotropy"
                / split_name
            )

        metrics = analyze_latent_isotropy(
            latents,
            save_dir=save_dir,
            prefix=save_name,
        )

        print(
            f"[Latent isotropy metrics: "
            f"{'validation' if is_val else 'training'}]"
        )

        for key, value in metrics.items():
            print(f"{key}: {value}")

        return {
            "metrics": metrics,
            "latents": latents,
            "indices": data["indices"],
        }


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
        
        
    def _prepare_proprio(
        self,
        proprio,
        proprio_key="proprio",
    ):
        if torch.is_tensor(proprio):
            proprio_np = proprio.detach().cpu().numpy()
        else:
            proprio_np = np.asarray(proprio)

        # historyが含まれる場合は最新時刻を使用
        if proprio_np.ndim == 2:
            proprio_np = proprio_np[-1]

        if proprio_np.shape != (8,):
            raise ValueError(
                "Expected proprio shape (8,) or (T,8), "
                f"got {proprio_np.shape}"
            )

        proprio_np = proprio_np[None, :]

        if (
            self.process is not None
            and proprio_key in self.process
        ):
            proprio_np = self.process[
                proprio_key
            ].transform(proprio_np)

        return torch.from_numpy(
            np.asarray(
                proprio_np,
                dtype=np.float32,
            )
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

    def _encode_pixels(self, pixels, pixel_key="pixels",):
        """
        pixels: (B, C, H, W)
        return: (B, D)
        """
        
        if pixel_key == "pixels":
            encoder = self.model.overhead_encoder
            projector = self.model.overhead_projector

        elif pixel_key == "wrist_pixels":
            encoder = self.model.wrist_encoder
            projector = self.model.wrist_projector

        else:
            raise ValueError(
                f"Unsupported pixel_key: {pixel_key}"
            )



        output = encoder(
            pixels,
            interpolate_pos_encoding=True,
        )

        if hasattr(output, "last_hidden_state"):
            z = output.last_hidden_state[:, 0]
        else:
            z = output[:, 0]

        z = projector(z)

        return z
    
    @torch.no_grad()
    def _encode_state(
        self,
        sample,
        pixel_key="pixels",
        wrist_pixel_key="wrist_pixels",
        proprio_key="proprio",
    ):
        if pixel_key not in sample:
            raise KeyError(
                f"{pixel_key!r} was not found. "
                f"Available keys: {list(sample.keys())}"
            )
            
        if wrist_pixel_key not in sample:
            raise KeyError(
                f"{wrist_pixel_key!r} was not found. "
                f"Available keys: {list(sample.keys())}"
            )

        if proprio_key not in sample:
            raise KeyError(
                f"{proprio_key!r} was not found. "
                f"Available keys: {list(sample.keys())}"
            )

        pixels = self._prepare_pixels(
            sample[pixel_key],
            pixel_key=pixel_key,
        )

        wrist_pixels = self._prepare_pixels(
            sample[wrist_pixel_key],
            pixel_key=wrist_pixel_key,
        )
        
        proprio = self._prepare_proprio(
            sample[proprio_key],
            proprio_key=proprio_key,
        )

        # JEPA.encode()の入力形式に合わせる
        # pixels:   (B=1, T=1, C, H, W)
        # proprio:  (B=1, T=1, 8)
        info = {
            "pixels": pixels.unsqueeze(1),
            "wrist_pixels": wrist_pixels.unsqueeze(1),
            "proprio": proprio.unsqueeze(1),
        }

        output = self.model.encode(info)

        # (1, 1, 224) -> (1, 224)
        return output["emb"][:, -1]


    @torch.no_grad()
    def _encode_pixels_patch_map(
        self,
        pixels,
        pixel_key="pixels",
        layer_idx=None,
    ):
        """
        Encoderのpatch tokenを空間特徴マップへ変換する。

        Args:
            pixels:
                (B, C, H, W)

            pixel_key:
                transformに使う画像キー。
                この関数内では基本的に使用しないが、
                呼び出し側とのインターフェース統一用。

            layer_idx:
                None:
                    最終層のhidden stateを使用

                int:
                    指定した中間層のhidden stateを使用

        Returns:
            feat_map:
                (B, C_feat, h_patch, w_patch)
        """
        
        if pixel_key == "pixels":
            encoder = self.model.overhead_encoder
        elif pixel_key == "wrist_pixels":
            encoder = self.model.wrist_encoder
        else:
            raise ValueError(
                f"Unsupported pixel_key: {pixel_key}"
            )
            
        out = encoder(
            pixels,
            interpolate_pos_encoding=True,
            output_hidden_states=True,
            return_dict=True,
        )

        if layer_idx is None:
            tokens = out.last_hidden_state
        else:
            if out.hidden_states is None:
                raise RuntimeError(
                    "Encoder did not return hidden_states. "
                    "Check output_hidden_states=True."
                )

            n_layers = len(out.hidden_states)

            if not (-n_layers <= layer_idx < n_layers):
                raise IndexError(
                    f"Invalid layer_idx={layer_idx}. "
                    f"Encoder returned {n_layers} hidden states."
                )

            tokens = out.hidden_states[layer_idx]

        if tokens.ndim != 3:
            raise ValueError(
                f"Expected token shape (B,N,D), got {tuple(tokens.shape)}"
            )

        # ViTの先頭にあるCLS tokenを除く
        patch_tokens = tokens[:, 1:, :]  # (B, N, D)
        
        B, N, D = patch_tokens.shape
        _, _, H, W = pixels.shape

        patch_size = encoder.config.patch_size

        if isinstance(patch_size, int):
            patch_h = H // patch_size
            patch_w = W // patch_size
        else:
            patch_h = H // patch_size[0]
            patch_w = W // patch_size[1]

        if patch_h * patch_w != N:
            raise ValueError(
                f"Patch shape mismatch: "
                f"input_size=({H}, {W}), "
                f"patch_size={patch_size}, "
                f"calculated=({patch_h}, {patch_w}), "
                f"calculated_tokens={patch_h * patch_w}, "
                f"actual_tokens={N}"
            )

        feat_map = (
            patch_tokens
            .reshape(B, patch_h, patch_w, D)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        return feat_map


    def _validate_dataset_sample(
        self,
        dataset,
        dataset_name,
    ):
        sample = dataset[0]

        required_keys = {
            "pixels",
            "wrist_pixels",
            "proprio",
            self.action_key,
        }

        missing_keys = (
            required_keys
            - set(sample.keys())
        )

        if missing_keys:
            raise KeyError(
                f"{dataset_name} dataset is missing keys: "
                f"{sorted(missing_keys)}. "
                f"Available keys: {list(sample.keys())}"
            )

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
        """
        Dataset中の連続画像について、Encoderのpatch表現を
        PCA-RGB画像として可視化し、MP4へ保存する。

        PCAは全時刻・全patchをまとめてfitする。
        したがって、すべてのフレームでRGB軸が共通になる。

        Args:
            start_idx:
                可視化を開始するdataset index

            horizon:
                可視化する最大フレーム数

            pixel_key:
                使用する画像のdataset key

            is_val:
                Trueならvalidation datasetを使用

            save_path:
                出力するGIFのパス

            layer_idx:
                NoneならEncoder最終層
                intなら指定した中間層

            upsample:
                元画像およびPCA-RGB画像の表示サイズ

            fps:
                GIFのフレームレート

        Returns:
            dict:
                save_path
                num_frames
                feature_shape
                explained_variance_ratio
                dataset_indices
        """


        dataset = self._get_dataset(is_val)

        if start_idx < 0 or start_idx >= len(dataset):
            raise ValueError(
                f"Invalid start_idx={start_idx}. "
                f"Dataset length is {len(dataset)}."
            )

        if horizon <= 0:
            raise ValueError(
                f"horizon must be positive, got {horizon}"
            )

        if save_path is None:
            if self.results_path is None:
                raise ValueError(
                    "Both save_path and self.results_path are None."
                )

            save_dir = (
                Path(self.results_path)
                / "probing"
                / "pca_rgb_video"
            )
            save_dir.mkdir(parents=True, exist_ok=True)

            split_name = "val" if is_val else "train"

            save_path = (
                save_dir
                / f"{split_name}_seq_{start_idx}_{horizon}_pca_rgb.mp4"
            )
        else:
            save_path = Path(save_path)
            save_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        def _minmax01(x, eps=1e-8):
            """
            配列全体を[0,1]へ正規化する。
            """
            x = x - x.min()
            return x / (x.max() + eps)

        def _resize_rgb(rgb_hw3, out_hw):
            """
            (H,W,3)のnumpy配列を指定サイズへ拡大する。
            Patch境界を保つためnearestを使用する。
            """
            if out_hw is None:
                return rgb_hw3

            rgb_tensor = (
                torch.from_numpy(rgb_hw3)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
            )

            rgb_tensor = F.interpolate(
                rgb_tensor,
                size=out_hw,
                mode="nearest",
            )

            return (
                rgb_tensor[0]
                .permute(1, 2, 0)
                .cpu()
                .numpy()
                .clip(0.0, 1.0)
            )

        feat_list = []
        img_list = []
        dataset_indices = []

        end_idx = min(
            start_idx + horizon,
            len(dataset),
        )

        for idx in tqdm(
            range(start_idx, end_idx),
            desc="Encoding Flip Mug PCA-RGB sequence",
        ):
            sample = dataset[idx]

            if pixel_key not in sample:
                raise KeyError(
                    f"{pixel_key!r} was not found. "
                    f"Available keys: {list(sample.keys())}"
                )

            pixels = sample[pixel_key]

            if not torch.is_tensor(pixels):
                pixels = torch.as_tensor(pixels)

            # history付きの(T,C,H,W)なら最新フレームを使用
            if pixels.ndim == 4:
                pixels = pixels[-1]

            if pixels.ndim != 3:
                raise ValueError(
                    "Expected image shape (C,H,W) or (T,C,H,W), "
                    f"got {tuple(pixels.shape)} at idx={idx}"
                )

            # transform前の画像を表示用に保存
            raw_img = pixels.detach().cpu()

            encoder_pixels = pixels

            if self.transform is not None:
                if pixel_key not in self.transform:
                    raise KeyError(
                        f"{pixel_key!r} is not in transform. "
                        f"Available transform keys: "
                        f"{list(self.transform.keys())}"
                    )

                encoder_pixels = self.transform[pixel_key](
                    encoder_pixels
                )

            encoder_pixels = (
                encoder_pixels
                .unsqueeze(0)
                .to(self.device)
            )

            feat_map = self._encode_pixels_patch_map(
                encoder_pixels,
                pixel_key=pixel_key,
                layer_idx=layer_idx,
            )  # (1,C_feat,h,w)

            feat_list.append(
                feat_map[0].detach().cpu()
            )
            img_list.append(raw_img)
            dataset_indices.append(idx)

        if len(feat_list) == 0:
            raise RuntimeError(
                "No frames were collected for PCA-RGB video."
            )

        # (T,C,h,w)
        feats = torch.stack(
            feat_list,
            dim=0,
        )

        num_frames, feature_dim, patch_h, patch_w = feats.shape

        print(
            "Flip Mug PCA video feature shape:",
            tuple(feats.shape),
        )

        # 全時刻・全patchを1つの標本集合としてPCA
        #
        # (T,C,h,w)
        #     -> (T,h,w,C)
        #     -> (T*h*w,C)
        features_flat = (
            feats
            .permute(0, 2, 3, 1)
            .reshape(
                num_frames * patch_h * patch_w,
                feature_dim,
            )
            .float()
            .numpy()
        )

        n_components = min(
            3,
            feature_dim,
            features_flat.shape[0],
        )

        pca = PCA(
            n_components=n_components
        )

        projected = pca.fit_transform(
            features_flat
        )

        explained_variance_ratio = (
            pca.explained_variance_ratio_
        )

        # 常にRGBの3チャンネルを用意
        projected_rgb = np.zeros(
            (
                num_frames * patch_h * patch_w,
                3,
            ),
            dtype=np.float32,
        )

        # 各PCA成分を、全時刻共通のmin/maxで[0,1]へ正規化
        for component_idx in range(n_components):
            projected_rgb[:, component_idx] = _minmax01(
                projected[:, component_idx]
            )

        rgb_sequence = projected_rgb.reshape(
            num_frames,
            patch_h,
            patch_w,
            3,
        )

        frames = []

        for t in tqdm(
            range(num_frames),
            desc="Rendering Flip Mug PCA-RGB frames",
        ):
            raw_img = img_list[t]

            # CHW -> HWC
            img = (
                raw_img
                .permute(1, 2, 0)
                .float()
                .numpy()
            )

            img_vis = img.copy()

            # 元画像を表示用に正規化
            for channel_idx in range(img_vis.shape[2]):
                img_vis[:, :, channel_idx] = _minmax01(
                    img_vis[:, :, channel_idx]
                )

            img_vis = _resize_rgb(
                np.clip(img_vis, 0.0, 1.0),
                upsample,
            )

            pca_rgb = _resize_rgb(
                rgb_sequence[t],
                upsample,
            )

            fig, axes = plt.subplots(
                1,
                2,
                figsize=(6, 3),
                dpi=120,
            )

            axes[0].imshow(
                img_vis,
                interpolation="nearest",
            )
            axes[0].set_title(
                f"Image idx={dataset_indices[t]}"
            )
            axes[0].axis("off")

            axes[1].imshow(
                pca_rgb,
                interpolation="nearest",
            )

            variance_text = ", ".join(
                f"{value:.3f}"
                for value in explained_variance_ratio
            )

            axes[1].set_title(
                f"PCA-RGB\n{variance_text}"
            )
            axes[1].axis("off")

            layer_name = (
                "last"
                if layer_idx is None
                else str(layer_idx)
            )

            fig.suptitle(
                f"Flip Mug | layer={layer_name}",
                fontsize=10,
            )

            fig.tight_layout()

            fig.canvas.draw()

            frame = (
                np.asarray(
                    fig.canvas.buffer_rgba()
                )[:, :, :3]
                .copy()
            )

            frames.append(frame)
            plt.close(fig)

        writer = imageio.get_writer(
            save_path,
            fps=fps,
            codec="libx264",
        )

        for frame in frames:
            writer.append_data(frame)

        writer.close()

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Saved PCA-RGB video to: {save_path}")

        return {
            "save_path": save_path,
            "num_frames": len(frames),
            "feature_shape": (
                num_frames,
                feature_dim,
                patch_h,
                patch_w,
            ),
            "explained_variance_ratio": (
                explained_variance_ratio
            ),
            "dataset_indices": np.asarray(
                dataset_indices
            ),
        }






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
        
        if self.config.plot_all_train_data:
            if self.config.plot_open_data:
                
                print("open pred in train data")
                rollout_data = self.collect_one_step_rollout_latents(
                    start_idx=0,
                    max_horizon=self.plot_max_horizon,
                    pixel_key="pixels",
                    action_key=self.action_key,
                    is_val=False,
                )
                
                current_z = rollout_data["current_z"]  # [z0, ..., z_{N-1}]
                true_z = rollout_data["true_z"]        # [z0, z1, ..., z_N]
                pred_z = rollout_data["pred_z"]        # [z0, zhat1, ..., zhat_N]

                assert true_z.shape == pred_z.shape
                assert true_z.shape[0] == current_z.shape[0] + 1

                pred_true_mse = np.mean(
                    (pred_z[1:] - true_z[1:]) ** 2,
                    axis=1,
                )

                pred_current_mse = np.mean(
                    (pred_z[1:] - current_z) ** 2,
                    axis=1,
                )
                copy_baseline = np.mean(
                    (current_z - true_z[1:]) ** 2,
                    axis=1,
                )

                print("pred vs true next :", pred_true_mse.mean())
                print("pred vs current   :", pred_current_mse.mean())
                print("current vs true z:", copy_baseline.mean())

                plot_result = plot_one_step_rollout_pca(
                    rollout_data,
                    save_path=save_dir / "train_one_step_pca.png",
                    title="Flip Mug One-step Dynamics",
                    draw_connections=False,
                )

                np.savez_compressed(
                    save_dir / "train_one_step_rollout_data.npz",
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
            
            if self.config.plot_closed_data.plot_whisker.plot:

                print("whisker closed pred in train data")
                episode_closed_data = self.collect_episode_closed_rollouts(
                    start_idx=0,
                    pred_step=self.config.plot_closed_data.plot_whisker.pred_step,
                    plot_interval=self.config.plot_closed_data.plot_whisker.plot_interval,
                    pixel_key="pixels",
                    action_key=self.action_key,
                    is_val=False,
                )

                closed_plot_result = (
                    plot_episode_closed_rollout_whiskers_pca(
                        episode_closed_data,
                        save_path=(
                            save_dir
                            / "train_closed_loop_whiskers_pca.png"
                        ),
                        title="Flip Mug 5-step Closed-loop Dynamics",
                        draw_true_segments=False,
                        draw_endpoint_connections=False,
                    )
                )
            
            if self.config.plot_closed_data.plot_pred_horizon.plot:
                print("long horizon closed pred in train data")                  
                closed_rollout_data = self.collect_closed_loop_rollout_latents(
                    start_idx=0,
                    pred_step=self.config.plot_closed_data.plot_pred_horizon.pred_horizon,
                    pixel_key="pixels",
                    action_key=self.action_key,
                    is_val=False,
                )

                closed_true_z = closed_rollout_data["true_z"]
                closed_pred_z = closed_rollout_data["pred_z"]

                assert closed_true_z.shape == closed_pred_z.shape

                # horizonごとの潜在予測MSE
                closed_mse_by_horizon = np.mean(
                    (closed_pred_z - closed_true_z) ** 2,
                    axis=1,
                )

                print("closed-loop MSE by horizon:")
                for h, mse in enumerate(closed_mse_by_horizon):
                    print(f"h={h:3d}: {mse:.8f}")

                print(
                    "closed-loop final MSE:",
                    closed_mse_by_horizon[-1],
                )

                closed_plot_result = plot_closed_loop_rollout_pca(
                    closed_rollout_data,
                    save_path=save_dir / "train_closed_loop_pca.png",
                    title="Flip Mug Closed-loop Dynamics",
                    draw_connections=False,
                )

                np.savez_compressed(
                    save_dir / "train_closed_loop_rollout_data.npz",
                    true_z=closed_true_z,
                    pred_z=closed_pred_z,
                    actions=closed_rollout_data["actions"],
                    indices=closed_rollout_data["indices"],
                    mse_by_horizon=closed_mse_by_horizon,
                    true_pca=closed_plot_result["true_pca"],
                    pred_pca=closed_plot_result["pred_pca"],
                    explained_variance_ratio=(
                        closed_plot_result["explained_variance_ratio"]
                    ),
                )



        if self.config.plot_all_val_data:
            if self.config.plot_open_data:
                save_dir = (
                    Path(self.results_path)
                    / "probing"
                    / "one_step_dynamics"
                )
                save_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                print("open pred in val data")
                rollout_data = self.collect_one_step_rollout_latents(
                    start_idx=0,
                    max_horizon=self.plot_max_horizon,
                    pixel_key="pixels",
                    action_key=self.action_key,
                    is_val=True,
                )
                
                current_z = rollout_data["current_z"]  # [z0, ..., z_{N-1}]
                true_z = rollout_data["true_z"]        # [z0, z1, ..., z_N]
                pred_z = rollout_data["pred_z"]        # [z0, zhat1, ..., zhat_N]

                assert true_z.shape == pred_z.shape
                assert true_z.shape[0] == current_z.shape[0] + 1

                pred_true_mse = np.mean(
                    (pred_z[1:] - true_z[1:]) ** 2,
                    axis=1,
                )

                pred_current_mse = np.mean(
                    (pred_z[1:] - current_z) ** 2,
                    axis=1,
                )
                copy_baseline = np.mean(
                    (current_z - true_z[1:]) ** 2,
                    axis=1,
                )

                print("pred vs true next :", pred_true_mse.mean())
                print("pred vs current   :", pred_current_mse.mean())
                print("current vs true z:", copy_baseline.mean())

                plot_result = plot_one_step_rollout_pca(
                    rollout_data,
                    save_path=save_dir / "val_one_step_pca.png",
                    title="Flip Mug One-step Dynamics",
                    draw_connections=False,
                )

                np.savez_compressed(
                    save_dir / "val_one_step_rollout_data.npz",
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
            
            if self.config.plot_closed_data.plot_whisker.plot:
                
                print("whisker closed pred in val data")
                episode_closed_data = self.collect_episode_closed_rollouts(
                    start_idx=0,
                    pred_step=self.config.plot_closed_data.plot_whisker.pred_step,
                    plot_interval=self.config.plot_closed_data.plot_whisker.plot_interval,
                    pixel_key="pixels",
                    action_key=self.action_key,
                    is_val=True,
                )

                closed_plot_result = (
                    plot_episode_closed_rollout_whiskers_pca(
                        episode_closed_data,
                        save_path=(
                            save_dir
                            / "val_closed_loop_whiskers_pca.png"
                        ),
                        title="Flip Mug 5-step Closed-loop Dynamics",
                        draw_true_segments=False,
                        draw_endpoint_connections=False,
                    )
                )
            
            if self.config.plot_closed_data.plot_pred_horizon.plot:

                print("long horizon closed pred in val data")  
                closed_rollout_data = self.collect_closed_loop_rollout_latents(
                    start_idx=0,
                    pred_step=self.config.plot_closed_data.plot_pred_horizon.pred_horizon,
                    pixel_key="pixels",
                    action_key=self.action_key,
                    is_val=True,
                )

                closed_true_z = closed_rollout_data["true_z"]
                closed_pred_z = closed_rollout_data["pred_z"]

                assert closed_true_z.shape == closed_pred_z.shape

                # horizonごとの潜在予測MSE
                closed_mse_by_horizon = np.mean(
                    (closed_pred_z - closed_true_z) ** 2,
                    axis=1,
                )

                print("closed-loop MSE by horizon:")
                for h, mse in enumerate(closed_mse_by_horizon):
                    print(f"h={h:3d}: {mse:.8f}")

                print(
                    "closed-loop final MSE:",
                    closed_mse_by_horizon[-1],
                )

                closed_plot_result = plot_closed_loop_rollout_pca(
                    closed_rollout_data,
                    save_path=save_dir / "val_closed_loop_pca.png",
                    title="Flip Mug Closed-loop Dynamics",
                    draw_connections=False,
                )

                np.savez_compressed(
                    save_dir / "val_closed_loop_rollout_data.npz",
                    true_z=closed_true_z,
                    pred_z=closed_pred_z,
                    actions=closed_rollout_data["actions"],
                    indices=closed_rollout_data["indices"],
                    mse_by_horizon=closed_mse_by_horizon,
                    true_pca=closed_plot_result["true_pca"],
                    pred_pca=closed_plot_result["pred_pca"],
                    explained_variance_ratio=(
                        closed_plot_result["explained_variance_ratio"]
                    ),
                )
                

        if self.config.encoder_rgb_pca.check:
            print("overhead_encoder_rgb_pca")
            pca_overhead_video_result = self.make_dataset_sequence_pca_rgb_video(
                start_idx=0,
                horizon=self.config.encoder_rgb_pca.horizon,
                pixel_key="pixels",
                is_val=self.config.encoder_rgb_pca.is_val,
                save_path=(
                    self.results_path
                    / "probing" / "flip_mug_overhead_encoder_pca_rgb.mp4"
                ),
                layer_idx=None,
                upsample=(224, 298),
                fps=10,
            )

            print("wrist_encoder_rgb_pca")
            pca_wrist_video_result = self.make_dataset_sequence_pca_rgb_video(
                start_idx=0,
                horizon=self.config.encoder_rgb_pca.horizon,
                pixel_key="wrist_pixels",
                is_val=self.config.encoder_rgb_pca.is_val,
                save_path=(
                    self.results_path
                    / "probing" / "flip_mug_wrist_encoder_pca_rgb.mp4"
                ),
                layer_idx=None,
                upsample=(224, 298),
                fps=10,
            )



        if self.config.encoder_isotropy.check:
            print("overhead encoder isotropy")
            train_isotropy = self.evaluate_encoder_isotropy(
                max_samples=self.config.encoder_isotropy.max_samples,
                pixel_key="pixels",
                is_val=False,
                save_name="train_overhead_encoder_latent",
                sample_interval=(
                    self.config.encoder_isotropy.sample_interval
                ),
            )

            isotropy_dir = (
                Path(self.results_path)
                / "probing"
                / "isotropy"
                / "train"
                / "overhead"
            )
            isotropy_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            np.savez_compressed(
                isotropy_dir / "train_overhead_encoder_latents.npz",
                latents=train_isotropy["latents"],
                indices=train_isotropy["indices"],
            )






            print("wrist encoder isotropy")
            train_isotropy = self.evaluate_encoder_isotropy(
                max_samples=self.config.encoder_isotropy.max_samples,
                pixel_key="wrist_pixels",
                is_val=False,
                save_name="train_wrist_encoder_latent",
                sample_interval=(
                    self.config.encoder_isotropy.sample_interval
                ),
            )

            isotropy_dir = (
                Path(self.results_path)
                / "probing"
                / "isotropy"
                / "train"
                / "wrist"
            )
            isotropy_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            np.savez_compressed(
                isotropy_dir / "train_wrist_encoder_latents.npz",
                latents=train_isotropy["latents"],
                indices=train_isotropy["indices"],
            )
            

        return






def analyze_latent_isotropy(
    latents,
    save_dir=None,
    prefix="encoder_latent",
    eps=1e-8,
):
    """
    Encoder出力の平均、分散、共分散、固有値を調べる。

    Args:
        latents:
            shape (N, D)

        save_dir:
            可視化・metricsの保存先

        prefix:
            保存ファイル名の接頭辞

    Notes:
        厳密な標準正規分布 N(0, I) なら、
            mean ≈ 0
            covariance ≈ I
        となる。

        等方性だけを見る場合は、共分散が
            covariance ≈ sigma^2 I
        であればよく、分散が必ず1である必要はない。
    """
    latents = np.asarray(latents, dtype=np.float64)

    if latents.ndim != 2:
        raise ValueError(
            f"latents must have shape (N,D), got {latents.shape}"
        )

    N, D = latents.shape

    if N < 2:
        raise ValueError(
            f"At least two samples are required, got N={N}"
        )

    if not np.isfinite(latents).all():
        raise ValueError(
            "latents contains NaN or infinite values"
        )

    mean = latents.mean(axis=0)
    std = latents.std(axis=0)
    var = latents.var(axis=0)

    centered = latents - mean[None, :]
    cov = centered.T @ centered / max(N - 1, 1)

    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]

    # 浮動小数点誤差による微小な負値を0にする
    eigvals = np.clip(eigvals, 0.0, None)

    diag = np.diag(cov)
    offdiag = cov - np.diag(diag)

    total_variance = eigvals.sum()
    explained = eigvals / (total_variance + eps)

    # スケール非依存の等方性評価用
    normalized_eigvals = eigvals / (eigvals.mean() + eps)

    # covarianceを対角成分で正規化した相関行列
    denom = np.sqrt(
        np.maximum(diag[:, None] * diag[None, :], eps)
    )
    corr = cov / denom
    corr_offdiag = corr - np.diag(np.diag(corr))

    positive_eigvals = eigvals[eigvals > eps]

    if len(positive_eigvals) > 0:
        effective_condition = (
            positive_eigvals.max() / positive_eigvals.min()
        )
    else:
        effective_condition = np.inf

    metrics = {
        "N": int(N),
        "D": int(D),

        # 中心が0に近いか
        "mean_abs_mean": float(np.mean(np.abs(mean))),
        "mean_l2_norm": float(np.linalg.norm(mean)),

        # 各座標軸の分散
        "std_mean": float(std.mean()),
        "var_mean": float(var.mean()),
        "var_std": float(var.std()),
        "var_min": float(var.min()),
        "var_max": float(var.max()),
        "var_cv": float(
            var.std() / (var.mean() + eps)
        ),

        # 元の座標系における共分散
        "offdiag_abs_mean": float(
            np.mean(np.abs(offdiag))
        ),
        "offdiag_abs_max": float(
            np.max(np.abs(offdiag))
        ),

        # スケールに依存しない相関
        "corr_offdiag_abs_mean": float(
            np.mean(np.abs(corr_offdiag))
        ),
        "corr_offdiag_abs_max": float(
            np.max(np.abs(corr_offdiag))
        ),

        # 共分散固有値
        "eig_mean": float(eigvals.mean()),
        "eig_std": float(eigvals.std()),
        "eig_min": float(eigvals.min()),
        "eig_max": float(eigvals.max()),
        "eig_cv": float(
            eigvals.std() / (eigvals.mean() + eps)
        ),
        "eig_condition": float(
            eigvals.max() / (eigvals.min() + eps)
        ),
        "effective_eig_condition": float(
            effective_condition
        ),
        "normalized_eig_min": float(
            normalized_eigvals.min()
        ),
        "normalized_eig_max": float(
            normalized_eigvals.max()
        ),

        # 上位主成分への偏り
        "top1_explained_ratio": float(
            explained[:1].sum()
        ),
        "top3_explained_ratio": float(
            explained[: min(3, D)].sum()
        ),
        "top10_explained_ratio": float(
            explained[: min(10, D)].sum()
        ),
    }

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 1. dimension-wise mean
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(mean)
        ax.axhline(
            0.0,
            linestyle="--",
            linewidth=1.0,
        )
        ax.set_title("Latent mean per dimension")
        ax.set_xlabel("latent dimension")
        ax.set_ylabel("mean")
        ax.grid(True, linestyle=":", alpha=0.5)
        fig.tight_layout()
        fig.savefig(
            save_dir / f"{prefix}_mean_per_dim.png",
            dpi=200,
        )
        plt.close(fig)

        # 2. dimension-wise variance
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(var)
        ax.axhline(
            var.mean(),
            linestyle="--",
            linewidth=1.0,
            label=f"mean variance={var.mean():.4f}",
        )
        ax.axhline(
            1.0,
            linestyle=":",
            linewidth=1.0,
            label="unit variance",
        )
        ax.set_title("Latent variance per dimension")
        ax.set_xlabel("latent dimension")
        ax.set_ylabel("variance")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.5)
        fig.tight_layout()
        fig.savefig(
            save_dir / f"{prefix}_var_per_dim.png",
            dpi=200,
        )
        plt.close(fig)

        # 3. covariance matrix
        cov_abs_max = max(
            float(np.max(np.abs(cov))),
            eps,
        )

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(
            cov,
            vmin=-cov_abs_max,
            vmax=cov_abs_max,
            # cmap="summer",
        )
        ax.set_title("Latent covariance matrix")
        ax.set_xlabel("latent dimension")
        ax.set_ylabel("latent dimension")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(
            save_dir / f"{prefix}_covariance.png",
            dpi=200,
        )
        plt.close(fig)

        # 4. correlation matrix
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(
            corr,
            vmin=-1.0,
            vmax=1.0,
            # cmap="coolwarm",
        )
        ax.set_title("Latent correlation matrix")
        ax.set_xlabel("latent dimension")
        ax.set_ylabel("latent dimension")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(
            save_dir / f"{prefix}_correlation.png",
            dpi=200,
        )
        plt.close(fig)

        # 5. eigenvalue spectrum
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(eigvals)
        ax.axhline(
            eigvals.mean(),
            linestyle="--",
            linewidth=1.0,
            label=f"mean={eigvals.mean():.4f}",
        )
        ax.set_title("Covariance eigenvalue spectrum")
        ax.set_xlabel("rank")
        ax.set_ylabel("eigenvalue")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.5)
        fig.tight_layout()
        fig.savefig(
            save_dir / f"{prefix}_eigenvalues.png",
            dpi=200,
        )
        plt.close(fig)

        # 6. normalized eigenvalue spectrum
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(normalized_eigvals)
        ax.axhline(
            1.0,
            linestyle="--",
            linewidth=1.0,
        )
        ax.set_title(
            "Normalized covariance eigenvalue spectrum"
        )
        ax.set_xlabel("rank")
        ax.set_ylabel("eigenvalue / mean eigenvalue")
        ax.grid(True, linestyle=":", alpha=0.5)
        fig.tight_layout()
        fig.savefig(
            save_dir
            / f"{prefix}_normalized_eigenvalues.png",
            dpi=200,
        )
        plt.close(fig)

        # 7. PCA explained variance ratio
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(explained)
        ax.axhline(
            1.0 / D,
            linestyle="--",
            linewidth=1.0,
            label=f"isotropic reference=1/{D}",
        )
        ax.set_title("PCA explained variance ratio")
        ax.set_xlabel("principal component")
        ax.set_ylabel("explained variance ratio")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.5)
        fig.tight_layout()
        fig.savefig(
            save_dir / f"{prefix}_explained_ratio.png",
            dpi=200,
        )
        plt.close(fig)

        # 8. PCA 3D scatter
        if min(N, D) >= 3:
            pca3 = PCA(n_components=3)
            latents_pca3 = pca3.fit_transform(latents)
            pca3_var = pca3.explained_variance_ratio_

            fig = plt.figure(figsize=(8, 7))
            ax = fig.add_subplot(
                111,
                projection="3d",
            )

            ax.scatter(
                latents_pca3[:, 0],
                latents_pca3[:, 1],
                latents_pca3[:, 2],
                s=8,
                alpha=0.45,
            )

            ax.set_title(
                "Latent PCA 3D scatter\n"
                f"variance explained: "
                f"{pca3_var[0]:.3f}, "
                f"{pca3_var[1]:.3f}, "
                f"{pca3_var[2]:.3f}"
            )
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_zlabel("PC3")

            fig.tight_layout()
            fig.savefig(
                save_dir / f"{prefix}_pca3d_scatter.png",
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)

            metrics["pca3_top1_explained_ratio"] = float(
                pca3_var[0]
            )
            metrics["pca3_top3_explained_ratio"] = float(
                pca3_var.sum()
            )

        # PCA 3Dの指標も含めて最後にJSON保存
        with open(
            save_dir / f"{prefix}_isotropy_metrics.json",
            "w",
        ) as file:
            json.dump(
                metrics,
                file,
                indent=2,
                allow_nan=True,
            )

    return metrics
