"""Cross Entropy Method solver for model-based planning."""

import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from loguru import logger as logging

from .solver import Costable

import matplotlib.pyplot as plt



class CEMSolver:
    """Cross Entropy Method solver for action optimization.

    Args:
        model: World model implementing the Costable protocol.
        batch_size: Number of environments to process in parallel.
        num_samples: Number of action candidates to sample per iteration.
        var_scale: Initial variance scale for the action distribution.
        n_steps: Number of CEM iterations.
        topk: Number of elite samples to keep for distribution update.
        device: Device for tensor computations.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        model: Costable,
        batch_size: int = 1,
        num_samples: int = 300,
        var_scale: float = 1,
        n_steps: int = 30,
        topk: int = 30,
        device: str | torch.device = "cpu",
        seed: int = 1234,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.var_scale = var_scale
        self.num_samples = num_samples
        self.n_steps = n_steps
        self.topk = topk
        self.device = device
        self.torch_gen = torch.Generator(device=device).manual_seed(seed)

    def configure(self, *, n_envs: int, config: Any, action_processor = None, action_space: gym.Space = None) -> None:
        """Configure the solver with environment specifications."""

        self._action_space = action_space
        self._action_processor = action_processor
        
        self._n_envs = n_envs
        self._config = config
        self._action_dim = int(np.prod(action_space.shape[1:]))
        self._configured = True
        
        
        low = action_space.low
        high = action_space.high

        mean = action_processor.mean_.reshape(1, -1)
        scale = action_processor.scale_.reshape(1, -1)

        norm_low = action_processor.normed_min_
        norm_high = action_processor.normed_max_

        self._norm_low = torch.as_tensor(norm_low, dtype=torch.float32, device=self.device)
        self._norm_high = torch.as_tensor(norm_high, dtype=torch.float32, device=self.device)
        self.clip_action = config.clip_action

        # print("mean:", mean)
        # print("scale:", scale)
        # print("norm_low:", self._norm_low)
        # print("norm_high:", self._norm_high)

        if not isinstance(action_space, Box):
            logging.warning(f"Action space is discrete, got {type(action_space)}. CEMSolver may not work as expected.")

    @property
    def n_envs(self) -> int:
        """Number of parallel environments."""
        return self._n_envs

    @property
    def action_dim(self) -> int:
        """Flattened action dimension including action_block grouping."""
        return self._action_dim * self._config.action_block

    @property
    def horizon(self) -> int:
        """Planning horizon in timesteps."""
        return self._config.horizon

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        """Make solver callable, forwarding to solve()."""
        return self.solve(*args, **kwargs)

    def init_action_distrib(
        self, actions: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialize the action distribution parameters (mean and variance)."""
        var = self.var_scale * torch.ones([self.n_envs, self.horizon, self.action_dim])
        mean = torch.zeros([self.n_envs, 0, self.action_dim]) if actions is None else actions

        remaining = self.horizon - mean.shape[1]
        if remaining > 0:
            device = mean.device
            new_mean = torch.zeros([self.n_envs, remaining, self.action_dim])
            mean = torch.cat([mean, new_mean], dim=1).to(device)

        return mean, var

    @torch.inference_mode()
    def solve(
        self, info_dict: dict, init_action: torch.Tensor | None = None, action_projector=None, projection_state=None,
    ) -> dict:
        """Solve the planning problem using Cross Entropy Method."""
        start_time = time.time()
        outputs = {
            "costs": [],
            "mean": [],  # History of means
            "var": [],  # History of vars
        }


        if (
            action_projector is not None
            and projection_state is None
        ):
            raise ValueError(
                "projection_state is required "
                "when action_projector is enabled"
            )
        


        # -- initialize the action distribution globally
        mean, var = self.init_action_distrib(init_action)
        mean = mean.to(self.device)
        var = var.to(self.device)

        total_envs = self.n_envs
        
        #CEM内の更新をログ
        outputs["mean_n_steps"] = [torch.empty_like(mean).cpu() for _ in range(self.n_steps)]
        outputs["std_n_steps"] = [torch.empty_like(var).cpu() for _ in range(self.n_steps)]
        outputs["elite_cost_mean_n_steps"] = [torch.empty(self.n_envs).cpu() for _ in range(self.n_steps)]
        outputs["elite_cost_min_n_steps"] = [torch.empty(self.n_envs).cpu() for _ in range(self.n_steps)]

        # --- Iterate over batches ---
        for start_idx in range(0, total_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_envs)
            current_bs = end_idx - start_idx

            # Slice Distribution Parameters for current batch
            batch_mean = mean[start_idx:end_idx]
            batch_var = var[start_idx:end_idx]

            # Expand Info Dict for current batch
            expanded_infos = {}
            for k, v in info_dict.items():
                # v is shape (n_envs, ...)
                # Slice batch
                v_batch = v[start_idx:end_idx]
                if torch.is_tensor(v):
                    # Add sample dim: (batch, 1, ...)
                    v_batch = v_batch.unsqueeze(1)
                    # Expand: (batch, num_samples, ...)
                    v_batch = v_batch.expand(current_bs, self.num_samples, *v_batch.shape[2:])
                elif isinstance(v, np.ndarray):
                    v_batch = np.repeat(v_batch[:, None, ...], self.num_samples, axis=1)
                expanded_infos[k] = v_batch

            # Optimization Loop
            final_batch_cost = None

            for step in range(self.n_steps):
                if step == 0:
                    # print("\n=== INIT ===")
                    # print("mean[0,0]:", batch_mean[0, 0].cpu().numpy())
                    # print("var[0,0]:", batch_var[0, 0].cpu().numpy())
                    pass
                
                # Sample action sequences: (Batch, Num_Samples, Horizon, Dim)
                candidates = torch.randn(
                    current_bs,
                    self.num_samples,
                    self.horizon,
                    self.action_dim,
                    generator=self.torch_gen,
                    device=self.device,
                )
                
                
                # print(f"\n=== STEP {step} ===")
                # print("candidates stats:")
                # print("min:", candidates.min().item())
                # print("max:", candidates.max().item())

                # Scale and shift: (Batch, N, H, D) * (Batch, 1, H, D) + (Batch, 1, H, D)
                candidates = candidates * batch_var.unsqueeze(1) + batch_mean.unsqueeze(1)

                # Force the first sample to be the current mean
                candidates[:, 0] = batch_mean
                
                #clamp
                low = self._norm_low[start_idx:end_idx].view(current_bs, 1, 1, -1)
                high = self._norm_high[start_idx:end_idx].view(current_bs, 1, 1, -1)
                if self.clip_action:
                    candidates = torch.maximum(candidates, low)
                    candidates = torch.minimum(candidates, high)
                
                

                current_info = expanded_infos.copy()


                # ----------------------------------
                # Action projection
                # ----------------------------------
                if action_projector is not None:

                    (
                        candidates_for_model,
                        feasible_mask,
                    ) = self._project_candidates(
                        candidates=candidates,
                        action_projector=action_projector,
                        projection_state=projection_state,
                        start_idx=start_idx,
                        current_bs=current_bs,
                    )

                else:
                    candidates_for_model = candidates
                    feasible_mask = None
                
                

                # Evaluate candidates
                costs = self.model.get_cost(current_info, candidates_for_model)

                if feasible_mask is not None:
                    costs = costs.masked_fill(~feasible_mask, float("inf"),)


                assert isinstance(costs, torch.Tensor), f"Expected cost to be a torch.Tensor, got {type(costs)}"
                assert costs.ndim == 2 and costs.shape[0] == current_bs and costs.shape[1] == self.num_samples, (
                    f"Expected cost to be of shape ({current_bs}, {self.num_samples}), got {costs.shape}"
                )

                # Select Top-K
                # topk_vals: (Batch, K), topk_inds: (Batch, K)
                topk_vals, topk_inds = torch.topk(costs, k=self.topk, dim=1, largest=False)

                # Gather Top-K Candidates
                # We need to select the specific candidates corresponding to topk_inds
                batch_indices = torch.arange(current_bs, device=self.device).unsqueeze(1).expand(-1, self.topk)

                # Indexing: candidates[batch_idx, sample_idx]
                # Result shape: (Batch, K, Horizon, Dim)
                # topk_candidates = candidates[batch_indices, topk_inds]
                if action_projector is not None:
                    topk_candidates = candidates_for_model[batch_indices, topk_inds]

                else:
                    topk_candidates = candidates[batch_indices, topk_inds]

                # Update Mean and Variance based on Top-K
                batch_mean = topk_candidates.mean(dim=1)
                batch_var = topk_candidates.std(dim=1)

                #clamp
                low_mean = self._norm_low[start_idx:end_idx].view(current_bs, 1, -1)
                high_mean = self._norm_high[start_idx:end_idx].view(current_bs, 1, -1)
                if self.clip_action:
                    batch_mean = torch.maximum(batch_mean, low_mean)
                    batch_mean = torch.minimum(batch_mean, high_mean)




                # Update final cost for logging
                # We average the cost of the top elites
                final_batch_cost = topk_vals.mean(dim=1).cpu().tolist()
                
                
            
                # print("updated mean stats:")
                # print("mean min:", batch_mean.min().item())
                # print("mean max:", batch_mean.max().item())

                # print("updated var stats:")
                # print("var min:", batch_var.min().item())
                # print("var max:", batch_var.max().item())
                                
                                
                outputs["mean_n_steps"][step][start_idx:end_idx] = batch_mean.detach().cpu()
                outputs["std_n_steps"][step][start_idx:end_idx] = batch_var.detach().cpu()
                outputs["elite_cost_mean_n_steps"][step][start_idx:end_idx] = topk_vals.mean(dim=1).detach().cpu()
                outputs["elite_cost_min_n_steps"][step][start_idx:end_idx] = topk_vals.min(dim=1).values.detach().cpu()
                
                

            # Write results back to global storage
            mean[start_idx:end_idx] = batch_mean
            var[start_idx:end_idx] = batch_var

            # Store history/metadata
            outputs["costs"].extend(final_batch_cost)

        outputs["actions"] = mean.detach().cpu()
        outputs["mean"] = [mean.detach().cpu()]
        outputs["var"] = [var.detach().cpu()]

        print(f"CEM solve time: {time.time() - start_time:.4f} seconds")
        return outputs



    def _project_candidates(
        self,
        candidates,
        action_projector,
        projection_state,
        start_idx,
        current_bs,
    ):
        """
        CEM candidateを物理空間へ戻し、
        ActionProjectorをhorizon方向に逐次適用した後、
        再び正規化空間へ戻す。

        Args:
            candidates:
                shape:
                (batch, num_samples, horizon, action_dim)

                normalized action space

        Returns:
            projected_candidates:
                candidatesと同shape
                projected後のnormalized action

            feasible_mask:
                shape (batch, num_samples)
        """

        if self._config.action_block != 1:
            raise NotImplementedError(
                "Action projection currently assumes "
                "action_block == 1"
            )

        # -----------------------------
        # normalized -> physical
        # -----------------------------
        candidates_np = (
            candidates
            .detach()
            .cpu()
            .numpy()
        )

        original_shape = candidates_np.shape

        flat_candidates = candidates_np.reshape(
            -1,
            original_shape[-1],
        )

        physical_flat = (
            self._action_processor
            .inverse_transform(flat_candidates)
        )

        physical_candidates = physical_flat.reshape(
            original_shape
        ).astype(np.float32)

        # -----------------------------
        # output buffers
        # -----------------------------
        projected_physical = np.empty_like(
            physical_candidates,
            dtype=np.float32,
        )

        feasible_mask = np.ones(
            (current_bs, self.num_samples),
            dtype=bool,
        )

        # -----------------------------
        # initial physical state
        # -----------------------------
        qpos_all = np.asarray(
            projection_state["qpos"],
            dtype=np.float32,
        )

        ee_all = np.asarray(
            projection_state["ee"],
            dtype=np.float32,
        )

        gripper_all = np.asarray(
            projection_state["gripper"],
            dtype=np.float32,
        )

        # n_envs=1のとき shape を揃える
        if qpos_all.ndim == 1:
            qpos_all = qpos_all[None, :]

        if ee_all.ndim == 1:
            ee_all = ee_all[None, :]

        if gripper_all.ndim == 0:
            gripper_all = gripper_all[None]

        # -----------------------------
        # each environment
        # -----------------------------
        for batch_idx in range(current_bs):

            env_idx = start_idx + batch_idx

            initial_qpos = qpos_all[env_idx]
            initial_ee = ee_all[env_idx]
            initial_gripper = float(
                gripper_all[env_idx]
            )

            # -------------------------
            # each CEM particle
            # -------------------------
            for sample_idx in range(
                self.num_samples
            ):

                simulated_qpos = (
                    initial_qpos.copy()
                )

                simulated_ee = (
                    initial_ee.copy()
                )

                simulated_gripper = (
                    initial_gripper
                )

                # ---------------------
                # rollout over horizon
                # ---------------------
                for horizon_idx in range(
                    self.horizon
                ):

                    raw_action = physical_candidates[
                        batch_idx,
                        sample_idx,
                        horizon_idx,
                    ]

                    result = action_projector.project(
                        action=raw_action,
                        current_qpos=simulated_qpos,
                        current_ee=simulated_ee,
                        current_gripper=simulated_gripper,
                    )

                    if not result.feasible:
                        feasible_mask[
                            batch_idx,
                            sample_idx,
                        ] = False

                        # このparticleはどうせ無効なので
                        # 残りの計算を止める
                        break

                    projected_physical[
                        batch_idx,
                        sample_idx,
                        horizon_idx,
                    ] = result.action

                    # 次stepの仮想状態
                    simulated_qpos = (
                        result.qpos.copy()
                    )

                    simulated_ee = (
                        result.ee.copy()
                    )

                    simulated_gripper = float(
                        result.action[7]
                    )

                # IK failureなどが起きた場合、
                # 未初期化領域を残さない
                if not feasible_mask[
                    batch_idx,
                    sample_idx,
                ]:
                    projected_physical[
                        batch_idx,
                        sample_idx,
                    ] = physical_candidates[
                        batch_idx,
                        sample_idx,
                    ]

        # -----------------------------
        # physical -> normalized
        # -----------------------------
        projected_flat = projected_physical.reshape(
            -1,
            original_shape[-1],
        )

        normalized_flat = (
            self._action_processor
            .transform(projected_flat)
        )

        projected_normalized = (
            normalized_flat
            .reshape(original_shape)
            .astype(np.float32)
        )

        projected_candidates = torch.as_tensor(
            projected_normalized,
            dtype=candidates.dtype,
            device=candidates.device,
        )

        feasible_mask = torch.as_tensor(
            feasible_mask,
            dtype=torch.bool,
            device=candidates.device,
        )

        return (
            projected_candidates,
            feasible_mask,
        )