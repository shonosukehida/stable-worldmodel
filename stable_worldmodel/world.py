"""World environment manager for vectorized Gymnasium environments."""

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

from stable_worldmodel.data.utils import get_cache_dir
from stable_worldmodel.policy import Policy

from .wrapper import MegaWrapper, SyncWorld, VariationWrapper

from stable_worldmodel.plot.plot import plot_task_result_xy, plot_joint_angle_comparison, plot_cartesian_comparison
import matplotlib.pyplot as plt

from stable_worldmodel.plot.plot import plot_cem_rollout_cost


def _make_env(env_name, max_episode_steps, wrappers, **kwargs):
    """Create a gymnasium environment with specified wrappers.

    Factory function for creating environments within a vectorized setup.
    Creates the base environment and applies wrappers in order.

    Args:
        env_name: Name of the gymnasium environment to create.
        max_episode_steps: Maximum steps per episode before truncation.
        wrappers: List of wrapper functions/classes to apply. Each wrapper
            should accept an environment and return a wrapped environment.
        **kwargs: Additional keyword arguments passed to gym.make.

    Returns:
        The wrapped gymnasium environment.

    Example:
        >>> from functools import partial
        >>> wrappers = [partial(MegaWrapper, image_shape=(64, 64))]
        >>> env = _make_env("CartPole-v1", max_episode_steps=500, wrappers=wrappers)
    """
    env = gym.make(env_name, max_episode_steps=max_episode_steps, **kwargs)
    print("env:", env)
    for wrapper in wrappers:
        env = wrapper(env)
    return env


class World:
    """High-level manager for vectorized Gymnasium environments.

    Manages a set of synchronized vectorized environments with automatic
    preprocessing (resizing, frame stacking, goal conditioning).

    Args:
        env_name: Name of the Gymnasium environment to create.
        num_envs: Number of parallel environments.
        image_shape: Target shape for image observations (H, W).
        goal_transform: Optional callable to transform goal observations.
        image_transform: Optional callable to transform image observations.
        seed: Random seed for reproducibility.
        history_size: Number of frames to stack.
        frame_skip: Number of frames to skip per step.
        max_episode_steps: Maximum steps per episode before truncation.
        verbose: Verbosity level (0: silent, >0: info).
        extra_wrappers: List of additional wrappers to apply to each env.
        goal_conditioned: Whether to separate goal from observation.
        **kwargs: Additional keyword arguments passed to `gym.make_vec`.
    """

    def __init__(
        self,
        env_name: str,
        num_envs: int,
        image_shape: tuple[int, int],
        goal_transform: Callable[[Any], Any] | None = None,
        image_transform: Callable[[Any], Any] | None = None,
        seed: int = 2349867,
        history_size: int = 1,
        frame_skip: int = 1,
        max_episode_steps: int = 100,
        verbose: int = 1,
        extra_wrappers: list[Callable] | None = None,
        goal_conditioned: bool = True,
        **kwargs: Any,
    ) -> None:
        wrappers = [
            partial(
                MegaWrapper,
                image_shape=image_shape,
                pixels_transform=image_transform,
                goal_transform=goal_transform,
                history_size=history_size,
                frame_skip=frame_skip,
                separate_goal=goal_conditioned,
            ),
            *(extra_wrappers or []),
        ]

        env_fn = partial(
            _make_env, env_name, max_episode_steps, wrappers, **kwargs
        )
        self.env_fn = env_fn
        env_fns = [env_fn for _ in range(num_envs)]
        # print("self.envs:", env_fns[0])
        self.envs: VectorEnv = VariationWrapper(SyncWorld(env_fns))
        self.envs.unwrapped.autoreset_mode = gym.vector.AutoresetMode.DISABLED
        
        # print("type(self.envs):", type(self.envs))
        

        self._history_size = history_size
        self.policy: Policy | None = None
        self.states: dict | None = None
        self.infos: dict = {}
        self.rewards: np.ndarray | None = None
        self.terminateds: np.ndarray | None = None
        self.truncateds: np.ndarray | None = None


        if verbose > 0:
            logging.info(f'🌍🌍🌍 World {env_name} initialized 🌍🌍🌍')

            logging.info('🕹️ 🕹️ 🕹️ Action space 🕹️ 🕹️ 🕹️')
            logging.info(f'{self.envs.action_space}')

            logging.info('👁️ 👁️ 👁️ Observation space 👁️ 👁️ 👁️')
            logging.info(f'{str(self.envs.observation_space)}')

            if self.envs.variation_space is not None:
                logging.info('⚗️ ⚗️ ⚗️ Variation space ⚗️ ⚗️ ⚗️')
                print(self.single_variation_space.to_str())
            else:
                logging.warning('No variation space provided!')

        self.seed = seed

    @property
    def num_envs(self) -> int:
        """Number of parallel environment instances."""
        return self.envs.num_envs

    @property
    def observation_space(self) -> gym.Space:
        """Batched observation space for all environments."""
        return self.envs.observation_space

    @property
    def action_space(self) -> gym.Space:
        """Batched action space for all environments."""
        return self.envs.action_space

    @property
    def variation_space(self) -> gym.Space | None:
        """Batched variation space for domain randomization."""
        return self.envs.variation_space

    @property
    def single_variation_space(self) -> gym.Space | None:
        """Variation space for a single environment instance."""
        return self.envs.single_variation_space

    @property
    def single_action_space(self) -> gym.Space:
        """Action space for a single environment instance."""
        return self.envs.single_action_space

    @property
    def single_observation_space(self) -> gym.Space:
        """Observation space for a single environment instance."""
        return self.envs.single_observation_space

    def close(self, **kwargs: Any) -> None:
        """Close all environments and clean up resources."""
        return self.envs.close(**kwargs)

    def step(self) -> None:
        """Advance all environments by one step using the current policy."""
        # note: reset happens before because of auto-reset, should fix that
        if self.policy is None:
            raise RuntimeError('No policy set. Call set_policy() first.')

        # print("(stable_worldmodel/world.py, step)self.infos.keys(): ", self.infos.keys())
        actions = self.policy.get_action(self.infos)
        
        (
            self.states,
            self.rewards,
            self.terminateds,
            self.truncateds,
            self.infos,
        ) = self.envs.step(actions)

    def reset(
        self,
        seed: int | list[int] | None = None,
        options: dict | None = None,
    ) -> None:
        """Reset all environments to initial states.

        Args:
            seed: Random seed(s) for the environments.
            options: Additional options passed to the environment reset.
        """
        
        
        self.states, self.infos = self.envs.reset(seed=seed, options=options)
        # print("(stable_worldmodel/world.py) self.env:", type(self.envs))
        # print("(stable_worldmodel/world.py) self.infos.keys():", self.infos.keys())

    def set_policy(self, policy: Policy, results_path = None) -> None:
        """Attach a policy to the world.

        Args:
            policy: The policy instance to use for determining actions.
        """
        self.policy = policy
        self.policy.set_env(self.envs)

        if hasattr(self.policy, 'seed') and self.policy.seed is not None:
            self.policy.set_seed(self.policy.seed)
            
        self.policy.results_path = results_path
        self.results_path = results_path

    def record_video(
        self,
        video_path: str | Path,
        max_steps: int = 500,
        fps: int = 30,
        viewname: str | list[str] = 'pixels',
        seed: int | None = None,
        extension: str = 'mp4',
        options: dict | None = None,
    ) -> None:
        """Record rollout videos for each environment under the current policy.

        Args:
            video_path: Directory path to save the videos.
            max_steps: Maximum steps to record per environment.
            fps: Frames per second for the output video.
            viewname: Key(s) in `infos` containing image data to render.
            seed: Random seed for reset.
            extension: Video file format ('mp4' or 'gif').
            options: Options for reset.
        """

        assert extension in ['mp4', 'gif'], (
            'Unsupported video format. Use "mp4" or "gif".'
        )

        import imageio

        viewname = [viewname] if isinstance(viewname, str) else viewname
        out = [
            imageio.get_writer(
                Path(video_path) / f'env_{i}.{extension}',
                fps=fps,
                codec='libx264',
            )
            for i in range(self.num_envs)
        ]

        self.reset(seed, options)

        for i, o in enumerate(out):
            frames_to_stack = []
            for v_name in viewname:
                frame_data = self.infos[v_name][i]
                # if frame_data has a history dimension, take the last frame
                if frame_data.ndim > 3:
                    frame_data = frame_data[-1]
                frames_to_stack.append(frame_data)
            frame = np.vstack(frames_to_stack)

            if 'goal' in self.infos:
                goal_data = self.infos['goal'][i]
                if goal_data.ndim > 3:
                    goal_data = goal_data[-1]
                frame = np.vstack([frame, goal_data])
            o.append_data(frame)

        for _ in range(max_steps):
            self.step()

            if np.any(self.terminateds) or np.any(self.truncateds):
                break

            for i, o in enumerate(out):
                frames_to_stack = []
                for v_name in viewname:
                    frame_data = self.infos[v_name][i]
                    # if frame_data has a history dimension, take the last frame
                    if frame_data.ndim > 3:
                        frame_data = frame_data[-1]
                    frames_to_stack.append(frame_data)
                frame = np.vstack(frames_to_stack)

                if 'goal' in self.infos:
                    goal_data = self.infos['goal'][i]
                    if goal_data.ndim > 3:
                        goal_data = goal_data[-1]
                    frame = np.vstack([frame, goal_data])
                o.append_data(frame)
        for o in out:
            o.close()
        print(f'Video saved to {video_path}')

    def record_dataset(
        self,
        dataset_name: str,
        episodes: int = 10,
        seed: int | None = None,
        cache_dir: os.PathLike | str | None = None,
        options: dict | None = None,
    ) -> None:
        """Records episodes from the environment into an HDF5 dataset.

        Args:
            dataset_name: Name of the dataset file (without extension).
            episodes: Total number of episodes to record.
            seed: Initial random seed.
            cache_dir: Directory to save the dataset. Defaults to standard cache.
            options: Reset options passed to environments.

        Raises:
            NotImplementedError: If history_size > 1.
        """
        if self._history_size > 1:
            raise NotImplementedError(
                'Frame history > 1 not supported for dataset recording.'
            )

        path = (
            get_cache_dir(cache_dir, sub_folder='datasets')
            / f'{dataset_name}.h5'
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        self.terminateds = np.zeros(self.num_envs, dtype=bool)
        self.truncateds = np.zeros(self.num_envs, dtype=bool)

        episode_buffers = [defaultdict(list) for _ in range(self.num_envs)]

        h5_kwargs = {
            'name': str(path),
            'mode': 'a' if path.exists() else 'w',
            'libver': 'latest',
        }

        if not path.exists():  # creation only args
            h5_kwargs.update(
                {'fs_strategy': 'page', 'fs_page_size': 4 * 1024 * 1024}
            )

        with h5py.File(**h5_kwargs) as f:
            f.swmr_mode = True  # avoid issue when killed

            if 'ep_len' in f:
                n_ep_recorded = f['ep_len'].shape[0]
                global_step_ptr = (
                    f['ep_offset'][-1] + f['ep_len'][-1]
                    if n_ep_recorded > 0
                    else 0
                )
                initialized = True
                seed = None if seed is None else (seed + n_ep_recorded)
                logging.info(
                    f'Resuming: {n_ep_recorded} episodes already on disk.'
                )
            else:
                n_ep_recorded = 0
                global_step_ptr = 0
                initialized = False

            self.reset(seed, options=options)
            seed = None if seed is None else (seed + self.num_envs)
            self._dump_step_data(episode_buffers)  # record initial state

            with tqdm(
                total=episodes, initial=n_ep_recorded, desc='Recording'
            ) as pbar:
                while n_ep_recorded < episodes:
                    self.step()
                    self._dump_step_data(episode_buffers)

                    for i in range(self.num_envs):
                        if self.terminateds[i] or self.truncateds[i]:
                            finished_ep = self._handle_done_ep(
                                episode_buffers, i, n_ep_recorded
                            )

                            # lazy dataset initialization
                            if not initialized:
                                self._init_h5_datasets(f, finished_ep)
                                initialized = True

                            # contiguous writing
                            steps_written = self._write_episode(
                                f, finished_ep, global_step_ptr
                            )
                            global_step_ptr += steps_written
                            n_ep_recorded += 1
                            pbar.update(1)

                            f.flush()  # flush metadata to avoid corruption

                            if n_ep_recorded >= episodes:
                                break

                            # reset terminated env and record initial state
                            n_seed = (
                                None
                                if seed is None
                                else (seed + n_ep_recorded)
                            )
                            self._reset_single_env(i, n_seed, options)
                            self._dump_step_data(episode_buffers, env_idx=i)

        logging.info(f'Recording complete. Total frames: {global_step_ptr}')

    def _init_h5_datasets(
        self, f: h5py.File, sample_episode: dict[str, list[Any]]
    ) -> None:
        """Initialize resizable HDF5 datasets based on the first episode.

        Args:
            f: The open HDF5 file handle.
            sample_episode: A dictionary containing data from a single episode,
                used to determine shapes and dtypes.
        """
        for key, data_list in sample_episode.items():
            if key in ['ep_len', 'ep_idx', 'policy']:
                continue

            key = key.replace('/', '_')  # sanitize keys for h5

            # determine array shape and dtype from sample data
            sample_data = np.array(data_list[0])
            shape = (0,) + sample_data.shape
            maxshape = (None,) + sample_data.shape

            # determine chunk size and compression
            if sample_data.ndim >= 2:
                chunks = (100,) + sample_data.shape
                compression = hdf5plugin.Blosc(
                    cname='lz4', clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE
                )

            else:
                chunks = (1000,) + sample_data.shape
                compression = None

            dtype = sample_data.dtype
            if np.issubdtype(dtype, np.str_) or np.issubdtype(
                dtype, np.bytes_
            ):
                dtype = h5py.string_dtype()

            f.create_dataset(
                key,
                shape=shape,
                maxshape=maxshape,
                dtype=dtype,
                chunks=chunks,
                compression=compression,
            )

        # index metadata
        f.create_dataset(
            'ep_offset', shape=(0,), maxshape=(None,), dtype=np.int64
        )
        f.create_dataset(
            'ep_len', shape=(0,), maxshape=(None,), dtype=np.int32
        )

        # per-step episode index
        f.create_dataset(
            'ep_idx',
            shape=(0,),
            maxshape=(None,),
            dtype=np.int32,
            chunks=(1000,),
        )

    def _reset_single_env(
        self,
        env_idx: int,
        seed: int | None = None,
        options: dict | None = None,
    ) -> None:
        """Reset a single environment and update infos dict.

        Args:
            env_idx: Index of the environment to reset.
            seed: Random seed for this specific environment.
            options: Reset options.
        """
        self.envs.unwrapped._autoreset_envs = np.zeros(self.num_envs)
        _, infos = self.envs.envs[env_idx].reset(seed=seed, options=options)

        for k, v in infos.items():
            if k in self.infos:
                self.infos[k][env_idx] = v

    def _handle_done_ep(
        self,
        tmp_buffer: list[dict[str, list[Any]]],
        env_idx: int,
        n_ep_recorded: int,
    ) -> dict[str, list[Any]]:
        """Prepare the episode buffer for writing.

        Args:
            tmp_buffer: List of dictionaries accumulating step data per env.
            env_idx: Index of the environment that finished an episode.
            n_ep_recorded: Number of episodes recorded so far.

        Returns:
            A dictionary containing the complete episode data.
        """
        ep_buffer = tmp_buffer[env_idx]

        # left-shift actions to align with observations i.e. (o_t, a_t)
        if 'action' in ep_buffer:
            actions = ep_buffer['action']
            nan = actions.pop(0)
            actions.append(nan)

        # Extract a copy and clear the temporary buffer
        out = {k: list(v) for k, v in ep_buffer.items()}
        ep_buffer.clear()
        self.terminateds[env_idx] = False
        self.truncateds[env_idx] = False

        # Add episode index to all steps
        ep_len = len(out['step_idx'])
        out['ep_idx'] = [n_ep_recorded] * ep_len

        return out

    def _write_episode(
        self, f: h5py.File, ep_data: dict[str, list[Any]], global_ptr: int
    ) -> int:
        """Write a single contiguous episode to the HDF5 file.

        Args:
            f: The open HDF5 file handle.
            ep_data: The episode data dictionary.
            global_ptr: The global step index where this episode starts.

        Returns:
            The length of the episode written.
        """
        ep_len = len(ep_data['step_idx'])

        # append data to each dataset
        for key in ep_data:
            h5_key = key.replace('/', '_')  # sanitize keys for h5
            if h5_key in ['ep_len', 'policy']:
                continue

            ds = f[h5_key]
            curr_size = ds.shape[0]
            ds.resize(curr_size + ep_len, axis=0)
            ds[curr_size:] = np.array(ep_data[key])

        # update metadata
        meta_idx = f['ep_offset'].shape[0]
        f['ep_offset'].resize(meta_idx + 1, axis=0)
        f['ep_len'].resize(meta_idx + 1, axis=0)

        f['ep_offset'][meta_idx] = global_ptr
        f['ep_len'][meta_idx] = ep_len

        return ep_len

    def _dump_step_data(
        self,
        tmp_buffer: list[dict[str, list[Any]]],
        env_idx: int | None = None,
    ) -> None:
        """Append current step data to temporary episode buffers.

        Args:
            tmp_buffer: List of dictionaries accumulating step data.
            env_idx: Optional index to dump data for a single environment.
                If None, dumps for all environments.
        """
        env_indices = range(self.num_envs) if env_idx is None else [env_idx]

        for col, data in self.infos.items():
            if col.startswith('_'):
                continue

            # normalize data shape and type
            if isinstance(data, np.ndarray):
                data = (
                    np.squeeze(data, axis=1)
                    if data.ndim > 1 and data.shape[1] == 1
                    else data
                )
                if data.dtype == object:
                    data = np.concatenate(data).tolist()

            # append to buffers
            for i in env_indices:
                env_data = (
                    data[i].copy()
                    if isinstance(data[i], np.ndarray)
                    else data[i]
                )
                tmp_buffer[i][col].append(env_data)

    def evaluate(
        self,
        episodes: int = 10,
        eval_keys: list[str] | None = None,
        seed: int | None = None,
        options: dict | None = None,
        dump_every: int = -1,
    ) -> dict:
        """Evaluate the current policy over multiple episodes.

        Args:
            episodes: Number of episodes to evaluate.
            eval_keys: List of keys in `infos` to collect and return.
            seed: Random seed for evaluation.
            options: Reset options.
            dump_every: Interval to save intermediate results (for long evals).

        Returns:
            Dictionary containing success rates, seeds, and collected keys.
        """
        options = options or {}

        results: dict = {
            'episode_count': 0,
            'success_rate': 0.0,
            'episode_successes': np.zeros(episodes),
            'seeds': np.zeros(episodes, dtype=np.int32),
        }

        if eval_keys:
            for key in eval_keys:
                results[key] = np.zeros(episodes)

        self.terminateds = np.zeros(self.num_envs)
        self.truncateds = np.zeros(self.num_envs)

        episode_idx = np.arange(self.num_envs)
        self.reset(seed=seed, options=options)
        root_seed = seed + self.num_envs if seed is not None else None

        eval_done = False

        # determine "unique" hash for this eval run
        config = {
            'episodes': episodes,
            'eval_keys': tuple(sorted(eval_keys)) if eval_keys else None,
            'seed': seed,
            'options': tuple(sorted(options.items())) if options else None,
            'dump_every': dump_every,
        }

        config_str = json.dumps(config, sort_keys=True)
        run_hash = hashlib.sha256(config_str.encode()).hexdigest()[:8]
        run_tmp_path = Path(f'eval_tmp_{run_hash}.npy')

        # load back intermediate results if file exists
        if run_tmp_path.exists():
            tmp_results = np.load(run_tmp_path, allow_pickle=True).item()
            results.update(tmp_results)

            ep_count = results['episode_count']
            episode_idx = np.arange(ep_count, ep_count + self.num_envs)

            # reset seed where we left off
            last_seed = seed + ep_count if seed is not None else None
            self.reset(seed=last_seed, options=options)

            logging.success(
                f'Found existing eval tmp file {run_tmp_path}, resuming from episode {ep_count}/{episodes}'
            )

        while True:
            self.step()

            # start new episode for done envs
            for i in range(self.num_envs):
                if self.terminateds[i] or self.truncateds[i]:
                    # record eval info
                    ep_idx = episode_idx[i]
                    results['episode_successes'][ep_idx] = self.terminateds[i]
                    results['seeds'][ep_idx] = self.envs.envs[
                        i
                    ].unwrapped.np_random_seed

                    if eval_keys:
                        for key in eval_keys:
                            assert key in self.infos, (
                                f'key {key} not found in infos'
                            )
                            results[key][ep_idx] = self.infos[key][i]

                    # determine new episode idx
                    # re-reset env with seed and options (no supported by auto-reset)
                    new_seed = (
                        root_seed + results['episode_count']
                        if seed is not None
                        else None
                    )
                    next_ep_idx = episode_idx.max() + 1
                    episode_idx[i] = next_ep_idx
                    results['episode_count'] += 1

                    # break if enough episodes evaluated
                    if results['episode_count'] >= episodes:
                        eval_done = True
                        if run_tmp_path.exists():
                            logging.info(
                                f'Eval done, deleting tmp file {run_tmp_path}'
                            )
                            os.remove(run_tmp_path)
                        break

                    # dump temporary results in a file
                    if dump_every > 0 and (
                        results['episode_count'] % dump_every == 0
                    ):
                        np.save(run_tmp_path, results)
                        logging.success(
                            f'Dumped intermediate eval results to {run_tmp_path} ({results["episode_count"]}/{episodes})'
                        )
                    self.envs.unwrapped._autoreset_envs = np.zeros(
                        (self.num_envs,)
                    )
                    _, infos = self.envs.envs[i].reset(
                        seed=new_seed, options=options
                    )

                    for k, v in infos.items():
                        if k not in self.infos:
                            continue
                        # Convert to array and extract scalar to preserve dtype
                        self.infos[k][i] = np.asarray(v)

            if eval_done:
                break

        # compute success rate
        results['success_rate'] = (
            float(np.sum(results['episode_successes'])) / episodes * 100.0
        )

        assert results['episode_count'] == episodes, (
            f'episode_count {results["episode_count"]} != episodes {episodes}'
        )

        assert np.unique(results['seeds']).shape[0] == episodes, (
            'Some episode seeds are identical!'
        )

        return results

    def evaluate_from_dataset(
        self,
        dataset: Any,
        episodes_idx: Sequence[int],
        start_steps: Sequence[int],
        goal_offset_steps: int,
        eval_budget: int,
        callables: list[dict] | None = None,
        save_video: bool = True,
        video_path: str | Path = './',
    ) -> dict:
        """Evaluate the policy starting from states sampled from a dataset.

        Args:
            dataset: The source dataset to sample initial states/goals from.
            episodes_idx: Indices of episodes to sample from.
            start_steps: Step indices within those episodes to start from.
            goal_offset_steps: Number of steps ahead to look for the goal.
            eval_budget: Maximum steps allowed for the agent to reach the goal.
            callables: Optional list of method calls to setup the env.
            save_video: Whether to save rollout videos.
            video_path: Path to save videos.

        Returns:
            Dictionary containing success rates and other metrics.

        Raises:
            ValueError: If input sequence lengths mismatch or don't match num_envs.
        """
        assert (
            self.envs.envs[0].spec.max_episode_steps is None
            or self.envs.envs[0].spec.max_episode_steps >= goal_offset_steps
        ), 'env max_episode_steps must be greater than eval_budget'

        ep_idx_arr = np.array(episodes_idx)
        start_steps_arr = np.array(start_steps)
        # We add +1 so that the last loaded frame will align with the last frame we encounter
        # when stepping through the rollout.
        end_steps = start_steps_arr + goal_offset_steps + 1

        if not (len(ep_idx_arr) == len(start_steps_arr)):
            raise ValueError(
                'episodes_idx and start_steps must have the same length'
            )

        if len(ep_idx_arr) != self.num_envs:
            raise ValueError(
                'Number of episodes to evaluate must match number of envs'
            )

        print("")
        data = dataset.load_chunk(ep_idx_arr, start_steps_arr, end_steps)
        columns = dataset.column_names
        # print("columns:", columns) 
        
        
        
        # keep relevant part of the chunk
        init_step_per_env: dict[str, list[Any]] = defaultdict(list)
        goal_step_per_env: dict[str, list[Any]] = defaultdict(list)

        for i, ep in enumerate(data):
            # print("ep[action].shape:", ep["action"].shape) #[26, 7]
            # print("ep[pixels].shape:", ep["pixels"].shape) #[26, 3, 64, 64]
            
            
            for col in columns:
                if col.startswith('goal'):
                    continue
                if col.startswith('pixels'):
                    # permute channel to be last
                    ep[col] = ep[col].permute(0, 2, 3, 1)

                if not isinstance(ep[col], (torch.Tensor | np.ndarray)):
                    continue

                init_data = ep[col][0]
                goal_data = ep[col][-1]

                # TODO handle that better
                if not isinstance(init_data, (np.ndarray | torch.Tensor)):
                    logging.warning(
                        f'Data type {type(init_data)} for column {col} not supported, yet skipping conversion'
                    )
                    continue

                init_data = (
                    init_data.numpy()
                    if isinstance(init_data, torch.Tensor)
                    else init_data
                )
                goal_data = (
                    goal_data.numpy()
                    if isinstance(goal_data, torch.Tensor)
                    else goal_data
                )

                init_step_per_env[col].append(init_data)
                goal_step_per_env[col].append(goal_data)

        init_step = {
            k: np.stack(v) for k, v in deepcopy(init_step_per_env).items()
        }

        goal_step = {}
        for key, value in goal_step_per_env.items():
            key = 'goal' if key == 'pixels' else f'goal_{key}'
            goal_step[key] = np.stack(value)

        # get dataset info
        seeds = init_step.get('seed')
        # get dataset variation
        vkey = 'variation.'
        variations_dict = {
            k.removeprefix(vkey): v
            for k, v in init_step.items()
            if k.startswith(vkey)
        }

        options = [{} for _ in range(self.num_envs)]

        if len(variations_dict) > 0:
            for i in range(self.num_envs):
                options[i]['variation'] = list(variations_dict.keys())
                options[i]['variation_values'] = {
                    k: v[i] for k, v in variations_dict.items()
                }

        init_step.update(deepcopy(goal_step))
        self.reset(seed=seeds, options=options)  # set seeds for all envs

        # apply callable list (e.g used for set initial position if not access to seed)
        callables = callables or []
        for i, env in enumerate(self.envs.unwrapped.envs):
            env_unwrapped = env.unwrapped

            for spec in callables:
                method_name = spec['method']
                if not hasattr(env_unwrapped, method_name):
                    logging.warning(
                        f'Env {env_unwrapped} has no method {method_name}, skipping callable'
                    )
                    continue

                method = getattr(env_unwrapped, method_name)
                args = spec.get('args', spec)

                # prepare args
                prepared_args = {}
                for args_name, args_data in args.items():
                    value = args_data.get('value', None)
                    is_in_datset = args_data.get('in_dataset', True)

                    if is_in_datset:
                        if value not in init_step:
                            logging.warning(
                                f'Col {value} not found in dataset, skipping callable for env {env_unwrapped}'
                            )
                            continue
                        prepared_args[args_name] = deepcopy(
                            init_step[value][i]
                        )
                    else:
                        prepared_args[args_name] = args_data.get('value')

                # call method with prepared args
                method(**prepared_args)

        for i, env in enumerate(self.envs.unwrapped.envs):
            env_unwrapped = env.unwrapped

            # TODO remove this
            if 'goal_state' in init_step and 'goal_state' in goal_step:
                assert np.array_equal(
                    init_step['goal_state'][i], goal_step['goal_state'][i]
                ), 'Goal state info does not match at reset'

        results: dict = {
            'success_rate': 0.0,
            'episode_successes': np.zeros(len(episodes_idx)),
            'seeds': seeds,
        }

        # expend all data to the right shape (x, y, (original_shape))
        shape_prefix = self.infos['pixels'].shape[:2]

        # TODO get the data from the previous step in the dataset for history
        init_step = {
            k: np.broadcast_to(v[:, None, ...], shape_prefix + v.shape[1:])
            for k, v in init_step.items()
        }
        goal_step = {
            k: np.broadcast_to(v[:, None, ...], shape_prefix + v.shape[1:])
            for k, v in goal_step.items()
        }

        # print("(stable_worldmodel/world.py)(before) self.infos.keys(): ", self.infos.keys())
        
        # update the reset with our new init and goal infos
        self.infos.update(deepcopy(init_step))
        self.infos.update(deepcopy(goal_step))
        # print("(stable_worldmodel/world.py)(after) self.infos.keys(): ", self.infos.keys())

        if 'goal' in goal_step and 'goal' in self.infos:
            assert np.allclose(self.infos['goal'], goal_step['goal']), (
                'Goal info does not match'
            )

        target_frames = torch.stack([ep['pixels'] for ep in data]).numpy()
        video_frames = self._run_evaluation_rollout(
            goal_step=goal_step,
            eval_budget=eval_budget,
            results=results,
        )

        n_episodes = len(episodes_idx)

        # compute success rate
        results['success_rate'] = (
            float(np.sum(results['episode_successes'])) / n_episodes * 100.0
        )

        # save video if required
        if save_video:
            self._save_evaluation_videos(
                video_frames=video_frames,
                target_frames=target_frames,
                video_path=video_path,
            )

        if results['seeds'] is not None:
            assert np.unique(results['seeds']).shape[0] == n_episodes, (
                'Some episode seeds are identical!'
            )

        return results

    def evaluate_zeroshot(
        self,
        start_positions: Sequence[Any],
        goal_positions: Sequence[Any],
        init_ee_poses: Sequence[Any],
        goal_ee_poses: Sequence[Any],
        eval_budget: int,
        start_option_name: str = 'state',
        goal_option_name: str = 'target_state',
        init_ee_option_name: str = 'init_ee_pos',
        goal_ee_option_name: str = 'goal_ee_pos',
        start_info_name: str | None = 'state',
        goal_info_name: str | None = 'goal_state',
        seeds: int | Sequence[int | None] | None = None,
        reset_options: Sequence[dict[str, Any] | None] | None = None,
        callables: list[dict[str, Any]] | None = None,
        save_video: bool = True,
        video_path: str | Path = './',
        plot_joint_compare_normed: bool = False,
        x_range = None,
        y_range = None, 
        z_range = None, 
    ) -> dict:
        """Evaluate the policy from manually specified start and goal positions.

        This is a dataset-free counterpart to ``evaluate_from_dataset``.
        The start and goal values are injected through each environment's
        ``reset(options=...)`` and the resulting goal observations are then
        kept fixed during the rollout.

        Args:
            start_positions: Per-environment start positions/states.
            goal_positions: Per-environment goal positions/states.
            eval_budget: Maximum steps allowed for the agent to reach the goal.
            start_option_name: Reset option name used for the initial state.
            goal_option_name: Reset option name used for the goal state.
            start_info_name: Info key to overwrite with ``start_positions``.
            goal_info_name: Info key to overwrite with ``goal_positions``.
            seeds: Optional seed or per-environment seeds for reset.
            reset_options: Additional reset options for each environment.
            callables: Optional list of per-env method calls, following the
                same schema as ``evaluate_from_dataset``.
            save_video: Whether to save rollout videos.
            video_path: Path to save videos.

        Returns:
            Dictionary containing success rates and other metrics.
        """
        start_arr = np.asarray(start_positions)
        goal_arr = np.asarray(goal_positions)
        init_ee_arr = np.asarray(init_ee_poses)
        goal_ee_arr = np.asarray(goal_ee_poses)
        # print("goal_ee_arr: ", goal_ee_arr)

        if start_arr.ndim == 0 or goal_arr.ndim == 0:
            raise ValueError('start_positions and goal_positions must be arrays')

        # Accept either:
        # - a single state vector, e.g. (state_dim,)
        # - one state per env, e.g. (num_envs, state_dim)
        if start_arr.ndim == 1:
            start_arr = np.broadcast_to(start_arr, (self.num_envs, *start_arr.shape))
        if goal_arr.ndim == 1:
            goal_arr = np.broadcast_to(goal_arr, (self.num_envs, *goal_arr.shape))

        if len(start_arr) != len(goal_arr):
            raise ValueError(
                'start_positions and goal_positions must have the same length'
            )

        if len(start_arr) != self.num_envs:
            raise ValueError(
                'Number of start/goal positions must match number of envs, '
                'or provide a single state vector to broadcast to all envs'
            )

        if reset_options is None:
            options = [{} for _ in range(self.num_envs)]
        else:
            if len(reset_options) != self.num_envs:
                raise ValueError(
                    'reset_options length must match number of envs'
                )
            options = [
                deepcopy(opt) if opt is not None else {}
                for opt in reset_options
            ]

        for i in range(self.num_envs):
            options[i][start_option_name] = np.array(start_arr[i], copy=True)
            options[i][goal_option_name] = np.array(goal_arr[i], copy=True)
            options[i][init_ee_option_name] = np.array(init_ee_arr[i], copy=True)
            options[i][goal_ee_option_name] = np.array(goal_ee_arr[i], copy=True)

        seed_list = self._expand_reset_seeds(seeds)
        result_seeds = None if seed_list is None else np.array(seed_list)

        self.reset(seed=seeds, options=options)
        # print("infos keys:", self.infos.keys())
        # print("start input:", start_arr)
        # print("goal input:", goal_arr)
        # print("actual bluebox:", self.infos["bluebox_pos"][:, -1])
        # print("goal_step goal_state:", goal_step.get("goal_state", None))

        callables = callables or []
        if callables:
            manual_step = {
                'start_positions': np.array(start_arr, copy=True),
                'goal_positions': np.array(goal_arr, copy=True),
            }
            if start_info_name is not None:
                manual_step[start_info_name] = np.array(start_arr, copy=True)
            if goal_info_name is not None:
                manual_step[goal_info_name] = np.array(goal_arr, copy=True)

            for i, env in enumerate(self.envs.unwrapped.envs):
                env_unwrapped = env.unwrapped

                for spec in callables:
                    method_name = spec['method']
                    if not hasattr(env_unwrapped, method_name):
                        logging.warning(
                            f'Env {env_unwrapped} has no method {method_name}, skipping callable'
                        )
                        continue

                    method = getattr(env_unwrapped, method_name)
                    args = spec.get('args', spec)

                    prepared_args = {}
                    for args_name, args_data in args.items():
                        value = args_data.get('value')
                        in_positions = args_data.get('in_positions', False)
                        in_dataset = args_data.get('in_dataset', False)

                        if in_positions or in_dataset:
                            if value not in manual_step:
                                logging.warning(
                                    f'Key {value} not found for callable {method_name}, skipping arg'
                                )
                                continue
                            prepared_args[args_name] = deepcopy(
                                manual_step[value][i]
                            )
                        else:
                            prepared_args[args_name] = args_data.get('value')

                    method(**prepared_args)

        goal_visuals = None
        if 'goal' not in self.infos: #ここでゴール画像作ってる
            goal_visuals = self._render_goal_observations(
                goal_positions=goal_arr,
                base_options=options,
                seed_list=seed_list,
                start_option_name=start_option_name,
                goal_option_name=goal_option_name,
            )
            # Restore the intended start states after rendering goal observations.
            self.reset(seed=seeds, options=options)
            
            # print("goal_visuals.shape: ", goal_visuals.shape)

            from PIL import Image
            from pathlib import Path

            save_dir = Path("./debug_goal_visuals")
            save_dir.mkdir(parents=True, exist_ok=True)

            img = goal_visuals[0]  # (64, 64, 3)

            # float画像なら 0~255 uint8 に変換
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).clip(0, 255)
                img = img.astype(np.uint8)

            Image.fromarray(img).save(save_dir / "goal_visual_0.png")
            print("saved:", save_dir / "goal_visual_0.png")
            
            

        shape_prefix = self.infos['pixels'].shape[:2]
        if start_info_name is not None:
            self.infos[start_info_name] = np.broadcast_to(
                start_arr[:, None, ...], shape_prefix + start_arr.shape[1:]
            )
        if goal_info_name is not None:
            self.infos[goal_info_name] = np.broadcast_to(
                goal_arr[:, None, ...], shape_prefix + goal_arr.shape[1:]
            )

        goal_step = {
            k: deepcopy(v)
            for k, v in self.infos.items()
            if k == 'goal' or k.startswith('goal_')
        }
        if goal_visuals is not None:
            goal_step['goal'] = np.broadcast_to(
                goal_visuals[:, None, ...],
                shape_prefix + goal_visuals.shape[1:],
            )
            self.infos['goal'] = deepcopy(goal_step['goal'])
        if goal_info_name is not None:
            goal_step[goal_info_name] = deepcopy(self.infos[goal_info_name])

        results: dict[str, Any] = {
            'success_rate': 0.0,
            'episode_successes': np.zeros(self.num_envs),
            'seeds': result_seeds,
            'start_positions': np.array(start_arr, copy=True),
            'goal_positions': np.array(goal_arr, copy=True),
            'init_ee_poses': np.array(init_ee_arr, copy=True),
        }

        video_frames, traj, joint_logs, cartesian_logs  = self._run_evaluation_rollout(
            goal_step=goal_step,
            eval_budget=eval_budget,
            results=results,
            collect_traj=True,
            collect_joint_logs=True,
        )
        

        results["traj_bluebox_pos"] = traj["bluebox_pos"]
        results["traj_bluebox_quat"] = traj["bluebox_quat"]
        results["traj_ee_pos"] = traj["ee_pos"]
        results["traj_target_action"] = joint_logs["target_action"]
        results["traj_actual_qpos"] = joint_logs["actual_qpos"]
        # results["traj_target_xyz"] = cartesian_logs["target_xyz"]
        # results["traj_actual_ee_pos"] = cartesian_logs["actual_ee_pos"]
        results['success_rate'] = (
            float(np.sum(results['episode_successes'])) / self.num_envs * 100.0
        )
        if cartesian_logs is not None:
            results["traj_target_xyz"] = cartesian_logs["target_xyz"]
            results["traj_actual_ee_pos"] = cartesian_logs["actual_ee_pos"]

        if save_video:
            target_frames = None
            if 'goal' in goal_step:
                target_frames = np.array(goal_step['goal'][:, -1:], copy=True)
            self._save_evaluation_videos_2split(
                video_frames=video_frames,
                target_frames=target_frames,
                video_path=video_path,
            )
        
        if "traj_bluebox_pos" in results and "traj_ee_pos" in results:
            plot_dir = Path(video_path) / "task_map"
            plot_dir.mkdir(parents=True, exist_ok=True)

            
            for env_idx in range(self.num_envs):
                plot_task_result_xy(
                    bluebox_traj=results["traj_bluebox_pos"][env_idx],
                    bluebox_quat_traj=results["traj_bluebox_quat"][env_idx],
                    ee_traj=results["traj_ee_pos"][env_idx],
                    goal_pos=results["goal_positions"][env_idx],
                    save_path=plot_dir / f"task_plot_{env_idx}.png",
                    title=f"Task Visualization env={env_idx}",
                    workspace_x=x_range, 
                    workspace_y=y_range, 
                )
        print("results.keys():", results.keys())
        if "traj_target_action" in results and "traj_actual_qpos" in results:
            joint_plot_dir = Path(video_path) / "joint_angle"
            joint_plot_dir.mkdir(parents=True, exist_ok=True)
            
            cart_plot_dir = Path(video_path) / "cartesian"
            cart_plot_dir.mkdir(parents=True, exist_ok=True)


            for env_idx in range(self.num_envs):
                plot_joint_angle_comparison(
                    target_actions=results["traj_target_action"][env_idx],   # (T, 7)
                    actual_qpos=results["traj_actual_qpos"][env_idx],       # (T+1, 7)
                    save_path=joint_plot_dir / f"joint_angle_env_{env_idx}.png",
                    title="Joint angle comparison",
                    episode_idx=env_idx,
                    normalize=plot_joint_compare_normed,
                )
                if cartesian_logs is not None:
                    plot_cartesian_comparison(
                        target_xyz=results["traj_target_xyz"][env_idx],
                        actual_ee_pos=results["traj_actual_ee_pos"][env_idx],
                        save_path=cart_plot_dir / f"cartesian_env_{env_idx}.png",
                        title="Cartesian target vs actual EE",
                        episode_idx=env_idx,
                        
                    )

        return results

    def _expand_reset_seeds(
        self, seeds: int | Sequence[int | None] | None
    ) -> list[int | None] | None:
        """Normalize reset seeds into a per-environment list."""
        if seeds is None:
            return None
        if isinstance(seeds, int):
            return [seeds + i for i in range(self.num_envs)]
        if isinstance(seeds, Sequence) and not isinstance(seeds, (str, bytes)):
            if len(seeds) != self.num_envs:
                raise ValueError('seeds length must match number of envs')
            return list(seeds)
        raise ValueError('Unsupported seeds format')

    def _render_goal_observations(
        self,
        goal_positions: np.ndarray,
        base_options: Sequence[dict[str, Any]],
        seed_list: Sequence[int | None] | None,
        start_option_name: str,
        goal_option_name: str,
    ) -> np.ndarray:
        """Render goal observations for envs that do not expose ``info['goal']``.

        Each environment is temporarily reset with its state placed at the goal
        so that the wrapped ``pixels`` output can be reused as the goal image.
        """
        goal_images: list[np.ndarray] = []
        for i in range(self.num_envs):
            goal_options = deepcopy(base_options[i])
            goal_state = np.array(goal_positions[i], copy=True)
            goal_options[start_option_name] = goal_state
            goal_options[goal_option_name] = np.array(goal_positions[i], copy=True)

            # print("(stable_worldmodel/world.py) self.envs:", self.envs)
            # print("(stable_worldmodel/world.py) self.envs.envs[i]:", self.envs.envs[i])
            
            print("(stable_worldmodel/world.py) goal_options.keys():", goal_options.keys()) #dict_keys(['box_pos', goal_marker_pos', 'init_ee_pos', 'goal_ee_pos'])
        
            
            if goal_options.get("goal_ee_pos", None) is not None:
                goal_options["init_ee_pos"] = np.array(goal_options["goal_ee_pos"], copy=True)
            
            _, goal_infos = self.envs.envs[i].reset(
                seed=None if seed_list is None else seed_list[i],
                options=goal_options,
            )

            if 'goal' in goal_infos:
                goal_img = goal_infos['goal']
            elif 'pixels' in goal_infos:
                goal_img = goal_infos['pixels']
            else:
                raise RuntimeError(
                    "Unable to infer a goal image: env reset returned neither 'goal' nor 'pixels'."
                )

            goal_img = np.asarray(goal_img)
            if goal_img.ndim > 3:
                goal_img = goal_img[-1]
            goal_images.append(goal_img)
            
            # print("[GOAL RENDER] goal_state:", goal_state)
            # print("[GOAL RENDER] returned info keys:", goal_infos.keys())


        # save_dir = "./debug_goal_images"
        # os.makedirs(save_dir, exist_ok=True)

        # for i, img in enumerate(goal_images):
        #     save_path = os.path.join(save_dir, f"goal_image_env_{i}.png")
        #     plt.imsave(save_path, img)
        #     print(f"[GOAL IMG] saved: {save_path}")

        return np.stack(goal_images)


    def _run_evaluation_rollout(
        self,
        goal_step: dict[str, Any],
        eval_budget: int,
        results: dict[str, Any],
        collect_traj: bool = False,
        collect_joint_logs: bool = False,
    ):
        video_frames = np.empty(
            (self.num_envs, eval_budget, *self.infos['pixels'].shape[-3:]),
            dtype=np.uint8,
        )

        traj = None
        if collect_traj:
            traj = {
                "bluebox_pos": [],
                "bluebox_quat": [],
                "ee_pos": [],
            }

            # reset直後も記録
            if "bluebox_pos" in self.infos:
                blue0 = np.asarray(self.infos["bluebox_pos"])
                if blue0.ndim > 2:
                    blue0 = blue0[:, -1]
                traj["bluebox_pos"].append(blue0.copy())
                
            if "bluebox_quat" in self.infos:
                quat0 = np.asarray(self.infos["bluebox_quat"])
                if quat0.ndim > 2:
                    quat0 = quat0[:, -1]
                traj["bluebox_quat"].append(quat0.copy())

            if "ee_pos" in self.infos:
                ee0 = np.asarray(self.infos["ee_pos"])
                if ee0.ndim > 2:
                    ee0 = ee0[:, -1]
                traj["ee_pos"].append(ee0.copy())
            elif "state" in self.infos:
                st0 = np.asarray(self.infos["state"])
                if st0.ndim > 2:
                    st0 = st0[:, -1]
                traj["ee_pos"].append(st0[:, -3:].copy())

        joint_logs = None
        if collect_joint_logs:
            joint_logs = {
                "target_action": [],
                "actual_qpos": [],
            }

            # reset直後の qpos を保存
            if "state" in self.infos:
                st0 = np.asarray(self.infos["state"])
                if st0.ndim > 2:
                    st0 = st0[:, -1]
                joint_logs["actual_qpos"].append(st0[:, :7].copy())

        cartesian_logs = None
        if collect_joint_logs:
            cartesian_logs = {
                "target_xyz": [],
                "actual_ee_pos": [],
            }
            
        cem_cost_over_time = []
        for i in range(eval_budget):
            video_frames[:, i] = self.infos['pixels'][:, -1]
            self.infos.update(deepcopy(goal_step))

            # ---- action を明示的に取得 ----
            if self.policy is None:
                raise RuntimeError("No policy set. Call set_policy() first.")


            actions, outputs = self.policy.get_action(self.infos)
            
            ####cemの最終コストをログ
            if outputs is not None:
                elite_mean = torch.stack(
                    outputs["elite_cost_mean_n_steps"]
                )[:, 0]

                elite_min = torch.stack(
                    outputs["elite_cost_min_n_steps"]
                )[:, 0]

                cem_cost_over_time.append({
                    "timestep": i,
                    "final_mean": elite_mean[-1].item(),
                    "final_min": elite_min[-1].item(),
                    "initial_mean": elite_mean[0].item(),
                    "initial_min": elite_min[0].item(),
                })
            ##########
            

            # if collect_joint_logs:
            #     joint_logs["target_action"].append(np.asarray(actions).copy())

            (
                self.states,
                self.rewards,
                self.terminateds,
                self.truncateds,
                self.infos,
            ) = self.envs.step(actions)
            
            if collect_joint_logs:
                # target_qpos を使う
                if "target_qpos" in self.infos:
                    tq = np.asarray(self.infos["target_qpos"])
                    if tq.ndim > 2:
                        tq = tq[:, -1]
                    joint_logs["target_action"].append(tq[:, :7].copy())

                # actualもinfoから
                if "actual_qpos" in self.infos:
                    aq = np.asarray(self.infos["actual_qpos"])
                    if aq.ndim > 2:
                        aq = aq[:, -1]
                    joint_logs["actual_qpos"].append(aq[:, :7].copy())
                    

                # if "target_xyz" in self.infos:
                #     tx = np.asarray(self.infos["target_xyz"])
                #     # print("tx.shape:", tx.shape)
                #     if tx.ndim == 2 and tx.shape[1] >= 3:
                    
                #         cartesian_logs["target_xyz"].append(tx[:, :3].copy())

                # if "actual_ee_pos" in self.infos:
                #     ee = np.asarray(self.infos["actual_ee_pos"])
                #     if ee.ndim > 2:
                #         ee = ee[:, -1]
                #     cartesian_logs["actual_ee_pos"].append(ee[:, :3].copy())
                # elif "ee_pos" in self.infos:
                #     ee = np.asarray(self.infos["ee_pos"])
                #     if ee.ndim > 2:
                #         ee = ee[:, -1]
                #     cartesian_logs["actual_ee_pos"].append(ee[:, :3].copy())

                if "target_xyz" in self.infos:
                    tx = np.asarray(self.infos["target_xyz"])

                    if tx.ndim == 2 and tx.shape[1] >= 3:
                        cartesian_logs["target_xyz"].append(tx[:, :3].copy())

                        if "actual_ee_pos" in self.infos:
                            ee = np.asarray(self.infos["actual_ee_pos"])
                            if ee.ndim > 2:
                                ee = ee[:, -1]
                            cartesian_logs["actual_ee_pos"].append(ee[:, :3].copy())

                        elif "ee_pos" in self.infos:
                            ee = np.asarray(self.infos["ee_pos"])
                            if ee.ndim > 2:
                                ee = ee[:, -1]
                            cartesian_logs["actual_ee_pos"].append(ee[:, :3].copy())





            results['episode_successes'] = np.logical_or(
                results['episode_successes'], self.terminateds
            )
            self.envs.unwrapped._autoreset_envs = np.zeros((self.num_envs,))

            if collect_traj:
                if "bluebox_pos" in self.infos:
                    blue = np.asarray(self.infos["bluebox_pos"])
                    if blue.ndim > 2:
                        blue = blue[:, -1]
                    traj["bluebox_pos"].append(blue.copy())

                if "ee_pos" in self.infos:
                    ee = np.asarray(self.infos["ee_pos"])
                    if ee.ndim > 2:
                        ee = ee[:, -1]
                    traj["ee_pos"].append(ee.copy())
                elif "state" in self.infos:
                    st = np.asarray(self.infos["state"])
                    if st.ndim > 2:
                        st = st[:, -1]
                    traj["ee_pos"].append(st[:, -3:].copy())

            # if collect_joint_logs:
            #     if "state" in self.infos:
            #         st = np.asarray(self.infos["state"])
            #         if st.ndim > 2:
            #             st = st[:, -1]
            #         joint_logs["actual_qpos"].append(st[:, :7].copy())

        if eval_budget > 0:
            video_frames[:, -1] = self.infos['pixels'][:, -1]

        out = [video_frames]

        if collect_traj:
            traj = {k: np.stack(v, axis=1) for k, v in traj.items()}
            out.append(traj)

        if collect_joint_logs:
            joint_logs = {
                k: np.stack(v, axis=1)
                for k, v in joint_logs.items()
            }
            out.append(joint_logs)

            # cartesian_logs は中身があるときだけ stack
            if (
                cartesian_logs is not None
                and len(cartesian_logs["target_xyz"]) > 0
                and len(cartesian_logs["actual_ee_pos"]) > 0
            ):
                cartesian_logs = {
                    k: np.stack(v, axis=1)
                    for k, v in cartesian_logs.items()
                }
            else:
                cartesian_logs = None

            out.append(cartesian_logs)
        
        #CEMのコスト遷移をプロット
        if len(cem_cost_over_time) > 0:
            plot_cem_rollout_cost(cem_cost_over_time, self.results_path)

        if len(out) == 1:
            return out[0]
        
        # print('type(joint_logs["actual_qpos"]):')
        # print('type(joint_logs["actual_qpos"]):', type(joint_logs["actual_qpos"]))
        return tuple(out)


    def _save_evaluation_videos(
        self,
        video_frames: np.ndarray,
        target_frames: np.ndarray | None,
        video_path: str | Path,
    ) -> None:
        """Save rollout videos with the current frame and target side by side."""
        import imageio

        if video_frames.shape[1] == 0:
            logging.warning('No rollout frames to save, skipping video export.')
            return

        if target_frames is None:
            target_frames = video_frames[:, -1:, ...]

        target_len = target_frames.shape[1]
        video_path_obj = Path(video_path)
        video_path_obj.mkdir(parents=True, exist_ok=True)
        for i in range(self.num_envs):
            out = imageio.get_writer(
                video_path_obj / f'rollout_{i}.mp4',
                fps=15,
                codec='libx264',
            )
            goals = np.vstack([target_frames[i, -1], target_frames[i, -1]])
            for t in range(video_frames.shape[1]):
                stacked_frame = np.vstack(
                    [video_frames[i, t], target_frames[i, t % target_len]]
                )
                frame = np.hstack([stacked_frame, goals])
                out.append_data(frame)
            out.close()
        print(f'Video saved to {video_path_obj}')

    def _save_evaluation_videos_2split(
        self,
        video_frames: np.ndarray,
        target_frames: np.ndarray | None,
        video_path: str | Path,
    ) -> None:
        """Save rollout videos with current frame and final goal frame side by side."""
        import imageio

        if video_frames.shape[1] == 0:
            logging.warning('No rollout frames to save, skipping video export.')
            return

        if target_frames is None:
            target_frames = video_frames[:, -1:, ...]

        video_path_obj = Path(video_path)
        video_path_obj.mkdir(parents=True, exist_ok=True)

        for i in range(self.num_envs):
            out = imageio.get_writer(
                video_path_obj / f'rollout_{i}.mp4',
                fps=15,
                codec='libx264',
            )

            # 目標画像は常に最終goalだけ
            goal = target_frames[i, -1]

            for t in range(video_frames.shape[1]):
                current = video_frames[i, t]

                # 左: タスク実行中, 右: 最終goal
                frame = np.hstack([current, goal])

                out.append_data(frame)

            out.close()

        print(f'Video saved to {video_path_obj}')