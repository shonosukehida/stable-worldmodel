from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from collections.abc import Callable

import numpy as np
import torch
from loguru import logger as logging
from torchvision import tv_tensors

import stable_worldmodel as swm
from stable_worldmodel.solver import Solver
import gymnasium as gym




from stable_worldmodel.plot.plot import plot_cem_cost_convergence, plot_cem_sequence_transition_colormap


@dataclass(frozen=True)
class PlanConfig:
    """Configuration for the MPC planning loop.

    Attributes:
        horizon: Planning horizon in number of steps.
        receding_horizon: Number of steps to execute before re-planning.
        history_len: Number of past observations to consider.
        action_block: Number of times each action is repeated (frameskip).
        warm_start: Whether to use the previous plan to initialize the next one.
    """

    horizon: int
    receding_horizon: int
    history_len: int = 1
    action_block: int = 1
    warm_start: bool = True
    action_space: str = ""
    clip_action: bool = True

    @property
    def plan_len(self) -> int:
        """Total plan length in environment steps."""
        return self.horizon * self.action_block


class Transformable(Protocol):
    """Protocol for reversible data transformations (e.g., normalizers, scalers)."""

    def transform(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover
        """Apply preprocessing to input data.

        Args:
            x: Input data as a numpy array.

        Returns:
            Preprocessed data as a numpy array.
        """
        ...

    def inverse_transform(
        self, x: np.ndarray
    ) -> np.ndarray:  # pragma: no cover
        """Reverse the preprocessing transformation.

        Args:
            x: Preprocessed data as a numpy array.

        Returns:
            Original data as a numpy array.
        """
        ...


class Actionable(Protocol):
    """Protocol for model action computation."""

    def get_action(info) -> torch.Tensor:  # pragma: no cover
        """Compute action from observation and goal"""
        ...


class BasePolicy:
    """Base class for agent policies.

    Attributes:
        env: The environment the policy is associated with.
        type: A string identifier for the policy type.
    """

    env: Any
    type: str

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the base policy.

        Args:
            **kwargs: Additional configuration parameters.
        """
        self.env = None
        self.type = 'base'
        for arg, value in kwargs.items():
            setattr(self, arg, value)

    def get_action(self, obs: Any, **kwargs: Any) -> np.ndarray:
        """Get action from the policy given the observation.

        Args:
            obs: The current observation from the environment.
            **kwargs: Additional parameters for action selection.

        Returns:
            Selected action as a numpy array.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError

    def set_env(self, env: Any) -> None:
        """Associate this policy with an environment.

        Args:
            env: The environment to associate.
        """
        self.env = env

    def _prepare_info(self, info_dict: dict) -> dict[str, torch.Tensor]:
        """Pre-process and transform observations.

        Applies preprocessing (via `self.process`) and transformations (via `self.transform`)
        to observation data. Used by subclasses like FeedForwardPolicy and WorldModelPolicy.

        Args:
            info_dict: Raw observation dictionary from the environment.

        Returns:
            A dictionary of processed tensors.

        Raises:
            ValueError: If an expected numpy array is missing for processing.
        """
        for k, v in info_dict.items():
            is_numpy = isinstance(v, (np.ndarray | np.generic))

            if hasattr(self, 'process') and k in self.process:
                if not is_numpy:
                    raise ValueError(
                        f"Expected numpy array for key '{k}' in process, got {type(v)}"
                    )

                # flatten extra dimensions if needed
                shape = v.shape
                if len(shape) > 2:
                    v = v.reshape(-1, *shape[2:])

                # process and reshape back
                v = self.process[k].transform(v)
                v = v.reshape(shape)

            # collapse env and time dimensions for transform (e, t, ...) -> (e * t, ...)
            # then restore after transform
            if hasattr(self, 'transform') and k in self.transform:
                shape = None
                if is_numpy or torch.is_tensor(v):
                    if v.ndim > 2:
                        shape = v.shape
                        v = v.reshape(-1, *shape[2:])
                if k.startswith('pixels') or k.startswith('goal'):
                    # permute channel first for transform
                    if is_numpy:
                        v = np.transpose(v, (0, 3, 1, 2))
                    else:
                        v = v.permute(0, 3, 1, 2)
                v = torch.stack(
                    [self.transform[k](tv_tensors.Image(x)) for x in v]
                )
                is_numpy = isinstance(v, (np.ndarray | np.generic))

                if shape is not None:
                    v = v.reshape(*shape[:2], *v.shape[1:])

            if is_numpy and v.dtype.kind not in 'USO':
                v = torch.from_numpy(v)

            info_dict[k] = v

        return info_dict


class RandomPolicy(BasePolicy):
    """Policy that samples random actions from the action space."""

    def __init__(self, seed: int | None = None, **kwargs: Any) -> None:
        """Initialize the random policy.

        Args:
            seed: Optional random seed for the action space.
            **kwargs: Additional configuration parameters.
        """
        super().__init__(**kwargs)
        self.type = 'random'
        self.seed = seed

    def get_action(self, obs: Any, **kwargs: Any) -> np.ndarray:
        """Get a random action from the environment's action space.

        Args:
            obs: The current observation (ignored).
            **kwargs: Additional parameters (ignored).

        Returns:
            A randomly sampled action.
        """
        return self.env.action_space.sample()

    def set_seed(self, seed: int) -> None:
        """Set the random seed for action sampling.

        Args:
            seed: The seed value.
        """
        if self.env is not None:
            self.env.action_space.seed(seed)


class ExpertPolicy(BasePolicy):
    """Policy using expert demonstrations or heuristics."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the expert policy.

        Args:
            **kwargs: Additional configuration parameters.
        """
        super().__init__(**kwargs)
        self.type = 'expert'

    def get_action(
        self, obs: Any, goal_obs: Any, **kwargs: Any
    ) -> np.ndarray | None:
        """Get action from the expert policy.

        Args:
            obs: The current observation.
            goal_obs: The goal observation.
            **kwargs: Additional parameters.

        Returns:
            The expert action, or None if not available.
        """
        # Implement expert policy logic here
        pass


class FeedForwardPolicy(BasePolicy):
    """Feed-Forward Policy using a neural network model.

    Actions are computed via a single forward pass through the model.
    Useful for imitation learning policies like Goal-Conditioned Behavioral Cloning (GCBC).

    Attributes:
        model: Neural network model implementing the Actionable protocol.
        process: Dictionary of data preprocessors for specific keys.
        transform: Dictionary of tensor transformations (e.g., image transforms).
    """

    def __init__(
        self,
        model: Actionable,
        process: dict[str, Transformable] | None = None,
        transform: dict[str, Callable[[torch.Tensor], torch.Tensor]]
        | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the feed-forward policy.

        Args:
            model: Neural network model with a `get_action` method.
            process: Dictionary of data preprocessors for specific keys.
            transform: Dictionary of tensor transformations (e.g., image transforms).
            **kwargs: Additional configuration parameters.
        """
        super().__init__(**kwargs)
        self.type = 'feed_forward'
        self.model = model.eval()
        self.process = process or {}
        self.transform = transform or {}

    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        """Get action via a forward pass through the neural network model.

        Args:
            info_dict: Current state information containing at minimum a 'goal' key.
            **kwargs: Additional parameters (unused).

        Returns:
            The selected action as a numpy array.

        Raises:
            AssertionError: If environment not set or 'goal' not in info_dict.
        """
        assert hasattr(self, 'env'), 'Environment not set for the policy'
        assert 'goal' in info_dict, "'goal' must be provided in info_dict"

        # Prepare the info dict (transforms and normalizes inputs)
        info_dict = self._prepare_info(info_dict)

        # Add goal_pixels key for GCBC model
        if 'goal' in info_dict:
            info_dict['goal_pixels'] = info_dict['goal']

        # Move all tensors to the model's device
        device = next(self.model.parameters()).device
        for k, v in info_dict.items():
            if torch.is_tensor(v):
                info_dict[k] = v.to(device)

        # Get action from model
        with torch.no_grad():
            action = self.model.get_action(info_dict)

        # Convert to numpy
        if torch.is_tensor(action):
            action = action.cpu().detach().numpy()

        # post-process action
        if 'action' in self.process:
            action = self.process['action'].inverse_transform(action)

        return action


class WorldModelPolicy(BasePolicy):
    """Policy using a world model and planning solver for action selection."""

    def __init__(
        self,
        solver: Solver,
        config: PlanConfig,
        process: dict[str, Transformable] | None = None,
        transform: dict[str, Callable[[torch.Tensor], torch.Tensor]]
        | None = None,
        action_projector=None,
        **kwargs: Any,
    ) -> None:
        """Initialize the world model policy.

        Args:
            solver: The planning solver to use.
            config: MPC planning configuration.
            process: Dictionary of data preprocessors for specific keys.
            transform: Dictionary of tensor transformations (e.g., image transforms).
            **kwargs: Additional configuration parameters.
        """
        super().__init__(**kwargs)

        self.type = 'world_model'
        self.cfg = config
        self.solver = solver
        
        self.action_projector = action_projector
        
        
        self.action_buffer: deque[torch.Tensor] = deque(
            maxlen=self.flatten_receding_horizon
        )
        self.process = process or {}
        self.transform = transform or {}
        self._action_buffer: deque[torch.Tensor] | None = None
        self._next_init: torch.Tensor | None = None
        
        # print("self.process:", self.process)
        
        
        self.results_path = None



    @property
    def flatten_receding_horizon(self) -> int:
        """Receding horizon in environment steps (with frameskip)."""
        return self.cfg.receding_horizon * self.cfg.action_block


    def set_action_projector(self, action_projector) -> None:
        self.action_projector = action_projector


    def set_env(self, env: Any) -> None:
        """Configure the policy and solver for the given environment.

        Args:
            env: The environment to associate with the policy.
        """
        self.env = env

        if self.cfg.action_space == "joint":
            self.action_space = self.env.action_space #実空間
            
            if ("action" in self.process.keys()):
                self.action_processor = self.process["action"] 
                # action_processor = self.process["action_joint"] 
            elif ("action_joint" in self.process.keys()): 
                self.action_processor = self.process["action_joint"] 
            
        elif self.cfg.action_space == "cartesian":
            self.action_space = gym.spaces.Box(
                low=np.array([[0.315, -0.2, 0.095]], dtype=np.float32),
                high=np.array([[0.715,  0.2, 0.105]], dtype=np.float32),
                dtype=np.float32,
            ) #実空間
            self.action_processor = self.process["action_cartesian"]
        else:
            raise ValueError(
                f"Unknown action_space: {self.cfg.action_space}. "
                "Expected 'joint' or 'cartesian'."
            )
        
        # print("self.process.keys():", self.process.keys())
            
        
        
        
        n_envs = getattr(env, 'num_envs', 1)
        
        self.solver.configure(
            n_envs=n_envs, config=self.cfg, action_processor=self.action_processor, action_space=self.action_space, 
        )

        self._action_buffer = deque(maxlen=self.flatten_receding_horizon)

        assert isinstance(self.solver, Solver), (
            'Solver must implement the Solver protocol'
        )

    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        """Get action via planning with the world model.

        Args:
            info_dict: Current state information from the environment.
            **kwargs: Additional parameters for planning.

        Returns:
            The selected action(s) as a numpy array.
        """
        
        assert hasattr(self, 'env'), 'Environment not set for the policy'
        assert 'pixels' in info_dict, "'pixels' must be provided in info_dict"
        assert 'goal' in info_dict, "'goal' must be provided in info_dict"


        # ActionProjector用の物理状態。(qpos, ee, gripper)
        # _prepare_info()には通さない。
        projection_state = kwargs.get(
            "projection_state",
            None,
        )

        info_dict = self._prepare_info(info_dict)
        # print("[stable-worldmodel/stable_worldmodel/policy.py] info_dict.keys() : ", info_dict.keys())
        # print("info_dict(step_idx)", info_dict["step_idx"][0][0])

        outputs = None
        # need to replan if action buffer is empty
        if len(self._action_buffer) == 0:
            outputs = self.solver(
                info_dict, 
                init_action=self._next_init,
                action_projector=self.action_projector,
                projection_state=projection_state,
            )
            
            
            #cem の内部状況をログ
            timestep = int(info_dict["step_idx"][0][0])
            save_dir = self.results_path / "cem"
            save_dir.mkdir(parents=True, exist_ok=True)
            print("save_dir:", save_dir)
            
            plot_cem_cost_convergence(outputs, env_idx=0, save_dir=save_dir, timestep=timestep)
            plot_cem_sequence_transition_colormap(outputs, env_idx=0, save_dir=save_dir, timestep=timestep, action_processor=self.action_processor)

            actions = outputs['actions']  # (num_envs, horizon, action_dim)
            
            
            actions_np = actions.cpu().numpy() if torch.is_tensor(actions) else actions

            for d in range(actions_np.shape[-1]):
                print(f"dim {d}: min={actions_np[..., d].min():.3f}, max={actions_np[..., d].max():.3f}") 
            


            keep_horizon = self.cfg.receding_horizon
            plan = actions[:, :keep_horizon]
            rest = actions[:, keep_horizon:]
            self._next_init = rest if self.cfg.warm_start else None

            # frameskip back to timestep
            plan = plan.reshape(
                self.env.num_envs, self.flatten_receding_horizon, -1
            )

            self._action_buffer.extend(plan.transpose(0, 1))

        action = self._action_buffer.popleft()
        action = action.reshape(*self.action_space.shape)
        action = action.numpy()


        # post-process action
        
        if 'action' in self.process: ##
            action = self.process['action'].inverse_transform(action)
            # action_joint = action
        elif "action_cartesian" in self.process:
            action = self.process["action_cartesian"].inverse_transform(action)
        elif "action_joint" in self.process:
            action = self.process["action_joint"].inverse_transform(action)

        return action, outputs  # (num_envs, action_dim)


def _load_model_with_attribute(run_name, attribute_name, cache_dir=None):
    """Helper function to load a model checkpoint and find a module with the specified attribute.

    Args:
        run_name: Path or name of the model run
        attribute_name: Name of the attribute to look for in the module (e.g., 'get_action', 'get_cost')
        cache_dir: Optional cache directory path

    Returns:
        The module with the specified attribute

    Raises:
        RuntimeError: If no module with the specified attribute is found
    """
    if Path(run_name).exists():
        run_path = Path(run_name)
    else:
        run_path = Path(
            cache_dir
            or swm.data.utils.get_cache_dir(sub_folder='checkpoints'),
            run_name,
        )

    if run_path.is_dir():
        ckpt_files = list(run_path.glob('*_object.ckpt'))
        ckpt_files.sort(key=lambda x: x.stat().st_ctime, reverse=True)
        path = ckpt_files[0]
        logging.info(f'Loading model from checkpoint: {path}')
    else:
        path = Path(f'{run_path}_object.ckpt')
        assert path.exists(), (
            f'Checkpoint path does not exist: {path}. Launch pretraining first.'
        )

    spt_module = torch.load(path, weights_only=False, map_location='cpu')

    def scan_module(module):
        if hasattr(module, attribute_name):
            if isinstance(module, torch.nn.Module):
                module = module.eval()
            return module
        for child in module.children():
            result = scan_module(child)
            if result is not None:
                return result
        return None

    result = scan_module(spt_module)
    if result is not None:
        return result

    raise RuntimeError(
        f"No module with '{attribute_name}' found in the loaded world model."
    )


def AutoActionableModel(
    run_name: str, cache_dir: str | Path | None = None
) -> torch.nn.Module:
    """Load a model checkpoint and return the module with a `get_action` method.

    Automatically scans the checkpoint for a module implementing the Actionable
    protocol (i.e., has a `get_action` method).

    Args:
        run_name: Path or name of the model run/checkpoint.
        cache_dir: Optional cache directory path. Defaults to STABLEWM_HOME.

    Returns:
        The module with a `get_action` method, set to eval mode.

    Raises:
        RuntimeError: If no module with `get_action` is found in the checkpoint.
    """
    return _load_model_with_attribute(run_name, 'get_action', cache_dir)


def AutoCostModel(
    run_name: str, cache_dir: str | Path | None = None
) -> torch.nn.Module:
    """Load a model checkpoint and return the module with a `get_cost` method.

    Automatically scans the checkpoint for a module implementing a cost function
    (i.e., has a `get_cost` method) for use with planning solvers.

    Args:
        run_name: Path or name of the model run/checkpoint.
        cache_dir: Optional cache directory path. Defaults to STABLEWM_HOME.

    Returns:
        The module with a `get_cost` method, set to eval mode.

    Raises:
        RuntimeError: If no module with `get_cost` is found in the checkpoint.
    """
    return _load_model_with_attribute(run_name, 'get_cost', cache_dir)


# Alias for backward compatibility and type hinting
Policy = BasePolicy

