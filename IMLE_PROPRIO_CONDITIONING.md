# IMLE proprio conditioning

Base: `cfe90c7` (`real_robot_multiview`). Worktree branch: `feature/imle-proprio-conditioning`.

`ActionGenerator(..., use_proprio=False)` and `IMLEPolicy(..., use_proprio=False, proprio_stats=None)` preserve the historical default, parameter shapes, initialization order, sampling and loss path. Existing positional process/transform arguments retain their positions.

When enabled, pass raw Cartesian proprio `(B,obs_horizon,8)` alongside images to training, predict_action and sample_action_sequences. IMLEPolicy applies the same validated checkpoint zscore in all three paths, then concatenates per-step `[overhead_features,wrist_features,normalized_proprio]`.

For feature_dim=128 and obs_horizon=8, condition changes from `(B,16,128)` to `(B,8,264)`, flattened to 2112 for the generator. Gripper-head temporal mean has width264. Noise remains `(B,pred_horizon,7)` with separate gripper, full output `(B,pred_horizon,8)`, and action slicing remains `obs_horizon-1`.

`imle/proprio.py` computes Train-only dimension-wise mean/population std in float64, saves float32 values and uses scale1 for std<1e-4. Metadata validates dimension8, xyz in meters, quaternion xyzw and gripper0=open/1=closed. Quaternion values/signs are not changed. Missing/malformed statistics or nonfinite values raise ValueError.

The companion LeWorldModel change saves/restores statistics, builds aligned training/runtime histories, exports offline histories and retains old checkpoint default=false. See `../LeWorldModel-imle-proprio/docs/IMLE_PROPRIO_CONDITIONING.md` for the full Japanese implementation/test report and deployment limitations.

Changed library files are restricted to `stable_worldmodel/imle/{model,policy,proprio}.py` and `tests/imle/test_proprio.py`. The shared policy, Diffusion, JEPA, GPC candidate scoring/reward/selection, normalization, execution, and visualization are unchanged.

CPU tests cover state conditioning, backward, seeded inference and malformed statistics. Companion tests also compare the original image-only policy outputs/RNG exactly, checkpoint/EMA round trips, synthetic training, offline modes, actual small JEPA and hardware-free rollout histories. No robot or camera connection, push or merge was performed. CUDA and trained production checkpoint comparisons are unverified.

Final companion CPU verification: 120 passed, 1 pre-existing failed, 2 skipped, 85 subtests passed across the combined and episode-boundary runs. The existing failure is the ProcessStats test's module-level AST execution of a function return; it also fails before these changes. Skips are CUDA and an unavailable trained DP checkpoint. New failures: zero. The protected shared source files were additionally checked byte-for-byte against the baseline commits.
