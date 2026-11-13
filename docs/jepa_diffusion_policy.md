# Frozen JEPA Diffusion Policy

`stable_worldmodel.diffusion.jepa` provides `FrozenJEPAEncoder` and
`JEPADiffusionPolicy`. The latter extends only the observation-dictionary hook;
noise scheduling, U-Net calls, training loss, action sampling, and GPC candidate
sampling use the existing `DiffusionPolicy` implementation.

The encoder runs the loaded world's `encode()` method on preprocessed overhead
and wrist images plus raw proprio. It normalizes proprio using saved world-model
training statistics and returns the concatenated overhead/projector,
wrist/projector, and proprio latents. All feature parameters are frozen, forward
uses no_grad, and train() always retains eval mode, including BatchNorm buffers.
`EmptyWristEncoder` is a zero-dimensional compatibility handle: wrist features
are already part of the combined encoder.

Existing ResNet18/Conv encoders and state_dict keys are unchanged. The base
policy's new `_encode_observation_info(info)` delegates to its historical
`_encode_observation(pixels, wrist_pixels)` implementation. JEPA is selected
explicitly by the application, not imported or loaded by legacy encoders.

Model-object restoration, provenance, Hydra configuration, checkpoint packaging,
training, offline evaluation, and real-robot integration live in the companion
LeWorldModel repository (`jepa_dp.py`, `JEPA_DP.md`). The integration tests there
cover synthetic and real JEPA checkpoints and exact legacy loss/actions/RNG
comparisons for ResNet18 and Conv.
