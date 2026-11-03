# IMLE provenance and adaptation

Reference checkout: `s3fmp` commit `7b795a36f4f640c2b7ad2a192790a5513b70b23d` (read only).
The generator dependency is `keio-crl/ml-networks` commit
`c672fb376b28b96c7c58cf6bbceb351176b0642d`, pinned by `s3fmp/uv.lock`.
No runtime import of either checkout is required.

| Reference | Port |
|---|---|
| `src/s3fmp/policy.py::rs_imle_loss` | `losses.py::rs_imle_loss` |
| `IMLEPolicy._generate`, `_gripper`, `_noise`, `inference` | `model.py::ActionGenerator`, `policy.py::IMLEPolicy` |
| `IMLEPolicy._shared_step` | `policy.py::IMLEPolicy.compute_loss` |
| `ml_networks/unet.py::ConditionalUnet1d`, `ConditionalResidualBlock1d`, `Downsample1d`, `Upsample1d` | `networks.py` |
| `ml_networks/layers.py::ConvNormActivation1d`, `Attention1d`, horizon shuffle | `networks.py` |
| `models/cfg/xarm/imle.yaml` | LeWorldModel `config/train/policy/imle_policy.yaml` |
| `PolicyBase.configure_optimizers`, `optimizer_step`, EMA/checkpoint hooks; `main.py::train` | LeWorldModel `train_imle_policy.py`, `imle_checkpoint.py` |
| `src/s3fmp/eval.py` load + `apply_ema()` | LeWorldModel `imle_checkpoint.py::load_imle_policy` |

The reduced network module retains the reference's parameter names, layer
initialization order, convolution/FiLM/attention/downsampling and upsampling
operations. Only the non-hypernetwork 1D path used by the IMLE config is
supported. Torch activations replace the generic activation dispatch; unused
2D, hypernetwork, and Lightning utilities are excluded.

RS loss uses Euclidean distance over the flattened non-gripper trajectory,
not MSE. Candidates at or below `epsilon * sqrt(T * G)` are penalized by the
detached global maximum distance before `min`. The `min` remains in the
computational graph: the selected candidate receives gradients. Teacher
examples whose selected distance is not strictly below that global maximum
are omitted. This also drops an otherwise valid maximum-only candidate; that
reference edge behavior is intentionally preserved. No extra rejection loop,
threshold update, or nearest-neighbor search across different observations is
added. Epsilon is fixed.

The independent gripper head receives teacher non-gripper actions during
training and generated non-gripper actions during inference. It uses
trajectory-summed L1, multiplied by `0.1 / pred_horizon`. This is present in
s3fmp, not a newly invented auxiliary loss. There is no trajectory consistency
mechanism in the inspected s3fmp IMLE path, and none is added here. The public
RSS implementation is not substituted for the local reference.

Adaptations:

- s3fmp uses a ConvNet `[16,32,64]`, pooled/projected to 128, plus a state MLP.
  The port uses the existing DP ResNet18 encoder independently for each camera,
  projected from 512 to the same configurable 128 features. Overhead/wrist
  features are separate tokens, interleaved by observation time. The state MLP
  is omitted to use the same image/action dataset fields as DP, without leaking
  teacher actions into conditioning. This is not encoder-level equivalence to
  the reference.
- Cartesian `[x,y,z,qx,qy,qz,qw,gripper]` replaces the reference robot's action
  interpretation. Gaussian noise is `(B,T,7)` with separate gripper enabled,
  or `(B,T,8)` otherwise. `noise_dim` in a checkpoint is the per-step dimension.
- Default horizons become prediction/observation/execution `16/8/8` instead of
  xarm reference `32/1/32`; execute slice is `[obs_horizon-1 : +action_horizon]`.
  U-Net horizons must fit both downsampling factors.
- Images use DP preprocessing at size 224; batch size is 16 instead of 64 for
  multiview/history memory. Normalization and random window split follow DP,
  not s3fmp's episode-split dataset. Those differences are explicit experiments,
  not an identical s3fmp end-to-end training setup.
- AdamW lr `1e-4`, default weight decay `.01`, Gaussian scale `1`, N `128`,
  epsilon `.01`, generator `[64,128,256]`, kernel 5, attention 8 heads,
  gripper `[32,64,128]`, kernel 3, EMA `.999` with update counting,
  500 epochs / patience 20 are retained from the inspected reference/defaults.
  There is no learning-rate scheduler in that reference path.
- Validation computes the reference loss with EMA weights, without its
  Lightning first-batch visualization samples. A null validation seed consumes
  the normal RNG stream; an explicit seed temporarily fixes and restores RNG.
- Lightning checkpoints become an explicit self-contained IMLE dictionary with
  model/encoders/optimizer/EMA/config/normalization/layout metadata. This is not
  a loader for old s3fmp checkpoints.

`tests/imle/s3fmp_reference.npz` contains CPU float32 outputs, loss, and gradients
for every generator/head parameter, computed from the reference AST, not the
port. The source SHA256 values and torch version are embedded. The fixture
uses smaller `[8,16,32]` widths, T=16, feature width=8 and the same architecture
operations to keep CPU tests small. It verifies generator/loss equivalence,
not camera encoder or full training equivalence. Regenerate explicitly with:

```bash
python tests/imle/generate_reference.py \
  --s3fmp /path/to/read-only/s3fmp \
  --ml-networks /path/to/ml-networks-at-c672fb3
```

## Configurable image encoders

The original adaptation above describes the default `encoder.type=resnet18`.
`encoder.type=convnet` now selects `obs_encoder.py::ConvNetObsEncoder`:
three biased Conv2d+Mish stages `[16,32,64]`, kernel/stride/padding `3/2/1`,
AdaptiveAvgPool2d(1), Flatten, and a biased 64→feature_dim Linear layer.
This matches `models/cfg/xarm/imle.yaml` and the pinned ml-networks
`vision.Encoder/ConvNet` path, without unused generic wrappers/Identity layers.
No normalization, dropout, or image attention is introduced.

Both options retain the existing IMLE two-camera/history adapter and the same
ImageNet normalization/resize preprocessing. The reference instead uses a
single 96x128 camera divided by 255 and state conditioning; encoder equivalence
on identical tensors is not end-to-end data-pipeline equivalence.

`tests/imle/convnet_reference.npz` is generated from the actual reference AST
with seed 137, two 96x128 RGB images, feature_dim 128, CPU float32. It contains
all initial parameters, output features, input gradients and every parameter
gradient. Source hashes and torch version are embedded. The port matches these
arrays exactly. Regenerate with `tests/imle/generate_convnet_reference.py
--ml-networks /path/to/pinned/checkout`. Reference checkouts remain read-only.

LeWorldModel checkpoints store a resolved `encoder_architecture` descriptor
plus the full encoder config and conditioning width. Missing encoder.type in
historical checkpoints means ResNet18. Existing ResNet construction order,
parameter keys, preprocessing, outputs, training update, and RNG are retained.
