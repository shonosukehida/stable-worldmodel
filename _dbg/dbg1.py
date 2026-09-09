



from diffusers.schedulers.scheduling_ddpm import (
    DDPMScheduler,
)

from stable_worldmodel.diffusion import (
    ConditionalUnet1D,
    ResNet18ObsEncoder,
)

from stable_worldmodel.policy import DiffusionPolicy
import torch




obs_horizon = 3
action_dim = 8

process = None
transform = None

obs_encoder = ResNet18ObsEncoder(
    pretrained=False,
)

obs_feature_dim = obs_encoder.output_shape()[0]

model = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=(
        obs_feature_dim * obs_horizon
    ),
    diffusion_step_embed_dim=256,
    down_dims=(256, 512, 1024),
    kernel_size=5,
    n_groups=8,
    cond_predict_scale=True,
)

noise_scheduler = DDPMScheduler(
    num_train_timesteps=100,
    beta_schedule="squaredcos_cap_v2",
    clip_sample=True,
    prediction_type="epsilon",
)


policy = DiffusionPolicy(
    model=model,
    obs_encoder=obs_encoder,
    noise_scheduler=noise_scheduler,
    pred_horizon=16,
    obs_horizon=3,
    action_horizon=8,
    action_dim=8,
    process=process,
    transform=transform,
)

print("policy initialized successfully")


print("\n==============================")
print("DiffusionPolicy test start")
print("==============================\n")

device = torch.device("cpu")

policy.model = policy.model.to(device)
policy.obs_encoder = policy.obs_encoder.to(device)

# -----------------------------
# Test settings
# -----------------------------
B = 2
K = 4

image_height = 224
image_width = 224

print("device:", device)
print("batch size B:", B)
print("num candidates K:", K)

# ============================================================
# 1. Test ConditionalUnet1D.forward()
# ============================================================

print("\n[1] ConditionalUnet1D.forward()")

sample = torch.randn(
    B,
    policy.pred_horizon,
    policy.action_dim,
    device=device,
)

global_cond = torch.randn(
    B,
    obs_feature_dim * policy.obs_horizon,
    device=device,
)

timestep = torch.randint(
    low=0,
    high=noise_scheduler.config.num_train_timesteps,
    size=(B,),
    device=device,
)

with torch.no_grad():
    model_output = policy.model(
        sample,
        timestep,
        local_cond=None,
        global_cond=global_cond,
    )

print("input sample shape :", sample.shape)
print("global_cond shape  :", global_cond.shape)
print("model output shape :", model_output.shape)

assert model_output.shape == sample.shape

print("OK")


# ============================================================
# 2. Test conditional_sample()
# ============================================================

print("\n[2] DiffusionPolicy.conditional_sample()")

condition_data = torch.zeros(
    B,
    policy.pred_horizon,
    policy.action_dim,
    device=device,
)

with torch.no_grad():
    sampled_trajectory = policy.conditional_sample(
        condition_data=condition_data,
        global_cond=global_cond,
    )

print(
    "sampled trajectory shape:",
    sampled_trajectory.shape,
)

assert sampled_trajectory.shape == (
    B,
    policy.pred_horizon,
    policy.action_dim,
)

print("OK")


# ============================================================
# 3. Create dummy observation
# ============================================================

print("\n[3] Create dummy observation")

# _prepare_info() が transform=None の場合は
# tensorをそのまま受け取れるので CHW で直接作る
dummy_pixels = torch.rand(
    B,
    policy.obs_horizon,
    3,
    image_height,
    image_width,
    dtype=torch.float32,
)

info_dict = {
    "pixels": dummy_pixels,
}

print(
    "pixels shape:",
    dummy_pixels.shape,
)


# ============================================================
# 4. Test predict_action()
# ============================================================

print("\n[4] DiffusionPolicy.predict_action()")

with torch.no_grad():
    result = policy.predict_action(
        info_dict
    )

print(
    "result keys:",
    result.keys(),
)

print(
    "action_pred shape:",
    result["action_pred"].shape,
)

print(
    "action shape:",
    result["action"].shape,
)

assert result["action_pred"].shape == (
    B,
    policy.pred_horizon,
    policy.action_dim,
)

assert result["action"].shape == (
    B,
    policy.action_horizon,
    policy.action_dim,
)

print("OK")


# ============================================================
# 5. Test get_action()
# ============================================================

print("\n[5] DiffusionPolicy.get_action()")

with torch.no_grad():
    action = policy.get_action(
        info_dict
    )

print(
    "get_action output shape:",
    action.shape,
)

# get_action() は action chunk の最初の1stepを返す
assert action.shape == (
    B,
    policy.action_dim,
)

print("OK")


# ============================================================
# 6. Test sample_action_sequences()
# ============================================================

print(
    "\n[6] DiffusionPolicy.sample_action_sequences()"
)

with torch.no_grad():
    candidate_actions = (
        policy.sample_action_sequences(
            info_dict,
            num_samples=K,
        )
    )

print(
    "candidate actions shape:",
    candidate_actions.shape,
)

assert candidate_actions.shape == (
    B,
    K,
    policy.pred_horizon,
    policy.action_dim,
)

print("OK")


# ============================================================
# 7. Test compute_loss()
# ============================================================

print("\n[7] DiffusionPolicy.compute_loss()")

dummy_actions = torch.randn(
    B,
    policy.pred_horizon,
    policy.action_dim,
    dtype=torch.float32,
)

batch = {
    "pixels": dummy_pixels,
    "action": dummy_actions,
}

loss = policy.compute_loss(
    batch
)

print(
    "loss:",
    loss.item(),
)

assert loss.ndim == 0

assert torch.isfinite(loss)

print("OK")


print("\n==============================")
print("All DiffusionPolicy tests passed!")
print("==============================")