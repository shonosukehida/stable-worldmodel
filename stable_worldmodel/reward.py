import torch

def latent_goal_reward(rollout_output: dict, info_dict: dict,) -> torch.Tensor:
    pred_emb = rollout_output["predicted_emb"]
    goal_emb = rollout_output["goal_emb"]

    pred_final = pred_emb[:, :, -1]
    goal_final = goal_emb[:, -1]

    rewards = -(
        (pred_final - goal_final[:, None]) ** 2
    ).sum(dim=-1)

    return rewards