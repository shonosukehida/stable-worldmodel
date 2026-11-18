from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class History:
    """Four packed observations and the three normalized transitions between them."""
    obs: torch.Tensor
    past_actions: torch.Tensor

    def action_window(self, next_action):
        return torch.cat((self.past_actions, next_action[:, None]), dim=1)

    def advance(self, image, window):
        return History(torch.cat((self.obs[:, 1:], image[:, None]), dim=1), window[:, 1:])
