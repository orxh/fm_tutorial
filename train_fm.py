import numpy as np
import torch
import torch.nn as nn
import utils
import matplotlib.pyplot as plt

BATCH_SIZE = 1024
STEPS = 5_000
HIDDEN = 64
LEARNING_RATE = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Flow(nn.Module):
    def __init__(self, x_dim: int = 2, h: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + 1, h),
            nn.ELU(),
            nn.Linear(h, h),
            nn.ELU(),
            nn.Linear(h, h),
            nn.ELU(),
            nn.Linear(h, x_dim),
        )

    def forward(self, t: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        """Predict velocity field at (t, x_t)."""
        h = torch.cat([x_t, t], dim=-1)
        return self.net(h)

    @torch.no_grad()
    def sample_path(self, x_0: torch.Tensor, num_steps: int = 200) -> torch.Tensor:
        """Return the entire path: (num_steps+1, B, 2)."""
        x_t = x_0.clone()
        path = [x_t.clone()]
        ts = torch.linspace(0, 1, num_steps + 1, device=x_0.device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            v = self.forward(ts[i : i + 1].repeat(x_t.size(0), 1), x_t)
            x_t = x_t + dt * v
            path.append(x_t.clone())
        return torch.stack(path)

    @torch.no_grad()
    def sample(self, x_0: torch.Tensor, num_steps: int = 200) -> torch.Tensor:
        return self.sample_path(x_0, num_steps)[-1]


def make_batch(
        batch_size: int,
        means_source: list[np.ndarray],
        covs_source: list[np.ndarray],
        weights_source: list[float],
        means_target: list[np.ndarray],
        covs_target: list[np.ndarray],
        weights_target: list[float]
    ) -> tuple[torch.Tensor, ...]:
    """Return tensors (t, x_t, dx_t)."""
    source_np = utils.sample_gaussian_mixture(
        batch_size, means_source, covs_source, weights_source
    )
    target_np = utils.sample_gaussian_mixture(
        batch_size, means_target, covs_target, weights_target
    )

    source = torch.from_numpy(source_np).float().to(DEVICE)
    target = torch.from_numpy(target_np).float().to(DEVICE)

    t = torch.rand(batch_size, 1, device=DEVICE)
    x_t = (1 - t) * source + t * target
    dx_t = target - source
    return t, x_t, dx_t


def train(config: dict):
    means_source = config.get("means_source")
    covs_source = config.get("covs_source")
    weights_source = config.get("weights_source", [1.0])
    means_target = config.get("means_target")
    covs_target = config.get("covs_target")
    weights_target = config.get("weights_target", [1.0])

    model = Flow(x_dim=2, h=HIDDEN).to(DEVICE)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE
    )

    training_losses = []

    for step in range(1, STEPS + 1):
        t, x_t, dx_t = make_batch(
            BATCH_SIZE,
            means_source,
            covs_source,
            weights_source,
            means_target,
            covs_target,
            weights_target
        )

        optimiser.zero_grad(set_to_none=True)
        loss = torch.mean((model(t, x_t) - dx_t) ** 2)
        loss.backward()
        optimiser.step()

        training_losses.append(loss.item())

    return model.eval(), training_losses
