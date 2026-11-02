"""Small supervised complex-image residual CNN, not a clinical model."""
import numpy as np
import torch
from torch import nn
from .core import phantom, acquisition, noisy_kspace, ifft, project


class ResidualCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(16, 2, 3, padding=1))

    def forward(self, x):
        return x+self.layers(x)


def channels(images):
    return torch.from_numpy(np.stack([images.real, images.imag], axis=1).astype(np.float32))


def train(seed, count=512, epochs=8, size=64):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed*100000+17)
    targets = np.stack([phantom(seed*100000+i, size) for i in range(count)])
    inputs = []
    for target in targets:
        mask, _ = acquisition(rng, size)
        inputs.append(ifft(noisy_kspace(target, rng)*mask[None, :]))
    x, y = channels(np.stack(inputs)), channels(targets)
    model = ResidualCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = []
    generator = torch.Generator().manual_seed(seed)
    model.train()
    for epoch in range(epochs):
        total = 0.
        for ids in torch.randperm(count, generator=generator).split(32):
            optimizer.zero_grad()
            loss = ((model(x[ids])-y[ids])**2).mean()
            loss.backward()
            optimizer.step()
            total += loss.item()*len(ids)
        history.append(total/count)
        print(f"seed={seed} epoch={epoch+1}/{epochs} train_mse={history[-1]:.6f}", flush=True)
    model.eval()
    return model, history


def predict(model, zero_filled, measured, mask):
    with torch.inference_mode():
        pred = model(channels(zero_filled[None]))[0].numpy()
    return project(pred[0]+1j*pred[1], measured, mask)
