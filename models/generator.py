import torch
import torch.nn as nn


class View(nn.Module):
    def __init__(self, *size):
        super().__init__()
        self.size = size

    def forward(self, tensor) -> torch.Tensor:
        return tensor.view(self.size)


class Generator(nn.Module):
    def __init__(self, latent_size: int):
        super().__init__()
        self.__sequential_blocks = [
            nn.Linear(latent_size, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Linear(50, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, 400),
            nn.ReLU()
        ]
        self.main = nn.Sequential(*self.__sequential_blocks)

    def forward(self, input_latent: torch.Tensor):
        decoded_images = self.main(input_latent)
        return decoded_images
