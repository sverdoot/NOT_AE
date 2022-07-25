from torch import nn

from not_ae.utils.general import REGISTRY


@REGISTRY.model.register()
class MLPAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.T = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.Conv2d(128, 1, kernel_size=5, padding=2),
        )

    def forward(self, x):
        return self.T(x)


@REGISTRY.model.register()
class MLPPotential(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.f = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.AvgPool2d(2),  #  128 x 8 x 8
            nn.Conv2d(128, 256, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.AvgPool2d(2),  #  256 x 4 x 4
            nn.Conv2d(256, 512, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.AvgPool2d(2),  #  512 x 2 x 2
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.AvgPool2d(2),  #  512 x 1 x 1
            nn.Conv2d(512, 1, kernel_size=1, padding=0),
            nn.Flatten(1),
        )

    def forward(self, x):
        return self.f(x).mean(1)
