import torch.cuda
import torch.nn as nn
from .build import NETWORK_REGISTRY
from dpcv import device


@NETWORK_REGISTRY.register()
class SpectrumConv1D(nn.Module):

    def __init__(self, channel=50, hidden_units=[64, 256, 1024]):
        super(SpectrumConv1D, self).__init__()

        self.conv_in = nn.Sequential(
            nn.Conv1d(
                in_channels=2, out_channels=hidden_units[0], kernel_size=(1, 7), padding=(0, 3)
            ),
            # nn.BatchNorm1d(hidden_units[0]),
            nn.ReLU(),
        )
        self.conv_up_scale = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_units[0], out_channels=hidden_units[1],
                kernel_size=(1, 5), padding=(0, 2),
            ),
            nn.ReLU(),

            nn.Conv1d(
                in_channels=hidden_units[1], out_channels=hidden_units[2],
                kernel_size=(1, 3), padding=(0, 1),
            ),
            nn.ReLU()
        )
        self.conv_down_scale = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_units[2], out_channels=hidden_units[1],
                kernel_size=(1, 3), padding=(0, 1),
            ),
            nn.ReLU(),
        )

        self.conv_out = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_units[1],  out_channels=1, kernel_size=(1, 1)
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=1, out_channels=1, kernel_size=(1, channel)
            ),
        )

    def forward(self, x):
        x = self.conv_in(x)           # (bs, 2, 5, 50) --> (bs, 64, 5, 50)
        x = self.conv_up_scale(x)     # (bs, 2, 5, 50) --> (bs, 64, 5, 50)
        x = self.conv_down_scale(x)   # (bs, 2, 5, 50) --> (bs, 64, 5, 50)
        x = self.conv_out(x)          # (bs, 2, 5, 50) --> (bs, 64, 5, 50)
        x = x.squeeze(1)              # (bs, 2, 5, 50) --> (bs, 64, 5, 50)
        x = x.squeeze()               # (bs, 2, 5, 50) --> (bs, 64, 5, 50)
        return x


@NETWORK_REGISTRY.register()
def spectrum_conv_model(cfg):
    # return SpectrumConv1D().to(device=torch.device("gpu" if torch.cuda.is_available() else "cpu"))
    sample_channel = 50
    return SpectrumConv1D(sample_channel).to(device=device)
