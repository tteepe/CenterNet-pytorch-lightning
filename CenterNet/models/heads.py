import torch.nn as nn


class HeadConv(nn.Module):
    def __init__(self, out_channels: int, intermediate_channel: int, head_conv: int):
        super().__init__()
        self.out_channels = out_channels

        self.fc = nn.Sequential(
            nn.Conv2d(
                intermediate_channel, head_conv, kernel_size=3, padding=1, bias=True
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, out_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.fc(x)

    def fill_fc_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class CenterHead(nn.Module):
    def __init__(self, heads, intermediate_channel, head_conv):
        super().__init__()

        self.heads = heads
        for name, out_channel in heads.items():
            self.__setattr__(name, HeadConv(out_channel, intermediate_channel, head_conv))

        self.init_weights()

    def forward(self, x):
        ret = {}
        for name in self.heads.keys():
            ret[name] = self.__getattr__(name)(x)

        return ret

    def init_weights(self):
        for name in self.heads.keys():
            if name.startswith("heatmap"):
                self.__getattr__(name).fc[-1].bias.data.fill_(-2.19)
            else:
                self.__getattr__(name).fill_fc_weights()
