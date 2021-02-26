import torch.nn as nn


class Head(nn.Module):
    def __init__(self, out_channels: int, intermediate_channel: int, head_conv: int):
        super().__init__()
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


class DetectionHead(nn.Module):
    def __init__(self, num_classes, intermediate_channel, head_conv):
        super().__init__()

        self.heatmap = Head(num_classes, intermediate_channel, head_conv)
        self.width_height = Head(2, intermediate_channel, head_conv)
        self.regression = Head(2, intermediate_channel, head_conv)
        self.init_weights()

    def forward(self, x):
        return {
            "heatmap": self.heatmap(x),
            "width_height": self.width_height(x),
            "regression": self.regression(x),
        }

    def init_weights(self):
        self.heatmap[-1].bias.data.fill_(-2.19)

        self.width_height.fill_fc_weights()
        self.regression.fill_fc_weights()


class PoseHead(nn.Module):
    def __init__(self, intermediate_channel, head_conv, num_joints=17):
        super().__init__()

        self.keypoints = Head(int(num_joints * 2), intermediate_channel, head_conv)
        self.heatmap_keypoints = Head(num_joints, intermediate_channel, head_conv)
        self.heatpoint_offset = Head(2, intermediate_channel, head_conv)
        self.init_weights()

    def forward(self, x):
        return {
            "keypoints": self.keypoints(x),
            "heatmap_keypoints": self.heatmap_keypoints(x),
            "heatpoint_offset": self.heatpoint_offset(x),
        }

    def init_weights(self):
        self.heatmap_keypoints[-1].bias.data.fill_(-2.19)

        self.keypoints.fill_fc_weights()
        self.heatpoint_offset.fill_fc_weights()
