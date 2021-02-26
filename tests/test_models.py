import torch
from models import create_model
from models.heads import CenterHead

supported_backbones = [
    "res_18", "res_101", "resdcn_18", "resdcn_101", "dla_34", "hourglass"
]


def test_models():
    sample_input = torch.rand((1, 3, 512, 512))

    for arch in supported_backbones:
        print(f"Testing: {arch}")
        model = create_model(arch)
        heads = {
            "heatmap": 1,
            "width_height": 2,
            "regression": 2,
            "heatmap_keypoints": 17,
            "heatpoint_offset": 2,
            "keypoints": 34,
        }

        head = CenterHead(heads, model.out_channels, 64)

        out_backbone = model(sample_input)
        output = head(out_backbone[-1])

        assert output

        for name, data in output.items():
            shape = getattr(head, name).out_channels
            head_shape = torch.Size(
                [1, shape, sample_input.shape[2] // 4, sample_input.shape[3] // 4]
            )
            assert data.shape == head_shape


if __name__ == "__main__":
    test_models()
