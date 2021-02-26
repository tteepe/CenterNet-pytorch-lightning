import torch
from models import create_model

supported_backbones = [
    "res_18", "res_101", "resdcn_18", "resdcn_101", "dla_34", "hourglass"
]


def test_models():
    heads = {"heatmap": 80, "width_height": 2, "regression": 2}
    sample_input = torch.rand((1, 3, 512, 512))
    for arch in supported_backbones:
        print(f"Testing: {arch}")
        model = create_model(arch, heads, 64)

        outputs = model(sample_input)

        assert outputs

        for output in outputs:
            for head, shape in heads.items():
                head_shape = torch.Size([1, shape, sample_input.shape[2] // 4, sample_input.shape[3] // 4])
                assert output[head].shape == head_shape


if __name__ == "__main__":
    test_models()
