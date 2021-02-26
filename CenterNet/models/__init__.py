from .backbones.msra_resnet import get_pose_net
from .backbones.pose_dla_dcn import get_pose_net as get_dla_dcn
from .backbones.resnet_dcn import get_pose_net as get_pose_net_dcn
from .backbones.large_hourglass import get_large_hourglass_net

_model_factory = {
    "res": get_pose_net,  # default Resnet with deconv
    "dla": get_dla_dcn,
    "resdcn": get_pose_net_dcn,
    "hourglass": get_large_hourglass_net
}


def create_model(arch):
    num_layers = int(arch[arch.find("_") + 1:]) if "_" in arch else 0
    arch = arch[: arch.find("_")] if "_" in arch else arch
    get_model = _model_factory[arch]
    model = get_model(num_layers=num_layers)
    return model
