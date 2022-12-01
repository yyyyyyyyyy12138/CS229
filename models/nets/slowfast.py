from pytorchvideo.models.slowfast import create_slowfast
import os
import torch


def get_slowfast(num_classes, cfg):
    net = create_slowfast(
        input_channels=(3, 3),
        model_depth=50,
        model_num_class=num_classes,
        dropout_rate=0.5,
        slowfast_fusion_conv_kernel_size=(7, 1, 1)
    )

    if cfg.pretrain:
        pretrain_path = os.path.join(cfg.root, "pretrain/SLOWFAST_8x8_R50.pyth")
        ckpt = torch.load(pretrain_path)
        weights = ckpt["model_state"]
        weights.pop("blocks.6.proj.weight")
        weights.pop("blocks.6.proj.bias")

        net.load_state_dict(weights, strict=False)

    return net
