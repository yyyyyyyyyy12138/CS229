from pytorchvideo.models.slowfast import create_slowfast
import os
import torch


def get_slowfast(num_classes, cfg):
    # create slowfast network with moma num_of_classes of [act]
    net = create_slowfast(
        input_chhoiels=(3, 3),
        model_depth=50,
        model_num_class=num_classes,
        dropout_rate=0.5,
        slowfast_fusion_conv_kernel_size=(7, 1, 1)
    )

    # load pretrain weight from kinetics-400
    if cfg.pretrain:
        pretrain_path = os.path.join(cfg.root, "pretrain/SLOWFAST_8x8_R50.pyth")  # 8*8 = T*Tau
        ckpt = torch.load(pretrain_path)
        weights = ckpt["model_state"]
        # remove the weights for last layer
        weights.pop("blocks.6.proj.weight")
        weights.pop("blocks.6.proj.bias")
        # apply the pretrain weight to our net
        net.load_state_dict(weights, strict=False)

    return net
