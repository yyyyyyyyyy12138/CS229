import os
import torch
from pytorchvideo.models.vision_transformers import create_multiscale_vision_transformers


def get_mvit(num_classes, cfg):
    # create mvit network with moma num_of_classes of [act]
    net = create_multiscale_vision_transformers(
        spatial_size=224,
        temporal_size=16,
        depth=16,
        input_channels=3,
        patch_embed_dim=96,
        num_heads=1,
        dropout_rate_block=0.0,
        droppath_rate_block=0.2,
        embed_dim_mul=[[1, 2.0], [3, 2.0], [14, 2.0]],
        atten_head_mul=[[1, 2.0], [3, 2.0], [14, 2.0]],
        pool_q_stride_size=[[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]],
        pool_kv_stride_size=None,
        pool_kv_stride_adaptive=[1, 8, 8],
        pool_kvq_kernel=[3, 3, 3],
        # head =
        head_dropout_rate=0.5,
        head_activation=None,
        head_num_classes=num_classes
    )

    # load pretrain weight from kinetics-400
    if cfg.pretrain:
        pretrain_path = os.path.join(cfg.root, "pretrain/MVIT_B_16x4.pyth")
        ckpt = torch.load(pretrain_path)
        weights = ckpt["model_state"]
        # remove the weights for last layer
        weights.pop('head.proj.weight')
        weights.pop('head.proj.bias')
        # apply the pretrain weight to our net
        net.load_state_dict(weights, strict=False)

    return net

