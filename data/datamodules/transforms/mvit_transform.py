from pytorchvideo.transforms import (
  ApplyTransformToKey,
  Div255,
  Normalize,
  Permute,
  RandomResizedCrop,
  ShortSideScale,
  UniformTemporalSubsample,
  UniformCropVideo,
  RemoveKey
)
from pytorchvideo.transforms.rand_augment import RandAugment
from pytorchvideo_trainer.datamodule.rand_erase_transform import RandomErasing
from torchvision.transforms import (
  CenterCrop,
  Compose,
  RandomHorizontalFlip
)

mean_Kinetics = [0.45, 0.45, 0.45]
std_Kinetics = [0.225, 0.225, 0.225]


def get_mvit_transform(cfg):
    train_transform = Compose([
        ApplyTransformToKey(
            key="video",
            transform=Compose(
                transforms=[
                    UniformTemporalSubsample(num_samples=cfg.T),
                    Div255(),
                    Permute(dims=(1, 0, 2, 3)),
                    RandAugment(magnitude=7, num_layers=4),
                    Permute(dims=(1, 0, 2, 3)),
                    Normalize(mean=mean_Kinetics, std=std_Kinetics),
                    RandomResizedCrop(target_height=224, target_width=224,
                                      scale=(0.08, 1.0), aspect_ratio=(0.75, 1.3333)),
                    RandomHorizontalFlip(0.5),
                    Permute(dims=(1, 0, 2, 3)),
                    RandomErasing(probability=0.25, mode='pixel', max_count=1, num_splits=1, device='cpu'),
                    Permute(dims=(1, 0, 2, 3))
                ]
            )
        ),
        RemoveKey(key="audio")
    ])

    val_transform = Compose([
        ApplyTransformToKey(
            key="video",
            transform=Compose(
                transforms=[
                    UniformTemporalSubsample(num_samples=cfg.T),
                    Div255(),
                    Normalize(mean=mean_Kinetics, std=std_Kinetics),
                    ShortSideScale(224),
                    CenterCrop(224)
                ]
            )
        ),
        RemoveKey(key="audio")
    ])

    test_transform = Compose([
        ApplyTransformToKey(
            key="video",
            transform=Compose(
                transforms=[
                    UniformTemporalSubsample(num_samples=cfg.T),
                    Div255(),
                    Normalize(mean=mean_Kinetics, std=std_Kinetics),
                    ShortSideScale(224),
                ]
            )
        ),
        UniformCropVideo(224),
        RemoveKey(key="audio")
    ])

    return train_transform, val_transform, test_transform
