from pytorchvideo.transforms import (
  ApplyTransformToKey,
  Div255,
  Normalize,
  RandomShortSideScale,
  ShortSideScale,
  UniformTemporalSubsample,
  UniformCropVideo,
  RemoveKey
)

from pytorchvideo_trainer.datamodule.transforms import SlowFastPackPathway
from torchvision.transforms import (
  CenterCrop,
  Compose,
  RandomCrop,
  RandomHorizontalFlip
)

mean_Kinetics = [0.45, 0.45, 0.45]
std_Kinetics = [0.225, 0.225, 0.225]


def get_slowfast_transform(cfg):
    train_transform = Compose([
        ApplyTransformToKey(
            key="video",
            transform=Compose(
                transforms=[
                    UniformTemporalSubsample(num_samples=cfg.T * cfg.alpha),
                    Div255(),
                    Normalize(mean=mean_Kinetics, std=std_Kinetics),
                    RandomShortSideScale(min_size=256, max_size=320),
                    RandomCrop(224),
                    RandomHorizontalFlip(0.5),
                    SlowFastPackPathway(alpha=cfg.alpha)
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
                    UniformTemporalSubsample(num_samples=cfg.T * cfg.alpha),
                    Div255(),
                    Normalize(mean=mean_Kinetics, std=std_Kinetics),
                    ShortSideScale(256),
                    CenterCrop(256),
                    SlowFastPackPathway(alpha=cfg.alpha)
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
                    UniformTemporalSubsample(num_samples=cfg.T * cfg.alpha),
                    Div255(),
                    Normalize(mean=mean_Kinetics, std=std_Kinetics),
                    ShortSideScale(256),
                ]
            )
        ),
        UniformCropVideo(256),
        ApplyTransformToKey(
            key="video",
            transform=SlowFastPackPathway(alpha=cfg.alpha)
        ),
        RemoveKey(key="audio")
    ])

    return train_transform, val_transform, test_transform
