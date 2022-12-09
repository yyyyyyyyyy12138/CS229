import os
import momaapi
from pytorchvideo.data import LabeledVideoDataset, make_clip_sampler
from torch.utils.data import DistributedSampler, RandomSampler, SequentialSampler


class MOMAVideoDataset(LabeledVideoDataset):
    # static variable
    num_classes = 20

    def __init__(self, cfg, split: str, transform=None):
        path = os.path.join(cfg.root, "moma-lrg")
        moma = momaapi.MOMA(path)

        # get activity instance IDs (train+val or test)
        if split == "train":
            act_ids = moma.get_ids_act(split="train")
            act_ids_val = moma.get_ids_act(split="val")
            act_ids.extend(act_ids_val)
        else:  # split is val or test
            act_ids = moma.get_ids_act(split="test")

        # get video paths for all activities
        paths = moma.get_paths(ids_act=act_ids)
        # get annotation of activities
        anns_acts = moma.get_anns_act(act_ids)
        # get corresponding class IDs for each activity in train/test set
        cids = [ann_act.cid for ann_act in anns_acts]

        # get dataset: list of tuples[(video path, dict{"cid": labels})]
        dataset = [(path, {"cid": cid, "act_id": act_id}) for path, cid, act_id in zip(paths, cids, act_ids)]

        if split == "train":
            clip_sampler = make_clip_sampler("random", cfg.T*cfg.tau/cfg.fps)
            # video_sampler = DistributedSampler if len(cfg.gpus) > 1 else RandomSampler
            # manually turn randomsampler off when run fit_twostream.py
            video_sampler = SequentialSampler
        elif split == "val":
            clip_sampler = make_clip_sampler("constant_clips_per_video", cfg.T * cfg.tau/cfg.fps, 1)
            video_sampler = SequentialSampler
        else:
            clip_sampler = make_clip_sampler("constant_clips_per_video", cfg.T*cfg.tau/cfg.fps, 1)
            video_sampler = SequentialSampler

        super().__init__(
            labeled_video_paths=dataset,
            clip_sampler=clip_sampler,
            video_sampler=video_sampler,
            transform=transform,
            decode_audio=False
        )
