import os
import ffmpeg
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.functional import one_hot
from PIL import Image
import momaapi
from pytorchvideo.data import LabeledVideoDataset, make_clip_sampler
from torch.utils.data import DistributedSampler, RandomSampler


class MOMAGraphDataset(Dataset):
    num_classes = 20

    def __init__(self, root_dir: str, train: bool):
        path = os.path.join(root_dir, "moma-lrg")
        moma = momaapi.MOMA(path)
        self.path = path
        self.train = train

        # get activity instance IDs (train+val or test)
        if self.train:
            act_ids = moma.get_ids_act(split="train")
            act_ids_val = moma.get_ids_act(split="val")
            act_ids.extend(act_ids_val)
        else:
            act_ids = moma.get_ids_act(split="test")
        self.act_ids = act_ids

        # get annotation of activities
        anns_acts = moma.get_anns_act(act_ids)
        # get corresponding class IDs for each activity in train/test set
        cids = [ann_act.cid for ann_act in anns_acts]

        self.cids = cids
        self.moma = moma

    def __len__(self):
        return len(self.cids)

    def __getitem__(self, idx):
        act_id = self.act_ids[idx]
        label = self.cids[idx]

        hoi_id = self.moma.get_ids_hoi(ids_act=[act_id])
        hois = self.moma.get_anns_hoi(ids_hoi=hoi_id)
        feature = []
        for hoi in hois:
            # TODO: get bounding box
            actors = hoi.actors
            actors_cids = [actor.cid for actor in actors]
            actors_cids = torch.tensor(actors_cids)
            actors_cids = one_hot(actors_cids.long(), num_classes=26)
            actors_cids = actors_cids.flatten()

            objects = hoi.objects
            objects_cids = [object.cid for object in objects]
            objects_cids = torch.tensor(objects_cids)
            objects_cids = one_hot(objects_cids.long(), num_classes=227)
            objects_cids = objects_cids.flatten()

            ias = hoi.ias
            atts = hoi.atts
            # TODO: question: different kinds?
            atts_cids = [att.cid for att in atts]
            ias_cids = [ia.cid + 4 for ia in ias]
            atts_cids.extend(ias_cids)
            atts_cids = torch.tensor(atts_cids)
            atts_cids = one_hot(atts_cids.long(), num_classes=13)  # len(att_names) = 4; len(ia_names) = 9
            atts_cids = atts_cids.flatten()

            tas = hoi.tas
            rels = hoi.rels
            rels_cids = [rel.cid for rel in rels]
            tas_cids = [ta.cid + 19 for ta in tas]
            rels_cids.extend(tas_cids)
            rels_cids = torch.tensor(rels_cids)
            rels_cids = one_hot(rels_cids.long(), num_classes=52)  # len(ta_names) = 33; len(rel_names) = 19

            rels_cids = rels_cids.flatten()

            feature.extend(actors_cids.tolist())
            feature.extend(objects_cids.tolist())
            feature.extend(atts_cids.tolist())
            feature.extend(rels_cids.tolist())
    
        return torch.tensor(feature), label


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
        dataset = [(path, {"cid": cid}) for path, cid in zip(paths, cids)]

        if split == "train":
            clip_sampler = make_clip_sampler("random", cfg.T*cfg.tau/cfg.fps)
        elif split == "val":
            clip_sampler = make_clip_sampler("uniform", cfg.T*cfg.tau/cfg.fps)
        else:
            clip_sampler = make_clip_sampler("constant_clips_per_video", cfg.T*cfg.tau/cfg.fps, cfg.num_clips, cfg.num_crops)

        super().__init__(
            labeled_video_paths=dataset,
            clip_sampler=clip_sampler,
            video_sampler=DistributedSampler if cfg.gpus not in [0, 1] else RandomSampler,
            transform=transform,
            decode_audio=False
        )


class MOMAFrameDataset(Dataset):
    # static variable
    num_classes = 20

    def __init__(self, root_dir: str, train: bool, transform=None):
        path = os.path.join(root_dir, "moma-lrg")
        moma = momaapi.MOMA(path)
        self.path = path
        # self.num_classes = moma.num_classes["act"]
        self.transform = transform
        self.train = train

        # get activity instance IDs (train+val or test)
        if self.train:
            act_ids = moma.get_ids_act(split="train")
            act_ids_val = moma.get_ids_act(split="val")
            act_ids.extend(act_ids_val)
        else:
            act_ids = moma.get_ids_act(split="test")

        # get video paths for all activities
        paths = moma.get_paths(ids_act=act_ids)
        # get annotation of activities
        anns_acts = moma.get_anns_act(act_ids)
        # get corresponding class IDs for each activity in train/test set
        cids = [ann_act.cid for ann_act in anns_acts]

        # get dataset: list of tuples[(video path, label)]
        dataset = [(path, cid) for path, cid in zip(paths, cids)]
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path, label = self.dataset[idx]
        assert os.path.exists(path)

        probe = ffmpeg.probe(path)
        video_streams = probe["streams"][0]

        # get total number of frames in the video
        num_frames = int(video_streams["nb_frames"])
        width = int(video_streams['width'])
        height = int(video_streams['height'])

        # if train, get a random frame, if test, get the mid frame index
        target_frame = random.randint(0, num_frames-1) if self.train else num_frames // 2

        # get the mid frame image
        out, _ = (
            ffmpeg
            .input(path)
            .filter("select", "gte(n,{})".format(target_frame))
            .output('pipe:', vframes=1, format='rawvideo', pix_fmt='rgb24', loglevel="quiet")
            .run(capture_stdout=True)
        )
        # turn the image type to numpy
        image = (
            np
            .frombuffer(out, np.uint8)
            .reshape([-1, height, width, 3])
        )
        image = image[0]
        # turn the image type to PIL (for transform purpose)
        image = Image.fromarray(image)
        # transform image:  resize, to Tensor, normalize
        if self.transform:
            image = self.transform(image)
        return image, label
