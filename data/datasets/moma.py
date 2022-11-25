import momaapi
from torch.utils.data import Dataset, DataLoader
import ffmpeg
import os
import numpy as np
from PIL import Image as im

class MOMADataset(Dataset):
    def __init__(self, root_dir: str, train: bool, transform=None):
        path = os.path.join(root_dir, "moma-lrg")
        moma = momaapi.MOMA(path)
        self.path = path
        self.num_classes = moma.num_classes()
        self.transform = transform

        # get classes ids for activity and train/test
        if train:
            ids = moma.get_cids("act", "train")
        else:
            ids = moma.get_cids("act", "test")

        # get dataset
        dataset = []  # dataset: list of tuples[(video path, label)]
        for id in ids:
            videos_paths = moma.get_paths(id)
            for video_path in videos_paths:
                dataset.append((video_path, id))
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
        # get the mid frame index
        target_frame = num_frames // 2
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
        image = im.fromarray(image)
        # transform image:  resize, to Tensor, normalize
        if self.transform:
            image = self.transform(image)
        return image, label
