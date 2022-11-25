import os
from torch.utils.data import Dataset, DataLoader
import glob
import ffmpeg
import numpy as np
from PIL import Image as im
from pprint import pprint


def split(root_dir: str, split_id: str) -> list:
    """
    output: list of txt(split) paths
    """
    name = f"*{split_id}.txt"
    pathname = os.path.join(root_dir, "hmdb51/splits", name)
    files = glob.glob(pathname)
    return files


class HMDB51Dataset(Dataset):
    def __init__(self, root_dir: str, split_id: str, train: bool, transform=None):
        # self.mid_frame = ;
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        # get all classes names
        classes = {}
        path = os.path.join(self.root_dir, "hmdb51/videos")
        classes_list = os.listdir(path)
        idx = 0
        for c in classes_list:
            classes[c] = idx
            idx += 1

        # get train/test dataset, list of tuples
        files = split(self.root_dir, split_id)
        self.dataset = self.get_videos(files, train, classes)
        self.num_classes = len(classes)
        # pprint(dataset)

    def get_videos(self, files: list, train: bool, classes: dict) -> list:
        """
        .txt file path --> class name, video name/path, used to train or test, class label
        output: list of tuples [(video path, class label)]
        """
        dataset = []
        for f in files:
            file = open(f, 'r')
            txt_name = f.split('/')[-1]
            class_name = txt_name.rsplit('_', 2)[0]
            lines = file.readlines()
            for line in lines:
                video_name, is_train = line.split()
                video_path = os.path.join(self.root_dir, "hmdb51/videos", class_name, video_name)
                index = classes[class_name]
                if (train and is_train == '1') or (not train and is_train == '2'):
                    dataset.append((video_path, index))
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Video --> a middle frame (target frame) --> transform
        Output: tuple -- image(tensor), label(int)
        """
        path, label = self.dataset[idx]
        assert os.path.exists(path)

        # get the metadata from the video
        probe = ffmpeg.probe(path)
        video_streams = probe["streams"][0]
        # get total number of frames in the video
        num_frames = int(video_streams["nb_frames"])
        width = int(video_streams['width'])
        height = int(video_streams['height'])
        # get the mid frame index
        target_frame = num_frames//2
        # get the mid frame image
        out, _ = (
            ffmpeg
            .input(path)
            .filter("select", "gte(n,{})".format(target_frame))
            .output('pipe:', vframes=1, format='rawvideo', pix_fmt='rgb24',loglevel="quiet")
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


