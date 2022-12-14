import os
from torch.utils.data import Dataset, DataLoader
import torch
import momaapi
import json


class MOMAObjectDataset(Dataset):
    num_classes = 20

    def __init__(self, root_dir: str, train: bool):
        # Test dataset?
        # data: list of dictionaries(useful info: act_id,score,category id)
        # first, split dataset into videos(according to same act_id)
        # then one video has feature as: detected objects(category id) and their scores, a vector/matrix of col number 251
        # objects->object features, and act_ids->cids to be labels, return features,labels in get_item()

        self.train = train
        moma_path = os.path.join(root_dir, "moma-lrg")
        moma = momaapi.MOMA(moma_path)

        # get activity instance IDs (train+val or test)
        if self.train:
            act_ids = moma.get_ids_act(split="train")
            act_ids_val = moma.get_ids_act(split="val")
            act_ids.extend(act_ids_val)

            mm_train_path = os.path.join(root_dir, "mmdetection", "prediction_bboxes_train.json")
            mm_val_path = os.path.join(root_dir, "mmdetection", "prediction_bboxes_val.json")
            f_train = open(mm_train_path)
            f_val = open(mm_val_path)
            mm_data = json.load(f_train)
            mm_data_val = json.load(f_val)
            mm_data.extend(mm_data_val)
        else:
            act_ids = moma.get_ids_act(split="test")
            mm_test_path = os.path.join(root_dir, "mmdetection", "prediction_bboxes_test.json")
            f_test = open(mm_test_path)
            mm_data = json.load(f_test)

        dict = {}  # key is act_id, val is extracted feature
        # split dataset into videos:
        for data in mm_data:
            act_id = data['act_id']
            score = data['score']
            category_id = data['category_id']
            if act_id not in dict:
                dict[act_id] = [0]*251
            dict[act_id][category_id] += score

        # get annotation of activities
        anns_acts = moma.get_anns_act(act_ids)
        # get corresponding class IDs for each activity in train/test set
        cids = [ann_act.cid for ann_act in anns_acts]

        self.cids = cids
        dataset = []
        for act_id in act_ids:
            dataset.append(dict[act_id])
        self.dataset = dataset

    def __len__(self):
        return len(self.cids)

    def __getitem__(self, idx):
        label = self.cids[idx]

        feature = self.dataset[idx]
        feature = torch.LongTensor(feature).float()

        return feature, label