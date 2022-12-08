import os
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.functional import one_hot
import momaapi


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

        actor_cids = list(set([actor.cid for hoi in hois for actor in hoi.actors]))
        object_cids = list(set([object.cid for hoi in hois for object in hoi.objects]))
        ia_cids = [x+4 for x in list(set([ia.cid for hoi in hois for ia in hoi.ias]))]
        ta_cids = [x+19 for x in list(set([ta.cid for hoi in hois for ta in hoi.tas]))]
        att_cids = list(set([att.cid for hoi in hois for att in hoi.atts]))+ia_cids
        rel_cids = list(set([rel.cid for hoi in hois for rel in hoi.rels]))+ta_cids

        actor_feature = torch.sum(one_hot(torch.LongTensor(actor_cids), num_classes=26), dim=0)
        object_feature = torch.sum(one_hot(torch.LongTensor(object_cids), num_classes=227), dim=0)
        att_feature = torch.sum(one_hot(torch.LongTensor(att_cids), num_classes=13), dim=0)
        rel_feature = torch.sum(one_hot(torch.LongTensor(rel_cids), num_classes=52), dim=0)

        feature = torch.cat((actor_feature, object_feature, att_feature, rel_feature)).float()

        return feature, label
